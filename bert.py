"""BERT model
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningModule
from torch import Tensor


class LayerNorm(nn.Module):
    """Layer Normalization

    ref: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    """

    def __init__(self, dim_size: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.scale = torch.nn.parameter.Parameter(torch.ones(dim_size))
        self.bias = torch.nn.parameter.Parameter(torch.zeros(dim_size))

    def forward(self, x: Tensor) -> Tensor:
        """Normalization with scale and bias
        Args:
            x (B, L, H)

        Returns:
            (B, L, H)
        """
        mean: Tensor = x.mean(dim=-1, keepdims=True)  # (B, L, H)
        std = x.std(dim=-1, keepdims=True)  # (B, L, H)
        return self.scale * (x - mean) / (std + self.eps) + self.bias


class BERTEmbedding(nn.Module):
    """Construct the embeddings of token, position, and segment

    ref:
        - huggingface BertEmbeddings
            - https://github.com/huggingface/transformers/src/transformers/models/bert/modeling_bert.py#L166
    """

    def __init__(self, vocab_size, emb_dim, seq_len, pad_id, dropout=0.1):
        super(BERTEmbedding, self).__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(seq_len, emb_dim)  # TODO
        self.register_buffer(
            "position_ids", torch.arange(seq_len).expand((1, -1))
        )  # 3 == seg labels for sent1/2 and padding
        self.seg_emb = nn.Embedding(3, emb_dim, padding_idx=0)
        self.layer_norm = LayerNorm(emb_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sentence, segment_label):
        """
        Args:
            sentence (B, L)
            segment_label (B, L)

        Returns:
            x (B, L, H)
        """
        seq_length = sentence.size(1)
        pos_embs = self.pos_emb(self.position_ids[:, :seq_length])
        x = self.tok_emb(sentence) + pos_embs + self.seg_emb(segment_label)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    """MultiHeadAttention module

    ref: https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention
    """

    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (B, L, d_model)
            k (B, L, d_model)
            v (B, L, d_model)
            mask (B, 1, d_model, d_model)

        Returns:
            out (B, L, d_model)
            attn (B, n_h, L, L)
        """
        bs = q.shape[0]

        # split q, k, v into n_heads
        # (B, L, d_model) -> (B, L, n_h, d_k) -> (B, n_h, L, d_k)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)

        # attention
        qk = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.d_k)  # (B, n_h, L, L)
        if mask is not None:
            qk = qk.masked_fill(mask == 0, -float("inf"))  # ignore padding tokens
        attention = F.softmax(qk, dim=-1)
        attention = self.dropout(attention)
        out = torch.matmul(attention, v)  # (B, n_h, L, d_k)
        out = out.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_k)  # (B, L, d_model)
        out = self.out_linear(out)  # (B, L, d_model)
        return out, attention


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.mh = MultiHeadAttention(d_model, n_heads)
        self.dropout_ff1 = nn.Dropout(dropout)
        self.dropout_ff2 = nn.Dropout(dropout)
        self.dropout_sa = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)

    def sa_block(self, x, attn_mask):
        """Self-attention block
        Args:
            x (B, L, H)
            attn_mask (B, 1, L, L)

        Returns:
            (B, L, H)
        """
        x, _attn = self.mh(q=x, k=x, v=x, mask=attn_mask)  # (B, L, H), (B, n_h, L, L)
        return self.dropout_sa(x)

    def ff_block(self, x):
        """Feed Forward block
        f(x) = max(0, xW1 + b1)W2 + b2

        Args:
            x (B, L, d_model)
        Returns:
            (B, L, d_model)
        """
        x = F.gelu(self.linear1(x))  # (B, L, d_ff)
        return self.linear2(self.dropout_ff1(x))  # (B, L, d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x (B, L, H)
            mask (B, 1, L, L)

        Returns:
            (B, L, H)
        """
        x = self.layer_norm1(x + self.sa_block(x, mask))  # (B, L, H)
        x = self.layer_norm2(x + self.ff_block(x))  # (B, L, H)
        return x


class BERT(nn.Module):
    def __init__(self, vocab_size, seq_len, pad_id, d_model=768, d_ff=3072, n_layers=12, n_heads=12):
        super(BERT, self).__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.d_model = d_model
        self.emb = BERTEmbedding(vocab_size, d_model, seq_len, pad_id)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff) for _ in range(n_layers)]
        )

    def forward(self, x, seg):
        """
        Args:
            x (B, L)
            seg (B, L)

        Returns:
            (B, L, H)
        """
        # attention masking for padding tokens
        pad_attn_mask = (x != self.pad_id).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)  # (B, 1, L, L)

        x = self.emb(x, seg)  # (B, L, H)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, pad_attn_mask)
        return x


class NextSentencePrediction(nn.Module):
    def __init__(self, d_model):
        super(NextSentencePrediction, self).__init__()
        self.linear = nn.Linear(d_model, 2)  # 2 = next sentence or not

    def forward(self, x):
        """
        Args:
            x (B, L, H)

        Returns:
            (B, 2)
        """
        return F.log_softmax(self.linear(x[:, 0, :]), dim=-1)


class MaskedLanguageModel(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(MaskedLanguageModel, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Args:
            x (B, L, H)

        Returns:
            (B, L, V)
        """
        return F.log_softmax(self.linear(x), dim=-1)


class BERTLM(LightningModule):
    """Main BERT module."""

    def __init__(self, vocab_size, seq_len, pad_id):
        super(BERTLM, self).__init__()
        self.save_hyperparameters()

        self.bert = BERT(vocab_size=vocab_size, seq_len=seq_len, pad_id=pad_id)
        self.nsp = NextSentencePrediction(self.bert.d_model)
        self.mlm = MaskedLanguageModel(self.bert.d_model, vocab_size)
        self.mlm_criterion = torch.nn.NLLLoss(ignore_index=pad_id)
        self.nsp_criterion = torch.nn.NLLLoss()

    def forward(self, x, seg_labels):
        """
        Args:
            x (B, L)
            seg_labels (B, L)
        Returns:
            pred_ns (B,): next sentence prediction's result
            pred_vocab (B, L, V): masked language model's result
        """
        x = self.bert(x, seg_labels)  # (B, L, H)
        pred_ns = self.nsp(x)  # (B,)
        pred_vocab = self.mlm(x)  # (B, L, V)
        return pred_ns, pred_vocab

    def _calc_loss(self, batch):
        x = self.bert(batch["input_ids"], batch["segment_labels"])
        pred_ns = self.nsp(x)  # (B,)
        pred_vocab = self.mlm(x)  # (B, L, V)
        nsp_loss = self.nsp_criterion(pred_ns, batch["is_next"])
        mlm_loss = self.mlm_criterion(pred_vocab.transpose(1, 2), batch["token_labels"])
        return nsp_loss, mlm_loss

    def training_step(self, batch, batch_idx):
        nsp_loss, mlm_loss = self._calc_loss(batch)
        loss = nsp_loss + mlm_loss
        self.log("train_loss", {"nsp_loss": nsp_loss, "mlm_loss": mlm_loss}, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        nsp_loss, mlm_loss = self._calc_loss(batch)
        # loss = nsp_loss + mlm_loss
        self.log("val_loss", {"nsp_loss": nsp_loss, "mlm_loss": mlm_loss})

    def configure_optimizers(self):
        """Configure optimizer

        bias and layer_norm should be excluded from weight decay. The following link may be helpful.
        https://stats.stackexchange.com/questions/153605/no-regularisation-term-for-bias-unit-in-neural-network
        https://github.com/huggingface/transformers/blob/27c1b656cca75efa0cc414d3bf4e6aacf24829de/examples/run_lm_finetuning.py#L210

        """
        no_decay = ["bias", "layer_norm"]
        params = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.999))

        return optimizer

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        """Manual warmup learning rate"""
        warmup_step = 10_000
        if self.trainer.global_step < warmup_step:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / warmup_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * 1e-4

        # update params
        optimizer.step(closure=optimizer_closure)

    def on_epoch_end(self):
        print("epoch end")
        pass


class BERTForSequenceClassification(LightningModule):
    def __init__(self, vocab_size, seq_len, pad_id, num_labels, dropout=0.1):
        super().__init__()
        self.bert = BERT(vocab_size=vocab_size, seq_len=seq_len, pad_id=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.d_model, num_labels)

        # freeze BERT except last transformer layer
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.transformer_blocks[-1].parameters():
            param.requires_grad = True

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x, seg_labels):
        """
        Args:
            x (B, L)
            seg_labels (B, L)

        Returns:
            (B, num_labels)
        """
        out = self.bert(x, seg_labels)  # (B, L, H)
        out = self.classifier(out[:, 0, :])  # (B, num_labels)
        return F.log_softmax(out, dim=-1)

    def configure_optimizers(self):
        # set small lr to BERT's pretrained transformer layer
        return torch.optim.Adam(
            [
                {"params": self.bert.transformer_blocks[-1].parameters(), "lr": 5e-5},
                {"params": self.classifier.parameters(), "lr": 1e-4},
            ]
        )

    def training_step(self, batch, batch_idex):
        input_ids, seg_labels, y = batch["input_ids"], batch["segment_labels"], batch["label"]
        preds = self(input_ids, seg_labels)
        loss = F.nll_loss(preds, y)
        self.train_acc(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idex):
        input_ids, seg_labels, y = batch["input_ids"], batch["segment_labels"], batch["label"]
        preds = self(input_ids, seg_labels)
        loss = F.nll_loss(preds, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.valid_acc(preds, y)
        self.log("val_acc", self.valid_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idex):
        input_ids, seg_labels, y = batch["input_ids"], batch["segment_labels"], batch["label"]
        preds = self(input_ids, seg_labels)
        loss = F.nll_loss(preds, y)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.test_acc(preds, y)
        self.log("test_acc", self.test_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
