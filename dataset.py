import random
from subprocess import check_output

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class BERTDataset(Dataset):
    def __init__(self, tokenizer, text_file, seq_len, encoding="utf-8"):
        self.tokenizer = tokenizer
        self.text_file = text_file
        self.seq_len = seq_len
        self.encoding = encoding
        self.n_text_lines = int(check_output(["wc", "-l", text_file]).split()[0])

        self.text_list = []
        print("opening", text_file, flush=True)
        with open(text_file, "r", encoding=encoding) as f:
            for line in tqdm(f.read().splitlines(), total=self.n_text_lines):
                if len(line) > 0 and not line.isspace():
                    self.text_list.append(line)

    def __len__(self):
        return self.n_text_lines // 2  # two lines works as an example

    def get_corpus_line(self, index, random_line=False):
        if random_line:
            s1 = self.text_list[index]
            s2 = self.text_list[random.randrange(len(self.text_list))]
        else:
            # TODO adhoc
            if index == len(self.text_list) - 1:
                index -= 1
            s1 = self.text_list[index]
            s2 = self.text_list[index + 1]

        s1_tokens = self.tokenizer.encode(s1.strip())
        s2_tokens = self.tokenizer.encode(s2.strip())

        return s1_tokens, s2_tokens

    def random_next_sent(self, index):
        random_line = True if random.random() > 0.5 else False
        s1_tokens, s2_tokens = self.get_corpus_line(index, random_line)
        is_next = int(not random_line)
        return (s1_tokens, s2_tokens), is_next

    def random_mask_word(self, tokens):
        """
        The training data generator
        chooses 15% of the token positions at random for
        prediction. If the i-th token is chosen, we replace
        the i-th token with (1) the [MASK] token 80% of
        the time (2) a random token 10% of the time (3)
        the unchanged i-th token 10% of the time. T

        Args:
            tokens (L,)
        Returns:
            (L,)
        """
        token_labels = []
        for i, tok in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15  # normalize to 1.0
                if prob < 0.8:  # replace the word with the [MASK] token
                    tokens[i] = self.tokenizer.mask_id
                elif prob < 0.9:  # Replace the word with a random word
                    tokens[i] = random.randrange(self.tokenizer.control_symbol_size, self.tokenizer.vocab_size)
                else:  # Keep the word unchanged
                    pass
                token_labels.append(tok)
            else:
                token_labels.append(self.tokenizer.pad_id)
        return tokens, token_labels

    def __getitem__(self, index):
        (s1_tokens, s2_tokens), is_next = self.random_next_sent(index)
        s1_tokens, s1_mask_labels = self.random_mask_word(s1_tokens)
        s2_tokens, s2_mask_labels = self.random_mask_word(s2_tokens)

        # concat s1 and s2
        # [BOS] + s1_tokens + [EOS] + s2_tokens + [EOS]
        t1 = [self.tokenizer.bos_id] + s1_tokens + [self.tokenizer.eos_id]
        t2 = s2_tokens + [self.tokenizer.eos_id]
        input_ids = (t1 + t2)[: self.seq_len]
        s1_mask_labels = [self.tokenizer.pad_id] + s1_mask_labels + [self.tokenizer.pad_id]
        s2_mask_labels = s2_mask_labels + [self.tokenizer.pad_id]
        token_labels = (s1_mask_labels + s2_mask_labels)[: self.seq_len]

        segment_labels = [1] * len(t1) + [2] * len(t2)
        segment_labels = segment_labels[: self.seq_len]

        # add padding
        padding = [self.tokenizer.pad_id] * max(0, self.seq_len - len(input_ids))
        input_ids.extend(padding), token_labels.extend(padding)
        zero_padding = [0] * len(padding)
        segment_labels.extend(zero_padding)

        return {
            "input_ids": torch.tensor(input_ids),
            "token_labels": torch.tensor(token_labels),
            "segment_labels": torch.tensor(segment_labels),
            "is_next": is_next,
        }


class BERTDatasetModule(LightningDataModule):
    def __init__(self, train_f, seq_len, batch_size, n_workers, tokenizer):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_set = BERTDataset(
            tokenizer=self.hparams.tokenizer, text_file=self.hparams.train_f, seq_len=self.hparams.seq_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.n_workers,
            shuffle=True,
        )


class SingleTextLabelDataset(Dataset):
    def __init__(self, text_file, tokenizer, l2i, seq_len=128, encoding="utf-8"):
        self.text_list = []
        self.tokenizer = tokenizer
        self.l2i = l2i
        print(self.l2i)
        self.seq_len = seq_len
        with open(text_file, "r", encoding=encoding) as f:
            next(f)
            for line in f.read().splitlines():
                line = line.split("\t")
                if len(line) != 2:
                    continue
                    print(f"{len(line)}, {line}")  # text\tlabel
                # assert len(line) == 2, f'{len(line)}, {line}' # text\tlabel
                text, label = line
                self.text_list.append((text, label))

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, index):
        text, label = self.text_list[index]
        input_ids = self.tokenizer.encode(text.strip())[: self.seq_len]

        segment_labels = [1] * len(input_ids)
        segment_labels = segment_labels[: self.seq_len]

        padding = [self.tokenizer.pad_id] * max(0, self.seq_len - len(input_ids))
        input_ids.extend(padding)
        zero_padding = [0] * len(padding)
        segment_labels.extend(zero_padding)

        label = self.l2i[label]
        return {
            "input_ids": torch.tensor(input_ids),
            "segment_labels": torch.tensor(segment_labels),
            "label": torch.tensor(label),
        }


class SingleTextLabelDatasetModule(LightningDataModule):
    def __init__(self, train_f, dev_f, test_f, batch_size, l2i, n_workers, tokenizer):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_set = SingleTextLabelDataset(self.hparams.train_f, self.hparams.tokenizer, self.hparams.l2i)
        self.dev_set = SingleTextLabelDataset(self.hparams.dev_f, self.hparams.tokenizer, self.hparams.l2i)
        self.test_set = SingleTextLabelDataset(self.hparams.test_f, self.hparams.tokenizer, self.hparams.l2i)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.n_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.n_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.n_workers,
            shuffle=False,
        )
