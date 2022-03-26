"""Sample finetuning script to classify livedoor news corpus.
"""
import argparse

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from bert import BERTForSequenceClassification
from dataset import SingleTextLabelDatasetModule
from tokenizers import SentencePieceTokenizer

SEED = 1111


def load_bert_dict(bert_for_cls, ckpt_path):
    bert_for_cls_dict = bert_for_cls.state_dict()
    print("before", bert_for_cls_dict["bert.emb.tok_emb.weight"][0, :8])
    bert_dict = bert_for_cls.state_dict()

    # load whole bert model with MLM/NSP layers
    print("Load", ckpt_path)
    pretrained_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))

    # filter out MLM/NSP
    pretrained_dict = {k: v for k, v in pretrained_dict["state_dict"].items() if k in bert_dict}
    assert pretrained_dict, "failed to load pretrained bert dict"

    # update parameters
    bert_for_cls_dict.update(pretrained_dict)
    bert_for_cls.load_state_dict(bert_for_cls_dict)

    print("after", bert_for_cls_dict["bert.emb.tok_emb.weight"][0, :8])

    return bert_for_cls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/jawiki.yaml",
        help="configuration yaml file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="checkpoint model path",
    )
    parser.add_argument(
        "-e",
        "--exp",
        type=str,
        default="livedoor",
        help="experiment name",
    )
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    torch.manual_seed(SEED)

    tp = config["training_params"]
    tokenizer = SentencePieceTokenizer(
        in_path=tp["train_f"],
        model_prefix=tp["model_prefix"],
        vocab_size=config["model_params"]["vocab_size"],
    )
    all_label = (
        "dokujo-tsushin",
        "it-life-hack",
        "kaden-channel",
        "livedoor-homme",
        "movie-enter",
        "peachy",
        "smax",
        "sports-watch",
        "topic-news",
    )
    num_labels = len(set(all_label))

    model = BERTForSequenceClassification(
        config["model_params"]["vocab_size"], seq_len=tp["seq_len"], pad_id=tokenizer.pad_id, num_labels=num_labels
    )
    model = load_bert_dict(model, args.checkpoint)

    l2i = {l: i for i, l in enumerate(set(all_label))}

    datamodule = SingleTextLabelDatasetModule(
        train_f="corpora/livedoor/train.tsv",
        dev_f="corpora/livedoor/dev.tsv",
        test_f="corpora/livedoor/test.tsv",
        batch_size=tp["batch_size"],
        n_workers=0,
        tokenizer=tokenizer,
        l2i=l2i,
    )
    exp = args.exp
    logger = TensorBoardLogger(save_dir="experiments", version=exp)
    trainer = Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=10,
        log_every_n_steps=tp["log_every_n_steps"],
        logger=logger,
        default_root_dir=f"./experiments/default/{exp}",
    )
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
