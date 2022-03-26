"""Masked Language Model example.
"""
import torch
import argparse

import yaml
import numpy as np
from bert import BERTLM
from tokenizers import SentencePieceTokenizer
import os


def fill_mask(bertlm, tokenizer, in_text, mask_pos):
    if type(in_text) == str:
        s_list = tokenizer.encode(in_text, str)

    label = s_list[mask_pos]
    label_id = tokenizer.sp.piece_to_id(label)

    s_list[mask_pos] = "<MASK>"
    x = torch.tensor(tokenizer.sp.piece_to_id(s_list)).unsqueeze(0)
    seg = torch.ones(1, len(x)).long()
    pred_ns, pred_vocab = bertlm(x, seg)
    pred_vocab = pred_vocab[0]
    pred_vocab.shape
    pred_id = torch.argmax(pred_vocab[mask_pos]).item()
    pred = tokenizer.sp.decode([pred_id])
    pred_prob = pred_vocab[mask_pos][pred_id].exp().item()
    label_prob = pred_vocab[mask_pos][label_id].exp().item()

    return {
        "pred": {
            "text": pred,
            "prob": pred_prob,
        },
        "label": {"text": label, "prob": label_prob, "nll": -np.log(label_prob)},
    }


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
        help="model checkpoint path",
    )
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        help="input text",
        default="日本の首都は東京で，アメリカの首都はワシントンです",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    tp = config["training_params"]
    tokenizer = SentencePieceTokenizer(
        in_path=tp["train_f"], model_prefix=tp["model_prefix"], vocab_size=config["model_params"]["vocab_size"]
    )

    assert os.path.isfile(args.checkpoint)
    bertlm = BERTLM.load_from_checkpoint(args.checkpoint)
    bertlm.eval()
    with torch.no_grad():
        s_list = tokenizer.encode(args.text, str)
        for i in range(len(s_list)):
            res = fill_mask(bertlm, tokenizer, args.text, i)
            masked = s_list[:]
            masked[i] = f"pred:\"{res['pred']['text']}\" / orig:\"{masked[i]}\""
            print(masked)


if __name__ == "__main__":
    main()
