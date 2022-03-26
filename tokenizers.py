import os

import sentencepiece as spm

SYMBOLS = {
    "CLS": "<CLS>",
    "MASK": "<MASK>",
}


class SentencePieceTokenizer:
    def __init__(self, in_path, model_prefix, vocab_size, control_symbols=[], force_train=False):
        self.vocabs = {}
        control_symbols += list(SYMBOLS.values())
        model_path = model_prefix + ".model"
        # If the sentencepiece model does not exist, this code trains it.
        if force_train or not os.path.isfile(model_path):
            self.train_model(in_path, model_prefix, vocab_size, control_symbols)

        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def train_model(self, in_path, model_prefix, vocab_size, control_symbols):
        spm.SentencePieceTrainer.train(
            input=in_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            control_symbols=control_symbols,
            pad_id=3,
        )

    def encode(self, in_str: str, out_type=int, seq_len=None):
        seq = self.sp.encode(in_str, out_type=out_type)
        if seq_len:
            seq = seq[:seq_len]
            seq += seq + [self.pad_id] * [seq_len - len(seq)]
        return seq

    def decode(self, in_ids):
        return self.sp.decode(in_ids)

    @property
    def bos_id(self):
        return self.sp.bos_id()

    @property
    def eos_id(self):
        return self.sp.eos_id()

    @property
    def pad_id(self):
        return self.sp.pad_id()

    @property
    def mask_id(self):
        return self.sp.piece_to_id(SYMBOLS["MASK"])

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    @property
    def control_symbol_size(self):
        # 4 = bos/eos/unk/pad
        return 4 + len(SYMBOLS)
