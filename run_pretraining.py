import argparse
import os
import re

import torch
import yaml
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from bert import BERTLM
from dataset import BERTDatasetModule
from tokenizers import SentencePieceTokenizer

SEED = 1111


class CheckpointEveryNSteps(Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    https://github.com/PyTorchLightning/pytorch-lightning/issues/2534#issuecomment-674582085
    """

    def __init__(
        self,
        ckpt_save_interval,
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.ckpt_save_interval = ckpt_save_interval
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: Trainer, _):
        """Check if we should save a checkpoint after every train batch"""
        # epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.ckpt_save_interval == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"gs={global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            # print('save', ckpt_path)
            trainer.save_checkpoint(ckpt_path)


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
        "-r",
        "--resume",
        type=str,
        help="resumed checkpoint model path",
    )
    parser.add_argument(
        "-e",
        "--exp",
        type=str,
        default="jawiki",
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

    # I only prepare training dataset. The training data is basically quite huge
    # and preparing the validation dataset may not be a good way to evaluate the model.
    dataset_module = BERTDatasetModule(
        batch_size=tp["batch_size"],
        train_f=tp["train_f"],
        seq_len=tp["seq_len"],
        n_workers=tp["n_workers"],
        tokenizer=tokenizer,
    )

    model_path = None
    if args.resume and os.path.isdir(args.resume):
        # Retrieve the largest global step model.
        max_step = 0
        for ckpt_name in os.listdir(os.path.join(args.resume, "checkpoints")):
            match = re.search(r"gs=(\d+)\.ckpt", ckpt_name)
            if match:
                step = int(match[1])
                if step > max_step:
                    max_step = step
                    model_path = os.path.join(args.resume, "checkpoints", ckpt_name)
        assert model_path is not None
        print("resume", model_path)
        bertlm = BERTLM.load_from_checkpoint(model_path)
    else:
        bertlm = BERTLM(
            vocab_size=config["model_params"]["vocab_size"], pad_id=tokenizer.pad_id, seq_len=tp["seq_len"]
        )
    logger = TensorBoardLogger(save_dir="experiments", version=args.exp)
    print("train_params:", tp)
    trainer = Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=tp["max_epochs"],
        log_every_n_steps=tp["log_every_n_steps"],
        logger=logger,
        default_root_dir=f"./experiments/default/{args.exp}",
        callbacks=[CheckpointEveryNSteps(ckpt_save_interval=tp["ckpt_save_interval"])],
    )
    trainer.fit(bertlm, dataset_module, ckpt_path=model_path)


if __name__ == "__main__":
    main()
