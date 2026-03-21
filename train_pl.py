#!/usr/bin/env python3
"""Training script for PARSeq with Pseudo-Labeling (PL) extended charset.

Key differences from train.py:
1. charset_train is extended with Unicode variant chars from unicode_mapping.json
2. Training data comes from PL LMDB (normalize_unicode=False to preserve extended chars)
3. Validation/test data uses original LMDB with standard charset
"""
import json
import math
import random

from pathlib import Path, PurePath

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

import torch

import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize

from strhub.data.dataset import build_tree_dataset
from strhub.data.module import SceneTextDataModule
from strhub.data.utils import PLCharsetAdapter
from strhub.models.base import BaseSystem
from strhub.models.utils import get_pretrained_weights


# Copied from OneCycleLR
def _annealing_cos(start, end, pct):
    'Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0.'
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


def get_swa_lr_factor(warmup_pct, swa_epoch_start, div_factor=25, final_div_factor=1e4) -> float:
    """Get the SWA LR factor for the given `swa_epoch_start`. Assumes OneCycleLR Scheduler."""
    total_steps = 1000  # Can be anything. We use 1000 for convenience.
    start_step = int(total_steps * warmup_pct) - 1
    end_step = total_steps - 1
    step_num = int(total_steps * swa_epoch_start) - 1
    pct = (step_num - start_step) / (end_step - start_step)
    return _annealing_cos(1, 1 / (div_factor * final_div_factor), pct)


def build_pl_charset(base_charset: str, unicode_mapping_path: str) -> tuple[str, dict]:
    """Extend charset_train with Unicode variant characters from unicode_mapping.json.

    Returns:
        (extended_charset, ext_to_base) where ext_to_base maps each Unicode variant to its base char.
    """
    with open(unicode_mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    ext_chars = ''.join(v['unicode'] for v in mapping.values())
    ext_to_base = {v['unicode']: v['base_char'] for v in mapping.values()}
    return base_charset + ext_chars, ext_to_base


class ValPredictionLogger(Callback):
    """Logs 5 random pred/gt pairs with images as a wandb table each validation."""

    def __init__(self):
        super().__init__()
        self.val_data = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        images, labels = batch
        with torch.no_grad():
            logits = pl_module(images)
        probs = logits.softmax(-1)
        preds, _ = pl_module.tokenizer.decode(probs)
        for img, pred_raw, gt in zip(images, preds, labels):
            pred_mapped = pl_module.charset_adapter(pred_raw)
            # Denormalize image (was normalized with mean=0.5, std=0.5)
            img = (img.cpu() * 0.5 + 0.5).clamp(0, 1)
            self.val_data.append((img, gt, pred_raw, pred_mapped))

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.val_data or not trainer.logger:
            return
        samples = random.sample(self.val_data, min(5, len(self.val_data)))
        table = wandb.Table(columns=['image', 'gt', 'pred', 'pred_raw', 'correct'])
        for img, gt, pred_raw, pred_mapped in samples:
            table.add_data(wandb.Image(img), gt, pred_mapped, pred_raw, gt == pred_mapped)
        trainer.logger.experiment.log({'val_samples': table}, step=trainer.global_step)
        self.val_data.clear()


class PLSceneTextDataModule(SceneTextDataModule):
    """Data module that uses PL LMDB for training (with normalize_unicode=False)
    and original LMDB for validation/test.

    If use_pl_data=False, falls back to standard SceneTextDataModule training data.
    """

    def __init__(self, pl_root_dir: str, use_pl_data: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.pl_root_dir = pl_root_dir
        self.use_pl_data = use_pl_data

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            transform = self.get_transform(self.img_size, self.augment)
            if self.use_pl_data:
                root = PurePath(self.pl_root_dir, 'train', self.train_dir)
                self._train_dataset = build_tree_dataset(
                    root,
                    self.charset_train,
                    self.max_label_length,
                    self.min_image_dim,
                    self.remove_whitespace,
                    False,  # normalize_unicode=False to preserve extended chars
                    transform=transform,
                )
            else:
                root = PurePath(self.root_dir, 'train', self.train_dir)
                self._train_dataset = build_tree_dataset(
                    root,
                    self.charset_train,
                    self.max_label_length,
                    self.min_image_dim,
                    self.remove_whitespace,
                    self.normalize_unicode,
                    transform=transform,
                )
        return self._train_dataset


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):
    trainer_strategy = 'auto'
    project_root = Path(hydra.utils.get_original_cwd())

    def resolve_path(p):
        """Resolve path: ~ → home, relative → project root, absolute → as-is."""
        p = Path(p).expanduser()
        if not p.is_absolute():
            p = project_root / p
        return str(p)

    with open_dict(config):
        # Validation every 1000 steps for PL training
        config.trainer.val_check_interval = 100
        # Data paths (~/data/STR/... or absolute)
        config.data.root_dir = resolve_path(config.data.root_dir)
        # PL config: pl_root_dir defaults to data.root_dir + '/PL'
        if not config.get('pl_root_dir'):
            config.pl_root_dir = config.data.root_dir + '/PL'
        else:
            config.pl_root_dir = resolve_path(config.pl_root_dir)
        # Unicode mapping (relative to project root)
        if not config.get('unicode_mapping'):
            config.unicode_mapping = 'confusion_pl_output/unicode_mapping.json'
        config.unicode_mapping = resolve_path(config.unicode_mapping)
        # Whether to use PL LMDB for training (default: True)
        if not config.get('use_pl_data'):
            config.use_pl_data = True
        # Special handling for GPU-affected config
        gpu = config.trainer.get('accelerator') == 'gpu'
        devices = config.trainer.get('devices', 0)
        if gpu:
            config.trainer.precision = 'bf16-mixed' if torch.get_autocast_gpu_dtype() is torch.bfloat16 else '16-mixed'
        if gpu and devices > 1:
            trainer_strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
            config.trainer.val_check_interval //= devices
            if config.trainer.get('max_steps', -1) > 0:
                config.trainer.max_steps //= devices

    # Special handling for PARseq
    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'

    # Extend charset_train with PL Unicode variants
    pl_charset, ext_to_base = build_pl_charset(config.model.charset_train, config.unicode_mapping)
    print(f'Base charset_train size: {len(config.model.charset_train)}')
    print(f'Extended charset_train size: {len(pl_charset)} (+{len(pl_charset) - len(config.model.charset_train)} PL chars)')
    with open_dict(config):
        config.model.charset_train = pl_charset

    model: BaseSystem = hydra.utils.instantiate(config.model)
    # If specified, use pretrained weights to initialize the model (partial load)
    if config.pretrained is not None:
        pretrained_state = get_pretrained_weights(config.pretrained)
        m = model.model if config.model._target_.endswith('PARSeq') else model
        # Filter out mismatched keys (e.g. head layer size changed due to extended charset)
        model_state = m.state_dict()
        filtered_state = {
            k: v for k, v in pretrained_state.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        skipped = set(pretrained_state.keys()) - set(filtered_state.keys())
        if skipped:
            print(f'Skipped {len(skipped)} pretrained keys due to shape mismatch: {skipped}')
        m.load_state_dict(filtered_state, strict=False)
    # Replace charset_adapter with PLCharsetAdapter (extended → base char mapping for eval)
    model.charset_adapter = PLCharsetAdapter(config.data.charset_test, ext_to_base)
    print(summarize(model, max_depth=2))

    # Create PL data module (PL LMDB for train, original LMDB for val/test)
    print(f'PL train data root: {config.pl_root_dir}')
    print(f'Val/test data root: {config.data.root_dir}')
    print(f'use_pl_data: {config.use_pl_data}')
    datamodule = PLSceneTextDataModule(
        pl_root_dir=config.pl_root_dir,
        use_pl_data=config.use_pl_data,
        root_dir=config.data.root_dir,
        train_dir=config.data.train_dir,
        img_size=config.data.img_size,
        max_label_length=config.data.max_label_length,
        charset_train=config.model.charset_train,
        charset_test=config.data.charset_test,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        augment=config.data.augment,
        remove_whitespace=config.data.get('remove_whitespace', True),
        normalize_unicode=config.data.get('normalize_unicode', True),
    )

    cwd = (
        HydraConfig.get().runtime.output_dir
        if config.ckpt_path is None
        else str(Path(config.ckpt_path).parents[1].absolute())
    )
    checkpoint = ModelCheckpoint(
        monitor='val_accuracy',
        mode='max',
        save_top_k=3,
        save_last=True,
        filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}',
        dirpath=cwd + '/checkpoints',
    )
    swa_epoch_start = 0.75
    swa_lr = config.model.lr * get_swa_lr_factor(config.model.warmup_pct, swa_epoch_start)
    swa = StochasticWeightAveraging(swa_lr, swa_epoch_start)
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=WandbLogger(
            project=config.wandb.project,
            group=config.wandb.group,
            name=config.wandb.name,
            save_dir=cwd,
        ),
        strategy=trainer_strategy,
        enable_model_summary=False,
        callbacks=[checkpoint, swa, ValPredictionLogger()],
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)


if __name__ == '__main__':
    main()
