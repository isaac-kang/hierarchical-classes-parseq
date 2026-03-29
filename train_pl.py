#!/usr/bin/env python3
"""Training script for PARSeq with Pseudo-Labeling (PL) extended charset.

Key differences from train.py:
1. charset_train is extended with Unicode variant chars from unicode_mapping.json
2. Training data comes from PL LMDB (normalize_unicode=False to preserve extended chars)
3. Validation/test data uses original LMDB with standard charset
"""
import glob
import json
import math
import os
import random

from pathlib import Path, PurePath

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

import torch
import torch.nn.functional as F

import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize

from torch.utils.data import DataLoader

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


def compute_token_freqs(datamodule, tokenizer, num_classes):
    """Count per-token frequency from training labels. Returns a 1-D tensor of counts."""
    freqs = torch.zeros(num_classes)
    dataset = datamodule.train_dataset
    for ds in dataset.datasets:
        for label in ds.labels:
            for c in label:
                idx = tokenizer._stoi.get(c)
                if idx is not None:
                    freqs[idx] += 1
    # EOS appears once per sample
    freqs[tokenizer.eos_id] = sum(len(ds.labels) for ds in dataset.datasets)
    return freqs


def patch_training_step(model, use_balanced_softmax=False, pairwise_margin=0.0, pairwise_weight=1.0,
                        confusion_margin=0.0, confusion_weight=0.0,
                        ext_to_base_map=None, ext_to_sim_char=None):
    """Monkey-patch model.training_step with optional balanced softmax and/or pairwise/confusion margin loss.

    Buffers used (must be registered before calling):
      - model.log_prior: (num_head_classes,) log-frequency prior for balanced softmax
      - model._is_ext:   (num_head_classes,) bool mask for ext token ids (ext class mode)
      - model._ext_to_sim: (num_head_classes,) maps ext token id -> confused-with token id (ext class mode)

    Args:
      ext_to_base_map: dict mapping ext char -> base char (for no-ext-class mode)
      ext_to_sim_char: dict mapping ext char -> confused_with char (for no-ext-class mode)
    """

    def _patched_training_step(batch, batch_idx):
        images, labels = batch

        # No-ext-class mode: detect ext positions in labels, replace with base chars
        # confusion_positions[sample_idx] = list of (char_pos, base_token_id, sim_token_id)
        confusion_positions = None
        if ext_to_base_map is not None and confusion_margin > 0:
            stoi = model.tokenizer._stoi
            confusion_positions = []
            cleaned_labels = []
            for label in labels:
                positions = []
                cleaned = []
                for ci, ch in enumerate(label):
                    if ch in ext_to_base_map:
                        base_ch = ext_to_base_map[ch]
                        sim_ch = ext_to_sim_char.get(ch)
                        base_id = stoi.get(base_ch)
                        sim_id = stoi.get(sim_ch) if sim_ch else None
                        if base_id is not None and sim_id is not None:
                            positions.append((ci, base_id, sim_id))
                        cleaned.append(base_ch)
                    else:
                        cleaned.append(ch)
                confusion_positions.append(positions)
                cleaned_labels.append(''.join(cleaned))
            labels = cleaned_labels
        elif ext_to_base_map is not None:
            # No confusion margin but still need to clean labels
            cleaned_labels = []
            for label in labels:
                cleaned_labels.append(''.join(ext_to_base_map.get(ch, ch) for ch in label))
            labels = cleaned_labels

        tgt = model.tokenizer.encode(labels, model._device)

        memory = model.model.encode(images)

        tgt_perms = model.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_padding_mask = (tgt_in == model.pad_id) | (tgt_in == model.eos_id)

        loss = 0
        loss_numel = 0
        n = (tgt_out != model.pad_id).sum().item()
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = model.generate_attn_masks(perm)
            out = model.model.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
            logits = model.model.head(out).flatten(end_dim=1)

            # CE loss (optionally balanced)
            if use_balanced_softmax:
                ce = F.cross_entropy(logits + model.log_prior, tgt_out.flatten(), ignore_index=model.pad_id)
            else:
                ce = F.cross_entropy(logits, tgt_out.flatten(), ignore_index=model.pad_id)
            loss += n * ce

            # Pairwise margin loss: max(0, m + z_sim - z_ext) for ext targets (ext class mode only)
            if pairwise_margin > 0:
                targets_flat = tgt_out.flatten()
                num_classes = logits.shape[1]
                valid = targets_flat < num_classes
                ext_mask = valid & model._is_ext[targets_flat.clamp(max=num_classes - 1)]
                if ext_mask.any():
                    pos = ext_mask.nonzero(as_tuple=True)[0]
                    tgt_ids = targets_flat[pos]
                    sim_ids = model._ext_to_sim[tgt_ids]
                    z_ext = logits[pos, tgt_ids]
                    z_sim = logits[pos, sim_ids]
                    m_loss = F.relu(pairwise_margin + z_sim - z_ext).mean()
                    loss += n * pairwise_weight * m_loss

            # Confusion margin loss: PL-guided, only at positions where PL predicted ext char
            if confusion_margin > 0 and confusion_positions is not None:
                seq_len = tgt_out.shape[1]
                all_base_ids = []
                all_sim_ids = []
                all_flat_pos = []
                for si, positions in enumerate(confusion_positions):
                    for ci, base_id, sim_id in positions:
                        if ci < seq_len:
                            flat_idx = si * seq_len + ci
                            all_flat_pos.append(flat_idx)
                            all_base_ids.append(base_id)
                            all_sim_ids.append(sim_id)
                if all_flat_pos:
                    flat_pos = torch.tensor(all_flat_pos, device=logits.device)
                    base_ids = torch.tensor(all_base_ids, device=logits.device)
                    sim_ids = torch.tensor(all_sim_ids, device=logits.device)
                    z_base = logits[flat_pos, base_ids]
                    z_sim = logits[flat_pos, sim_ids]
                    c_loss = F.relu(confusion_margin - (z_base - z_sim)).mean()
                    loss += n * confusion_weight * c_loss

            loss_numel += n
            if i == 1:
                tgt_out = torch.where(tgt_out == model.eos_id, model.pad_id, tgt_out)
                n = (tgt_out != model.pad_id).sum().item()
        loss /= loss_numel

        model.log('loss', loss)
        return loss

    import types
    model.training_step = types.MethodType(
        lambda self, batch, batch_idx: _patched_training_step(batch, batch_idx), model
    )


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

    def __init__(self, pl_root_dir: str, use_pl_data: bool = True,
                 ext_chars: set[str] = None, oversample_factor: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.pl_root_dir = pl_root_dir
        self.use_pl_data = use_pl_data
        self.ext_chars = ext_chars or set()
        self.oversample_factor = oversample_factor

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

    def train_dataloader(self):
        if self.oversample_factor <= 1.0 or not self.ext_chars:
            return super().train_dataloader()

        from torch.utils.data import WeightedRandomSampler

        dataset = self.train_dataset
        # Build per-sample weights from ConcatDataset
        weights = []
        for ds in dataset.datasets:
            for label in ds.labels:
                has_ext = any(c in self.ext_chars for c in label)
                weights.append(self.oversample_factor if has_ext else 1.0)

        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )


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
        config.trainer.val_check_interval = 1000
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
        if 'use_pl_data' not in config:
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

    # Extend charset_train with PL Unicode variants (unless use_ext_classes=false)
    use_ext_classes = config.get('use_ext_classes', True)
    pl_charset, ext_to_base = build_pl_charset(config.model.charset_train, config.unicode_mapping)
    print(f'Base charset_train size: {len(config.model.charset_train)}')
    if use_ext_classes:
        print(f'Extended charset_train size: {len(pl_charset)} (+{len(pl_charset) - len(config.model.charset_train)} PL chars)')
        with open_dict(config):
            config.model.charset_train = pl_charset
    else:
        print(f'use_ext_classes=false: keeping base charset, ext chars used for confusion margin only')

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
    _train_root = (
        str(PurePath(config.pl_root_dir, 'train', config.data.train_dir))
        if config.use_pl_data
        else str(PurePath(config.data.root_dir, 'train', config.data.train_dir))
    )
    print(f'Train data root: {_train_root}')
    print(f'Val data root:   {PurePath(config.data.root_dir, "val")}')
    print(f'use_pl_data: {config.use_pl_data}')
    oversample_factor = config.get('oversample_factor', 1.0)
    ext_chars = set(ext_to_base.keys())
    print(f'Ext char oversampling: {oversample_factor}x ({len(ext_chars)} ext chars)')
    # Datamodule always uses pl_charset so ext chars in labels are not filtered out
    datamodule = PLSceneTextDataModule(
        pl_root_dir=config.pl_root_dir,
        use_pl_data=config.use_pl_data,
        ext_chars=ext_chars,
        oversample_factor=oversample_factor,
        root_dir=config.data.root_dir,
        train_dir=config.data.train_dir,
        img_size=config.data.img_size,
        max_label_length=config.data.max_label_length,
        charset_train=pl_charset,
        charset_test=config.data.charset_test,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        augment=config.data.augment,
        remove_whitespace=config.data.get('remove_whitespace', True),
        normalize_unicode=config.data.get('normalize_unicode', True),
    )

    # --- Loss patches: Balanced Softmax & Pairwise Margin & Confusion Margin ---
    use_balanced_softmax = config.get('balanced_softmax', False)
    pairwise_margin = config.get('pairwise_margin', 0.0)
    pairwise_weight = config.get('pairwise_weight', 1.0)
    confusion_margin = config.get('confusion_margin', 0.0)
    confusion_weight = config.get('confusion_weight', 0.0)
    num_classes = len(model.tokenizer)
    num_head_classes = num_classes - 2  # head excludes BOS and PAD

    # Balanced Softmax: compute log-prior from training data token frequencies
    if use_balanced_softmax:
        freqs = compute_token_freqs(datamodule, model.tokenizer, num_classes)
        # Drop BOS and PAD indices (higher first to preserve lower indices)
        exclude_ids = sorted([model.tokenizer.bos_id, model.tokenizer.pad_id], reverse=True)
        head_freqs = freqs.clone()
        for idx in exclude_ids:
            head_freqs = torch.cat([head_freqs[:idx], head_freqs[idx + 1:]])
        log_prior = torch.log(head_freqs + 1)  # +1 smoothing to avoid log(0)
        model.register_buffer('log_prior', log_prior)
        # Print stats
        ext_indices = [model.tokenizer._stoi[c] for c in ext_chars if c in model.tokenizer._stoi]
        base_indices = [i for i in range(num_classes) if i not in set(ext_indices)
                        and i not in {model.tokenizer.bos_id, model.tokenizer.eos_id, model.tokenizer.pad_id}]
        ext_freq = freqs[ext_indices].sum().item() if ext_indices else 0
        base_freq = freqs[base_indices].sum().item() if base_indices else 0
        total_freq = freqs.sum().item()
        print(f'Balanced Softmax: enabled ({num_head_classes} head classes)')
        print(f'  Token freqs — base: {base_freq:.0f} ({100*base_freq/total_freq:.1f}%), '
              f'ext: {ext_freq:.0f} ({100*ext_freq/total_freq:.1f}%), total: {total_freq:.0f}')
    else:
        print('Balanced Softmax: disabled')

    # Pairwise Margin Loss: build ext_id -> sim_id lookup
    if pairwise_margin > 0:
        with open(config.unicode_mapping, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        stoi = model.tokenizer._stoi
        ext_to_sim = torch.zeros(num_head_classes, dtype=torch.long)
        is_ext = torch.zeros(num_head_classes, dtype=torch.bool)
        pair_count = 0
        for entry in mapping.values():
            ext_id = stoi.get(entry['unicode'])
            sim_id = stoi.get(entry['confused_with'])
            if ext_id is not None and sim_id is not None and ext_id < num_head_classes and sim_id < num_head_classes:
                ext_to_sim[ext_id] = sim_id
                is_ext[ext_id] = True
                pair_count += 1
        model.register_buffer('_ext_to_sim', ext_to_sim)
        model.register_buffer('_is_ext', is_ext)
        print(f'Pairwise Margin Loss: margin={pairwise_margin}, weight={pairwise_weight}, {pair_count} pairs')
    else:
        print('Pairwise Margin Loss: disabled')

    # Confusion Margin Loss setup
    ext_to_base_map = None
    ext_to_sim_char = None
    if confusion_margin > 0:
        if use_ext_classes:
            # Ext class mode: not supported (confusion margin is for no-ext-class mode)
            print('WARNING: confusion_margin with use_ext_classes=true is not supported. Ignoring.')
            confusion_margin = 0.0
        else:
            # No-ext-class mode: build ext_char -> sim_char mapping for PL-guided margin
            with open(config.unicode_mapping, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            ext_to_base_map = ext_to_base  # already built by build_pl_charset
            ext_to_sim_char = {v['unicode']: v['confused_with'] for v in mapping.values()}
            print(f'Confusion Margin Loss (PL-guided): margin={confusion_margin}, weight={confusion_weight}, '
                  f'{len(ext_to_sim_char)} ext->sim pairs')
    else:
        print('Confusion Margin Loss: disabled')

    # No-ext-class mode still needs ext_to_base_map to clean labels even without confusion margin
    if not use_ext_classes and ext_to_base_map is None:
        ext_to_base_map = ext_to_base

    # Apply monkey-patch if any loss modification is active
    if use_balanced_softmax or pairwise_margin > 0 or confusion_margin > 0 or ext_to_base_map is not None:
        patch_training_step(model, use_balanced_softmax, pairwise_margin, pairwise_weight,
                            confusion_margin, confusion_weight,
                            ext_to_base_map=ext_to_base_map if not use_ext_classes else None,
                            ext_to_sim_char=ext_to_sim_char)

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
    def list_lmdbs(root: str) -> list[str]:
        return sorted(
            str(Path(p).parent.relative_to(root))
            for p in glob.glob(str(Path(root) / '**/data.mdb'), recursive=True)
        )

    train_root = _train_root
    val_root = str(PurePath(config.data.root_dir, 'val'))

    overrides = HydraConfig.get().overrides.task
    wandb_logger = WandbLogger(
        project=config.wandb.project,
        group=config.wandb.group,
        name=config.wandb.name,
        save_dir=cwd,
        log_model=False,
    )
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        wandb_logger.experiment.config.update({
            'overrides': list(overrides),
            'dataset/train_root': train_root,
            'dataset/train_lmdbs': list_lmdbs(train_root),
            'dataset/val_root': val_root,
            'dataset/val_lmdbs': list_lmdbs(val_root),
        })
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=wandb_logger,
        strategy=trainer_strategy,
        enable_model_summary=False,
        callbacks=[checkpoint, swa, ValPredictionLogger()],
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)


if __name__ == '__main__':
    main()
