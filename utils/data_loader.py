"""Data loading and augmentation utilities for Phase 3a (Warm-up Training).

Provides:
- get_transforms(split): returns torchvision transform pipeline for 'train'/'val'/'test'
- EmotionDataset: custom dataset reading from data_splits/<split>/<emotion>/...jpg
- build_dataloaders: builds train/val/test DataLoader objects with class imbalance handling

Assumes directory structure:
 data_splits/
   train/angry/*.jpg
   val/angry/*.jpg
   test/angry/*.jpg
   ... other emotion folders

Class labels order imported from config.EMOTION_LABELS.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from config import EMOTION_LABELS, INPUT_SIZE, AUGMENTATION, BATCH_SIZE

# Import both augmentation configs for staged training
try:
    from config import AUGMENTATION_WARMUP, AUGMENTATION_FINETUNE
    _HAS_STAGED_AUGMENTATION = True
except ImportError:
    _HAS_STAGED_AUGMENTATION = False
    AUGMENTATION_WARMUP = AUGMENTATION
    AUGMENTATION_FINETUNE = AUGMENTATION


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str, augmentation_mode: str = 'warmup') -> transforms.Compose:
    """Return torchvision transforms for given split.
    Args:
        split: 'train', 'val', or 'test'
        augmentation_mode: 'warmup' (conservative) or 'finetune' (aggressive)
    """

    split = split.lower()

    if split == 'train':
        # Select augmentation config based on training phase
        if augmentation_mode == 'finetune' and _HAS_STAGED_AUGMENTATION:
            AUG_CONFIG = AUGMENTATION_FINETUNE
            print("[INFO] Using AGGRESSIVE augmentation (fine-tuning mode)")
        else:
            AUG_CONFIG = AUGMENTATION_WARMUP
            print("[INFO] Using CONSERVATIVE augmentation (warmup mode)")
        
        # Use augmentation parameters from selected config
        rotation_range = AUG_CONFIG.get('rotation_range', 10)
        horizontal_flip = AUG_CONFIG.get('horizontal_flip', True)
        brightness_range = AUG_CONFIG.get('brightness_range', (0.8, 1.2))
        contrast_range = AUG_CONFIG.get('contrast_range', 0.2)
        saturation_range = AUG_CONFIG.get('saturation_range', 0.2)
        hue_range = AUG_CONFIG.get('hue_range', 0.05)
        width_shift = AUG_CONFIG.get('width_shift_range', 0.1)
        height_shift = AUG_CONFIG.get('height_shift_range', 0.1)
        translate = (width_shift, height_shift)
        zoom_range = AUG_CONFIG.get('zoom_range', 0.1)
        resized_crop_scale = AUG_CONFIG.get('resized_crop_scale', (0.9, 1.0))
        random_erasing_prob = AUG_CONFIG.get('random_erasing_prob', 0.2)
        random_erasing_scale = AUG_CONFIG.get('random_erasing_scale', (0.02, 0.1))
        perspective_distortion = AUG_CONFIG.get('perspective_distortion', 0.0)
        perspective_prob = AUG_CONFIG.get('perspective_prob', 0.0)
        gaussian_blur_prob = AUG_CONFIG.get('gaussian_blur_prob', 0.0)
        grayscale_prob = AUG_CONFIG.get('grayscale_prob', 0.0)
        
        # Enhanced augmentation pipeline (staged based on training phase)
        transform_list = [
            # RandomResizedCrop: Crop random portion then resize to 224x224
            transforms.RandomResizedCrop(INPUT_SIZE, scale=resized_crop_scale),
            transforms.RandomRotation(degrees=rotation_range),
            transforms.RandomAffine(degrees=0, translate=translate, scale=(1-zoom_range, 1+zoom_range)),
        ]
        
        # Add perspective transform only in finetune mode
        if perspective_prob > 0:
            transform_list.append(
                transforms.RandomPerspective(distortion_scale=perspective_distortion, p=perspective_prob)
            )
        
        transform_list.extend([
            # Enhanced ColorJitter: brightness, contrast, saturation, hue
            transforms.ColorJitter(brightness=brightness_range, contrast=contrast_range, 
                                 saturation=saturation_range, hue=hue_range),
        ])
        
        # Add grayscale only in finetune mode
        if grayscale_prob > 0:
            transform_list.append(transforms.RandomGrayscale(p=grayscale_prob))

        if horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip())

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        
        # Add gaussian blur only in finetune mode
        if gaussian_blur_prob > 0:
            transform_list.append(
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=gaussian_blur_prob)
            )
        
        # Add random erasing (always apply, but with different intensity)
        transform_list.append(
            transforms.RandomErasing(p=random_erasing_prob, scale=random_erasing_scale, value='random')
        )

        return transforms.Compose(transform_list)
    
    else:
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class EmotionDataset(Dataset):
    """Custom dataset for facial emotion images stored in split/<emotion>/ folders."""
    def __init__(self, root_dir: str | Path, transform: Optional[transforms.Compose] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.class_counts: Dict[int, int] = {}
        self._build_index()

    def _build_index(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.root_dir}")
        
        for class_idx, emotion in enumerate(EMOTION_LABELS):
            emotion_dir = self.root_dir / emotion

            if not emotion_dir.exists():
                # Skip missing emotion folder but continue
                continue

            for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
                for img_path in emotion_dir.glob(f'*{ext}'):
                    self.samples.append((img_path, class_idx))

            self.class_counts[class_idx] = sum(1 for _ in emotion_dir.glob('*.jpg')) + \
                                          sum(1 for _ in emotion_dir.glob('*.jpeg')) + \
                                          sum(1 for _ in emotion_dir.glob('*.png')) + \
                                          sum(1 for _ in emotion_dir.glob('*.bmp'))
            
        # Remove empty classes from counts
        self.class_counts = {k: v for k, v in self.class_counts.items() if v > 0}
        if len(self.samples) == 0:
            raise RuntimeError(f"No image files found under {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        with Image.open(img_path) as img:
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_class_counts(self) -> Dict[int, int]:
        return self.class_counts

    def get_class_weights(self) -> torch.Tensor:
        # Weight inversely proportional to frequency
        total = sum(self.class_counts.values())
        weights_per_class = {cls: total / count for cls, count in self.class_counts.items()}
        sample_weights = [weights_per_class[label] for _, label in self.samples]

        return torch.tensor(sample_weights, dtype=torch.float)


def build_dataloader(dataset: EmotionDataset, batch_size: int, shuffle: bool, use_weighted_sampler: bool, num_workers: int, pin_memory: bool) -> DataLoader:
    if use_weighted_sampler:
        weights = dataset.get_class_weights()
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
    
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


def build_dataloaders(data_splits_root: str | Path, batch_size: int = BATCH_SIZE, use_weighted_sampler: bool = True, num_workers: Optional[int] = None, pin_memory: bool = True, augmentation_mode: str = 'warmup') -> Dict[str, DataLoader]:
    """Build train/val/test dataloaders.
    Args:
        data_splits_root: root containing 'train', 'val', 'test' folders
        batch_size: per-batch size
        use_weighted_sampler: apply class imbalance handling to train loader
        num_workers: worker count (auto if None)
        pin_memory: set True when using CUDA
        augmentation_mode: 'warmup' (conservative) or 'finetune' (aggressive)
    Returns: dict with keys 'train', 'val', 'test'
    """

    root = Path(data_splits_root)
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 4)

    loaders: Dict[str, DataLoader] = {}

    for split in ['train', 'val', 'test']:
        split_path = root / split
        transform = get_transforms(split, augmentation_mode=augmentation_mode if split == 'train' else 'warmup')
        dataset = EmotionDataset(split_path, transform=transform)

        if split == 'train':
            loaders['train'] = build_dataloader(dataset, batch_size, shuffle=not use_weighted_sampler, use_weighted_sampler=use_weighted_sampler, num_workers=num_workers, pin_memory=pin_memory)

        else:
            loaders[split] = build_dataloader(dataset, batch_size, shuffle=False, use_weighted_sampler=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return loaders

__all__ = [
    'get_transforms',
    'EmotionDataset',
    'build_dataloaders'
]
