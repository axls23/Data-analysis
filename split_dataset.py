#!/usr/bin/env python3
"""
Dataset Splitting Pipeline
============================

This script implements Phase 1 (Task 3) of the Facial Emotion Detection project:
Create train/validation/test splits with balanced emotion classes.

The script:
- Creates 80/10/10 train/val/test splits
- Maintains balanced emotion class distribution (stratified split)
- Preserves folder structure
- Generates split statistics

Usage:
    python split_dataset.py --input_dir preprocessed_data --output_dir data_splits
    python split_dataset.py --input_dir preprocessed_data --output_dir data_splits --train_ratio 0.7 --val_ratio 0.15
    python split_dataset.py --help

Author: AI Assistant
Date: November 2025
"""

import os
import shutil
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm


def collect_images_by_emotion(input_dir):
    """
    Collect all images organized by emotion class
    
    Args:
        input_dir: Directory containing preprocessed images
    
    Returns:
        emotion_files: Dict mapping emotion -> list of (dataset, image_path) tuples
    """
    input_path = Path(input_dir)
    emotion_folders = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    
    emotion_files = defaultdict(list)
    
    # Iterate through all dataset folders
    for dataset_folder in sorted(input_path.glob('d*')):
        if not dataset_folder.is_dir():
            continue
        
        dataset_name = dataset_folder.name
        
        # Collect images from each emotion folder
        for emotion in emotion_folders:
            emotion_path = dataset_folder / emotion
            if not emotion_path.exists():
                continue
            
            for img_path in emotion_path.glob('*.jpg'):
                emotion_files[emotion].append((dataset_name, img_path))
    
    return emotion_files


def stratified_split(emotion_files, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Create stratified train/val/test splits
    
    Args:
        emotion_files: Dict mapping emotion -> list of image paths
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        seed: Random seed for reproducibility
    
    Returns:
        splits: Dict with 'train', 'val', 'test' keys mapping to lists of (emotion, image_path) tuples
    """
    random.seed(seed)
    test_ratio = 1.0 - train_ratio - val_ratio
    
    if test_ratio < 0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")
    
    splits = {'train': [], 'val': [], 'test': []}
    
    # Split each emotion class independently
    for emotion, files in emotion_files.items():
        # Shuffle files
        shuffled_files = files.copy()
        random.shuffle(shuffled_files)
        
        n_total = len(shuffled_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split
        train_files = shuffled_files[:n_train]
        val_files = shuffled_files[n_train:n_train + n_val]
        test_files = shuffled_files[n_train + n_val:]
        
        # Add to splits with emotion label
        splits['train'].extend([(emotion, f) for f in train_files])
        splits['val'].extend([(emotion, f) for f in val_files])
        splits['test'].extend([(emotion, f) for f in test_files])
    
    # Shuffle splits (while maintaining emotion distribution)
    for split_name in splits:
        random.shuffle(splits[split_name])
    
    return splits


def copy_split_files(splits, input_dir, output_dir):
    """
    Copy files to train/val/test directories
    
    Args:
        splits: Dictionary with train/val/test splits
        input_dir: Source directory
        output_dir: Destination directory
    
    Returns:
        stats: Statistics about the split
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    stats = {
        'train': defaultdict(int),
        'val': defaultdict(int),
        'test': defaultdict(int)
    }
    
    # Create output directories
    for split_name in ['train', 'val', 'test']:
        for emotion in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']:
            (output_path / split_name / emotion).mkdir(parents=True, exist_ok=True)
    
    # Copy files
    for split_name, files in splits.items():
        print(f"\n[INFO] Copying {split_name} split ({len(files)} images)...")
        
        for emotion, (dataset_name, src_path) in tqdm(files, desc=f"Copying {split_name}"):
            # Destination path
            dst_path = output_path / split_name / emotion / src_path.name
            
            # Copy file
            shutil.copy2(src_path, dst_path)
            
            # Update stats
            stats[split_name][emotion] += 1
    
    return stats


def print_split_statistics(stats, splits):
    """Print detailed split statistics"""
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    
    print("\n" + "="*70)
    print("DATASET SPLIT STATISTICS")
    print("="*70)
    
    # Calculate totals
    total_train = sum(stats['train'].values())
    total_val = sum(stats['val'].values())
    total_test = sum(stats['test'].values())
    total_all = total_train + total_val + total_test
    
    print(f"\nOverall Distribution:")
    print(f"  Train: {total_train:5d} images ({total_train/total_all*100:5.1f}%)")
    print(f"  Val:   {total_val:5d} images ({total_val/total_all*100:5.1f}%)")
    print(f"  Test:  {total_test:5d} images ({total_test/total_all*100:5.1f}%)")
    print(f"  Total: {total_all:5d} images")
    
    print(f"\nPer-Emotion Distribution:")
    print("-"*70)
    print(f"{'Emotion':<12s} {'Train':>8s} {'Val':>8s} {'Test':>8s} {'Total':>8s}")
    print("-"*70)
    
    for emotion in emotions:
        train_count = stats['train'][emotion]
        val_count = stats['val'][emotion]
        test_count = stats['test'][emotion]
        total_count = train_count + val_count + test_count
        
        print(f"{emotion:<12s} {train_count:8d} {val_count:8d} {test_count:8d} {total_count:8d}")
    
    print("-"*70)
    print(f"{'Total':<12s} {total_train:8d} {total_val:8d} {total_test:8d} {total_all:8d}")
    print("="*70)


def save_split_info(splits, stats, output_dir, seed):
    """Save split information to JSON file"""
    output_path = Path(output_dir)
    
    # Create summary
    summary = {
        'seed': seed,
        'total_images': sum(len(files) for files in splits.values()),
        'splits': {
            split_name: {
                'total': len(files),
                'emotions': dict(Counter(emotion for emotion, _ in files))
            }
            for split_name, files in splits.items()
        },
        'statistics': {
            split_name: dict(emotion_counts)
            for split_name, emotion_counts in stats.items()
        }
    }
    
    # Save to file
    info_file = output_path / 'split_info.json'
    with open(info_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[INFO] Split information saved to {info_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Split preprocessed dataset into train/val/test sets - Phase 1 Task 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default 80/10/10 split
  python split_dataset.py --input_dir preprocessed_data --output_dir data_splits
  
  # Custom split ratios
  python split_dataset.py --input_dir preprocessed_data --output_dir data_splits --train_ratio 0.7 --val_ratio 0.15
  
  # Use different random seed
  python split_dataset.py --input_dir preprocessed_data --output_dir data_splits --seed 123
        """
    )
    
    parser.add_argument('--input_dir', type=str, default='preprocessed_data',
                       help='Input directory with preprocessed images (default: preprocessed_data)')
    parser.add_argument('--output_dir', type=str, default='data_splits',
                       help='Output directory for splits (default: data_splits)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training data ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation data ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Validate ratios
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0 or test_ratio > 1:
        print(f"[ERROR] Invalid ratios: train={args.train_ratio}, val={args.val_ratio}")
        return 1
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"[ERROR] Input directory does not exist: {args.input_dir}")
        return 1
    
    print("="*70)
    print("FACIAL EMOTION DETECTION - DATASET SPLITTING")
    print("="*70)
    print(f"Input directory:   {args.input_dir}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Train ratio:       {args.train_ratio:.1%}")
    print(f"Validation ratio:  {args.val_ratio:.1%}")
    print(f"Test ratio:        {test_ratio:.1%}")
    print(f"Random seed:       {args.seed}")
    print("="*70 + "\n")
    
    # Collect images by emotion
    print("[INFO] Collecting images...")
    emotion_files = collect_images_by_emotion(args.input_dir)
    
    if not emotion_files:
        print("[ERROR] No images found in input directory!")
        return 1
    
    total_images = sum(len(files) for files in emotion_files.values())
    print(f"[INFO] Found {total_images} images across {len(emotion_files)} emotion classes")
    
    for emotion, files in sorted(emotion_files.items()):
        print(f"  {emotion:12s}: {len(files):4d} images")
    
    # Create stratified splits
    print("\n[INFO] Creating stratified splits...")
    splits = stratified_split(
        emotion_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    # Copy files to split directories
    print("\n[INFO] Copying files to split directories...")
    stats = copy_split_files(splits, args.input_dir, args.output_dir)
    
    # Print statistics
    print_split_statistics(stats, splits)
    
    # Save split info
    save_split_info(splits, stats, args.output_dir, args.seed)
    
    print(f"\n✓ Dataset splitting complete!")
    print(f"  Train images: {args.output_dir}/train/")
    print(f"  Val images:   {args.output_dir}/val/")
    print(f"  Test images:  {args.output_dir}/test/")
    
    return 0


if __name__ == '__main__':
    exit(main())
