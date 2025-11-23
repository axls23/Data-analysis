"""Training monitoring and visualization utilities.

Provides tools for:
- Plotting training curves (loss, accuracy, learning rate)
- Detecting overfitting patterns
- Visualizing model performance metrics
"""

from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import pandas as pd
import numpy as np


def plot_training_curves(
    log_path: str | Path,
    output_path: Optional[str | Path] = None,
    show_lr: bool = False
) -> None:
    """Plot training and validation curves from CSV log file.
    
    Creates a multi-panel plot showing:
    - Loss curves (train vs val)
    - Accuracy curves (train vs val)
    - Learning rate schedule (if show_lr=True)
    
    Args:
        log_path: Path to CSV log file with columns: epoch, train_loss, train_acc, val_loss, val_acc
        output_path: Path to save plot (default: same directory as log_path with .png extension)
        show_lr: Whether to include learning rate subplot (requires 'lr' column in CSV)
    
    Example:
        >>> plot_training_curves('results/logs/mobilenet_warmup.csv')
        >>> plot_training_curves('results/logs/resnet50_finetune.csv', show_lr=True)
    """
    log_path = Path(log_path)
    
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    
    # Read CSV
    df = pd.read_csv(log_path)
    
    # Default output path (results/curves/)
    if output_path is None:
        curves_dir = log_path.parents[1] / 'curves'  # results/curves/
        curves_dir.mkdir(parents=True, exist_ok=True)
        output_path = curves_dir / f"{log_path.stem}_curves.png"
    else:
        output_path = Path(output_path)
    
    # Create figure
    n_plots = 3 if show_lr and 'lr' in df.columns else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4*n_plots))
    
    if n_plots == 2:
        axes = [axes[0], axes[1]]
    else:
        axes = [axes[0], axes[1], axes[2]]
    
    # Plot loss
    axes[0].plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(df['epoch'], df['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(df['epoch'], df['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(df['epoch'], df['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # Plot learning rate if requested
    if show_lr and 'lr' in df.columns:
        axes[2].plot(df['epoch'], df['lr'], 'g-', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Training curves saved to: {output_path}")


def detect_overfitting(
    log_path: str | Path,
    gap_threshold: float = 0.15,
    window_size: int = 3
) -> Dict[str, any]:
    """Detect overfitting patterns from training logs.
    
    Overfitting indicators:
    1. Train-val accuracy gap > threshold
    2. Val loss increasing while train loss decreasing
    3. Val accuracy plateauing or decreasing
    
    Args:
        log_path: Path to CSV log file
        gap_threshold: Maximum acceptable train-val accuracy gap (default: 0.15 = 15%)
        window_size: Moving average window for trend detection (default: 3 epochs)
    
    Returns:
        Dictionary with overfitting analysis:
        - is_overfitting: bool
        - max_gap: float (maximum train-val accuracy gap)
        - gap_trend: str ('increasing', 'stable', 'decreasing')
        - val_loss_trend: str ('increasing', 'stable', 'decreasing')
        - recommendations: List[str]
    
    Example:
        >>> result = detect_overfitting('results/logs/resnet50_finetune.csv')
        >>> if result['is_overfitting']:
        >>>     print(f"Overfitting detected! Max gap: {result['max_gap']:.2%}")
        >>>     for rec in result['recommendations']:
        >>>         print(f"  - {rec}")
    """
    log_path = Path(log_path)
    
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    
    df = pd.read_csv(log_path)
    
    # Calculate train-val gaps
    acc_gap = df['train_acc'] - df['val_acc']
    max_gap = acc_gap.max()
    latest_gap = acc_gap.iloc[-1]
    
    # Detect gap trend (last window_size epochs)
    if len(acc_gap) >= window_size:
        recent_gaps = acc_gap.iloc[-window_size:]
        gap_slope = np.polyfit(range(len(recent_gaps)), recent_gaps, 1)[0]
        
        if gap_slope > 0.02:
            gap_trend = 'increasing'
        elif gap_slope < -0.02:
            gap_trend = 'decreasing'
        else:
            gap_trend = 'stable'
    else:
        gap_trend = 'insufficient_data'
    
    # Detect val loss trend
    if len(df) >= window_size:
        recent_val_loss = df['val_loss'].iloc[-window_size:]
        val_loss_slope = np.polyfit(range(len(recent_val_loss)), recent_val_loss, 1)[0]
        
        if val_loss_slope > 0.01:
            val_loss_trend = 'increasing'
        elif val_loss_slope < -0.01:
            val_loss_trend = 'decreasing'
        else:
            val_loss_trend = 'stable'
    else:
        val_loss_trend = 'insufficient_data'
    
    # Determine if overfitting
    is_overfitting = (
        max_gap > gap_threshold or
        (gap_trend == 'increasing' and val_loss_trend == 'increasing') or
        latest_gap > gap_threshold
    )
    
    # Generate recommendations
    recommendations = []
    
    if max_gap > gap_threshold:
        recommendations.append(f"Train-val accuracy gap ({max_gap:.2%}) exceeds threshold ({gap_threshold:.2%})")
        recommendations.append("→ Consider: Increase weight_decay, dropout, or data augmentation")
    
    if gap_trend == 'increasing':
        recommendations.append("Train-val gap is widening over time")
        recommendations.append("→ Consider: Early stopping, reduce model complexity, or add regularization")
    
    if val_loss_trend == 'increasing' and df['train_loss'].iloc[-1] < df['train_loss'].iloc[-window_size]:
        recommendations.append("Validation loss increasing while train loss decreasing")
        recommendations.append("→ Consider: Stop training now, or reduce learning rate")
    
    if df['val_acc'].iloc[-1] < df['val_acc'].iloc[-window_size]:
        recommendations.append("Validation accuracy is decreasing")
        recommendations.append("→ Consider: Revert to earlier checkpoint")
    
    if not is_overfitting:
        recommendations.append("No significant overfitting detected")
        if latest_gap < 0.05:
            recommendations.append("→ Model is generalizing well")
        else:
            recommendations.append(f"→ Current gap ({latest_gap:.2%}) is acceptable but monitor closely")
    
    return {
        'is_overfitting': is_overfitting,
        'max_gap': max_gap,
        'latest_gap': latest_gap,
        'gap_trend': gap_trend,
        'val_loss_trend': val_loss_trend,
        'recommendations': recommendations
    }


def plot_model_comparison(
    summary_csv: str | Path,
    output_path: Optional[str | Path] = None,
    metric: str = 'test_acc'
) -> None:
    """Plot bar chart comparing multiple models from summary CSV.
    
    Args:
        summary_csv: Path to summary CSV (e.g., summary_phase3a.csv)
        output_path: Path to save plot (default: same directory with _comparison.png)
        metric: Metric to plot ('test_acc', 'best_val_acc', 'test_loss', etc.)
    
    Example:
        >>> plot_model_comparison('results/summary_phase3a.csv', metric='test_acc')
    """
    summary_csv = Path(summary_csv)
    
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")
    
    df = pd.read_csv(summary_csv)
    
    # Default output path
    if output_path is None:
        output_path = summary_csv.parent / f"{summary_csv.stem}_comparison.png"
    else:
        output_path = Path(output_path)
    
    # Get unique models (in case CSV has multiple runs)
    df_latest = df.groupby('model').tail(1)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df_latest['model'].values
    values = df_latest[metric].values
    
    bars = ax.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Model Comparison: {metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Model comparison plot saved to: {output_path}")


def print_overfitting_report(log_path: str | Path) -> None:
    """Print formatted overfitting analysis report to console.
    
    Args:
        log_path: Path to training log CSV file
    
    Example:
        >>> print_overfitting_report('results/logs/resnet50_finetune.csv')
    """
    result = detect_overfitting(log_path)
    
    print("\n" + "="*70)
    print("OVERFITTING ANALYSIS REPORT")
    print("="*70)
    print(f"Log file: {log_path}")
    print(f"\nOverfitting detected: {'YES ⚠️' if result['is_overfitting'] else 'NO ✅'}")
    print(f"Maximum train-val gap: {result['max_gap']:.2%}")
    print(f"Latest train-val gap: {result['latest_gap']:.2%}")
    print(f"Gap trend: {result['gap_trend']}")
    print(f"Val loss trend: {result['val_loss_trend']}")
    print("\nRecommendations:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"  {i}. {rec}")
    print("="*70 + "\n")


__all__ = [
    'plot_training_curves',
    'detect_overfitting',
    'plot_model_comparison',
    'print_overfitting_report'
]
