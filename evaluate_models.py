"""Phase 4: Model Evaluation and Visualization

Comprehensive evaluation script for emotion detection models.
Generates:
- Confusion matrices for all models
- Per-emotion metrics (precision, recall, F1-score)
- Classification reports
- Comparison visualizations

Usage:
    python evaluate_models.py --models resnet50,resnet18
    python evaluate_models.py --models all --save_predictions
    python evaluate_models.py --checkpoint_dir results/checkpoints --stage deep
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from config import EMOTION_LABELS, BATCH_SIZE, USE_GPU, GPU_ID, DATA_SPLITS_DIR
from models import create_model
from utils.data_loader import build_dataloaders


AVAILABLE_MODELS = ['mobilenet', 'efficientnet', 'resnet18', 'resnet50']
CHECKPOINT_STAGES = {
    'warmup': '{}_warmup_best.pt',
    'progressive': '{}_finetune_progressive_best.pt',
    'deep': '{}_finetune_deep_best.pt'
}


def get_device():
    if USE_GPU and torch.cuda.is_available():
        try:
            torch.cuda.set_device(GPU_ID)
        except Exception:
            pass
        return torch.device(f'cuda:{GPU_ID}')
    return torch.device('cpu')


def load_model_checkpoint(model_name: str, checkpoint_dir: Path, stage: str = 'deep', device=None):
    """Load trained model from checkpoint."""
    if device is None:
        device = get_device()
    
    # Get checkpoint filename pattern
    checkpoint_pattern = CHECKPOINT_STAGES.get(stage)
    if checkpoint_pattern is None:
        raise ValueError(f"Invalid stage: {stage}. Choose from: {list(CHECKPOINT_STAGES.keys())}")
    
    checkpoint_path = checkpoint_dir / checkpoint_pattern.format(model_name)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading {model_name} from {checkpoint_path}")
    
    # Create model
    model = create_model(model_name, num_classes=7, pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint.get('val_acc', 0.0)


def evaluate_single_model(model, dataloader, device, emotion_labels):
    """Evaluate model and return predictions and ground truth."""
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Get probabilities and predictions
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    return all_preds, all_labels, all_probs


def plot_confusion_matrix(y_true, y_pred, labels, model_name, save_path):
    """Generate and save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize to percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create annotations with % symbol
    annot_labels = np.array([[f'{val:.1f}%' for val in row] for row in cm_percent])
    
    # Plot heatmap with custom annotations
    sns.heatmap(cm_percent, annot=annot_labels, fmt='', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, 
                cbar_kws={'label': 'Percentage (%)'}, ax=ax,
                annot_kws={'fontsize': 9})
    
    ax.set_xlabel('Predicted Emotion', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Emotion', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name.upper()}', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")


def plot_per_emotion_metrics(metrics_dict, save_path):
    """Plot per-emotion metrics for all models."""
    emotions = list(EMOTION_LABELS)
    models = list(metrics_dict.keys())
    
    metrics_types = ['precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, metric in enumerate(metrics_types):
        ax = axes[idx]
        
        # Prepare data
        x = np.arange(len(emotions))
        width = 0.2
        
        for i, model in enumerate(models):
            values = [metrics_dict[model]['per_emotion'][emotion][metric] * 100 
                     for emotion in emotions]
            offset = (i - len(models)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model.upper(), alpha=0.8)
        
        ax.set_xlabel('Emotion', fontweight='bold')
        ax.set_ylabel(f'{metric.capitalize()} (%)', fontweight='bold')
        ax.set_title(f'Per-Emotion {metric.capitalize()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(emotions, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Per-emotion metrics saved to: {save_path}")


def plot_model_comparison(metrics_dict, save_path):
    """Plot overall model comparison."""
    models = list(metrics_dict.keys())
    
    # Extract overall metrics
    accuracy = [metrics_dict[m]['accuracy'] * 100 for m in models]
    precision = [metrics_dict[m]['precision'] * 100 for m in models]
    recall = [metrics_dict[m]['recall'] * 100 for m in models]
    f1 = [metrics_dict[m]['f1'] * 100 for m in models]
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#2E86AB')
    ax.bar(x - 0.5*width, precision, width, label='Precision', color='#A23B72')
    ax.bar(x + 0.5*width, recall, width, label='Recall', color='#F18F01')
    ax.bar(x + 1.5*width, f1, width, label='F1-Score', color='#C73E1D')
    
    ax.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score (%)', fontweight='bold', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Add value labels on bars
    for i, (acc, prec, rec, f) in enumerate(zip(accuracy, precision, recall, f1)):
        ax.text(i - 1.5*width, acc + 1, f'{acc:.1f}', ha='center', va='bottom', fontsize=8)
        ax.text(i - 0.5*width, prec + 1, f'{prec:.1f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + 0.5*width, rec + 1, f'{rec:.1f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + 1.5*width, f + 1, f'{f:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison saved to: {save_path}")


def evaluate_and_visualize(models_to_eval, checkpoint_dir, data_splits_dir, stage, save_predictions, output_dir):
    """Main evaluation pipeline."""
    device = get_device()
    
    # Load test data
    print("\nLoading test dataset...")
    loaders = build_dataloaders(data_splits_dir, batch_size=BATCH_SIZE, 
                                use_weighted_sampler=False, pin_memory=device.type == 'cuda')
    test_loader = loaders['test']
    
    print(f"Test set size: {len(test_loader.dataset)} images")
    
    # Create output directories
    cm_dir = output_dir / 'confusion_matrices'
    cm_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all metrics
    all_metrics = {}
    
    for model_name in models_to_eval:
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_name.upper()}")
        print(f"{'='*70}")
        
        try:
            # Load model
            model, val_acc = load_model_checkpoint(model_name, checkpoint_dir, stage, device)
            
            # Evaluate
            preds, labels, probs = evaluate_single_model(model, test_loader, device, EMOTION_LABELS)
            
            # Calculate metrics
            accuracy = accuracy_score(labels, preds)
            precision, recall, f1, support = precision_recall_fscore_support(
                labels, preds, average='weighted', zero_division=0
            )
            
            # Per-emotion metrics
            per_emotion_metrics = {}
            # Get per-class metrics for all emotions at once
            precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                labels, preds, average=None, zero_division=0, labels=list(range(len(EMOTION_LABELS)))
            )
            
            for idx, emotion in enumerate(EMOTION_LABELS):
                per_emotion_metrics[emotion] = {
                    'precision': precision_per_class[idx],
                    'recall': recall_per_class[idx],
                    'f1': f1_per_class[idx],
                    'support': int(support_per_class[idx])
                }
            
            # Store metrics
            all_metrics[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'val_acc': val_acc,
                'per_emotion': per_emotion_metrics
            }
            
            # Print results
            print(f"\nOverall Metrics:")
            print(f"  Test Accuracy:  {accuracy*100:.2f}%")
            print(f"  Precision:      {precision*100:.2f}%")
            print(f"  Recall:         {recall*100:.2f}%")
            print(f"  F1-Score:       {f1*100:.2f}%")
            
            print(f"\nPer-Emotion Accuracy:")
            for emotion in EMOTION_LABELS:
                em_acc = per_emotion_metrics[emotion]['f1'] * 100
                print(f"  {emotion.capitalize():12s}: {em_acc:.2f}%")
            
            # Generate confusion matrix
            cm_path = cm_dir / f'{model_name}_confusion_matrix.png'
            plot_confusion_matrix(labels, preds, EMOTION_LABELS, model_name, cm_path)
            
            # Save predictions if requested
            if save_predictions:
                pred_path = output_dir / f'{model_name}_predictions.csv'
                with open(pred_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['image_idx', 'true_emotion', 'predicted_emotion', 'confidence'] + 
                                   [f'prob_{em}' for em in EMOTION_LABELS])
                    
                    for i, (true_idx, pred_idx, prob_vector) in enumerate(zip(labels, preds, probs)):
                        row = [
                            i,
                            EMOTION_LABELS[true_idx],
                            EMOTION_LABELS[pred_idx],
                            f"{prob_vector[pred_idx]:.4f}"
                        ] + [f"{p:.4f}" for p in prob_vector]
                        writer.writerow(row)
                
                print(f"Predictions saved to: {pred_path}")
        
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate comparison visualizations
    if len(all_metrics) > 1:
        print(f"\n{'='*70}")
        print("Generating comparison visualizations...")
        print(f"{'='*70}")
        
        comparison_path = output_dir / 'model_comparison.png'
        plot_model_comparison(all_metrics, comparison_path)
        
        per_emotion_path = output_dir / 'per_emotion_metrics.png'
        plot_per_emotion_metrics(all_metrics, per_emotion_path)
    
    # Save metrics to JSON
    metrics_json_path = output_dir / 'evaluation_metrics.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Evaluation complete! Results saved to: {output_dir}")
    print(f"{'='*70}\n")
    
    return all_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Phase 4: Model Evaluation and Visualization')
    
    parser.add_argument('--models', type=str, default='all',
                       help='Comma-separated list of models or "all" (default: all)')
    parser.add_argument('--checkpoint_dir', type=str, default='results/checkpoints',
                       help='Directory containing model checkpoints')
    parser.add_argument('--data_splits', type=str, default=DATA_SPLITS_DIR,
                       help='Path to data_splits directory')
    parser.add_argument('--stage', type=str, default='deep',
                       choices=list(CHECKPOINT_STAGES.keys()),
                       help='Checkpoint stage to evaluate (default: deep)')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save detailed predictions to CSV')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Parse models
    if args.models.lower() == 'all':
        models_to_eval = AVAILABLE_MODELS
    else:
        models_to_eval = [m.strip() for m in args.models.split(',') if m.strip() in AVAILABLE_MODELS]
    
    if not models_to_eval:
        print("No valid models specified.")
        return 1
    
    print('='*70)
    print('PHASE 4: MODEL EVALUATION AND VISUALIZATION')
    print('='*70)
    print(f"Models: {', '.join(models_to_eval)}")
    print(f"Checkpoint stage: {args.stage}")
    print(f"Output directory: {args.output_dir}")
    print('='*70)
    
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    metrics = evaluate_and_visualize(
        models_to_eval,
        checkpoint_dir,
        args.data_splits,
        args.stage,
        args.save_predictions,
        output_dir
    )
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
