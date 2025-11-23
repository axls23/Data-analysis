"""Training utilities for Phase 3a warm-up training.

Includes:
- Trainer class handling training/validation loops
- Learning rate warmup scheduler
- Early stopping
- Checkpointing (best validation accuracy)
- CSV logging of metrics
- Gradient clipping
"""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR


class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device, 
        criterion: nn.Module, 
        optimizer: Optimizer, 
        scheduler: Optional[LinearLR] = None, 
        model_name: str = 'model', 
        results_dir: str | Path = 'results',
        warmup_epochs: int = 3,
        use_warmup: bool = True,
        grad_clip_max_norm: float = 1.0,
        phase: str = 'warmup'  # 'warmup' or 'finetune'
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.model_name = model_name
        self.phase = phase
        self.results_dir = Path(results_dir)
        self.checkpoint_dir = self.results_dir / 'checkpoints'
        self.log_dir = self.results_dir / 'logs'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f'{self.model_name}_{phase}.csv'
        self.best_ckpt_path = self.checkpoint_dir / f'{self.model_name}_{phase}_best.pt'
        self.grad_clip_max_norm = grad_clip_max_norm
        
        # Learning rate warmup scheduler
        if use_warmup:
            # LinearLR: scales LR from start_factor*base_lr to end_factor*base_lr over total_iters epochs
            # start_factor=0.01 means start at 1% of base_lr (1e-4 * 0.01 = 1e-6)
            # end_factor=1.0 means end at 100% of base_lr (1e-4)
            self.warmup_scheduler = LinearLR(
                optimizer, 
                start_factor=0.01, 
                end_factor=1.0, 
                total_iters=warmup_epochs
            )
            print(f"[INFO] Using LinearLR warmup: {optimizer.param_groups[0]['lr']*0.01:.2e} → {optimizer.param_groups[0]['lr']:.2e} over {warmup_epochs} epochs")
        else:
            self.warmup_scheduler = None
        
        # Optional additional scheduler (e.g., ReduceLROnPlateau)
        self.main_scheduler = scheduler
        
        self._init_csv()

    def _init_csv(self):
        if not self.log_path.exists():
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    @staticmethod
    def _accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
        preds = outputs.argmax(dim=1)
        correct = (preds == targets).sum().item()
        return correct / targets.size(0)

    def train_epoch(self, loader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for inputs, targets in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent explosions
            if self.grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)
            
            self.optimizer.step()
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    def eval_epoch(self, loader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (outputs.argmax(dim=1) == targets).sum().item()
                total_samples += batch_size
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    def fit(self, train_loader, val_loader, epochs: int = 20, patience: int = 20) -> Dict[str, float]:
        best_val_acc = 0.0
        best_val_loss = float('inf')
        epochs_no_improve = 0
        history = {}
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.eval_epoch(val_loader)
            
            # Step warmup scheduler (linear ramp for first N epochs)
            if self.warmup_scheduler is not None:
                self.warmup_scheduler.step()
            
            # Step main scheduler (e.g., ReduceLROnPlateau)
            if self.main_scheduler is not None:
                self.main_scheduler.step(val_loss)
            
            # Get current LR for logging
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to CSV
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.4f}", f"{val_loss:.6f}", f"{val_acc:.4f}"])
            print(f"[Epoch {epoch:03d}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | LR: {current_lr:.2e}")
            improved = False
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                epochs_no_improve = 0
                improved = True
                self._save_checkpoint(val_acc, val_loss, epoch)
            else:
                epochs_no_improve += 1
            if improved:
                print(f"  -> New best model saved (val_acc={best_val_acc:.4f})")
            if epochs_no_improve >= patience:
                print(f"[EarlyStopping] No improvement for {patience} epochs. Stopping.")
                break
        history['best_val_acc'] = best_val_acc
        history['best_val_loss'] = best_val_loss
        history['epochs_trained'] = epoch
        return history

    def _save_checkpoint(self, val_acc: float, val_loss: float, epoch: int):
        torch.save({
            'model_state': self.model.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'epoch': epoch
        }, self.best_ckpt_path)
    
    def load_checkpoint(self, checkpoint_path: str | Path):
        """Load model weights from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file (.pt)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"[INFO] Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        print(f"[INFO] Checkpoint loaded successfully")

    def evaluate_test(self, test_loader) -> Tuple[float, float]:
        # Load best checkpoint if exists
        if self.best_ckpt_path.exists():
            ckpt = torch.load(self.best_ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state'])
        test_loss, test_acc = self.eval_epoch(test_loader)
        return test_loss, test_acc

__all__ = ['Trainer']
