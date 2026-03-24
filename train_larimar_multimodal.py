import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from larimar_base.models import create_multimodal_classifier, count_parameters
from modules.multimodal_datasets import create_data_module

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Optional bitsandbytes
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("Warning: bitsandbytes not available. Install with: pip install bitsandbytes")

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Trainer
# ---------------------------------------------------------
class MultimodalClassifierTrainer:
    """Trainer for MultimodalClassifier"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.setup_directories()
        self.setup_wandb()

        self.model = None
        self.data_module = None
        self.optimizer = None
        self.scheduler = None

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    def setup_directories(self):
        """Create output directories"""
        output_cfg = self.config.get("output", {})
        for dir_name in ["checkpoint_dir", "log_dir", "results_dir"]:
            dir_path = Path(output_cfg.get(dir_name, f"./outputs/{dir_name}"))
            dir_path.mkdir(parents=True, exist_ok=True)
            setattr(self, dir_name, dir_path)

    def setup_wandb(self):
        """Setup wandb if configured"""
        self.use_wandb = False

        if not WANDB_AVAILABLE:
            logger.info("wandb not installed; logging locally only")
            return

        wandb_cfg = self.config.get("wandb", {})
        use_wandb = (
            wandb_cfg.get("project") is not None or
            wandb_cfg.get("api_key") is not None or
            os.getenv("WANDB_API_KEY") is not None
        )

        if not use_wandb:
            logger.info("wandb not configured; logging locally only")
            return

        try:
            if wandb_cfg.get("api_key"):
                os.environ["WANDB_API_KEY"] = wandb_cfg["api_key"]

            wandb.init(
                project=wandb_cfg.get(
                    "project", "Multimodal synthesis data detection"),
                config=self.config,
                name=wandb_cfg.get("run_name", None)
            )
            self.use_wandb = True
            logger.info("wandb initialized successfully")
        except Exception as e:
            logger.warning(f"wandb initialization failed: {e}")
            self.use_wandb = False

    def setup_model_and_data(self, max_samples: Optional[int] = None):
        """Initialize model and dataloaders"""
        logger.info("Setting up model and data...")

        self.model = create_multimodal_classifier(self.config["model"])
        self.model.to(self.device)

        param_count = count_parameters(self.model)
        logger.info(f"Model parameters: {param_count}")

        self.data_module = create_data_module(self.config["data"])
        self.data_module.setup(max_samples=max_samples)

        self.setup_optimizer()

    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        training_cfg = self.config["training"]

        if BITSANDBYTES_AVAILABLE:
            self.optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=training_cfg["learning_rate"],
                weight_decay=training_cfg["weight_decay"],
                betas=(0.9, 0.999),
                eps=1e-8
            )
            logger.info("Using AdamW8bit optimizer")
        else:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=training_cfg["learning_rate"],
                weight_decay=training_cfg["weight_decay"],
                betas=(0.9, 0.999),
                eps=1e-8
            )
            logger.info("Using AdamW optimizer")

        if training_cfg.get("scheduler", "cosine") == "cosine":
            train_loader = self.data_module.train_dataloader()
            steps_per_epoch = len(train_loader)
            total_steps = steps_per_epoch * training_cfg["max_epochs"]

            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=training_cfg.get("min_lr", 1e-6)
            )
            self.scheduler_step_mode = "step"
            logger.info(
                f"Using cosine scheduler with {total_steps} total steps")
        else:
            self.scheduler = None
            self.scheduler_step_mode = "epoch"

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------
    @staticmethod
    def _compute_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
        return (preds == labels).float().mean().item()

    @staticmethod
    def _compute_precision_recall_f1(preds: torch.Tensor, labels: torch.Tensor, positive_label: int = 1):
        preds = preds.view(-1)
        labels = labels.view(-1)

        tp = ((preds == positive_label) & (
            labels == positive_label)).sum().item()
        fp = ((preds == positive_label) & (
            labels != positive_label)).sum().item()
        fn = ((preds != positive_label) & (
            labels == positive_label)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1

    def _compute_memory_entropy(self, memory_usage: torch.Tensor) -> float:
        """Entropy of memory usage distribution"""
        try:
            if memory_usage is None or memory_usage.numel() == 0:
                return 0.0

            usage_sum = memory_usage.sum()
            if usage_sum <= 1e-8:
                return 0.0

            probs = memory_usage / usage_sum
            entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
            return entropy if np.isfinite(entropy) else 0.0
        except Exception as e:
            logger.warning(f"Memory entropy computation failed: {e}")
            return 0.0

    def _compute_cross_modal_similarity(
        self,
        fused_text_tokens: torch.Tensor,
        fused_image_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Cosine similarity between pooled fused text and pooled fused image.
        """
        try:
            if fused_text_tokens is None or fused_image_tokens is None:
                return 0.0

            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                text_pooled = (fused_text_tokens * mask).sum(dim=1) / \
                    mask.sum(dim=1).clamp(min=1e-6)
            else:
                text_pooled = fused_text_tokens.mean(dim=1)

            image_pooled = fused_image_tokens.mean(dim=1)

            if text_pooled.shape[-1] != image_pooled.shape[-1]:
                return 0.0

            sim = torch.cosine_similarity(
                text_pooled, image_pooled, dim=1).mean().item()
            return sim if np.isfinite(sim) else 0.0
        except Exception as e:
            logger.warning(f"Cross-modal similarity computation failed: {e}")
            return 0.0

    # -----------------------------------------------------
    # Train / Validate
    # -----------------------------------------------------
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        train_loader = self.data_module.train_dataloader()

        losses = []
        accs = []
        precisions = []
        recalls = []
        f1s = []

        epoch_metrics = {
            "train_loss": 0.0,
            "train_accuracy": 0.0,
            "train_precision": 0.0,
            "train_recall": 0.0,
            "train_f1": 0.0,
            "memory_usage_entropy": 0.0,
            "cross_modal_similarity": 0.0
        }

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            try:
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)

                # IMPORTANT: changed vision_features -> pixel_values
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                    mode="train"
                )

                loss = outputs["loss"]
                if loss is None or not torch.isfinite(loss):
                    logger.warning(f"Invalid loss at step {self.global_step}")
                    self.global_step += 1
                    continue

                self.optimizer.zero_grad()
                loss.backward()

                if self.config["training"].get("gradient_clip_val", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["training"]["gradient_clip_val"]
                    )

                self.optimizer.step()

                if self.scheduler and self.scheduler_step_mode == "step":
                    self.scheduler.step()

                preds = outputs["preds"]
                labels = batch["labels"]

                acc = self._compute_accuracy(preds, labels)
                precision, recall, f1 = self._compute_precision_recall_f1(
                    preds, labels)

                losses.append(loss.item())
                accs.append(acc)
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

                if outputs.get("memory_usage") is not None:
                    epoch_metrics["memory_usage_entropy"] += self._compute_memory_entropy(
                        outputs["memory_usage"])

                if outputs.get("fused_text_tokens") is not None and outputs.get("fused_image_tokens") is not None:
                    epoch_metrics["cross_modal_similarity"] += self._compute_cross_modal_similarity(
                        outputs["fused_text_tokens"],
                        outputs["fused_image_tokens"],
                        batch.get("attention_mask", None)
                    )

                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{acc:.4f}",
                    "f1": f"{f1:.4f}"
                })

                if self.use_wandb and batch_idx % self.config.get("wandb", {}).get("log_every_n_steps", 50) == 0:
                    wandb.log({
                        "train/step_loss": loss.item(),
                        "train/step_accuracy": acc,
                        "train/step_precision": precision,
                        "train/step_recall": recall,
                        "train/step_f1": f1,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "global_step": self.global_step
                    })

                self.global_step += 1

                if self.global_step % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(
                    f"Training batch {batch_idx} failed at step {self.global_step}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.global_step += 1
                continue

        num_batches = max(len(train_loader), 1)

        epoch_metrics["train_loss"] = float(
            np.mean(losses)) if losses else float("inf")
        epoch_metrics["train_accuracy"] = float(np.mean(accs)) if accs else 0.0
        epoch_metrics["train_precision"] = float(
            np.mean(precisions)) if precisions else 0.0
        epoch_metrics["train_recall"] = float(
            np.mean(recalls)) if recalls else 0.0
        epoch_metrics["train_f1"] = float(np.mean(f1s)) if f1s else 0.0
        epoch_metrics["memory_usage_entropy"] /= num_batches
        epoch_metrics["cross_modal_similarity"] /= num_batches

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return epoch_metrics

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        val_loaders = self.data_module.val_dataloader()

        losses = []
        accs = []
        precisions = []
        recalls = []
        f1s = []

        val_metrics = {
            "val_loss": 0.0,
            "val_accuracy": 0.0,
            "val_precision": 0.0,
            "val_recall": 0.0,
            "val_f1": 0.0,
            "val_memory_entropy": 0.0,
            "val_cross_modal_similarity": 0.0
        }

        with torch.no_grad():
            for loader_idx, val_loader in enumerate(val_loaders):
                logger.info(
                    f"Validating on dataset {loader_idx + 1}/{len(val_loaders)}")

                for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Validation-{loader_idx+1}")):
                    try:
                        for key in batch:
                            if torch.is_tensor(batch[key]):
                                batch[key] = batch[key].to(self.device)

                        outputs = self.model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            mode="read"   # validation should read memory only
                        )

                        if outputs["loss"] is not None and torch.isfinite(outputs["loss"]):
                            losses.append(outputs["loss"].item())

                        preds = outputs["preds"]
                        labels = batch["labels"]

                        acc = self._compute_accuracy(preds, labels)
                        precision, recall, f1 = self._compute_precision_recall_f1(
                            preds, labels)

                        accs.append(acc)
                        precisions.append(precision)
                        recalls.append(recall)
                        f1s.append(f1)

                        if outputs.get("memory_usage") is not None:
                            val_metrics["val_memory_entropy"] += self._compute_memory_entropy(
                                outputs["memory_usage"])

                        if outputs.get("fused_text_tokens") is not None and outputs.get("fused_image_tokens") is not None:
                            val_metrics["val_cross_modal_similarity"] += self._compute_cross_modal_similarity(
                                outputs["fused_text_tokens"],
                                outputs["fused_image_tokens"],
                                batch.get("attention_mask", None)
                            )

                    except Exception as e:
                        logger.warning(
                            f"Validation batch {batch_idx} in loader {loader_idx} failed: {e}")
                        continue

        total_batches = max(sum(len(loader) for loader in val_loaders), 1)

        val_metrics["val_loss"] = float(
            np.mean(losses)) if losses else float("inf")
        val_metrics["val_accuracy"] = float(np.mean(accs)) if accs else 0.0
        val_metrics["val_precision"] = float(
            np.mean(precisions)) if precisions else 0.0
        val_metrics["val_recall"] = float(np.mean(recalls)) if recalls else 0.0
        val_metrics["val_f1"] = float(np.mean(f1s)) if f1s else 0.0
        val_metrics["val_memory_entropy"] /= total_batches
        val_metrics["val_cross_modal_similarity"] /= total_batches

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model.train()
        return val_metrics

    # -----------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        try:
            checkpoint = {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "best_val_loss": self.best_val_loss,
                "config": self.config
            }

            checkpoint_path = self.checkpoint_dir / \
                f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)

            latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
            torch.save(checkpoint, latest_path)

            if is_best:
                best_path = self.checkpoint_dir / "best_checkpoint.pt"
                torch.save(checkpoint, best_path)
                logger.info(f"New best checkpoint saved: {best_path}")

            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self.scheduler and checkpoint["scheduler_state_dict"] is not None:
                self.scheduler.load_state_dict(
                    checkpoint["scheduler_state_dict"])

            self.current_epoch = checkpoint["epoch"]
            self.global_step = checkpoint["global_step"]
            self.best_val_loss = checkpoint["best_val_loss"]

            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return self.current_epoch
        except Exception as e:
            logger.error(
                f"Failed to load checkpoint from {checkpoint_path}: {e}")
            return 0

    # -----------------------------------------------------
    # Main loop
    # -----------------------------------------------------
    def train(self, max_samples: Optional[int] = None):
        logger.info("Starting MultimodalClassifier training...")

        self.setup_model_and_data(max_samples=max_samples)

        for epoch in range(self.current_epoch, self.config["training"]["max_epochs"]):
            logger.info(
                f"\nEpoch {epoch + 1}/{self.config['training']['max_epochs']}")

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)

            if self.scheduler and self.scheduler_step_mode == "epoch":
                self.scheduler.step()

            all_metrics = {
                **train_metrics,
                **val_metrics,
                "epoch": epoch,
                "learning_rate": self.optimizer.param_groups[0]["lr"]
            }

            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"Train Acc : {train_metrics['train_accuracy']:.4f}")
            logger.info(f"Train F1  : {train_metrics['train_f1']:.4f}")
            logger.info(f"Val Loss  : {val_metrics['val_loss']:.4f}")
            logger.info(f"Val Acc   : {val_metrics['val_accuracy']:.4f}")
            logger.info(f"Val F1    : {val_metrics['val_f1']:.4f}")

            if self.use_wandb:
                wandb.log(all_metrics)

            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]

            self.save_checkpoint(epoch, is_best=is_best)

        if self.use_wandb:
            wandb.finish()

        logger.info("Training completed!")


# ---------------------------------------------------------
# Config + main
# ---------------------------------------------------------
def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Train MultimodalClassifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/larimar_classifier_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Override max epochs from config"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Override batch size from config"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size

    trainer = MultimodalClassifierTrainer(config)

    if args.resume:
        trainer.setup_model_and_data(max_samples=args.max_samples)
        trainer.load_checkpoint(args.resume)

    trainer.train(max_samples=args.max_samples)


if __name__ == "__main__":
    main()
