#!/usr/bin/env python3
"""
TrainModel.py - Temporal Fusion Transformer for Stock Price Prediction

This script trains a TFT model to predict future closing prices of multiple stocks.
The model is trained on historical financial data with technical indicators and fundamental metrics.

Author: Data Scientist
Date: 2024
"""

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, TQDMProgressBar, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import pickle
import time
import warnings
import os
import traceback

warnings.filterwarnings('ignore')


# Create a wrapper class to make TFT compatible with PyTorch Lightning
class TFTLightningWrapper(pl.LightningModule):
    def __init__(self, tft_model):
        super().__init__()
        self.tft = tft_model
        self.loss_fn = QuantileLoss()
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def forward(self, x):
        return self.tft(x)

    def training_step(self, batch, batch_idx):
        x, y_tuple = batch
        y = y_tuple[0]  # Extract the target from the tuple

        # Ensure data is on the same device as model
        x = self._move_to_device(x)
        y = y.to(self.device)

        output = self.tft(x)

        # TFT returns a tuple (prediction, x) - we need the prediction part
        if isinstance(output, tuple):
            prediction = output[0]
        else:
            prediction = output

        # Debug: Check for extreme values
        if torch.isnan(prediction).any() or torch.isinf(prediction).any():
            print(f"WARNING: Invalid predictions detected")
            print(f"Prediction range: {prediction.min().item():.4f} to {prediction.max().item():.4f}")
            print(f"Target range: {y.min().item():.4f} to {y.max().item():.4f}")

        loss = self.loss_fn(prediction, y)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_tuple = batch
        y = y_tuple[0]  # Extract the target from the tuple

        # Ensure data is on the same device as model
        x = self._move_to_device(x)
        y = y.to(self.device)

        output = self.tft(x)

        # TFT returns a tuple (prediction, x) - we need the prediction part
        if isinstance(output, tuple):
            prediction = output[0]
        else:
            prediction = output

        loss = self.loss_fn(prediction, y)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        # Track best validation loss and save best model state
        current_val_loss = self.trainer.callback_metrics.get('val_loss')
        if current_val_loss is not None:
            current_val_loss = current_val_loss.item()
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                # Save the best model state
                self.best_model_state = {k: v.cpu().clone() for k, v in self.tft.state_dict().items()}
                print(f"🎉 New best validation loss: {current_val_loss:.4f}")

                # Check if we reached the target
                if current_val_loss <= 75.0:
                    print("🎯 TARGET ACHIEVED: Validation loss <= 75.0!")

    def _move_to_device(self, data):
        """Move all tensors in the data dictionary to the correct device"""
        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return data

    def configure_optimizers(self):
        # Use Adam optimizer with careful learning rate
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.001,  # Conservative learning rate
            weight_decay=1e-6,
            eps=1e-8
        )

        # Learning rate scheduler with careful settings
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,  # Gentle reduction
            patience=5,
            min_lr=1e-6,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.tft.predict(batch)

    def on_train_start(self):
        """Ensure model is on the correct device at training start"""
        self.tft = self.tft.to(self.device)
        print(f"Model moved to device: {self.device}")

    def on_validation_start(self):
        """Ensure model is on the correct device at validation start"""
        self.tft = self.tft.to(self.device)

    def _save_best_model(self):
        """Save the best model when training is interrupted"""
        if self.best_model_state is not None:
            # Restore best model weights
            self.tft.load_state_dict(self.best_model_state)
            print(f"Restored best model with validation loss: {self.best_val_loss:.4f}")

            # Save the best model
            torch.save({
                'model_state_dict': self.tft.state_dict(),
                'best_val_loss': self.best_val_loss,
            }, 'tft_model_interrupted.pt')

            print(f"💾 Best model saved as 'tft_model_interrupted.pt'")
            print(f"📈 Best validation loss achieved: {self.best_val_loss:.4f}")


def get_optimal_device():
    """
    Determine the best available device with memory optimization.
    MODIFIED: This function is fixed to avoid the MPS bug by falling back to CPU.
    """
    if torch.cuda.is_available():
        # CUDA GPU available
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3  # GB
        print(f"CUDA GPU detected: {gpu_name} with {gpu_memory:.1f}GB memory")

        if gpu_memory >= 16:  # High-end GPU (RTX 3080, A100, etc.)
            return 'cuda', 32, 128, 4, 60
        elif gpu_memory >= 8:  # Mid-range GPU (RTX 2070, etc.)
            return 'cuda', 16, 96, 3, 50
        else:  # Low-end GPU
            return 'cuda', 8, 64, 2, 40

    else:  # CPU fallback
        print("No CUDA GPU detected, using CPU. This will be slower.")
        return 'cpu', 16, 64, 2, 45


def main():
    """Main function to execute the TFT model training pipeline."""
    start_time = time.time()

    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)

    print("=" * 60)
    print("TRAINING TFT MODEL FOR STOCK PRICE PREDICTION")
    print("=" * 60)
    print("Target validation loss: 75.0")
    print("Auto-detecting optimal hardware configuration...")
    print("=" * 60)

    # 1. SETUP AND DATA LOADING
    print("Loading and preprocessing data...")

    # Load training and validation data from Processed_Data folder
    train_df = pd.read_csv('Processed_Data/train_data.csv')
    val_df = pd.read_csv('Processed_Data/val_data.csv')

    # Filter out 'UNKNOWN' symbol from both datasets
    train_df = train_df[train_df['Symbol'] != 'UNKNOWN']
    val_df = val_df[val_df['Symbol'] != 'UNKNOWN']

    # Reset indices after filtering
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # 2. CHECK DATA QUALITY AND TIME CONSISTENCY
    print("Checking data quality and time consistency...")

    print(f"NaN values in train: {train_df.isna().sum().sum()}")
    print(f"NaN values in val: {val_df.isna().sum().sum()}")

    # Convert Date column to datetime
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    val_df['Date'] = pd.to_datetime(val_df['Date'])

    # Check time ranges
    print(f"Train date range: {train_df['Date'].min()} to {train_df['Date'].max()}")
    print(f"Validation date range: {val_df['Date'].min()} to {val_df['Date'].max()}")

    # Create consistent time index across both datasets
    min_date = min(train_df['Date'].min(), val_df['Date'].min())
    train_df['time_idx'] = (train_df['Date'] - min_date).dt.days
    val_df['time_idx'] = (val_df['Date'] - min_date).dt.days

    print(f"Train time_idx range: {train_df['time_idx'].min()} to {train_df['time_idx'].max()}")
    print(f"Validation time_idx range: {val_df['time_idx'].min()} to {val_df['time_idx'].max()}")

    # 3. FILTER DATA PROPERLY
    # Filter validation data to only include symbols present in training data
    valid_symbols = set(train_df['Symbol'].unique())
    val_df = val_df[val_df['Symbol'].isin(valid_symbols)]

    # Filter sectors
    train_sectors = set(train_df['Sector'].unique())
    val_sectors = set(val_df['Sector'].unique())
    valid_sectors = train_sectors.intersection(val_sectors)
    train_df = train_df[train_df['Sector'].isin(valid_sectors)]
    val_df = val_df[val_df['Sector'].isin(valid_sectors)]

    # Reset indices again after additional filtering
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"Train samples after filtering: {len(train_df):,}")
    print(f"Validation samples after filtering: {len(val_df):,}")

    # Check target statistics after filtering
    print(f"Target stats - Train: mean={train_df['future_close'].mean():.2f}, std={train_df['future_close'].std():.2f}")
    print(f"Target stats - Val: mean={val_df['future_close'].mean():.2f}, std={val_df['future_close'].std():.2f}")

    mean_diff = abs(train_df['future_close'].mean() - val_df['future_close'].mean())
    if mean_diff > 500:
        print(f"⚠️  WARNING: Large difference between train and validation target means: {mean_diff:.2f}")
        print("This suggests different market conditions or data leakage")

    # 4. CATEGORICAL VARIABLES
    print("Processing categorical variables...")

    all_symbols = pd.concat([train_df['Symbol'], val_df['Symbol']]).unique()
    all_sectors = pd.concat([train_df['Sector'], val_df['Sector']]).unique()

    train_df['Symbol'] = pd.Categorical(train_df['Symbol'], categories=all_symbols)
    val_df['Symbol'] = pd.Categorical(val_df['Symbol'], categories=all_symbols)
    train_df['Sector'] = pd.Categorical(train_df['Sector'], categories=all_sectors)
    val_df['Sector'] = pd.Categorical(val_df['Sector'], categories=all_sectors)

    # 5. AUTO-DETECT OPTIMAL HARDWARE CONFIGURATION
    print("Auto-detecting optimal hardware configuration...")
    accelerator, batch_size, hidden_size, attention_heads, encoder_length = get_optimal_device()

    print(f"Optimal configuration:")
    print(f"  - Accelerator: {accelerator}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Attention heads: {attention_heads}")
    print(f"  - Encoder length: {encoder_length}")

    # 6. TIMESERIESDATASET CONFIGURATION
    print("Creating TimeSeriesDataSet objects...")

    # Use essential features only for stability
    basic_features = [
        "Open", "High", "Low", "Close", "Volume", "MA_5", "MA_20", "MA_50",
        "price_change", "volatility_20", "RSI", "volume_ma_20"
    ]

    existing_columns = [col for col in basic_features if col in train_df.columns]

    # Create training dataset with optimized settings
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="future_close",
        group_ids=["Symbol"],
        max_encoder_length=encoder_length,
        max_prediction_length=7,
        static_categoricals=["Sector"],
        time_varying_known_reals=[],
        time_varying_unknown_reals=existing_columns,
        target_normalizer=GroupNormalizer(
            groups=["Symbol"],
            transformation="softplus",
            center=True,
            scale_by_group=True
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        allow_missing_timesteps=True,
        min_encoder_length=max(10, encoder_length // 3),
    )

    # Create validation dataset
    validation = TimeSeriesDataSet.from_dataset(
        training, val_df, predict=False, stop_randomization=True
    )

    # 7. DATALOADERS - Use single worker to avoid multiprocessing issues
    print("Creating DataLoaders...")
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # 8. MODEL SETUP - Optimized for detected hardware
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=hidden_size,
        attention_head_size=attention_heads,
        dropout=0.2,  # Regularization
        loss=QuantileLoss(),
        log_interval=100,
        reduce_on_plateau_patience=4,
        output_size=7,
    )

    lightning_tft = TFTLightningWrapper(tft)

    # 9. CALLBACKS
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.05,  # Smaller minimum improvement
        patience=15,  # More patient early stopping
        verbose=True,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='tft-best-{epoch:03d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        every_n_epochs=1,
        save_last=True
    )

    lr_logger = LearningRateMonitor()

    # Custom progress bar with detailed information
    class DetailedProgressBar(TQDMProgressBar):
        def __init__(self):
            super().__init__()
            self.enable = True

        def on_train_epoch_start(self, trainer, pl_module):
            super().on_train_epoch_start(trainer, pl_module)
            # FIX: The check 'trainer.resumed_from_checkpoint' is removed as it no longer exists.
            # Checking if the current epoch is 0 is sufficient to know it's a new run.
            if trainer.current_epoch == 0:
                print(f"\n🚀 Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"⏰ Estimated training time: 4-8 hours")
                print("🎯 Target validation loss: 75.0")
                print(f"💻 Training on {accelerator.upper()}")
                print("📊 Training until validation loss stops improving")
                print("=" * 60)

    progress_bar = DetailedProgressBar()

    # 10. TRAINER SETUP - Optimized for detected hardware
    trainer_args = {
        'max_epochs': 200,
        'callbacks': [early_stop_callback, checkpoint_callback, lr_logger, progress_bar],
        'enable_progress_bar': True,
        'num_sanity_val_steps': 0,  # Disable sanity check to avoid issues
        'check_val_every_n_epoch': 1,
        'gradient_clip_val': 0.3,
        'accelerator': accelerator,
        'devices': 1,
        'log_every_n_steps': 25,
    }

    # Set precision based on accelerator
    if accelerator == 'cuda':
        trainer_args['precision'] = '16-mixed'  # Mixed precision for CUDA
    else:
        trainer_args['precision'] = '32'  # Full precision for MPS/CPU

    trainer = pl.Trainer(**trainer_args)

    # 11. TRAINING
    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")
    print(f"Using device: {accelerator}")

    # ADDED: Check for a checkpoint to resume training
    ckpt_path = "checkpoints/last.ckpt"
    if os.path.exists(ckpt_path):
        print(f"🔄 Found checkpoint. Resuming training from: {ckpt_path}")
        resume_from_checkpoint = ckpt_path
    else:
        print("🚀 No checkpoint found. Starting a new training session.")
        resume_from_checkpoint = None

    print("Starting training...")

    try:
        trainer.fit(
            lightning_tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=resume_from_checkpoint  # This will resume training if a checkpoint is found
        )

        # 12. SAVE FINAL MODEL
        print("Saving best model...")

        # Load the best model from checkpoint
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            print(f"Loading best model from: {best_model_path}")
            best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        else:
            print("No best model checkpoint found. Using the last state of the model.")
            best_tft = lightning_tft.tft

        # Save model using torch.save
        torch.save({
            'model_state_dict': best_tft.state_dict(),
            'training_dataset_parameters': training.get_parameters(),
            'best_val_loss': lightning_tft.best_val_loss,
        }, 'tft_model.pt')

        print(f"✅ Model saved as 'tft_model.pt'")
        print(f"🏆 Best validation loss achieved: {lightning_tft.best_val_loss:.4f}")

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user. Saving best model from this session...")
        lightning_tft._save_best_model()

    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        print("💾 Saving current model state for debugging...")
        torch.save({
            'model_state_dict': lightning_tft.tft.state_dict(),
            'training_dataset_parameters': training.get_parameters(),
        }, 'tft_model_error.pt')
        print("🔧 Debug model saved as 'tft_model_error.pt'")
        traceback.print_exc()

    # 13. FINAL OUTPUT
    training_time = time.time() - start_time
    hours, rem = divmod(training_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("=" * 60)
    print("🏁 TRAINING COMPLETE")
    print("=" * 60)
    print(f"⏱️  Total training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

    if hasattr(lightning_tft, 'best_val_loss') and lightning_tft.best_val_loss != float('inf'):
        print(f"📊 Best validation loss: {lightning_tft.best_val_loss:.4f}")

        if lightning_tft.best_val_loss <= 75.0:
            print("🎯 TARGET ACHIEVED! Validation loss <= 75.0")
        else:
            print(f"📉 Target not reached. Best was: {lightning_tft.best_val_loss:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()