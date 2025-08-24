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

warnings.filterwarnings('ignore')


# Create a wrapper class to make TFT compatible with PyTorch Lightning
class TFTLightningWrapper(pl.LightningModule):
    def __init__(self, tft_model):
        super().__init__()
        self.tft = tft_model
        self.loss_fn = QuantileLoss()

    def forward(self, x):
        return self.tft(x)

    def training_step(self, batch, batch_idx):
        # TimeSeriesDataSet returns (x, y) where y is a tuple (target, weight)
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

        loss = self.loss_fn(prediction, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # TimeSeriesDataSet returns (x, y) where y is a tuple (target, weight)
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
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def _move_to_device(self, data):
        """Move all tensors in the data dictionary to the correct device"""
        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return data

    def configure_optimizers(self):
        # Use AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=0.001,  # Reasonable learning rate
            weight_decay=1e-5,
            eps=1e-8  # Numerical stability
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-6
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


def main():
    """Main function to execute the TFT model training pipeline."""

    start_time = time.time()

    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)

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

    # Filter validation data to only include symbols present in training data
    valid_symbols = set(train_df['Symbol'].unique())
    val_df = val_df[val_df['Symbol'].isin(valid_symbols)]

    # Also filter training and validation data to only include sectors present in both datasets
    train_sectors = set(train_df['Sector'].unique())
    val_sectors = set(val_df['Sector'].unique())
    valid_sectors = train_sectors.intersection(val_sectors)

    train_df = train_df[train_df['Sector'].isin(valid_sectors)]
    val_df = val_df[val_df['Sector'].isin(valid_sectors)]

    # Reset indices again after additional filtering
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # 2. ESSENTIAL PREPROCESSING
    print("Checking data quality...")

    # Check for NaN values and basic statistics
    print(f"NaN values in train: {train_df.isna().sum().sum()}")
    print(f"NaN values in val: {val_df.isna().sum().sum()}")
    print(f"Target stats - Train: mean={train_df['future_close'].mean():.2f}, std={train_df['future_close'].std():.2f}")
    print(f"Target stats - Val: mean={val_df['future_close'].mean():.2f}, std={val_df['future_close'].std():.2f}")

    # Convert Date column to datetime
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    val_df['Date'] = pd.to_datetime(val_df['Date'])

    # Create consistent time index across both datasets
    min_date = min(train_df['Date'].min(), val_df['Date'].min())
    train_df['time_idx'] = (train_df['Date'] - min_date).dt.days
    val_df['time_idx'] = (val_df['Date'] - min_date).dt.days

    # 3. HANDLE CATEGORICAL VARIABLES ROBUSTLY
    print("Processing categorical variables...")

    # Ensure both datasets have the same categories for Symbol and Sector
    all_symbols = pd.concat([train_df['Symbol'], val_df['Symbol']]).unique()
    all_sectors = pd.concat([train_df['Sector'], val_df['Sector']]).unique()

    # Convert to categorical with all possible categories
    train_df['Symbol'] = pd.Categorical(train_df['Symbol'], categories=all_symbols)
    val_df['Symbol'] = pd.Categorical(val_df['Symbol'], categories=all_symbols)

    train_df['Sector'] = pd.Categorical(train_df['Sector'], categories=all_sectors)
    val_df['Sector'] = pd.Categorical(val_df['Sector'], categories=all_sectors)

    # 4. TIMESERIESDATASET CONFIGURATION
    print("Creating TimeSeriesDataSet objects...")

    # Define time-varying unknown real features
    time_varying_unknown_reals = [
        "Open", "High", "Low", "Close", "Volume", "MA_5", "MA_20", "MA_50",
        "price_change", "volatility_20", "volume_ma_20", "volume_ratio", "RSI",
        "Market Cap", "Current Price", "High_fund", "Low_fund", "Stock P/E",
        "Book Value", "Dividend Yield", "ROE", "EPS", "Debt to equity",
        "Price to book value", "Volume_fund", "avg_sentiment", "news_count",
        "sentiment_std", "has_news"
    ]

    # Filter to only include columns that actually exist in the data
    existing_columns = [col for col in time_varying_unknown_reals if col in train_df.columns]

    # Create training dataset
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="future_close",
        group_ids=["Symbol"],
        max_encoder_length=60,
        max_prediction_length=7,
        static_categoricals=["Sector"],
        time_varying_known_reals=[],  # No known future features
        time_varying_unknown_reals=existing_columns,
        target_normalizer=GroupNormalizer(
            groups=["Symbol"],
            transformation="softplus",
            center=False
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        allow_missing_timesteps=True,
    )

    # Create validation dataset with same parameters
    validation = TimeSeriesDataSet.from_dataset(
        training, val_df, predict=True, stop_randomization=True
    )

    # 5. CREATE PYTORCH DATALOADERS
    print("Creating DataLoaders...")

    batch_size = 32
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )

    # 6. MODEL TRAINING AND OPTIMIZATION
    print("Setting up model and trainer...")

    # Configure callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
        verbose=True,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='tft-best-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        every_n_epochs=1,
        save_last=True
    )

    lr_logger = LearningRateMonitor()

    # Create a custom progress bar with time estimation
    class CustomProgressBar(TQDMProgressBar):
        def __init__(self):
            super().__init__()
            self.enable = True

        def on_train_epoch_start(self, trainer, pl_module):
            super().on_train_epoch_start(trainer, pl_module)
            if trainer.current_epoch == 0:
                print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print("Estimated training time: 1-3 hours (with GPU), 5-8 hours (CPU only)")

    progress_bar = CustomProgressBar()

    # 7. TRAINER SETUP - Proper Apple Silicon MPS support
    trainer_args = {
        'max_epochs': 50,
        'callbacks': [early_stop_callback, checkpoint_callback, lr_logger, progress_bar],
        'enable_progress_bar': True,
        'num_sanity_val_steps': 0,
        'check_val_every_n_epoch': 1,
        'gradient_clip_val': 0.1,
    }

    # Proper device detection with Apple Silicon MPS support
    if torch.backends.mps.is_available():
        trainer_args['accelerator'] = 'mps'
        trainer_args['devices'] = 1
        print("MPS (Apple Silicon) detected - training on MPS GPU")
    elif torch.cuda.is_available():
        trainer_args['accelerator'] = 'gpu'
        trainer_args['devices'] = 1
        print("CUDA GPU detected - training on GPU")
    else:
        trainer_args['accelerator'] = 'cpu'
        trainer_args['devices'] = 1
        print("No GPU detected - training on CPU")

    trainer = pl.Trainer(**trainer_args)

    # Initialize TFT model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=128,
        attention_head_size=4,
        dropout=0.1,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
        output_size=7,
    )

    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")
    print("Using learning rate of 0.001 with AdamW optimizer")

    # Wrap the TFT model for compatibility
    lightning_tft = TFTLightningWrapper(tft)

    # Train the model
    print("Starting training...")

    try:
        trainer.fit(
            lightning_tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        # 8. SAVE THE FINAL MODEL
        print("Saving best model...")

        # Extract the actual TFT model from the wrapper
        best_tft = lightning_tft.tft

        # Save model using torch.save to avoid pickle issues
        torch.save({
            'model_state_dict': best_tft.state_dict(),
            'training_dataset_parameters': training.get_parameters(),
            'model_config': {
                'hidden_size': 128,
                'attention_head_size': 4,
                'dropout': 0.1,
                'output_size': 7,
            }
        }, 'tft_model.pt')

        print("Model saved as 'tft_model.pt'")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving latest model...")
        best_tft = lightning_tft.tft
        torch.save({
            'model_state_dict': best_tft.state_dict(),
            'training_dataset_parameters': training.get_parameters(),
        }, 'tft_model_interrupted.pt')
        print("Model saved as 'tft_model_interrupted.pt'")

    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        print("Saving current model state for debugging...")
        torch.save({
            'model_state_dict': lightning_tft.tft.state_dict(),
            'training_dataset_parameters': training.get_parameters(),
        }, 'tft_model_error.pt')
        print("Model saved as 'tft_model_error.pt'")

    # 9. PROVIDE CONFIRMATION
    training_time = time.time() - start_time
    hours, rem = divmod(training_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("Training complete.")
    print(f"Total training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

    # Print final validation loss for reference
    if hasattr(trainer, 'callback_metrics') and 'val_loss' in trainer.callback_metrics:
        val_loss = trainer.callback_metrics['val_loss'].item()
        print(f"Final validation loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()