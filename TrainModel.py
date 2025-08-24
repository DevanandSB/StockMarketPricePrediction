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
        # Properly handle TFT output format
        x, y = batch
        output = self.tft(x)

        # TFT returns a tuple (prediction, x) - we need the prediction part
        if isinstance(output, tuple):
            prediction = output[0]
        else:
            prediction = output

        loss = self.loss_fn(prediction, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Properly handle TFT output format
        x, y = batch
        output = self.tft(x)

        # TFT returns a tuple (prediction, x) - we need the prediction part
        if isinstance(output, tuple):
            prediction = output[0]
        else:
            prediction = output

        loss = self.loss_fn(prediction, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Use a more sophisticated optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=0.0001,  # Lower learning rate
            weight_decay=1e-5  # Add weight decay for regularization
        )

        # Use a simple step-based scheduler instead of ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,  # Reduce LR every 10 epochs
            gamma=0.5  # Reduce by half
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # Update after each epoch
            }
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.tft.predict(batch)


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
    # First, identify all unique categories across both datasets
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

    # Create training dataset with different normalization
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
            center=False  # Don't center to avoid negative values
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
        patience=15,  # Increased patience
        verbose=True,
        mode="min"
    )

    # Add model checkpointing
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
                print("Estimated training time: 5-8 hours (depending on hardware)")

    progress_bar = CustomProgressBar()

    # Setup trainer with progress bar - removed gradient_clip_val
    trainer = pl.Trainer(
        max_epochs=50,  # Increased max epochs
        callbacks=[early_stop_callback, checkpoint_callback, lr_logger, progress_bar],
        enable_progress_bar=True,
        accelerator='cpu',
        devices=1,
        num_sanity_val_steps=1,  # Run validation at start
        check_val_every_n_epoch=1,  # Check validation every epoch
    )

    # Initialize TFT model with better hyperparameters
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.0001,  # Lower learning rate
        hidden_size=128,
        attention_head_size=4,
        dropout=0.2,  # Increased dropout for regularization
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
        output_size=7,  # Explicit output size for 7-day prediction
    )

    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")
    print("Using learning rate of 0.0001 with AdamW optimizer and StepLR scheduler")

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

        # 7. SAVE THE FINAL MODEL
        print("Saving best model...")

        # Extract the actual TFT model from the wrapper
        best_tft = lightning_tft.tft

        # Save model using pickle
        with open('tft_model.pkl', 'wb') as f:
            pickle.dump(best_tft, f)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving latest model...")
        # Save the current model state
        best_tft = lightning_tft.tft
        with open('tft_model_interrupted.pkl', 'wb') as f:
            pickle.dump(best_tft, f)
        print("Model saved as 'tft_model_interrupted.pkl'")

    # 8. PROVIDE CONFIRMATION
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