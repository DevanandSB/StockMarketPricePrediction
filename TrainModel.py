#!/usr/bin/env python3
"""
TrainModel.py - Transformer for Stock Price Prediction with M1 GPU Support

This script trains a Transformer model to predict future closing prices of multiple stocks.
Optimized for Apple M1 GPU acceleration.

Author: Devanand S B
Date: 2025
"""

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, TQDMProgressBar, ModelCheckpoint
import pickle
import time
import warnings
import os
import signal
import sys
from pathlib import Path
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import math
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# Set device for M1 GPU acceleration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"✅ Using Apple M1 GPU (Metal)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using CUDA GPU")
else:
    device = torch.device("cpu")
    print(f"⚠️  Using CPU (no GPU available)")


# Custom Transformer Model
class StockTransformer(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim=256, num_layers=4, num_heads=8, dropout=0.1, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        # Transformer layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # Better for MPS
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Output layers - predict only 1 value (next day's close)
        self.output_projection = nn.Linear(hidden_dim, 64)
        self.output_layer = nn.Linear(64, 1)

        self.loss_fn = nn.MSELoss()
        self.best_val_loss = float('inf')

    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, input_dim]
        x = self.input_projection(x)
        x = self.pos_encoder(x)

        # Transformer expects [seq_len, batch_size, dim] for mask, but we use batch_first=True
        if mask is not None:
            mask = mask.transpose(0, 1)  # Convert to [seq_len, seq_len]

        x = self.transformer_encoder(x, mask=mask)

        # Use the last time step's output for prediction
        x = x[:, -1, :]  # Take the last time step

        x = torch.nn.functional.gelu(self.output_projection(x))
        x = self.output_layer(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = self.loss_fn(predictions, y)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = self.loss_fn(predictions, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_rmse', torch.sqrt(loss), prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
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


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# Custom Dataset - Optimized for GPU
class StockDataset(Dataset):
    def __init__(self, df, sequence_length=60, prediction_horizon=1, feature_columns=None):
        self.df = df
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.feature_columns = feature_columns or [
            "Open", "High", "Low", "Close", "Volume", "MA_5", "MA_20", "MA_50",
            "price_change", "volatility_20", "RSI", "volume_ma_20"
        ]
        self.sequences = []
        self.targets = []
        self._prepare_data()

    def _prepare_data(self):
        # Pre-allocate arrays for better performance
        total_sequences = 0
        for symbol, group in self.df.groupby('Symbol'):
            group = group.sort_values('time_idx')
            total_sequences += max(0, len(group) - self.sequence_length - self.prediction_horizon + 1)

        # Pre-allocate arrays
        self.sequences = np.zeros((total_sequences, self.sequence_length, len(self.feature_columns)), dtype=np.float32)
        self.targets = np.zeros(total_sequences, dtype=np.float32)

        idx = 0
        for symbol, group in self.df.groupby('Symbol'):
            group = group.sort_values('time_idx')
            values = group[self.feature_columns].values.astype(np.float32)
            targets = group['future_close_scaled'].values.astype(np.float32)

            for i in range(len(group) - self.sequence_length - self.prediction_horizon + 1):
                self.sequences[idx] = values[i:i + self.sequence_length]
                self.targets[idx] = targets[i + self.sequence_length]
                idx += 1

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.sequences[idx]).float()
        target = torch.tensor(self.targets[idx]).float()
        return seq, target


def signal_handler(signal, frame):
    print("\n⏹️  Training interrupted by user. Saving best model...")
    sys.exit(0)


def analyze_data_quality(df, dataset_name):
    """Analyze data quality before training"""
    print(f"\n📊 {dataset_name} DATA QUALITY ANALYSIS:")
    print(f"Samples: {len(df):,}")
    print(f"Symbols: {len(df['Symbol'].unique())}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Check for NaN values
    nan_counts = df.isna().sum()
    if nan_counts.any():
        print("⚠️  NaN values found:")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"   {col}: {count} NaN values")

    # Check target distribution
    print(f"Target (future_close) stats:")
    print(f"   Mean: {df['future_close'].mean():.4f}")
    print(f"   Std: {df['future_close'].std():.4f}")
    print(f"   Min: {df['future_close'].min():.4f}")
    print(f"   Max: {df['future_close'].max():.4f}")

    return df.dropna()


def main():
    signal.signal(signal.SIGINT, signal_handler)
    start_time = time.time()

    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("=" * 60)
    print("TRAINING TRANSFORMER MODEL FOR STOCK PRICE PREDICTION")
    print("=" * 60)
    print(f"Device: {device}")
    print("Target validation loss: 10.0")
    print("Using custom Transformer architecture with M1 GPU acceleration")
    print("=" * 60)

    # Load and preprocess data
    print("Loading and preprocessing data...")

    train_df = pd.read_csv('Processed_Data/train_data.csv')
    val_df = pd.read_csv('Processed_Data/val_data.csv')

    # Filter data
    train_df = train_df[train_df['Symbol'] != 'UNKNOWN']
    val_df = val_df[val_df['Symbol'] != 'UNKNOWN']

    # Convert dates and create time index
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    val_df['Date'] = pd.to_datetime(val_df['Date'])
    min_date = min(train_df['Date'].min(), val_df['Date'].min())
    train_df['time_idx'] = (train_df['Date'] - min_date).dt.days
    val_df['time_idx'] = (val_df['Date'] - min_date).dt.days

    # Filter symbols
    valid_symbols = set(train_df['Symbol'].unique())
    val_df = val_df[val_df['Symbol'].isin(valid_symbols)]

    # Analyze data quality
    train_df = analyze_data_quality(train_df, "TRAINING")
    val_df = analyze_data_quality(val_df, "VALIDATION")

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"Train samples after cleaning: {len(train_df):,}")
    print(f"Validation samples after cleaning: {len(val_df):,}")

    # Feature columns
    feature_columns = [
        "Open", "High", "Low", "Close", "Volume",
        "MA_5", "MA_20", "MA_50", "price_change", "RSI"
    ]

    # Save feature list
    print(f"✅ Saving the feature list ({len(feature_columns)} features) to model_features.pkl...")
    with open('model_features.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    val_df[feature_columns] = scaler.transform(val_df[feature_columns])

    # Scale target
    target_scaler = StandardScaler()
    train_df['future_close_scaled'] = target_scaler.fit_transform(train_df[['future_close']])
    val_df['future_close_scaled'] = target_scaler.transform(val_df[['future_close']])

    # Create datasets
    print("Creating datasets...")
    train_dataset = StockDataset(
        train_df,
        sequence_length=60,
        prediction_horizon=1,
        feature_columns=feature_columns
    )
    val_dataset = StockDataset(
        val_df,
        sequence_length=60,
        prediction_horizon=1,
        feature_columns=feature_columns
    )

    print(f"Train sequences: {len(train_dataset):,}")
    print(f"Validation sequences: {len(val_dataset):,}")

    # Create dataloaders with optimized settings for MPS
    batch_size = 128  # Increased batch size for GPU
    num_workers = 0  # MPS doesn't support multiple workers well

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Create model
    input_dim = len(feature_columns)
    model = StockTransformer(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.3,
        learning_rate=0.0005
    )

    # Move model to device
    model.to(device)

    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=15,
        mode="min",
        verbose=True,
        min_delta=0.001
    )
    checkpoint = ModelCheckpoint(
        dirpath='checkpoints',
        filename='transformer-best-{epoch:03d}-{val_loss:.4f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    lr_monitor = LearningRateMonitor()

    # Trainer configuration for MPS
    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=[early_stop, checkpoint, lr_monitor],
        accelerator='mps' if torch.backends.mps.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        precision='32',  # MPS works best with float32
        deterministic=True,
    )

    # Train
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Starting training with M1 GPU acceleration...")

    try:
        trainer.fit(model, train_loader, val_loader)

        # Load best model
        best_model_path = checkpoint.best_model_path
        if best_model_path:
            print(f"Loading best model from: {best_model_path}")
            model = StockTransformer.load_from_checkpoint(best_model_path)
            model.to(device)

        # Save models
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_scaler': scaler,
            'target_scaler': target_scaler,
            'feature_columns': feature_columns,
            'best_val_loss': trainer.callback_metrics.get('val_loss', float('inf')).item()
        }, 'models/transformer_best_model.pt')

        best_val_loss = trainer.callback_metrics.get('val_loss', float('inf')).item()
        print(f"✅ Best model saved with validation loss: {best_val_loss:.4f}")

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted. Saving current model...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_scaler': scaler,
            'target_scaler': target_scaler,
            'feature_columns': feature_columns
        }, 'models/transformer_interrupted.pt')
        print("💾 Model saved")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_scaler': scaler,
            'target_scaler': target_scaler
        }, 'models/transformer_error.pt')
        print("🔧 Error model saved")

    # Final output
    training_time = time.time() - start_time
    hours, minutes, seconds = int(training_time // 3600), int((training_time % 3600) // 60), int(training_time % 60)

    print("=" * 60)
    print("🏁 TRAINING COMPLETE")
    print("=" * 60)
    print(f"⏱️  Time: {hours:02d}:{minutes:02d}:{seconds:02d}")

    best_val_loss = trainer.callback_metrics.get('val_loss', float('inf')).item()
    print(f"📊 Best validation loss: {best_val_loss:.4f}")

    if best_val_loss <= 10.0:
        print("🎯 TARGET ACHIEVED!")
    else:
        print("📉 Target not reached")

    print("=" * 60)


if __name__ == "__main__":
    main()