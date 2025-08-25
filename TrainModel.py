#!/usr/bin/env python3
"""
COMPLETE Stock Prediction Transformer - With All Features & Fixed Data
"""

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
import pickle
import time
import warnings
import os
import signal
import sys
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, RobustScaler
import math

warnings.filterwarnings('ignore')

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Using device: {device}")


def signal_handler(signal, frame):
    print("\n⏹️  Training interrupted. Saving model...")
    sys.exit(0)


def create_time_based_split(df, train_ratio=0.8):
    """Create proper time-based split for each symbol"""
    df = df.sort_values(['Symbol', 'Date'])

    train_dfs, val_dfs = [], []

    for symbol, symbol_data in df.groupby('Symbol'):
        symbol_data = symbol_data.sort_values('Date')
        split_idx = int(len(symbol_data) * train_ratio)

        train_dfs.append(symbol_data.iloc[:split_idx])
        val_dfs.append(symbol_data.iloc[split_idx:])

    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)

    return train_df, val_df


# ENHANCED TRANSFORMER WITH ALL FEATURES
class CompleteTransformer(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.2, learning_rate=0.0003):
        super().__init__()
        self.save_hyperparameters()

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output layers
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.input_proj(x)
        x = self.input_dropout(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # Last time step
        return self.output(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-5
        )
        return optimizer


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
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


# DATASET
class StockDataset(Dataset):
    def __init__(self, df, sequence_length=20, feature_columns=None):
        self.df = df
        self.seq_len = sequence_length
        self.feature_columns = feature_columns
        self.sequences, self.targets = self._prepare_data()

    def _prepare_data(self):
        sequences = []
        targets = []

        for symbol, group in self.df.groupby('Symbol'):
            group = group.sort_values('Date')

            values = group[self.feature_columns].values.astype(np.float32)
            future_closes = group['future_close_scaled'].values.astype(np.float32)

            for i in range(len(group) - self.seq_len - 1):
                seq = values[i:i + self.seq_len]
                target = future_closes[i + self.seq_len]

                sequences.append(seq)
                targets.append(target)

        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.targets[idx]])
        )


def prepare_data():
    """Prepare data with proper time-based split and ALL features"""
    print("📊 Loading data with proper time-based split...")

    # Load all data and split properly
    all_data = pd.read_csv('Processed_Data/train_data.csv')
    all_data = pd.concat([all_data, pd.read_csv('Processed_Data/val_data.csv')])
    all_data = all_data[all_data['Symbol'] != 'UNKNOWN'].copy()
    all_data['Date'] = pd.to_datetime(all_data['Date'])

    # Create proper time-based split
    train_df, val_df = create_time_based_split(all_data, train_ratio=0.8)

    print(f"✅ Proper split - Train: {len(train_df):,}, Val: {len(val_df):,}")

    # Get ALL features (technical + fundamental + sentiment)
    exclude_cols = ['Date', 'Symbol', 'time_idx', 'future_close', 'future_close_scaled']
    all_features = [col for col in train_df.columns if col not in exclude_cols]

    # Clean numeric data
    for col in all_features:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        val_df[col] = pd.to_numeric(val_df[col], errors='coerce')

    # Fill NaN values
    train_df = train_df.fillna(0)
    val_df = val_df.fillna(0)

    print(f"✅ Using {len(all_features)} features:")
    print("   - Technical indicators")
    print("   - Fundamental analysis")
    print("   - Sentiment analysis")

    # Scale features
    scaler = RobustScaler()
    train_df[all_features] = scaler.fit_transform(train_df[all_features])
    val_df[all_features] = scaler.transform(val_df[all_features])

    # Scale target
    target_scaler = RobustScaler()
    train_df['future_close_scaled'] = target_scaler.fit_transform(train_df[['future_close']])
    val_df['future_close_scaled'] = target_scaler.transform(val_df[['future_close']])

    return train_df, val_df, all_features, scaler, target_scaler


def main():
    signal.signal(signal.SIGINT, signal_handler)
    start_time = time.time()

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("=" * 60)
    print("🎯 COMPLETE STOCK PREDICTION TRANSFORMER")
    print("🎯 With Technical + Fundamental + Sentiment Analysis")
    print("=" * 60)

    # Prepare data with proper split and ALL features
    train_df, val_df, all_features, scaler, target_scaler = prepare_data()

    # Create datasets
    train_dataset = StockDataset(train_df, sequence_length=20, feature_columns=all_features)
    val_dataset = StockDataset(val_df, sequence_length=20, feature_columns=all_features)

    print(f"📊 Sequences - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Complete model with all features
    model = CompleteTransformer(
        input_dim=len(all_features),
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.2,
        learning_rate=0.0003
    ).to(device)

    # Callbacks
    checkpoint = ModelCheckpoint(
        dirpath='checkpoints',
        filename='complete-model-{epoch}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        min_delta=0.001,
        verbose=True
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[early_stop, checkpoint, LearningRateMonitor()],
        accelerator='mps' if torch.backends.mps.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=20,
        enable_progress_bar=True
    )

    print(f"🚀 Starting training with {sum(p.numel() for p in model.parameters()):,} parameters")
    print("=" * 60)

    try:
        trainer.fit(model, train_loader, val_loader)

        if checkpoint.best_model_path:
            model = CompleteTransformer.load_from_checkpoint(checkpoint.best_model_path)
            best_val_loss = checkpoint.best_model_score.item() if checkpoint.best_model_score else float('inf')
            print(f"✅ Best model validation loss: {best_val_loss:.4f}")

        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_scaler': scaler,
            'target_scaler': target_scaler,
            'feature_columns': all_features,
            'best_val_loss': best_val_loss
        }, 'models/complete_transformer.pt')

        print("💾 Complete model saved with ALL features")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    # RESULTS
    training_time = time.time() - start_time
    hours, minutes, seconds = int(training_time // 3600), int((training_time % 3600) // 60), int(training_time % 60)

    print("=" * 60)
    print("🏁 TRAINING COMPLETE")
    print("=" * 60)
    print(f"⏱️  Time: {hours:02d}:{minutes:02d}:{seconds:02d}")

    if checkpoint.best_model_score:
        final_loss = checkpoint.best_model_score.item()
        print(f"📊 Best validation loss: {final_loss:.4f}")

        if final_loss < 0.1:
            print("🎉 EXCELLENT PERFORMANCE! (Loss < 0.1)")
        elif final_loss < 0.5:
            print("✅ GOOD PERFORMANCE! (Loss < 0.5)")
        else:
            print("⚠️  NEEDS IMPROVEMENT")

    print("=" * 60)


if __name__ == "__main__":
    main()