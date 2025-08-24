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
        x, y = batch
        output = self.tft(x)
        prediction = output[0] if isinstance(output, tuple) else output
        loss = self.loss_fn(prediction, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.tft(x)
        prediction = output[0] if isinstance(output, tuple) else output
        loss = self.loss_fn(prediction, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=0.0001,
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }


def main():
    start_time = time.time()

    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)

    # 1. SETUP AND DATA LOADING
    print("Loading and preprocessing data...")

    # Load data
    train_df = pd.read_csv('Processed_Data/train_data.csv')
    val_df = pd.read_csv('Processed_Data/val_data.csv')

    # Filter data
    train_df = train_df[train_df['Symbol'] != 'UNKNOWN']
    val_df = val_df[val_df['Symbol'] != 'UNKNOWN']
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Filter validation data to only include symbols present in training data
    valid_symbols = set(train_df['Symbol'].unique())
    val_df = val_df[val_df['Symbol'].isin(valid_symbols)]

    # Filter sectors
    train_sectors = set(train_df['Sector'].unique())
    val_sectors = set(val_df['Sector'].unique())
    valid_sectors = train_sectors.intersection(val_sectors)
    train_df = train_df[train_df['Sector'].isin(valid_sectors)]
    val_df = val_df[val_df['Sector'].isin(valid_sectors)]
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # 2. PREPROCESSING
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    val_df['Date'] = pd.to_datetime(val_df['Date'])
    min_date = min(train_df['Date'].min(), val_df['Date'].min())
    train_df['time_idx'] = (train_df['Date'] - min_date).dt.days
    val_df['time_idx'] = (val_df['Date'] - min_date).dt.days

    # 3. CATEGORICAL VARIABLES
    all_symbols = pd.concat([train_df['Symbol'], val_df['Symbol']]).unique()
    all_sectors = pd.concat([train_df['Sector'], val_df['Sector']]).unique()
    train_df['Symbol'] = pd.Categorical(train_df['Symbol'], categories=all_symbols)
    val_df['Symbol'] = pd.Categorical(val_df['Symbol'], categories=all_symbols)
    train_df['Sector'] = pd.Categorical(train_df['Sector'], categories=all_sectors)
    val_df['Sector'] = pd.Categorical(val_df['Sector'], categories=all_sectors)

    # 4. TIMESERIESDATASET CONFIGURATION
    time_varying_unknown_reals = [
        "Open", "High", "Low", "Close", "Volume", "MA_5", "MA_20", "MA_50",
        "price_change", "volatility_20", "volume_ma_20", "volume_ratio", "RSI",
        "Market Cap", "Current Price", "High_fund", "Low_fund", "Stock P/E",
        "Book Value", "Dividend Yield", "ROE", "EPS", "Debt to equity",
        "Price to book value", "Volume_fund", "avg_sentiment", "news_count",
        "sentiment_std", "has_news"
    ]

    existing_columns = [col for col in time_varying_unknown_reals if col in train_df.columns]

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="future_close",
        group_ids=["Symbol"],
        max_encoder_length=60,
        max_prediction_length=7,
        static_categoricals=["Sector"],
        time_varying_known_reals=[],
        time_varying_unknown_reals=existing_columns,
        target_normalizer=GroupNormalizer(groups=["Symbol"], transformation="softplus", center=False),
        add_relative_time_idx=True,
        add_target_scales=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)

    # 5. DATALOADERS
    batch_size = 32
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # 6. MODEL SETUP
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.0001,
        hidden_size=128,
        attention_head_size=4,
        dropout=0.2,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
        output_size=7,
    )

    lightning_tft = TFTLightningWrapper(tft)

    # 7. CALLBACKS WITH CHECKPOINTING
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=15,
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
        save_last=True  # Save the last checkpoint as well
    )

    lr_logger = LearningRateMonitor()
    progress_bar = TQDMProgressBar()

    # 8. TRAINER SETUP
    trainer = pl.Trainer(
        max_epochs=50,
        gradient_clip_val=0.5,
        callbacks=[early_stop_callback, checkpoint_callback, lr_logger, progress_bar],
        enable_progress_bar=True,
        accelerator='cpu',
        devices=1,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=2,
    )

    # 9. TRAINING
    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")
    print("Starting training...")

    try:
        trainer.fit(
            lightning_tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        # 10. SAVE FINAL MODEL
        print("Saving best model...")
        best_tft = lightning_tft.tft
        with open('tft_model.pkl', 'wb') as f:
            pickle.dump(best_tft, f)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving latest model...")
        # Save the current model state
        best_tft = lightning_tft.tft
        with open('tft_model_interrupted.pkl', 'wb') as f:
            pickle.dump(best_tft, f)
        print("Model saved as 'tft_model_interrupted.pkl'")

    # 11. FINAL OUTPUT
    training_time = time.time() - start_time
    hours, rem = divmod(training_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("Training complete.")
    print(f"Total training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

    if hasattr(trainer, 'callback_metrics') and 'val_loss' in trainer.callback_metrics:
        val_loss = trainer.callback_metrics['val_loss'].item()
        print(f"Final validation loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()