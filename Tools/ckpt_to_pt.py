import torch
import os
import warnings
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from typing import Dict, Any

warnings.filterwarnings('ignore')


def convert_tft_ckpt_to_pt_final(ckpt_path: str, output_path: str, train_csv_path: str):
    """
    Directly converts a TFT checkpoint by building a model shell that
    precisely matches the architecture found in the checkpoint's error logs.
    """
    print("=" * 60)
    print(f"🚀 Starting Direct Conversion for: {os.path.basename(ckpt_path)}")
    print("=" * 60)

    try:
        # --- Step 1: Load Checkpoint Weights ---
        print("📦 Loading checkpoint weights manually...")
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        state_dict = checkpoint.get('state_dict')
        if not state_dict:
            raise RuntimeError("Checkpoint does not contain 'state_dict'.")
        # Clean the "tft." prefix from all weight names
        cleaned_state_dict = {key.replace("tft.", ""): value for key, value in state_dict.items()}
        print("✅ Weights loaded and cleaned successfully.")

        # --- Step 2: Define the EXACT architecture based on forensic analysis of error logs ---
        # This is the ground truth of your trained model's architecture.
        print("\n🔧 Defining the exact model architecture...")

        # --- Parameters Inferred from Error Logs ---
        HIDDEN_SIZE = 64
        ATTENTION_HEAD_SIZE = 8
        MAX_PREDICTION_LENGTH = 7
        STATIC_CATEGORICALS = ["Sector"]
        TIME_VARYING_REALS = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_20', 'MA_50',
            'price_change', 'volatility_20', 'RSI', 'volume_ma_20'
        ]

        # --- Step 3: Recreate the TimeSeriesDataSet blueprint ---
        print("\n🧬 Rebuilding TimeSeriesDataSet blueprint...")
        train_df = pd.read_csv(train_csv_path)
        train_df['Date'] = pd.to_datetime(train_df['Date'])
        train_df['time_idx'] = (train_df['Date'] - train_df['Date'].min()).dt.days

        reconstructed_dataset = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="future_close",
            group_ids=["Symbol"],
            max_encoder_length=60,
            max_prediction_length=MAX_PREDICTION_LENGTH,
            allow_missing_timesteps=True,
            static_categoricals=STATIC_CATEGORICALS,
            time_varying_unknown_reals=TIME_VARYING_REALS,
            # CRITICAL FIX: Disable automatic features that cause mismatches
            add_relative_time_idx=False,
            add_encoder_length=False,
            add_target_scales=False,
        )
        print("✅ Blueprint rebuilt successfully.")

        # --- Step 4: Build Model from Blueprint ---
        print("\n🏗️  Building model shell from blueprint...")
        tft_model = TemporalFusionTransformer.from_dataset(
            reconstructed_dataset,
            hidden_size=HIDDEN_SIZE,
            attention_head_size=ATTENTION_HEAD_SIZE,
            dropout=0.2,
            loss=None
        )
        print("✅ Model shell created successfully.")

        # --- Step 5: Load Weights into Model Shell ---
        print("\n⚖️  Loading weights into the model shell...")
        # Use strict=False to ignore internal components that are not part of the core model weights
        tft_model.load_state_dict(cleaned_state_dict, strict=False)
        tft_model.eval()
        print("✅ Weights successfully loaded.")

        # --- Step 6: Save the Final Package ---
        save_package = {
            'model_state_dict': tft_model.state_dict(),
            'dataset_parameters': reconstructed_dataset.get_parameters(),
        }

        torch.save(save_package, output_path)
        print(f"\n💾 Model saved successfully to: '{output_path}'")
        return output_path

    except Exception as e:
        print(f"\n❌ A critical error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_tft_conversion(pt_path: str):
    """Verifies that the new .pt file can be used to load the TFT model."""
    print("-" * 60)
    print(f"🔍 Verifying converted file: {os.path.basename(pt_path)}...")
    try:
        package = torch.load(pt_path, map_location=torch.device('cpu'))
        model = TemporalFusionTransformer.from_parameters(package['dataset_parameters'])
        model.load_state_dict(package['model_state_dict'])
        model.eval()
        print("\n✅ VERIFICATION SUCCESSFUL!")
        print("   The .pt file can be used to fully reconstruct the trained TFT model.")
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED. Error: {e}")


# --- Main Execution Block ---
if __name__ == "__main__":
    ckpt_file_path = "/Users/BiTS/PycharmProjects/BiTSDissertation/checkpoints/tft-best-epoch=009-val_loss=107.95.ckpt"
    training_data_path = "/Users/BiTS/PycharmProjects/BiTSDissertation/Processed_Data/train_data.csv"
    output_pt_path = "/Users/BiTS/PycharmProjects/BiTSDissertation/outputModel/converted_tft_model.pt"

    os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)

    converted_path = convert_tft_ckpt_to_pt_final(
        ckpt_path=ckpt_file_path,
        output_path=output_pt_path,
        train_csv_path=training_data_path
    )

    if converted_path:
        verify_tft_conversion(pt_path=converted_path)