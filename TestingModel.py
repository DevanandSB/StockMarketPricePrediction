import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Try to import the TFT model
try:
    from tft_model import TFTModel

    print("✓ Successfully imported TFTModel")
except ImportError:
    print("❌ Could not import TFTModel, creating it inline...")

    # Define the model classes inline if import fails
    import math


    class PositionalEncoding(torch.nn.Module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            pe = torch.zeros(1, max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(0).unsqueeze(2)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        def forward(self, x):
            return x + self.pe[:, :x.size(1), :]


    class TransformerEncoderLayer(torch.nn.Module):
        def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
            super().__init__()
            self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
            self.dropout = torch.nn.Dropout(dropout)
            self.linear2 = torch.nn.Linear(dim_feedforward, d_model)
            self.norm1 = torch.nn.LayerNorm(d_model)
            self.norm2 = torch.nn.LayerNorm(d_model)
            self.dropout1 = torch.nn.Dropout(dropout)
            self.dropout2 = torch.nn.Dropout(dropout)

        def forward(self, src, src_mask=None, src_key_padding_mask=None):
            src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            return src


    class TransformerEncoder(torch.nn.Module):
        def __init__(self, encoder_layer, num_layers):
            super().__init__()
            self.layers = torch.nn.ModuleList([encoder_layer for _ in range(num_layers)])

        def forward(self, src, mask=None, src_key_padding_mask=None):
            output = src
            for layer in self.layers:
                output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            return output


    class TFTModel(torch.nn.Module):
        def __init__(self, input_dim=12, hidden_dim=128, num_layers=3, num_heads=4, dropout=0.2, output_dim=7):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim

            self.input_projection = torch.nn.Linear(input_dim, hidden_dim)
            self.pos_encoder = PositionalEncoding(hidden_dim)

            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout
            )
            self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

            self.output_projection = torch.nn.Linear(hidden_dim, hidden_dim)
            self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)

            batch_size, seq_len, input_dim = x.shape
            x = self.input_projection(x)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            x = x[:, -1, :]
            x = self.output_projection(x)
            x = self.output_layer(x)
            return x


class TFTTester:
    def __init__(self, model_path, data_dir="Processed_Data"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model_path = model_path
        self.data_dir = Path(data_dir)
        self.model = None
        self.hyperparameters = None
        self.results = {}

    def load_model(self):
        """Load model with direct state dict manipulation"""
        try:
            print("Loading model with direct state dict manipulation...")
            model_data = torch.load(self.model_path, map_location=self.device)

            self.hyperparameters = model_data.get('hyperparameters', {})
            state_dict = model_data['state_dict']

            # Create model
            self.model = TFTModel(
                input_dim=12,
                hidden_dim=128,
                num_layers=3,
                num_heads=4,
                dropout=0.2,
                output_dim=7
            )

            # Get model's state dict
            model_sd = self.model.state_dict()

            # Manually copy weights that have matching shapes
            for key in model_sd.keys():
                if key in state_dict and state_dict[key].shape == model_sd[key].shape:
                    model_sd[key] = state_dict[key]
                    print(f"✓ Copied: {key}")
                else:
                    print(f"✗ Skipped: {key} (not found or shape mismatch)")

            # Load the modified state dict
            self.model.load_state_dict(model_sd)
            self.model.to(self.device)
            self.model.eval()
            print("✓ Model loaded with manual weight copying")

        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")
            raise e

    # TestingModel.py (updated load_test_data method)
    def load_test_data(self):
        """Load test data and handle input dimension mismatch"""
        print("Loading test data...")
        test_data_path = self.data_dir / "test_data.csv"

        if not test_data_path.exists():
            raise FileNotFoundError(f"Test data not found at {test_data_path}")

        # Load data
        df = pd.read_csv(test_data_path)
        print(f"Test data shape: {df.shape}")

        # Identify potential target columns
        target_candidates = ['target', 'target_return', 'future_close', 'Close', 'Price', 'return']
        target_col = None

        for candidate in target_candidates:
            if candidate in df.columns:
                target_col = candidate
                break

        if target_col is None:
            # Use last numeric column as target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                target_col = numeric_cols[-1]
            else:
                raise ValueError("No target column found")

        print(f"Using target column: {target_col}")

        # Get the expected input dimension from the model
        expected_input_dim = self.model.input_dim
        print(f"Model expects input dimension: {expected_input_dim}")

        # Prepare features - exclude non-numeric and target columns
        feature_cols = []
        for col in df.columns:
            if col != target_col:
                try:
                    # Try to convert to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].dtype in [np.int64, np.float64]:
                        feature_cols.append(col)
                except:
                    # Skip columns that can't be converted
                    continue

        print(f"Available feature columns: {len(feature_cols)}")

        # Select the most important features or use feature selection
        if len(feature_cols) > expected_input_dim:
            print(f"Too many features ({len(feature_cols)}), selecting top {expected_input_dim}...")

            # Option 1: Use correlation with target to select most important features
            correlations = {}
            for col in feature_cols:
                try:
                    correlation = abs(df[col].corr(df[target_col]))
                    correlations[col] = correlation
                except:
                    correlations[col] = 0

            # Sort by correlation and select top features
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feat[0] for feat in sorted_features[:expected_input_dim]]

            print(f"Selected top {expected_input_dim} features by correlation:")
            for i, feat in enumerate(selected_features):
                print(f"  {i + 1}. {feat} (corr: {correlations[feat]:.3f})")

            feature_cols = selected_features

        elif len(feature_cols) < expected_input_dim:
            print(f"Warning: Only {len(feature_cols)} features available, but model expects {expected_input_dim}")
            print("Using all available features and padding with zeros")
            # Pad with zeros to match expected dimension
            padding_needed = expected_input_dim - len(feature_cols)
            for i in range(padding_needed):
                df[f'padding_{i}'] = 0
                feature_cols.append(f'padding_{i}')

        # Handle missing values
        df[feature_cols] = df[feature_cols].fillna(0)
        df[target_col] = df[target_col].fillna(0)

        # Convert to numpy arrays
        self.X_test = df[feature_cols].values.astype(np.float32)
        self.y_test = df[target_col].values.astype(np.float32)

        print(f"Final features shape: {self.X_test.shape}")
        print(f"Target shape: {self.y_test.shape}")

        # Convert to tensors
        self.X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
        self.y_test_tensor = torch.FloatTensor(self.y_test).to(self.device)

        return df

    def predict(self):
        """Make predictions on test data"""
        print("Making predictions...")
        with torch.no_grad():
            predictions = self.model(self.X_test_tensor)
            self.predictions = predictions.cpu().numpy()

            # Your model outputs 7 features, but you probably want 1 prediction
            # If it's multi-output, take the first column or average
            if self.predictions.ndim == 2 and self.predictions.shape[1] > 1:
                print(f"Model output shape: {self.predictions.shape}")
                print("Taking first column as prediction (adjust if needed)")
                self.predictions = self.predictions[:, 0]  # Use first output

            print(f"Final predictions shape: {self.predictions.shape}")

        return self.predictions

    def calculate_metrics(self):
        """Calculate evaluation metrics"""
        print("Calculating metrics...")

        # Ensure shapes match
        if len(self.predictions) != len(self.y_test):
            min_len = min(len(self.predictions), len(self.y_test))
            self.predictions = self.predictions[:min_len]
            self.y_test = self.y_test[:min_len]

        # Calculate metrics
        mae = mean_absolute_error(self.y_test, self.predictions)
        mse = mean_squared_error(self.y_test, self.predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((self.y_test - self.predictions) / (np.abs(self.y_test) + 1e-8))) * 100
        r2 = r2_score(self.y_test, self.predictions)

        self.results = {
            'MAE': float(mae),
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAPE': float(mape),
            'R2': float(r2),
            'samples': len(self.y_test)
        }

        return self.results

    def plot_results(self, save_path="results"):
        """Plot actual vs predicted values"""
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)

        plt.figure(figsize=(15, 10))

        # Plot 1: Actual vs Predicted
        plt.subplot(2, 2, 1)
        plt.scatter(self.y_test, self.predictions, alpha=0.6)
        max_val = max(self.y_test.max(), self.predictions.max())
        min_val = min(self.y_test.min(), self.predictions.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted (R² = {self.results["R2"]:.3f})')

        # Plot 2: Residuals
        residuals = self.y_test - self.predictions
        plt.subplot(2, 2, 2)
        plt.scatter(self.predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')

        # Plot 3: Time series (first 100 samples)
        plt.subplot(2, 2, 3)
        sample_indices = min(100, len(self.y_test))
        plt.plot(self.y_test[:sample_indices], label='Actual', alpha=0.8)
        plt.plot(self.predictions[:sample_indices], label='Predicted', alpha=0.8)
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.title('Time Series Comparison')

        # Plot 4: Error distribution
        plt.subplot(2, 2, 4)
        plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')

        plt.tight_layout()
        plot_path = save_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Plots saved to: {plot_path}")

    def save_results(self, save_path="results"):
        """Save results to JSON and CSV"""
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)

        # Add metadata
        results_with_meta = {
            **self.results,
            'model_path': str(self.model_path),
            'test_date': datetime.now().isoformat(),
            'hyperparameters': self.hyperparameters,
            'device': str(self.device)
        }

        # Save JSON
        json_path = save_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump(results_with_meta, f, indent=2)

        # Save CSV predictions
        predictions_df = pd.DataFrame({
            'actual': self.y_test,
            'predicted': self.predictions,
            'error': self.y_test - self.predictions
        })
        csv_path = save_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        predictions_df.to_csv(csv_path, index=False)

        print(f"✓ Results saved to: {json_path}")
        print(f"✓ Predictions saved to: {csv_path}")

    def run_complete_test(self):
        """Run complete testing pipeline"""
        print("=" * 60)
        print("STARTING COMPLETE MODEL TESTING")
        print("=" * 60)

        start_time = datetime.now()

        try:
            # Load model
            self.load_model()

            # Load data
            self.load_test_data()

            # Make predictions
            predictions = self.predict()

            # Calculate metrics
            metrics = self.calculate_metrics()

            # Display results
            print("\n" + "=" * 40)
            print("TEST RESULTS")
            print("=" * 40)
            for metric, value in metrics.items():
                if metric != 'samples':
                    print(f"{metric}: {value:.6f}")
            print(f"Samples: {metrics['samples']}")

            # Save results and plots
            self.save_results()
            self.plot_results()

            # Print summary
            print("\n" + "=" * 40)
            print("TESTING SUMMARY")
            print("=" * 40)
            print(f"Model: {self.model_path}")
            print(f"Test samples: {len(self.y_test)}")
            print(f"Best metric: RMSE = {metrics['RMSE']:.6f}")
            print(f"R² Score: {metrics['R2']:.4f}")
            print(f"Testing time: {(datetime.now() - start_time).total_seconds():.2f} seconds")
            print("=" * 40)

            return metrics

        except Exception as e:
            print(f"❌ Testing failed: {str(e)}")
            raise e


def main():
    """Main function to run testing"""
    # Find the latest model in models directory
    models_dir = Path("models")
    model_files = list(models_dir.glob("*.pt"))

    if not model_files:
        raise FileNotFoundError("No .pt files found in models directory")

    # Use the first model found
    model_path = model_files[0]
    print(f"Using model: {model_path}")

    # Initialize and run tester
    tester = TFTTester(model_path)
    results = tester.run_complete_test()

    return results


if __name__ == "__main__":
    main()