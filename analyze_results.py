# analyze_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


def analyze_test_results(results_dir="results"):
    """Analyze and compare test results"""
    results_dir = Path(results_dir)
    json_files = list(results_dir.glob("test_results_*.json"))

    if not json_files:
        print("No result files found")
        return

    # Load the latest result
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)

    with open(latest_file, 'r') as f:
        results = json.load(f)

    print("=" * 60)
    print("MODEL PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Display current results
    print("📊 CURRENT MODEL PERFORMANCE:")
    print(f"   R² Score: {results['R2']:.6f}")
    print(f"   RMSE: {results['RMSE']:.6f}")
    print(f"   MAE: {results['MAE']:.6f}")
    print(f"   MAPE: {results['MAPE']:.6f}%")
    print(f"   Samples: {results['samples']}")

    # Define benchmark values (you should adjust these based on your domain)
    BENCHMARKS = {
        'Excellent': {'R2': 0.9, 'RMSE': 0.5, 'MAE': 0.3, 'MAPE': 5.0},
        'Good': {'R2': 0.7, 'RMSE': 1.0, 'MAE': 0.7, 'MAPE': 10.0},
        'Fair': {'R2': 0.5, 'RMSE': 2.0, 'MAE': 1.5, 'MAPE': 20.0},
        'Poor': {'R2': 0.3, 'RMSE': 3.0, 'MAE': 2.5, 'MAPE': 30.0},
        'Very Poor': {'R2': 0.0, 'RMSE': 5.0, 'MAE': 4.0, 'MAPE': 50.0}
    }

    # Create comparison table
    comparison_data = []
    for level, benchmarks in BENCHMARKS.items():
        comparison_data.append({
            'Performance Level': level,
            'R² (Benchmark)': benchmarks['R2'],
            'R² (Your Model)': results['R2'],
            'RMSE (Benchmark)': benchmarks['RMSE'],
            'RMSE (Your Model)': results['RMSE'],
            'MAE (Benchmark)': benchmarks['MAE'],
            'MAE (Your Model)': results['MAE']
        })

    # Create DataFrame for nice formatting
    df_comparison = pd.DataFrame(comparison_data)

    print("\n" + "=" * 60)
    print("📈 PERFORMANCE COMPARISON TABLE")
    print("=" * 60)
    print(df_comparison.to_string(index=False))

    # Additional analysis
    print("\n" + "=" * 60)
    print("🔍 ADDITIONAL ANALYSIS")
    print("=" * 60)

    # Load predictions
    csv_files = list(results_dir.glob("predictions_*.csv"))
    if csv_files:
        latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
        predictions_df = pd.read_csv(latest_csv)

        # ======== ADD THIS DEBUG CODE RIGHT HERE ========
        print("\n" + "=" * 50)
        print("🔍 DEBUG DATA CHECK:")
        print("=" * 50)

        actual = predictions_df['actual']
        predicted = predictions_df['predicted']

        print(f"Actual values range: {actual.min():.6f} to {actual.max():.6f}")
        print(f"Predicted values range: {predicted.min():.6f} to {predicted.max():.6f}")
        print(f"Actual mean: {actual.mean():.6f}, std: {actual.std():.6f}")
        print(f"Predicted mean: {predicted.mean():.6f}, std: {predicted.std():.6f}")
        print(f"NaN in actual: {actual.isna().sum()}")
        print(f"NaN in predicted: {predicted.isna().sum()}")
        print(f"Infinite values in actual: {np.isinf(actual).sum()}")
        print(f"Infinite values in predicted: {np.isinf(predicted).sum()}")

        # Check scaling issues
        if abs(actual.mean()) > 10 or abs(predicted.mean()) > 10:
            print("⚠️  CRITICAL: Values are not properly scaled!")
            print("   Your target variable needs StandardScaler!")
        elif abs(actual.mean()) > 5 or abs(predicted.mean()) > 5:
            print("⚠️  WARNING: Values might not be properly scaled")

        if actual.std() < 0.1 or predicted.std() < 0.1:
            print("⚠️  WARNING: Very low variance - check scaling!")

        # Check if predictions are constant
        if predicted.std() < 0.001:
            print("❌ CRITICAL: Model is predicting constant values!")
            print("   This means the model didn't learn anything")

        print("=" * 50)
        # ======== END OF DEBUG CODE ========

        # Calculate additional metrics
        mean_actual = predictions_df['actual'].mean()
        std_actual = predictions_df['actual'].std()
        mean_predicted = predictions_df['predicted'].mean()
        std_predicted = predictions_df['predicted'].std()

        print(f"Mean of actual values: {mean_actual:.6f}")
        print(f"Std of actual values: {std_actual:.6f}")
        print(f"Mean of predicted values: {mean_predicted:.6f}")
        print(f"Std of predicted values: {std_predicted:.6f}")
        print(f"Correlation coefficient: {predictions_df['actual'].corr(predictions_df['predicted']):.6f}")

        # Check for scale issues
        if abs(mean_actual - mean_predicted) > 2 * std_actual:
            print("⚠️  WARNING: Large mean difference between actual and predicted values")
        if abs(std_actual - std_predicted) > 0.5 * std_actual:
            print("⚠️  WARNING: Significant variance mismatch between actual and predicted values")

    # Interpretation
    print("\n" + "=" * 60)
    print("💡 INTERPRETATION & RECOMMENDATIONS")
    print("=" * 60)

    if results['R2'] < -1:
        print("❌ CRITICAL: Model performance is very poor (R² < -1)")
        print("   Possible causes:")
        print("   - Target column mismatch")
        print("   - Feature scaling issues")
        print("   - Model architecture mismatch")
        print("   - Data leakage or preprocessing errors")
        print("   - The model is worse than predicting the mean")

    elif results['R2'] < 0:
        print("⚠️  POOR: Model performance is negative (R² < 0)")
        print("   The model performs worse than simply predicting the mean")
        print("   Recommendations:")
        print("   - Check feature engineering")
        print("   - Verify data preprocessing steps")
        print("   - Consider simpler baseline models")

    elif results['R2'] < 0.3:
        print("⚠️  WEAK: Model explains little variance (R² < 0.3)")
        print("   The model has limited predictive power")
        print("   Recommendations:")
        print("   - Feature selection and engineering")
        print("   - Hyperparameter tuning")
        print("   - Try different model architectures")

    elif results['R2'] < 0.7:
        print("✅ MODERATE: Reasonable performance (0.3 ≤ R² < 0.7)")
        print("   The model has decent predictive power")
        print("   Recommendations:")
        print("   - Fine-tune hyperparameters")
        print("   - Add more relevant features")
        print("   - Ensemble methods")

    else:
        print("🎉 EXCELLENT: Strong predictive power (R² ≥ 0.7)")
        print("   The model explains most of the variance")
        print("   Consider deployment if validation holds")

    # Create visual comparison
    create_performance_chart(results, BENCHMARKS)



def create_performance_chart(results, benchmarks):
    """Create visual performance comparison chart"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # R² comparison
    benchmark_r2 = [benchmarks[level]['R2'] for level in benchmarks]
    your_r2 = results['R2']
    ax1.bar(range(len(benchmarks)), benchmark_r2, alpha=0.7, label='Benchmark')
    ax1.axhline(y=your_r2, color='red', linestyle='--', linewidth=2, label='Your Model')
    ax1.set_title('R² Score Comparison')
    ax1.set_xticks(range(len(benchmarks)))
    ax1.set_xticklabels(list(benchmarks.keys()), rotation=45)
    ax1.legend()

    # RMSE comparison
    benchmark_rmse = [benchmarks[level]['RMSE'] for level in benchmarks]
    your_rmse = results['RMSE']
    ax2.bar(range(len(benchmarks)), benchmark_rmse, alpha=0.7, label='Benchmark')
    ax2.axhline(y=your_rmse, color='red', linestyle='--', linewidth=2, label='Your Model')
    ax2.set_title('RMSE Comparison')
    ax2.set_xticks(range(len(benchmarks)))
    ax2.set_xticklabels(list(benchmarks.keys()), rotation=45)
    ax2.legend()

    # MAE comparison
    benchmark_mae = [benchmarks[level]['MAE'] for level in benchmarks]
    your_mae = results['MAE']
    ax3.bar(range(len(benchmarks)), benchmark_mae, alpha=0.7, label='Benchmark')
    ax3.axhline(y=your_mae, color='red', linestyle='--', linewidth=2, label='Your Model')
    ax3.set_title('MAE Comparison')
    ax3.set_xticks(range(len(benchmarks)))
    ax3.set_xticklabels(list(benchmarks.keys()), rotation=45)
    ax3.legend()

    # MAPE comparison (log scale due to extremely high values)
    benchmark_mape = [benchmarks[level]['MAPE'] for level in benchmarks]
    your_mape = results['MAPE']
    ax4.bar(range(len(benchmarks)), benchmark_mape, alpha=0.7, label='Benchmark')
    ax4.axhline(y=min(your_mape, 1000), color='red', linestyle='--', linewidth=2, label='Your Model (capped at 1000)')
    ax4.set_title('MAPE Comparison (Log Scale)')
    ax4.set_yscale('log')
    ax4.set_xticks(range(len(benchmarks)))
    ax4.set_xticklabels(list(benchmarks.keys()), rotation=45)
    ax4.legend()

    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Performance comparison chart saved as 'performance_comparison.png'")




if __name__ == "__main__":
    analyze_test_results()