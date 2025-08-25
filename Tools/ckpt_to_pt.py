import torch
import os
from pathlib import Path
import argparse


def convert_ckpt_to_pt():
    """
    Convert PyTorch Lightning CKPT to PT format
    """
    # Define paths
    checkpoints_dir = Path("../checkpoints")
    models_dir = Path("../models")

    # Create models directory if it doesn't exist
    models_dir.mkdir(exist_ok=True)

    # Find all .ckpt files
    ckpt_files = list(checkpoints_dir.glob("*.ckpt"))

    if not ckpt_files:
        print(f"No .ckpt files found in {checkpoints_dir}")
        return

    for ckpt_file in ckpt_files:
        try:
            print(f"Converting {ckpt_file.name}...")

            # Load checkpoint
            checkpoint = torch.load(ckpt_file, map_location='cpu')

            # Extract state dict and clean keys
            state_dict = checkpoint['state_dict']
            cleaned_state_dict = {}

            for key, value in state_dict.items():
                # Remove 'model.' prefix if it exists (common in Lightning checkpoints)
                if key.startswith('model.'):
                    cleaned_key = key[6:]  # Remove 'model.' prefix
                else:
                    cleaned_key = key
                cleaned_state_dict[cleaned_key] = value

            # Extract hyperparameters
            hyperparameters = checkpoint.get('hyper_parameters', {})

            # Create output filename
            output_filename = ckpt_file.stem + ".pt"
            output_path = models_dir / output_filename

            # Save as .pt format
            torch.save({
                'state_dict': cleaned_state_dict,
                'hyperparameters': hyperparameters,
                'epoch': checkpoint.get('epoch', 0),
                'global_step': checkpoint.get('global_step', 0),
                'original_file': ckpt_file.name
            }, output_path)

            print(f"✓ Successfully converted to: {output_path}")
            print(f"  - Parameters: {len(cleaned_state_dict)}")
            print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"  - Hyperparameters: {list(hyperparameters.keys())}")
            print()

        except Exception as e:
            print(f"✗ Failed to convert {ckpt_file.name}: {str(e)}")
            print()


def verify_conversion():
    """
    Verify the converted .pt files
    """
    models_dir = Path("../models")
    pt_files = list(models_dir.glob("*.pt"))

    if not pt_files:
        print("No .pt files found for verification")
        return

    print("Verifying converted files...")
    print("=" * 50)

    for pt_file in pt_files:
        try:
            model_data = torch.load(pt_file, map_location='cpu')
            print(f"✓ {pt_file.name}:")
            print(f"  - Keys: {list(model_data.keys())}")
            print(f"  - Parameters: {len(model_data['state_dict'])}")
            print(f"  - Epoch: {model_data.get('epoch', 'N/A')}")
            print()

        except Exception as e:
            print(f"✗ Failed to verify {pt_file.name}: {str(e)}")
            print()


if __name__ == "__main__":
    # Simple conversion without command line arguments
    print("Starting CKPT to PT conversion...")
    print("=" * 50)

    # Convert all checkpoints
    convert_ckpt_to_pt()

    # Verify the conversion
    print("=" * 50)
    verify_conversion()

    print("Conversion process completed!")