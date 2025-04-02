#!/usr/bin/env python
"""Script to collect data for the paper tables.

This script evaluates baseline adversarial attacks on pretrained models
and collects metrics for the paper tables, including success rates,
perturbation metrics (L2, L-inf, SSIM), and computational metrics
(iterations, gradient calls, runtime).
"""

import os
import sys
import argparse
import torch
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our custom dataset and model wrappers
from src.datasets.imagenet import get_dataset, get_dataloader
from src.models.wrappers import get_model

# Import baseline attacks
from src.attacks.baseline.attack_fgsm import FGSM
from src.attacks.baseline.attack_ffgsm import FFGSM
from src.attacks.baseline.attack_deepfool import DeepFool
from src.attacks.baseline.attack_cw import CW

# Import evaluation framework
from analysis.baseline_eval import AttackEvaluator

# We don't need the class mapping when using the full dataset
# with proper labels or when using simulation mode


def get_models(device):
    """Load pretrained models for evaluation using our custom wrappers."""
    model_dict = {
        "ResNet-18": get_model("resnet18").to(device),
        "ResNet-50": get_model("resnet50").to(device),
        "VGG-16": get_model("vgg16").to(device),
        "EfficientNet-B0": get_model("efficientnet_b0").to(device),
        "MobileNet-V3": get_model("mobilenet_v3_large").to(device),
    }
    return model_dict


def get_base_models(models_dict):
    """
    Extract the base torchvision models from our wrappers to avoid double normalization.

    This is crucial when working with datasets that are already normalized with
    ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
    """
    base_models = {}
    for name, model in models_dict.items():
        # Access the underlying torchvision model directly
        if hasattr(model, "_model"):
            base_model = model._model
        else:
            base_model = model

        # Set to evaluation mode
        base_model.eval()
        base_model.to(model.device)

        base_models[name] = base_model

    return base_models


def prepare_dataset(data_dir, num_samples=200):
    """Load ImageNet dataset using our custom implementation."""
    # Get the dataset using our custom implementation
    dataset = get_dataset(
        dataset_name="imagenet",
        data_dir=os.path.dirname(
            data_dir
        ),  # The parent directory of the ImageNet folder
        max_samples=num_samples,
    )

    # No need to create a Subset as our implementation already handles max_samples
    return dataset


def test_model_accuracy(models, base_models, dataset, device):
    """Test the accuracy of models on the dataset to verify our setup."""
    # Create a DataLoader with batch size matching dataset size
    dataloader = get_dataloader(
        dataset, batch_size=min(len(dataset), 32), shuffle=False, num_workers=4
    )

    # Get a batch of images
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    results = {}
    for model_name, model in models.items():
        base_model = base_models[model_name]
        base_model.eval()

        with torch.no_grad():
            # Use the base model directly to avoid double normalization
            outputs = base_model(images)
            _, predicted = torch.max(outputs, 1)

            # Calculate accuracy with original labels
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / len(images)

            results[model_name] = {"original_accuracy": accuracy}

            print(f"{model_name} accuracy: {accuracy:.2f}%")

    return results


class DirectModelAttackWrapper(torch.nn.Module):
    """
    Wrapper that accesses the underlying model directly for attacks.

    This avoids double normalization when working with already normalized datasets.
    """

    def __init__(self, model):
        super().__init__()
        # Store the wrapped model
        self.wrapped_model = model

        # Access the base model directly
        if hasattr(model, "_model"):
            self.model = model._model
        else:
            self.model = model

        # Set to evaluation mode
        self.model.eval()
        self.model.to(model.device if hasattr(model, "device") else "cuda")

        # Set device attribute required by attacks
        self.device = (
            model.device
            if hasattr(model, "device")
            else next(self.model.parameters()).device
        )

    def forward(self, x):
        """Forward pass directly to base model."""
        return self.model(x)


def get_initial_predictions(models, base_models, dataset, device):
    """Get initial model predictions to use as 'correct' labels for attack evaluation."""
    # Create a DataLoader with small batch size
    dataloader = get_dataloader(dataset, batch_size=32, shuffle=False, num_workers=4)

    model_predictions = {}

    # Store all initial predictions
    for model_name, model in models.items():
        print(f"Getting initial predictions for {model_name}...")
        base_model = base_models[model_name]
        base_model.eval()

        # Initialize tensor to store all predictions
        all_preds = torch.zeros(len(dataset), dtype=torch.long).to(device)
        all_scores = torch.zeros(len(dataset)).to(device)

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                start_idx = batch_idx * dataloader.batch_size
                end_idx = min(start_idx + dataloader.batch_size, len(dataset))

                # Get model predictions using base model to avoid double normalization
                images = images.to(device)
                outputs = base_model(images)

                # Get class with highest confidence
                scores, preds = torch.max(outputs, dim=1)

                # Store predictions and confidence scores
                all_preds[start_idx:end_idx] = preds
                all_scores[start_idx:end_idx] = scores

        # Store results for this model
        model_predictions[model_name] = {
            "predictions": all_preds,
            "confidence": all_scores,
        }

    return model_predictions


class SimulatedDataset(torch.utils.data.Dataset):
    """A dataset wrapper that replaces original labels with model predictions."""

    def __init__(self, base_dataset, model_predictions, model_name):
        self.base_dataset = base_dataset
        self.predictions = model_predictions[model_name]["predictions"]
        self.class_names = base_dataset.class_names
        self.image_paths = base_dataset.image_paths

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]
        # Use the model prediction as the "true" label
        label = self.predictions[idx].item()
        return image, label


def evaluate_model(
    model_name, model, dataset, base_model, attack_configs, args, device_id=None
):
    """Evaluate all attacks for a single model.
    This function can be run in parallel for multiple models.
    """
    try:
        # Use specified device ID if provided, otherwise use default
        if device_id is not None:
            device = torch.device(
                f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
            )
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\n\n{'#'*50}")
        print(f"Evaluating attacks on {model_name} (device: {device})")
        print(f"{'#'*50}")

        # Move model to the correct device
        model = model.to(device)

        # For simplicity, create simulated dataset using original dataset
        # This could be updated to use initial predictions if needed
        model_dataset = dataset

        # Create a direct model wrapper for attacks to avoid double normalization
        direct_model = DirectModelAttackWrapper(model)
        direct_model.to(device)

        # Create evaluation framework with model-specific dataset
        evaluator = AttackEvaluator({model_name: direct_model}, model_dataset, device)

        # Evaluate each attack in both untargeted and targeted modes
        results = {}
        for attack_name, attack_fn in attack_configs:
            attack_results = {}

            print(f"\n{'='*50}")
            print(f"Evaluating {attack_name} (untargeted)...")
            start_time = time.time()
            attack_results["untargeted"] = evaluator.evaluate_attack(
                attack_name, attack_fn, targeted=False
            )
            attack_time = time.time() - start_time
            print(f"Completed in {attack_time:.2f}s")

            # Skip evaluating targeted mode for DeepFool as it doesn't support it
            if "DeepFool" not in attack_name:
                print(f"\n{'-'*50}")
                print(f"Evaluating {attack_name} (targeted)...")
                start_time = time.time()
                attack_results["targeted"] = evaluator.evaluate_attack(
                    attack_name, attack_fn, targeted=True
                )
                attack_time = time.time() - start_time
                print(f"Completed in {attack_time:.2f}s")

            results[attack_name] = attack_results

        # Export results for this model
        model_output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        evaluator.export_results_to_tables(model_output_dir)

        # Visualize some example perturbations
        vis_dir = os.path.join(model_output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Visualize FGSM and CW examples
        evaluator.visualize_perturbations(
            "FGSM-0.03", model_name, num_samples=3, output_dir=vis_dir
        )
        evaluator.visualize_perturbations(
            "CW-kappa0", model_name, num_samples=3, output_dir=vis_dir
        )

        return {model_name: results}
    except Exception as e:
        print(f"Error evaluating model {model_name}: {e}")
        import traceback

        traceback.print_exc()
        return {model_name: str(e)}


def main(args):
    """Main evaluation function."""
    # Determine available devices
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s) available")

        if num_gpus > 1:
            print("Will use parallel evaluation across multiple GPUs")
        else:
            print("Will use single GPU for evaluation")
    else:
        num_gpus = 0
        print("No GPUs found, using CPU")

    # Load models and dataset
    print("Loading models...")
    all_models_dict = get_models(torch.device("cpu"))  # Initially load on CPU

    # Filter models based on command-line argument
    if args.model_name:
        if args.model_name not in all_models_dict:
            print(
                f"Error: Model '{args.model_name}' not found. Available models: {list(all_models_dict.keys())}"
            )
            sys.exit(1)
        print(f"Evaluating only the {args.model_name} model as requested")
        models_dict = {args.model_name: all_models_dict[args.model_name]}
    else:
        models_dict = all_models_dict
        print(f"Evaluating all models: {list(models_dict.keys())}")

    # Get base models for direct evaluation (avoid double normalization)
    base_models = get_base_models(models_dict)

    print(f"Loading dataset from {args.data_dir}...")
    dataset = prepare_dataset(args.data_dir, args.num_samples)

    # Print warning about normalization
    print("\nNOTE: Using base models directly to avoid double normalization")
    print("This is crucial when evaluating already-normalized datasets")

    # Test model accuracy on one device first to see if we need simulation mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    first_model = next(iter(models_dict.values())).to(device)
    first_base_model = next(iter(base_models.values())).to(device)

    print("\nTesting model accuracy on sample batch...")
    dataloader = get_dataloader(
        dataset, batch_size=min(32, len(dataset)), shuffle=False
    )
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = first_base_model(images)
        _, preds = torch.max(outputs, 1)
        accuracy = 100 * (preds == labels).sum().item() / len(labels)

    print(f"Sample accuracy: {accuracy:.2f}%")

    # If accuracy is very poor, warn about metrics but continue anyway
    if accuracy < 25.0:
        print("\nWARNING: Models show very low accuracy on this dataset!")
        print("Attack metrics may be affected - some images are already misclassified.")
        print("For best results, use a dataset the models can classify correctly.")
        print("\nContinuing with evaluation...")

    # Define attack configurations
    all_attack_configs = [
        # FGSM with different epsilon values
        ("FGSM-0.01", lambda model: FGSM(model, eps=0.01)),
        ("FGSM-0.03", lambda model: FGSM(model, eps=0.03)),
        # FFGSM
        ("FFGSM-0.03", lambda model: FFGSM(model, eps=0.03, alpha=0.02)),
        # DeepFool
        ("DeepFool", lambda model: DeepFool(model, steps=100, overshoot=0.05)),
        # CW with different parameters
        ("CW-kappa0", lambda model: CW(model, c=1.0, kappa=0.0, steps=100, lr=0.01)),
        ("CW-kappa2", lambda model: CW(model, c=1.0, kappa=2.0, steps=100, lr=0.01)),
    ]

    # Filter attacks based on command-line argument
    if args.attack_names:
        filtered_attack_configs = []
        for attack_name, attack_fn in all_attack_configs:
            base_name = attack_name.split("-")[0]
            if base_name in args.attack_names:
                filtered_attack_configs.append((attack_name, attack_fn))
        attack_configs = filtered_attack_configs
        print(f"Evaluating only these attacks: {[name for name, _ in attack_configs]}")
    else:
        attack_configs = all_attack_configs
        print(f"Evaluating all attacks: {[name for name, _ in attack_configs]}")

    # Evaluate attacks on models (in parallel if multiple GPUs available)
    if num_gpus > 1 and len(models_dict) > 1:
        # Parallelize model evaluation across GPUs
        print(
            f"\nRunning parallel evaluation across {min(num_gpus, len(models_dict))} GPUs"
        )

        all_results = {}
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for i, (model_name, model) in enumerate(models_dict.items()):
                # Assign each model to a different GPU in round-robin fashion
                device_id = i % num_gpus
                futures.append(
                    executor.submit(
                        evaluate_model,
                        model_name,
                        model,
                        dataset,
                        base_models[model_name],
                        attack_configs,
                        args,
                        device_id,
                    )
                )

            # Collect results as they complete
            for future in futures:
                result = future.result()
                if result:
                    all_results.update(result)
    else:
        # Sequential evaluation on a single device
        all_results = {}
        for model_name, model in models_dict.items():
            result = evaluate_model(
                model_name,
                model,
                dataset,
                base_models[model_name],
                attack_configs,
                args,
            )
            if result:
                all_results.update(result)

    print(f"\nAll results saved to {args.output_dir}")

    # Return to reference device for cleanup
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Fix multiprocessing issue in Windows
    if sys.platform == "win32":
        mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Collect data for paper tables.")
    parser.add_argument(
        "--data_dir", type=str, default="data/imagenet", help="Path to ImageNet dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/paper_data",
        help="Directory to save results",
    )
    parser.add_argument(
        "--num_samples", type=int, default=200, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        choices=["ResNet-18", "ResNet-50", "VGG-16", "EfficientNet-B0", "MobileNet-V3"],
        help="Specify a single model to evaluate (default: evaluate all models)",
    )
    parser.add_argument(
        "--attack-names",
        type=str,
        nargs="+",
        default=None,
        choices=["FGSM", "FFGSM", "DeepFool", "CW"],
        help="Specify specific attacks to evaluate (default: evaluate all attacks)",
    )
    args = parser.parse_args()

    main(args)
