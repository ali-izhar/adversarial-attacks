#!/usr/bin/env python
"""Script to collect data for the paper tables.

This script evaluates baseline adversarial attacks on pretrained models
and collects metrics for the paper tables, including success rates,
perturbation metrics (L2, L-inf, SSIM), and computational metrics
(iterations, gradient calls, runtime).

Optimized for high-end GPUs like RTX 4090.
"""

import os
import sys
import argparse
import torch
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import yaml
import gc
from functools import partial
from contextlib import nullcontext  # Add nullcontext for context management

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
from src.attacks.baseline.attack_mifgsm import MIFGSM

# Import evaluation framework
from analysis.baseline_eval import AttackEvaluator


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def detect_gpu_capabilities():
    """Detect GPU capabilities to optimize settings."""
    gpu_info = {
        "device": torch.device("cpu"),
        "high_end": False,
        "mixed_precision": False,
        "memory_gb": 0,
        "batch_size": 32,
        "parallel_mode": "thread",  # thread, process, or single
    }

    if torch.cuda.is_available():
        gpu_info["device"] = torch.device("cuda")
        gpu_info["count"] = torch.cuda.device_count()

        # Check capabilities of first GPU
        props = torch.cuda.get_device_properties(0)
        gpu_info["name"] = props.name
        gpu_info["memory_gb"] = props.total_memory / (1024**3)
        gpu_info["compute_capability"] = (props.major, props.minor)

        # Check for high-end GPU (> 16GB VRAM)
        if gpu_info["memory_gb"] > 16 or "RTX" in props.name:
            gpu_info["high_end"] = True

            # Higher batch size for high-end GPUs
            if (
                "3090" in props.name
                or "4090" in props.name
                or gpu_info["memory_gb"] > 20
            ):
                gpu_info["batch_size"] = 128
            elif "3080" in props.name or gpu_info["memory_gb"] > 10:
                gpu_info["batch_size"] = 64

        # Check for mixed precision support
        if props.major >= 7:  # Volta or newer architecture
            gpu_info["mixed_precision"] = True

        # Choose parallel mode based on GPU count
        if gpu_info["count"] > 1:
            # For multiple high-end GPUs, process-based parallelism can be better
            if gpu_info["high_end"] and sys.platform != "win32":
                gpu_info["parallel_mode"] = "process"
            else:
                gpu_info["parallel_mode"] = "thread"
        else:
            gpu_info["parallel_mode"] = "single"

    # Print detected capabilities
    if gpu_info["device"].type == "cuda":
        print(f"GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.1f}GB)")
        print(
            f"Compute capability: {gpu_info['compute_capability'][0]}.{gpu_info['compute_capability'][1]}"
        )
        print(
            f"Mixed precision: {'Enabled' if gpu_info['mixed_precision'] else 'Disabled'}"
        )
        print(f"Batch size: {gpu_info['batch_size']}")
        print(f"Parallel mode: {gpu_info['parallel_mode']}")
    else:
        print("No GPU detected, running on CPU")

    return gpu_info


def get_models(device, model_names=None):
    """Load pretrained models for evaluation using our custom wrappers.

    Args:
        device: Device to load models on
        model_names: Optional list of model names to load; if None, load all available models

    Returns:
        Dictionary mapping model names to model instances
    """

    # Move model loading to a separate function for better memory management
    def load_model(name, model_type):
        # Explicitly run garbage collection before loading a new model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = get_model(model_type).to(device)
        return name, model

    all_model_configs = {
        "ResNet-18": "resnet18",
        "ResNet-50": "resnet50",
        "VGG-16": "vgg16",
        "EfficientNet-B0": "efficientnet_b0",
        "MobileNet-V3": "mobilenet_v3_large",
    }

    # Filter model configs if names are provided
    if model_names:
        model_configs = {
            name: model_type
            for name, model_type in all_model_configs.items()
            if name in model_names
        }
    else:
        model_configs = all_model_configs

    # Load models one by one to avoid OOM
    models = {}
    for name, model_type in model_configs.items():
        print(f"Loading model: {name}")
        name, model = load_model(name, model_type)
        models[name] = model

    return models


def prepare_dataset(data_dir, num_samples=200, gpu_info=None):
    """Load ImageNet dataset using our custom implementation."""
    print(f"Loading dataset from {data_dir}, samples={num_samples}")

    # If data_dir includes 'imagenet', extract the base directory
    base_dir = data_dir
    if "imagenet" in data_dir.lower():
        # Find the part of the path before 'imagenet'
        base_dir = data_dir.split("imagenet")[0].rstrip("/\\")
        if not base_dir:  # If empty, use the current directory
            base_dir = "."

    dataset = get_dataset(
        dataset_name="imagenet",
        data_dir=base_dir,
        max_samples=num_samples,
    )

    return dataset


def test_model_accuracy(models, dataset, device, batch_size=32):
    """Test the accuracy of models on the dataset to verify our setup."""
    dataloader = get_dataloader(
        dataset,
        batch_size=min(len(dataset), batch_size),
        shuffle=False,
        num_workers=min(8, os.cpu_count() or 4),
    )

    images, labels = next(iter(dataloader))
    images, labels = images.to(device, non_blocking=True).float(), labels.to(
        device, non_blocking=True
    )

    results = {}
    for model_name, model in models.items():
        model.eval()

        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / len(images)

            results[model_name] = {"accuracy": accuracy}
            print(f"{model_name} accuracy: {accuracy:.2f}%")

    return results


def create_attack_config(config, args, gpu_info=None):
    """Create attack configurations based on the provided YAML config."""
    # Parse attack configurations from config file
    attack_configs = []

    # Use command line specified attack type if provided, else use config
    method = args.attack_names[0] if args.attack_names else config["attack"]["method"]

    print(f"Creating attack configurations for method: {method}")

    params = config["attack"]["params"]

    # Adjust parameters based on GPU capabilities
    if gpu_info and gpu_info["high_end"]:
        # For high-end GPUs, we can use more steps for optimization-based attacks
        params["DeepFool"]["max_iter"] = min(100, params["DeepFool"]["max_iter"])
        params["CW"]["max_iter"] = min(1000, params["CW"]["max_iter"])
        if "MIFGSM" in params:
            params["MIFGSM"]["steps"] = min(20, params["MIFGSM"].get("steps", 10))

    # Create attack configs based on method (case-insensitive)
    method = method.upper()

    if method in ["ALL", "FGSM"]:
        # Add FGSM attacks
        for norm_type in config["attack"]["norm_types"]:
            for eps in params["FGSM"]["eps_values"][norm_type]:
                attack_name = f"FGSM-{eps}-{norm_type}"
                attack_fn = lambda model, eps=eps: FGSM(model, eps=eps)
                attack_configs.append((attack_name, attack_fn))

    if method in ["ALL", "FFGSM"]:
        # Add FFGSM attacks
        for norm_type in config["attack"]["norm_types"]:
            # Use separate alpha for L2 and Linf norms
            alpha_param = params["FFGSM"].get(
                f"alpha_{norm_type.lower()}", params["FFGSM"].get("alpha_linf", 0.2)
            )

            for eps in params["FFGSM"]["eps_values"][norm_type]:
                alpha = alpha_param * eps  # Scale alpha proportionally
                attack_name = f"FFGSM-{eps}-{norm_type}"
                attack_fn = lambda model, eps=eps, alpha=alpha: FFGSM(
                    model, eps=eps, alpha=alpha
                )
                attack_configs.append((attack_name, attack_fn))

    if method in ["ALL", "DEEPFOOL"]:
        # Add DeepFool attack with multiple overshoot values
        overshoot_values = params["DeepFool"].get("overshoot_values", [0.02])
        if isinstance(overshoot_values, (int, float)):
            overshoot_values = [overshoot_values]

        steps = params["DeepFool"]["max_iter"]
        early_stopping = params["DeepFool"].get("early_stopping", True)

        for overshoot in overshoot_values:
            attack_name = f"DeepFool-over{overshoot}"
            # Use the exact parameter names from the DeepFool constructor
            attack_fn = lambda model, overshoot=overshoot, steps=steps, early_stopping=early_stopping: DeepFool(
                model, steps=steps, overshoot=overshoot, early_stopping=early_stopping
            )
            attack_configs.append((attack_name, attack_fn))

    if method in ["ALL", "CW"]:
        # Add CW attacks with multiple confidence values
        confidence_values = params["CW"].get("confidence_values", [0.0])
        if isinstance(confidence_values, (int, float)):
            confidence_values = [confidence_values]

        c = params["CW"]["c_init"]
        steps = params["CW"]["max_iter"]
        lr = params["CW"]["learning_rate"]
        binary_search_steps = params["CW"].get("binary_search_steps", 5)

        for conf in confidence_values:
            attack_name = f"CW-kappa{conf}"
            attack_fn = lambda model, conf=conf, bss=binary_search_steps: CW(
                model, c=c, kappa=conf, steps=steps, lr=lr, binary_search_steps=bss
            )
            attack_configs.append((attack_name, attack_fn))

    if method in ["ALL", "MIFGSM"] and "MIFGSM" in params:
        # Add MIFGSM (Momentum Iterative FGSM) attacks
        try:
            steps = params["MIFGSM"]["steps"]
            momentum = params["MIFGSM"].get("momentum", 0.9)
            step_size = params["MIFGSM"].get("alpha", 0.01)

            for norm_type in config["attack"]["norm_types"]:
                for eps in params["MIFGSM"]["eps_values"][norm_type]:
                    attack_name = f"MIFGSM-{eps}-{norm_type}"
                    attack_fn = lambda model, eps=eps: MIFGSM(
                        model,
                        eps=eps,
                        steps=steps,
                        alpha=step_size,
                        decay_factor=momentum,
                    )
                    attack_configs.append((attack_name, attack_fn))
        except ImportError:
            print("MIFGSM attack is configured but not implemented yet. Skipping.")

    # Filter based on command line if provided
    if args.attack_names and len(attack_configs) > 0:
        filtered_configs = []
        for attack_name, attack_fn in attack_configs:
            # Extract the base attack name (case-insensitive matching)
            base_name = attack_name.split("-")[0].upper()
            if any(
                base_name == requested_name.upper()
                for requested_name in args.attack_names
            ):
                filtered_configs.append((attack_name, attack_fn))

        if filtered_configs:
            attack_configs = filtered_configs
        else:
            print(
                f"Warning: No configurations matched the requested attack names: {args.attack_names}"
            )

    print(f"Created {len(attack_configs)} attack configurations")
    # Print the attack names that were created
    if attack_configs:
        print("Attack configurations:")
        for attack_name, _ in attack_configs:
            print(f"  - {attack_name}")
    else:
        print("WARNING: No attack configurations were created! Check your attack name.")

    return attack_configs


def evaluate_model(
    model_name, model, dataset, attack_configs, args, config, gpu_info, device_id=None
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
            device = gpu_info["device"]

        print(f"\n\n{'#'*50}")
        print(f"Evaluating attacks on {model_name} (device: {device})")
        print(f"{'#'*50}")

        # Move model to the correct device
        model = model.to(device)
        model.eval()  # Ensure model is in evaluation mode

        # Enable cuDNN benchmarking for optimal performance
        torch.backends.cudnn.benchmark = True

        # Create model dict with just this model
        model_dict = {model_name: model}

        # Create evaluation framework with optimized batch size
        batch_size = gpu_info["batch_size"] if gpu_info else 32
        num_workers = (
            min(16, os.cpu_count() or 8) if gpu_info and gpu_info["high_end"] else 8
        )

        evaluator = AttackEvaluator(
            model_dict,
            dataset,
            device,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Evaluate each attack in both untargeted and targeted modes
        results = {}
        attack_count = 0

        print(f"Starting evaluation of {len(attack_configs)} attack configurations")
        for attack_name, attack_fn in attack_configs:
            # Skip if this specific attack is excluded
            if args.exclude_attacks and any(
                ex in attack_name for ex in args.exclude_attacks
            ):
                print(f"Skipping {attack_name} (excluded by command line argument)")
                continue

            attack_count += 1
            attack_results = {}

            # Use mixed precision for compatible attacks if supported
            # (Note: some attacks may not work correctly with mixed precision)
            use_amp = (
                gpu_info
                and gpu_info["mixed_precision"]
                and not attack_name.startswith(("DeepFool", "CW"))
            )

            # Check if this is a gradient-based attack that needs gradients
            needs_grads = attack_name.startswith(("DeepFool", "CW"))

            # Evaluate untargeted attack first
            print(f"\n{'='*50}")
            print(f"Evaluating {attack_name} (untargeted)...")
            start_time = time.time()

            # Explicitly run garbage collection before each evaluation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Move model back to device in case it was moved
            model = model.to(device)
            model.eval()

            # Use appropriate context based on attack type
            with (
                torch.cuda.amp.autocast(enabled=use_amp)
                if use_amp
                else (nullcontext() if needs_grads else torch.no_grad())
            ):
                attack_results["untargeted"] = evaluator.evaluate_attack(
                    attack_name, attack_fn, targeted=False
                )

            attack_time = time.time() - start_time
            print(f"Completed in {attack_time:.2f}s")

            # Skip targeted evaluation for DeepFool as it doesn't support it
            if not attack_name.startswith("DeepFool"):
                print(f"\n{'-'*50}")
                print(f"Evaluating {attack_name} (targeted)...")
                start_time = time.time()

                # Explicitly run garbage collection before each evaluation
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Move model back to device in case it was moved
                model = model.to(device)
                model.eval()

                # Use appropriate context based on attack type
                with (
                    torch.cuda.amp.autocast(enabled=use_amp)
                    if use_amp
                    else (nullcontext() if needs_grads else torch.no_grad())
                ):
                    attack_results["targeted"] = evaluator.evaluate_attack(
                        attack_name, attack_fn, targeted=True
                    )

                attack_time = time.time() - start_time
                print(f"Completed in {attack_time:.2f}s")

            results[attack_name] = attack_results

        # Export results for this model - only if we have results
        model_output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        print(f"\nExporting results for {model_name}, processed {attack_count} attacks")
        if attack_count > 0:
            print(f"Results structure has {len(results)} attack entries")
            for attack_name in results:
                print(f"  {attack_name}: {list(results[attack_name].keys())}")
        else:
            print("Warning: No attacks were evaluated, results will be empty")

        evaluator.export_results_to_tables(model_output_dir)

        # Visualize some example perturbations if results contain examples
        vis_dir = os.path.join(model_output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Try to visualize a few key attacks if they were run
        key_attacks = [
            attack_name
            for attack_name, _ in attack_configs
            if "FGSM-0.03" in attack_name
            or "CW-kappa0" in attack_name
            or attack_name == "DeepFool"
        ]

        for attack_name in key_attacks:
            if attack_name in results:
                evaluator.visualize_perturbations(
                    attack_name, model_name, num_samples=3, output_dir=vis_dir
                )

        # Clear GPU memory before returning
        model = model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {model_name: results}

    except Exception as e:
        print(f"Error evaluating model {model_name}: {e}")
        import traceback

        traceback.print_exc()
        return {model_name: str(e)}


def main(args):
    """Main evaluation function."""
    # Load configuration
    config = load_config(args.config_file)
    print(f"Loaded configuration from {args.config_file}")

    # Get GPU capabilities for optimization
    gpu_info = detect_gpu_capabilities()

    # Determine available devices and parallelization strategy
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s) available")

        if num_gpus > 1:
            print(
                f"Will use parallel evaluation with {gpu_info['parallel_mode']} parallelism"
            )
        else:
            print("Will use single GPU for evaluation")
    else:
        num_gpus = 0
        print("No GPUs found, using CPU")

    # Set device for initial loading
    device = gpu_info["device"]

    # Load models based on config
    model_names = None
    if args.model_name:
        model_names = [args.model_name]
    elif config["models"]:
        # Map from config model names to our model class names
        model_map = {
            "resnet18": "ResNet-18",
            "resnet50": "ResNet-50",
            "vgg16": "VGG-16",
            "efficientnet": "EfficientNet-B0",
            "mobilenet": "MobileNet-V3",
        }
        model_names = [model_map.get(name, name) for name in config["models"]]

    print(f"Loading models: {model_names or 'all available'}")
    models_dict = get_models(device, model_names)

    # Load dataset
    data_dir = (
        config["dataset"]["image_dir"]
        if config["dataset"]["image_dir"]
        else args.data_dir
    )
    num_samples = args.num_samples or config["dataset"]["num_images"]

    dataset = prepare_dataset(data_dir, num_samples, gpu_info)

    # Test model accuracy to verify everything is working
    # batch_size = gpu_info["batch_size"] if gpu_info else 32
    # print("\nTesting model accuracy on dataset...")
    # model_accuracy = test_model_accuracy(models_dict, dataset, device, batch_size)

    # Create attack configurations from config file
    attack_configs = create_attack_config(config, args, gpu_info)

    # Evaluate attacks on models (in parallel if multiple GPUs available)
    if gpu_info["parallel_mode"] == "process" and len(models_dict) > 1:
        # Process pool is better for heavy GPU workloads on multiple GPUs
        print(
            f"\nRunning parallel evaluation with process pool across {min(num_gpus, len(models_dict))} GPUs"
        )

        # Function with preset arguments, only model_name and model will be provided later
        worker_fn = partial(
            evaluate_model,
            dataset=dataset,
            attack_configs=attack_configs,
            args=args,
            config=config,
            gpu_info=gpu_info,
        )

        all_results = {}
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for i, (model_name, model) in enumerate(models_dict.items()):
                # Assign each model to a different GPU in round-robin fashion
                device_id = i % num_gpus
                futures.append(
                    executor.submit(
                        worker_fn,
                        model_name,
                        model,
                        device_id=device_id,
                    )
                )

            # Collect results as they complete
            for future in futures:
                result = future.result()
                if result:
                    all_results.update(result)

    elif gpu_info["parallel_mode"] == "thread" and len(models_dict) > 1:
        # Thread pool for lighter workloads or Windows
        print(
            f"\nRunning parallel evaluation with thread pool across {min(num_gpus, len(models_dict))} GPUs"
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
                        attack_configs,
                        args,
                        config,
                        gpu_info,
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
                attack_configs,
                args,
                config,
                gpu_info,
            )
            if result:
                all_results.update(result)

    print(f"\nAll results saved to {args.output_dir}")

    # Return to reference device for cleanup
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Fix multiprocessing issue in Windows
    if sys.platform == "win32":
        mp.set_start_method("spawn", force=True)

    # Set environment variables for performance
    os.environ["OMP_NUM_THREADS"] = str(min(8, os.cpu_count() or 4))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

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
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (overrides config)",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="analysis/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        choices=["ResNet-18", "ResNet-50", "VGG-16", "EfficientNet-B0", "MobileNet-V3"],
        help="Specify a single model to evaluate (default: use models from config)",
    )
    parser.add_argument(
        "--attack-names",
        type=str,
        nargs="+",
        default=None,
        choices=["FGSM", "FFGSM", "DeepFool", "CW", "MIFGSM"],
        help="Specify specific attacks to evaluate (default: use attacks from config)",
    )
    parser.add_argument(
        "--exclude-attacks",
        type=str,
        nargs="+",
        default=None,
        help="Specify attacks to exclude (e.g., 'DeepFool' to exclude DeepFool)",
    )
    args = parser.parse_args()

    main(args)
