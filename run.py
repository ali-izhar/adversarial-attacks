"""
Comprehensive evaluation script for adversarial attack methods.
Collects all metrics required for the paper tables across multiple models.
Organized by table data collection requirements.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import csv
import json
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    VGG16_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
)

from src.attacks.baseline.attack_cw import CW
from src.attacks.baseline.attack_deepfool import DeepFool
from src.attacks.baseline.attack_fgsm import FGSM
from src.attacks.baseline.attack_ffgsm import FFGSM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization parameters
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])


def load_models():
    """
    Load all models required for evaluation.
    Includes all 5 models mentioned in the paper.

    Returns:
        Dictionary of model name -> model
    """
    print("Loading all models for evaluation...")
    models_dict = {}

    # All models mentioned in Table 1 with fixed weight initialization
    models_dict["resnet18"] = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    models_dict["resnet18"].eval()
    models_dict["resnet18"].to(device)

    models_dict["resnet50"] = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    models_dict["resnet50"].eval()
    models_dict["resnet50"].to(device)

    models_dict["vgg16"] = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    models_dict["vgg16"].eval()
    models_dict["vgg16"].to(device)

    # Fixed EfficientNet loading
    models_dict["efficientnet"] = models.efficientnet_b0(
        weights=EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    models_dict["efficientnet"].eval()
    models_dict["efficientnet"].to(device)

    models_dict["mobilenet"] = models.mobilenet_v3_large(
        weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1
    )
    models_dict["mobilenet"].eval()
    models_dict["mobilenet"].to(device)

    return models_dict


def load_image(image_path="data/images/panda.jpg"):
    """
    Load and preprocess an image for adversarial attack testing.

    Args:
        image_path: Path to the image file

    Returns:
        Preprocessed image tensor
    """
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Load the image
    img = Image.open(image_path)

    # Preprocess the image
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor


def get_prediction(model, img_tensor):
    """
    Get model prediction and confidence for an image.

    Args:
        model: The neural network model
        img_tensor: Input image tensor

    Returns:
        Dictionary with class name, confidence, and index
    """
    # Load class labels from local file
    imagenet_classes_path = "data/imagenet/imagenet_classes.txt"

    # Check if file exists
    if not os.path.exists(imagenet_classes_path):
        raise FileNotFoundError(
            f"ImageNet classes file not found at {imagenet_classes_path}"
        )

    with open(imagenet_classes_path) as f:
        classes = [line.strip() for line in f.readlines()]

    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]

    # Get top prediction
    confidence, index = torch.max(probabilities, 0)
    class_name = classes[index.item()]

    return {
        "class": class_name,
        "confidence": confidence.item(),
        "index": index.item(),
        "probabilities": probabilities.cpu().numpy(),  # Keep full probabilities for detailed analysis
    }


def calculate_ssim(img1, img2, mean, std):
    """
    Calculate structural similarity between two images.

    Args:
        img1, img2: Image tensors
        mean, std: Normalization parameters

    Returns:
        SSIM value between 0 and 1
    """
    # Denormalize images
    img1_np = denormalize_to_numpy(img1.squeeze(0).cpu(), mean, std)
    img2_np = denormalize_to_numpy(img2.squeeze(0).cpu(), mean, std)

    # Convert to grayscale for SSIM calculation
    img1_gray = np.mean(img1_np, axis=2)
    img2_gray = np.mean(img2_np, axis=2)

    # Calculate SSIM
    ssim_value = ssim(img1_gray, img2_gray, data_range=1.0)
    return ssim_value


def denormalize_to_numpy(tensor, mean, std):
    """
    Denormalize a tensor and convert to numpy for visualization/measurement.

    Args:
        tensor: Input tensor [C, H, W]
        mean, std: Normalization parameters

    Returns:
        Numpy array [H, W, C] in range [0, 1]
    """
    # Denormalize
    if mean.dim() == 1:
        mean = mean.view(-1, 1, 1)
    if std.dim() == 1:
        std = std.view(-1, 1, 1)

    img = tensor.clone().detach()
    img = img * std + mean
    img = torch.clamp(img, 0, 1)

    # Convert to numpy and transpose
    return img.permute(1, 2, 0).numpy()


def calculate_metrics(original_img, adv_img, attack_metrics):
    """
    Calculate various perturbation metrics for Table 2.

    Args:
        original_img: Original image tensor
        adv_img: Adversarial image tensor
        attack_metrics: Metrics dictionary from the attack

    Returns:
        Dictionary with all calculated metrics
    """
    # Calculate perturbation
    perturbation = adv_img - original_img

    # Calculate norms
    l2_norm = (
        torch.norm(perturbation.view(perturbation.shape[0], -1), p=2, dim=1)
        .mean()
        .item()
    )
    linf_norm = (
        torch.norm(perturbation.view(perturbation.shape[0], -1), p=float("inf"), dim=1)
        .mean()
        .item()
    )

    # Calculate SSIM
    ssim_value = calculate_ssim(original_img, adv_img, mean, std)

    # Return all metrics needed for Table 2
    return {
        # Include original metrics from the attack
        **attack_metrics,
        # Add perturbation metrics
        "l2_norm": l2_norm,
        "linf_norm": linf_norm,
        "ssim": ssim_value,
    }


def create_attack(method_name, config, model):
    """
    Create attack instance based on method name and configuration.

    Args:
        method_name: Name of the attack method
        config: Configuration dictionary
        model: Target model

    Returns:
        Attack instance
    """
    if method_name == "cw":
        attack = CW(
            model=model,
            confidence=config["confidence"],
            c_init=config["c_init"],
            binary_search_steps=config["binary_search_steps"],
            max_iter=config["max_iter"],
            learning_rate=0.01,
            targeted=config["targeted"],
            device=device,
        )

        # Set normalization parameters
        attack.mean = mean.view(1, 3, 1, 1).to(device)
        attack.std = std.view(1, 3, 1, 1).to(device)

        return attack
    # Add other attack methods as they are implemented
    else:
        raise ValueError(f"Unknown attack method: {method_name}")


def get_method_configs(method_name):
    """
    Get attack configurations based on method name.

    Args:
        method_name: Name of the attack method

    Returns:
        List of configuration dictionaries
    """
    if method_name == "cw":
        return [
            # Untargeted configuration for main tables
            {
                "name": "Untargeted Baseline",
                "confidence": 0,
                "c_init": 1.0,
                "binary_search_steps": 5,
                "max_iter": 1000,
                "targeted": False,
            },
            # Other configurations for exploration
            {
                "name": "Targeted Baseline",
                "confidence": 0,
                "c_init": 0.1,
                "binary_search_steps": 5,
                "max_iter": 1000,
                "targeted": True,
            },
            {
                "name": "Medium confidence",
                "confidence": 20,
                "c_init": 0.1,
                "binary_search_steps": 5,
                "max_iter": 1000,
                "targeted": True,
            },
            {
                "name": "High confidence",
                "confidence": 50,
                "c_init": 0.1,
                "binary_search_steps": 5,
                "max_iter": 1000,
                "targeted": True,
            },
            {
                "name": "Small c_init",
                "confidence": 0,
                "c_init": 0.01,
                "binary_search_steps": 5,
                "max_iter": 1000,
                "targeted": True,
            },
            {
                "name": "Medium c_init",
                "confidence": 0,
                "c_init": 1.0,
                "binary_search_steps": 5,
                "max_iter": 1000,
                "targeted": True,
            },
            {
                "name": "Large c_init",
                "confidence": 0,
                "c_init": 10.0,
                "binary_search_steps": 5,
                "max_iter": 1000,
                "targeted": True,
            },
            {
                "name": "More search steps",
                "confidence": 0,
                "c_init": 1.0,
                "binary_search_steps": 9,
                "max_iter": 1000,
                "targeted": True,
            },
            {
                "name": "Fewer iterations",
                "confidence": 0,
                "c_init": 1.0,
                "binary_search_steps": 5,
                "max_iter": 500,
                "targeted": True,
            },
            {
                "name": "High conf + Large c",
                "confidence": 50,
                "c_init": 10.0,
                "binary_search_steps": 5,
                "max_iter": 1000,
                "targeted": True,
            },
        ]
    # Add configurations for other methods as they are implemented
    else:
        raise ValueError(f"Unknown attack method: {method_name}")


def save_to_csv(results, attack_method, primary_config=None):
    """
    Save results to CSV file.

    Args:
        results: List of dictionaries with results
        attack_method: Name of the attack method
        primary_config: Name of the primary configuration for table data
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{attack_method}_{timestamp}.csv"

    # Ensure all dictionaries have the same keys
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(all_keys))
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {filename}")

    # If primary config is specified, also save table data separately
    if primary_config:
        # Filter results for the primary configuration
        primary_results = [r for r in results if r["config_name"] == primary_config]

        # Extract and save table data
        if primary_results:
            save_table_data(primary_results, attack_method)


def save_table_data(results, method_name):
    """
    Extract and save data for each table from results.

    Args:
        results: List of result dictionaries for the primary configuration
        method_name: Name of the attack method
    """
    table_data = {
        "method": method_name,
        "table1_success_rates": {},  # Success rates per model
        "table2_perturbation": {  # Perturbation metrics
            "l2_norm": 0.0,
            "linf_norm": 0.0,
            "ssim": 0.0,
        },
        "table3_computation": {  # Computational metrics
            "iterations": 0,
            "gradient_calls": 0,
            "runtime": 0.0,
        },
    }

    # Process results for all models
    models_count = 0
    total_l2 = total_linf = total_ssim = 0.0
    total_iter = total_grad = total_time = 0.0

    # Track success rates for each model
    for result in results:
        model_name = result["model_name"]

        # Table 1: Success rates per model
        table_data["table1_success_rates"][model_name] = int(result["success"]) * 100

        # Accumulate for Table 2 and 3 averages
        total_l2 += result["l2_norm"]
        total_linf += result["linf_norm"]
        total_ssim += result["ssim"]
        total_iter += result.get("iterations", 0)
        total_grad += result.get("gradient_calls", 0)
        total_time += result["time"]
        models_count += 1

    # Calculate averages for Table 2 and 3
    if models_count > 0:
        table_data["table2_perturbation"]["l2_norm"] = total_l2 / models_count
        table_data["table2_perturbation"]["linf_norm"] = total_linf / models_count
        table_data["table2_perturbation"]["ssim"] = total_ssim / models_count

        table_data["table3_computation"]["iterations"] = total_iter / models_count
        table_data["table3_computation"]["gradient_calls"] = total_grad / models_count
        table_data["table3_computation"]["runtime"] = total_time / models_count

    # Save table data to JSON
    with open(f"table_data_{method_name}.json", "w") as f:
        json.dump(table_data, f, indent=2)

    # Print table data for quick reference
    print("\n===== DATA FOR PAPER TABLES =====")
    print(f"Method: {method_name}")
    print("\nTable 1 - Success Rates (%):")
    for model, rate in table_data["table1_success_rates"].items():
        print(f"  {model}: {rate}")

    print("\nTable 2 - Perturbation Metrics:")
    t2 = table_data["table2_perturbation"]
    print(f"  L2 Norm: {t2['l2_norm']:.3f}")
    print(f"  Linf Norm: {t2['linf_norm']:.3f}")
    print(f"  SSIM: {t2['ssim']:.3f}")

    print("\nTable 3 - Computational Requirements:")
    t3 = table_data["table3_computation"]
    print(f"  Iterations: {int(t3['iterations'])}")
    print(f"  Gradient Calls: {int(t3['gradient_calls'])}")
    print(f"  Runtime (s): {t3['runtime']:.2f}")


def collect_table1_data(method_name, config, models_dict, image):
    """
    Collect data for Table 1: Success rates across all models.
    Run attacks against each model and record success rates.

    Args:
        method_name: Name of the attack method
        config: Configuration dictionary to use
        models_dict: Dictionary of models to test
        image: Input image tensor

    Returns:
        List of result dictionaries with success information for each model
    """
    print(f"\n=== Collecting success rate data (Table 1) for {method_name} ===")
    results = []

    # Test each model
    for model_name, model in tqdm(models_dict.items()):
        print(f"\nTesting {model_name}...")

        # Get original prediction
        original_pred = get_prediction(model, image)
        print(
            f"Original prediction: {original_pred['class']} with {original_pred['confidence']*100:.2f}% confidence"
        )

        # Create attack for this model
        attack = create_attack(method_name, config, model)

        # Set target based on attack type
        if config["targeted"]:
            # Target class: flamingo (ID 130) for targeted attacks
            target_index = 130
            target = torch.tensor([target_index]).to(device)
        else:
            # Use original class for untargeted attacks
            target_index = None
            target = torch.tensor([original_pred["index"]]).to(device)

        # Measure runtime
        start_time = time.time()

        # Generate adversarial example
        adv_img, attack_metrics = attack.generate(image, target)

        # Record elapsed time
        elapsed_time = time.time() - start_time

        # Get adversarial prediction
        adv_pred = get_prediction(model, adv_img)

        # Determine success based on attack type
        if config["targeted"]:
            success = adv_pred["index"] == target_index
        else:
            success = adv_pred["index"] != original_pred["index"]

        # Calculate metrics
        metrics = calculate_metrics(image, adv_img, attack_metrics)

        # Store result
        result = {
            "method": method_name,
            "model_name": model_name,
            "config_name": config["name"],
            "success": success,
            "original_class": original_pred["class"],
            "original_confidence": original_pred["confidence"],
            "adversarial_class": adv_pred["class"],
            "adversarial_confidence": adv_pred["confidence"],
            "l2_norm": metrics["l2_norm"],
            "linf_norm": metrics["linf_norm"],
            "ssim": metrics["ssim"],
            "time": elapsed_time,
            "iterations": metrics.get(
                "iterations", 0
            ),  # Might need to fix in attack implementation
            "gradient_calls": metrics.get(
                "gradient_calls", 0
            ),  # Might need to fix in attack implementation
            "targeted": config["targeted"],
        }

        # Print result
        print(f"Attack {'successful' if success else 'failed'}")
        print(
            f"Adversarial prediction: {adv_pred['class']} with {adv_pred['confidence']*100:.2f}% confidence"
        )
        print(
            f"L2 norm: {metrics['l2_norm']:.4f}, Linf norm: {metrics['linf_norm']:.4f}, SSIM: {metrics['ssim']:.4f}"
        )
        print(f"Runtime: {elapsed_time:.2f}s")

        results.append(result)

    return results


def run_comprehensive_evaluation(
    method_name, image_path="data/images/panda.jpg", visualize=False
):
    """
    Run comprehensive evaluation to collect data for all tables.

    Args:
        method_name: Name of the attack method to evaluate
        image_path: Path to the test image
        visualize: Whether to generate visualizations (default: False)
    """
    print(f"Starting comprehensive evaluation of {method_name}...")

    # Load all models
    models_dict = load_models()

    # Load test image
    image = load_image(image_path)

    # Get configurations for the method
    configs = get_method_configs(method_name)

    # For tables: Collect data for both untargeted (traditional baseline)
    primary_configs = {
        "untargeted": "Untargeted Baseline",
    }

    # Collect results for each primary configuration
    all_results = []

    for config_type, config_name in primary_configs.items():
        print(f"\n===== Evaluating {config_type} configuration: {config_name} =====")

        # Get the configuration dictionary
        primary_config_dict = next(
            (c for c in configs if c["name"] == config_name), configs[0]
        )

        # Table 1: Success rates across all models
        success_results = collect_table1_data(
            method_name, primary_config_dict, models_dict, image
        )
        all_results.extend(success_results)

        # Save intermediate results for this configuration
        save_to_csv(
            [r for r in all_results if r["config_name"] == config_name],
            f"{method_name}_{config_type}",
            config_name,
        )

    # Save all results to CSV
    save_to_csv(all_results, method_name)

    # If visualization is requested, generate visualizations
    if visualize:
        visualize_results(method_name, configs, models_dict, image)


def visualize_results(method_name, configs, models_dict, image):
    """
    Generate visualizations of attack results.

    Args:
        method_name: Name of the attack method
        configs: List of configuration dictionaries
        models_dict: Dictionary of models to test
        image: Input image tensor
    """
    if method_name == "cw":
        from src.plot.compare import visualize_cw_attack_grid

        # Use ResNet-50 for visualizing parameter effects
        model = models_dict["resnet50"]
        original_pred = get_prediction(model, image)

        originals = []
        perturbations = []
        adversarials = []
        results_list = []
        l2_norms = []
        config_names = []

        # Generate visualizations for each configuration
        for config in configs:
            # Create attack
            attack = create_attack(method_name, config, model)

            # Set target
            if config["targeted"]:
                target_index = 130  # flamingo
                target = torch.tensor([target_index]).to(device)
            else:
                target_index = None
                target = torch.tensor([original_pred["index"]]).to(device)

            # Generate adversarial example
            adv_img, attack_metrics = attack.generate(image, target)

            # Calculate perturbation
            perturbation = adv_img - image

            # Get prediction
            adv_pred = get_prediction(model, adv_img)

            # Calculate metrics
            metrics = calculate_metrics(image, adv_img, attack_metrics)

            # Store visualization data
            l2_norms.append(metrics["l2_norm"])
            config_names.append(config["name"])

            # Move tensors to CPU for visualization
            originals.append(image.squeeze(0).cpu())
            perturbations.append(perturbation.squeeze(0).cpu())
            adversarials.append(adv_img.squeeze(0).cpu())

            results_list.append(
                {
                    "original_class": original_pred["class"],
                    "original_confidence": original_pred["confidence"],
                    "adversarial_class": adv_pred["class"],
                    "adversarial_confidence": adv_pred["confidence"],
                }
            )

        # Visualize all results
        visualize_cw_attack_grid(
            originals=originals,
            perturbations=perturbations,
            adversarials=adversarials,
            results_list=results_list,
            l2_norms=l2_norms,
            config_names=config_names,
            method_name="C&W L2",
            mean=mean,
            std=std,
            save_path=f"{method_name}_attack_results.png",
            show=True,
        )

        # Create bar chart
        plt.figure(figsize=(12, 6))
        y_pos = np.arange(len(config_names))

        # Sort by L2 norm
        sorted_indices = np.argsort(l2_norms)
        sorted_config_names = [config_names[i] for i in sorted_indices]
        sorted_l2_norms = [l2_norms[i] for i in sorted_indices]

        plt.barh(y_pos, sorted_l2_norms, align="center")
        plt.yticks(y_pos, sorted_config_names)
        plt.xlabel("L2 Norm (smaller is better)")
        plt.title(f"{method_name.upper()} Attack Performance Comparison")
        plt.tight_layout()
        plt.savefig(f"{method_name}_l2_comparison.png", bbox_inches="tight", dpi=150)
        plt.show()


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Run comprehensive adversarial attack evaluation."
    )
    parser.add_argument(
        "--method", type=str, default="cw", help="Attack method to evaluate"
    )
    parser.add_argument(
        "--image", type=str, default="data/images/panda.jpg", help="Path to image file"
    )
    parser.add_argument(
        "--viz", dest="visualize", action="store_true", help="Enable visualizations"
    )
    parser.set_defaults(visualize=False)

    args = parser.parse_args()

    # Run evaluation
    run_comprehensive_evaluation(args.method, args.image, args.visualize)
