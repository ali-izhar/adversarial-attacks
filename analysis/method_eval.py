"""
Adversarial Attack Method Evaluation Script

Usage:
    python method_eval.py --config path/to/config.yaml
"""

import sys
import argparse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import time
import csv
import json
import yaml
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

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import attack methods
from src.attacks.baseline import CW, DeepFool, FFGSM, FGSM

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization parameters
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate adversarial attack methods."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="results/config.yaml",
        help="Path to configuration YAML file",
    )
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_models(model_names):
    """Load specified models from config."""
    print("Loading models...")
    models_dict = {}

    model_weights = {
        "resnet18": ResNet18_Weights.IMAGENET1K_V1,
        "resnet50": ResNet50_Weights.IMAGENET1K_V1,
        "vgg16": VGG16_Weights.IMAGENET1K_V1,
        "efficientnet": EfficientNet_B0_Weights.IMAGENET1K_V1,
        "mobilenet": MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    }

    model_functions = {
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
        "vgg16": models.vgg16,
        "efficientnet": models.efficientnet_b0,
        "mobilenet": models.mobilenet_v3_large,
    }

    for model_name in model_names:
        if model_name not in model_functions:
            print(f"Warning: Model {model_name} not recognized. Skipping.")
            continue

        print(f"Loading {model_name}...")
        model = model_functions[model_name](weights=model_weights[model_name])
        model.mean = mean
        model.std = std
        model.eval()
        model.to(device)
        models_dict[model_name] = model

    return models_dict


def load_imagenet_samples(num_images=100, image_dir="data/imagenet/images"):
    """
    Load a sample of ImageNet images.
    If sample images don't exist locally, download a sample set.
    """
    print(f"Loading {num_images} ImageNet sample images...")
    
    os.makedirs(image_dir, exist_ok=True)
    
    # If no images in directory, download a small sample set
    if len([f for f in os.listdir(image_dir) if f.endswith('.JPEG')]) < num_images:
        print("Downloading ImageNet sample images...")
        # This URL contains sample ImageNet images (adjust if needed)
        import requests
        import zipfile
        from io import BytesIO
        
        # Get a sample set of images from the github repo
        sample_url = "https://github.com/EliSchwartz/imagenet-sample-images/archive/master.zip"
        
        try:
            r = requests.get(sample_url)
            z = zipfile.ZipFile(BytesIO(r.content))
            z.extractall("data/imagenet/")
            
            # Move images to the correct directory
            import shutil
            sample_dir = "data/imagenet/imagenet-sample-images-master"
            for f in os.listdir(sample_dir):
                if f.endswith('.JPEG'):
                    shutil.copy(os.path.join(sample_dir, f), image_dir)
            
            print(f"Downloaded sample images to {image_dir}")
        except Exception as e:
            print(f"Error downloading sample images: {e}")
            print("Please manually download images to the specified directory.")
            return []
    
    # Load and preprocess images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    image_tensors = []
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                  if f.endswith('.JPEG') or f.endswith('.jpg') or f.endswith('.png')]
    
    for i, img_path in enumerate(image_paths[:num_images]):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            image_tensors.append(img_tensor)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    print(f"Loaded {len(image_tensors)} images")
    return image_tensors


def get_prediction(model, img_tensor):
    """Get model prediction for an image."""
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        confidence, index = torch.max(probabilities, 0)
    
    return {
        "index": index.item(),
        "confidence": confidence.item(),
        "probabilities": probabilities.cpu(),
    }


def calculate_ssim(img1, img2):
    """Calculate structural similarity between two images."""
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
    """Denormalize a tensor and convert to numpy for visualization."""
    # Denormalize
    if mean.dim() == 1:
        mean = mean.view(-1, 1, 1)
    if std.dim() == 1:
        std = std.view(-1, 1, 1)

    img = tensor.clone()
    img = img * std + mean
    img = torch.clamp(img, 0, 1)

    # Convert to numpy and transpose
    return img.permute(1, 2, 0).numpy()


def create_attack(config, model, norm):
    """
    Dynamically create the attack method based on config.
    
    Args:
        config: Configuration dictionary
        model: The model to attack
        norm: Norm to use ('L2' or 'Linf')
        
    Returns:
        Instantiated attack method
    """
    method = config["attack"]["method"]
    targeted = config["attack"]["targeted"]
    params = config["attack"]["params"][method]
    
    if method == "FGSM":
        return FGSM(
            model=model,
            norm=norm,
            eps=0.01,  # Default, will be overridden during evaluation
            targeted=targeted,
            loss_fn=params.get("loss_fn", "cross_entropy"),
            verbose=False,
            device=device,
        )
    elif method == "FFGSM":
        return FFGSM(
            model=model,
            norm=norm,
            eps=0.01,  # Default, will be overridden during evaluation
            alpha=params.get("alpha", 0.2),
            targeted=targeted,
            loss_fn=params.get("loss_fn", "cross_entropy"),
            verbose=False,
            device=device,
        )
    elif method == "DeepFool":
        return DeepFool(
            model=model,
            norm=norm,
            num_classes=params.get("num_classes", 1000),
            overshoot=params.get("overshoot", 0.02),
            max_iter=params.get("max_iter", 50),
            verbose=False,
            device=device,
        )
    elif method == "CW":
        return CW(
            model=model,
            confidence=params.get("confidence", 0.0),
            c_init=params.get("c_init", 0.01),
            max_iter=params.get("max_iter", 1000),
            binary_search_steps=params.get("binary_search_steps", 5),
            learning_rate=params.get("learning_rate", 0.01),
            targeted=targeted,
            abort_early=params.get("abort_early", True),
            verbose=False,
            device=device,
        )
    else:
        raise ValueError(f"Unknown attack method: {method}")


def run_attack_evaluation(config, models_dict, images):
    """
    Run attack evaluation on all models with parameters from config.
    
    Args:
        config: Configuration dictionary
        models_dict: Dictionary of models to test
        images: List of input image tensors
        
    Returns:
        Dictionary of results for all tables
    """
    method = config["attack"]["method"]
    targeted = config["attack"]["targeted"]
    norm_types = config["attack"]["norm_types"]
    
    all_results = {}
    
    # For each norm type specified in config
    for norm in norm_types:
        print(f"\nEvaluating {method} attack with {norm} norm ({'targeted' if targeted else 'untargeted'})")
        
        # Get epsilon values for this norm
        if method in ["FGSM", "FFGSM"]:
            eps_values = config["attack"]["params"][method]["eps_values"][norm]
        else:
            # For methods that don't use epsilon (like DeepFool), use a dummy value
            eps_values = [0.0]
        
        # Initialize results structure for all tables
        results = {
            "table1": {model_name: {str(eps): 0 for eps in eps_values} for model_name in models_dict},
            "table2": {str(eps): {"l2_norm": 0, "linf_norm": 0, "ssim": 0} for eps in eps_values},
            "table3": {str(eps): {"iterations": 0, "gradient_calls": 0, "runtime": 0} for eps in eps_values}
        }
        
        # Initialize counters for averaging
        attack_counts = {str(eps): 0 for eps in eps_values}
        
        # Create a tqdm progress bar for the total number of evaluations
        total_evals = len(models_dict) * len(images) * len(eps_values)
        pbar = tqdm(total=total_evals, desc=f"{method} {norm} Evaluation")
        
        # Initialize detailed results for CSV export
        detailed_results = []
        
        # For each model and image
        for model_name, model in models_dict.items():
            print(f"\nEvaluating {method} on {model_name}")
            
            for image_idx, image in enumerate(images):
                # Get original prediction
                orig_pred = get_prediction(model, image)
                
                # For each epsilon value
                for eps in eps_values:
                    eps_str = str(eps)
                    
                    # Create attack instance
                    attack = create_attack(config, model, norm)
                    
                    # For methods that use epsilon, set it
                    if hasattr(attack, "eps"):
                        attack.eps = eps
                    
                    # Set target label (for either targeted or untargeted attack)
                    if targeted:
                        # For targeted attack, target a random class different from original
                        target_class = (orig_pred["index"] + 1) % 1000
                        target = torch.tensor([target_class]).to(device)
                    else:
                        # For untargeted attack, use the original class
                        target = torch.tensor([orig_pred["index"]]).to(device)
                    
                    # Measure runtime
                    start_time = time.time()
                    
                    # Generate adversarial example
                    try:
                        adv_image, attack_metrics = attack.generate(image, target)
                        
                        # Record runtime
                        runtime = time.time() - start_time
                        
                        # Get new prediction
                        adv_pred = get_prediction(model, adv_image)
                        
                        # Determine success (for untargeted: prediction changed, for targeted: prediction matches target)
                        if targeted:
                            success = adv_pred["index"] == target_class
                        else:
                            success = adv_pred["index"] != orig_pred["index"]
                        
                        # Calculate perturbation metrics
                        perturbation = adv_image - image
                        l2_norm = torch.norm(perturbation.view(perturbation.shape[0], -1), p=2).item()
                        linf_norm = torch.norm(perturbation.view(perturbation.shape[0], -1), p=float("inf")).item()
                        ssim_value = calculate_ssim(image, adv_image)
                        
                        # Get computational metrics
                        if hasattr(attack, "total_iterations"):
                            iterations = attack.total_iterations
                        else:
                            iterations = 1  # Default for single-step methods
                            
                        if hasattr(attack, "total_gradient_calls"):
                            gradient_calls = attack.total_gradient_calls
                        else:
                            gradient_calls = 1  # Default for simple methods
                        
                        # Update results for Table 1 (success rates per model)
                        if success:
                            results["table1"][model_name][eps_str] += 1
                        
                        # Update results for Table 2 (perturbation metrics)
                        results["table2"][eps_str]["l2_norm"] += l2_norm
                        results["table2"][eps_str]["linf_norm"] += linf_norm
                        results["table2"][eps_str]["ssim"] += ssim_value
                        
                        # Update results for Table 3 (computational metrics)
                        results["table3"][eps_str]["iterations"] += iterations
                        results["table3"][eps_str]["gradient_calls"] += gradient_calls
                        results["table3"][eps_str]["runtime"] += runtime
                        
                        # Increment counter
                        attack_counts[eps_str] += 1
                        
                        # Record detailed result
                        detailed_result = {
                            "model": model_name,
                            "image_idx": image_idx,
                            "epsilon": eps,
                            "norm": norm,
                            "targeted": targeted,
                            "success": success,
                            "l2_norm": l2_norm,
                            "linf_norm": linf_norm,
                            "ssim": ssim_value,
                            "runtime": runtime,
                            "iterations": iterations,
                            "gradient_calls": gradient_calls,
                            "original_class": orig_pred["index"],
                            "original_confidence": orig_pred["confidence"],
                            "adversarial_class": adv_pred["index"],
                            "adversarial_confidence": adv_pred["confidence"],
                        }
                        detailed_results.append(detailed_result)
                        
                    except Exception as e:
                        print(f"Error during attack: {e}")
                        detailed_result = {
                            "model": model_name,
                            "image_idx": image_idx,
                            "epsilon": eps,
                            "norm": norm,
                            "targeted": targeted,
                            "error": str(e)
                        }
                        detailed_results.append(detailed_result)
                    
                    # Update progress bar
                    pbar.update(1)
        
        pbar.close()
        
        # Calculate averages for Tables 1, 2, and 3
        for eps_str in attack_counts.keys():
            # Skip if no successful attacks for this epsilon
            if attack_counts[eps_str] == 0:
                continue
            
            # Table 1: Convert counts to percentages per model
            for model_name in results["table1"]:
                results["table1"][model_name][eps_str] = (results["table1"][model_name][eps_str] / len(images)) * 100
            
            # Table 2: Average perturbation metrics
            results["table2"][eps_str]["l2_norm"] /= attack_counts[eps_str]
            results["table2"][eps_str]["linf_norm"] /= attack_counts[eps_str]
            results["table2"][eps_str]["ssim"] /= attack_counts[eps_str]
            
            # Table 3: Average computational metrics
            results["table3"][eps_str]["iterations"] /= attack_counts[eps_str]
            results["table3"][eps_str]["gradient_calls"] /= attack_counts[eps_str]
            results["table3"][eps_str]["runtime"] /= attack_counts[eps_str]
        
        # Save detailed results to CSV
        if config["evaluation"]["save_results"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = config["evaluation"]["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            
            detailed_filename = f"{output_dir}/{method}_{norm}_{'targeted' if targeted else 'untargeted'}_{timestamp}.csv"
            
            with open(detailed_filename, "w", newline="") as csvfile:
                if detailed_results:
                    writer = csv.DictWriter(csvfile, fieldnames=detailed_results[0].keys())
                    writer.writeheader()
                    writer.writerows(detailed_results)
            
            print(f"Detailed results saved to {detailed_filename}")
            
            # Save table results to JSON
            table_filename = f"{output_dir}/{method}_{norm}_tables_{'targeted' if targeted else 'untargeted'}_{timestamp}.json"
            with open(table_filename, "w") as jsonfile:
                json.dump(results, jsonfile, indent=2)
            
            print(f"Table results saved to {table_filename}")
        
        # Store results for this norm
        all_results[norm] = {
            "results": results,
            "detailed_results": detailed_results
        }
    
    return all_results


def format_paper_tables(results, config, eps_value=None, norm="Linf"):
    """Format results into paper tables format for easy viewing."""
    method = config["attack"]["method"]
    targeted = config["attack"]["targeted"]
    
    # If eps_value not provided, use a default from config if available
    if eps_value is None:
        if method in ["FGSM", "FFGSM"]:
            eps_value = config["attack"]["params"][method]["eps_values"][norm][1]  # Use second value (typically mid-range)
        else:
            eps_value = 0.0  # Default for methods that don't use epsilon
    
    # Get the specific epsilon results
    eps_str = str(eps_value)
    
    # Ensure results for this epsilon exist
    if eps_str not in results["table2"]:
        print(f"No results found for epsilon = {eps_value}")
        return
    
    print(f"\n===== PAPER TABLE DATA FOR {method} =====")
    print(f"Epsilon: {eps_value}, Norm: {norm}, {'Targeted' if targeted else 'Untargeted'}")
    
    # Table 1: Success rates per model
    print("\nTable 1: Success Rates (%)")
    print("-" * 50)
    print(f"{'Model':<15} | {'Success Rate (%)':<15}")
    print("-" * 50)
    for model, rates in results["table1"].items():
        if eps_str in rates:
            print(f"{model:<15} | {rates[eps_str]:.2f}")
    
    # Table 2: Perturbation metrics
    print("\nTable 2: Perturbation Metrics")
    print("-" * 50)
    metrics = results["table2"][eps_str]
    print(f"L2 Norm:    {metrics['l2_norm']:.4f}")
    print(f"Linf Norm:   {metrics['linf_norm']:.4f}")
    print(f"SSIM:        {metrics['ssim']:.4f}")
    
    # Table 3: Computational metrics
    print("\nTable 3: Computational Requirements")
    print("-" * 50)
    comp = results["table3"][eps_str]
    print(f"Iterations:     {comp['iterations']:.1f}")
    print(f"Gradient Calls: {comp['gradient_calls']:.1f}")
    print(f"Runtime (s):    {comp['runtime']:.4f}")


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load models
    models = load_models(config["models"])
    
    # Load sample images
    dataset_config = config["dataset"]
    images = load_imagenet_samples(
        num_images=dataset_config["num_images"],
        image_dir=dataset_config["image_dir"]
    )
    
    if not images:
        print("No images loaded, cannot continue.")
        return
    
    # Run attack evaluation
    all_results = run_attack_evaluation(config, models, images)
    
    # Format and display results for paper tables
    for norm in config["attack"]["norm_types"]:
        results = all_results[norm]["results"]
        
        # Choose an epsilon value to display
        if config["attack"]["method"] in ["FGSM", "FFGSM"]:
            eps_values = config["attack"]["params"][config["attack"]["method"]]["eps_values"][norm]
            eps_to_display = eps_values[1] if len(eps_values) > 1 else eps_values[0]  # Use second value if available
        else:
            eps_to_display = 0.0
        
        # Display formatted tables
        format_paper_tables(results, config, eps_value=eps_to_display, norm=norm)
    
    print("\nEvaluation complete. Results are saved in the output directory.")


if __name__ == "__main__":
    main()
