#! /usr/bin/env python

"""
This script demonstrates adversarial attacks on a pre-trained model using PyTorch.

USAGE::
    >>> python demo.py --image-path path/to/image.jpg --method fgsm --epsilon 0.03

To see all available options, run:
    >>> python demo.py --help
"""


import argparse
import os
import sys
import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_skimage

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.datasets.imagenet import (
    get_imagenet_transforms,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImageNetDataset,
)
from src.models.wrappers import get_model
from src.attacks import FGSM, FFGSM, DeepFool, CW, PGD, CG


def denormalize_tensor(tensor, mean, std):
    """Denormalizes a tensor image with mean and standard deviation."""
    if tensor.ndim == 3:
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
    elif tensor.ndim == 4:  # Batch of images
        for i in range(tensor.size(0)):
            for t, m, s in zip(tensor[i], mean, std):
                t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def load_and_transform_image(image_path, transform):
    """Loads a single image and applies transformations."""
    try:
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0)  # Add batch dimension
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def plot_adv_images(
    original_img_tensor,
    adv_img_tensor,
    perturbation_tensor,
    original_label_str,
    adv_label_str,
    linf_norm,
    l2_norm,
    ssim_val,
    output_path,
    title_prefix="",
):
    """Plots original, perturbation, and adversarial images."""
    # Denormalize for plotting
    orig_plot = denormalize_tensor(
        original_img_tensor.cpu().clone().squeeze(0), IMAGENET_MEAN, IMAGENET_STD
    )
    adv_plot = denormalize_tensor(
        adv_img_tensor.cpu().clone().squeeze(0), IMAGENET_MEAN, IMAGENET_STD
    )
    pert_plot = perturbation_tensor.cpu().clone().squeeze(0)

    # Visualize perturbation: amplify and center around 0.5
    # This makes 0 perturbation mid-gray. Positive perturbations -> lighter, Negative -> darker.
    # The factor (e.g., 5 or 10) controls sensitivity.
    amplification_factor = 10
    pert_plot_viz = (pert_plot * amplification_factor) + 0.5
    pert_plot_viz = torch.clamp(pert_plot_viz, 0, 1)  # Clip to [0,1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{title_prefix} Attack Results", fontsize=16)

    axes[0].imshow(orig_plot.permute(1, 2, 0).numpy())
    axes[0].set_title(f"Original\n{original_label_str}")
    axes[0].axis("off")

    axes[1].imshow(pert_plot_viz.permute(1, 2, 0).numpy())
    axes[1].set_title(f"Perturbation\L$\infty$: {linf_norm:.4f}, L2: {l2_norm:.4f}")
    axes[1].axis("off")

    axes[2].imshow(adv_plot.permute(1, 2, 0).numpy())
    axes[2].set_title(f"Adversarial\n{adv_label_str}\nSSIM: {ssim_val:.4f}")
    axes[2].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo script for adversarial attacks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        required=True,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--attack-method",
        "-a",
        type=str,
        required=True,
        choices=["fgsm", "ffgsm", "deepfool", "cw", "pgd", "cg"],
        help="Attack method to use.",
    )
    parser.add_argument(
        "--model-name", "-m", type=str, default="resnet18", help="Model architecture."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="results",
        help="Directory to save attack results and visualizations.",
    )
    parser.add_argument(
        "--config-file",
        "-c",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file for default attack parameters.",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        help="Device to use ('cuda' or 'cpu'). Autodetects if not set.",
    )
    parser.add_argument(
        "--true-label",
        "-l",
        type=int,
        default=None,
        help="Optional: True label ID of the input image for targeted attacks or reference.",
    )

    # Common attack parameters
    parser.add_argument(
        "--norm",
        "-n",
        type=str,
        choices=["Linf", "L2"],
        default="Linf",
        help="Norm for perturbation constraint (Note: CW attack typically uses L2, FGSM/FFGSM typically use Linf internally).",
    )
    parser.add_argument(
        "--epsilon",
        "-e",
        type=lambda x: eval(x) if "/" in x else float(x),
        help="Epsilon for perturbation budget (e.g., 0.03 or '8/255').",
    )
    parser.add_argument(
        "--steps", "-s", type=int, help="Number of iterations for iterative attacks."
    )

    # FGSM/FFGSM specific
    parser.add_argument(
        "--alpha",
        "-al",
        type=lambda x: eval(x) if "/" in x else float(x),
        help="Alpha for FFGSM's random start, or step size for CG.",
    )

    # DeepFool specific
    parser.add_argument(
        "--overshoot", "-os", type=float, help="Overshoot parameter for DeepFool."
    )

    # C&W specific
    parser.add_argument("--c-val", "-cv", type=float, help="C constant for C&W attack.")
    parser.add_argument(
        "--confidence", "-k", type=float, help="Confidence (kappa) for C&W attack."
    )
    parser.add_argument(
        "--learning-rate", "-lr", type=float, help="Learning rate for C&W optimizer."
    )

    # PGD specific
    parser.add_argument(
        "--step-size",
        "-ss",
        type=lambda x: eval(x) if "/" in x else float(x),
        help="Step size for PGD iterations (e.g., 0.01 or '2/255').",
    )

    # Targeted attack options
    parser.add_argument(
        "--targeted", "-t", action="store_true", help="Perform a targeted attack."
    )
    parser.add_argument(
        "--target-class",
        "-tc",
        type=int,
        help="Specify a target class ID for targeted attacks (used if --target-method is 'specific').",
    )
    parser.add_argument(
        "--target-method",
        "-tm",
        type=str,
        choices=["random", "least-likely", "specific"],
        default="specific",
        help="Target selection method for targeted attacks. 'specific' requires --target-class to be set.",
    )

    return parser.parse_args()


def get_config_params(
    config_path, method_name, norm_type, targeted, param_key, cli_val, default_val=None
):
    """Helper to get params: CLI > config > default."""
    if cli_val is not None:
        return cli_val
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        attack_config = (
            config.get("attack", {}).get("params", {}).get(method_name.upper(), {})
        )
        mode_config = attack_config.get("targeted" if targeted else "untargeted", {})

        if param_key == "epsilon":
            val = mode_config.get("eps_values", {}).get(norm_type, [default_val])[0]
        elif param_key == "step_size":  # PGD specific for norm
            val = mode_config.get(f"step_size_{norm_type.lower()}", default_val)
        elif param_key == "alpha_ffgsm":  # FFGSM specific
            val = mode_config.get(
                "alpha_linf", default_val
            )  # Assuming Linf for demo, adjust if L2 alpha exists
        else:
            val = mode_config.get(param_key, default_val)

        if isinstance(val, str) and "/" in val:  # handles fractions like "8/255"
            return eval(val)
        return val if val is not None else default_val
    except FileNotFoundError:
        # print(f"Config file {config_path} not found. Using default/CLI for {param_key}.")
        return default_val
    except Exception as e:
        # print(f"Error reading {param_key} from config {config_path}: {e}. Using default/CLI.")
        return default_val


def get_attack_instance(args, model):
    method = args.attack_method.lower()
    norm_type = args.norm  # Linf or L2

    # Get parameters, prioritizing CLI, then config, then hardcoded defaults aligned with README
    cfg_path = args.config_file

    epsilon = get_config_params(
        cfg_path, method, norm_type, args.targeted, "epsilon", args.epsilon, 0.03137
    )  # Default 8/255
    steps = get_config_params(
        cfg_path, method, norm_type, args.targeted, "steps", args.steps, 40
    )

    if method == "fgsm":
        print(
            f"Initializing FGSM (Linf): eps={epsilon:.4f}. CLI --norm '{norm_type}' is noted but FGSM standard is Linf."
        )
        return FGSM(model, eps=epsilon)
    elif method == "ffgsm":
        alpha_ffgsm = get_config_params(
            cfg_path,
            method,
            norm_type,
            args.targeted,
            "alpha_ffgsm",
            args.alpha,
            epsilon * 0.1,
        )
        print(
            f"Initializing FFGSM (Linf): eps={epsilon:.4f}, alpha={alpha_ffgsm:.4f}. CLI --norm '{norm_type}' is noted but FFGSM standard is Linf."
        )
        return FFGSM(model, eps=epsilon, alpha=alpha_ffgsm)
    elif method == "deepfool":
        steps_deepfool = get_config_params(
            cfg_path, method, norm_type, args.targeted, "steps", args.steps, 50
        )
        overshoot = get_config_params(
            cfg_path,
            method,
            norm_type,
            args.targeted,
            "overshoot",
            args.overshoot,
            0.02,
        )
        print(
            f"Initializing DeepFool: steps={steps_deepfool}, overshoot={overshoot:.4f}"
        )
        return DeepFool(
            model, steps=steps_deepfool, overshoot=overshoot
        )  # Norm typically L2, handled by class
    elif method == "cw":
        c_val = get_config_params(
            cfg_path,
            method,
            norm_type,
            args.targeted,
            "c_val",
            args.c_val,
            10.0 if args.targeted else 1.0,
        )
        confidence = get_config_params(
            cfg_path,
            method,
            norm_type,
            args.targeted,
            "confidence",
            args.confidence,
            5.0 if args.targeted else 0.0,
        )
        lr = get_config_params(
            cfg_path,
            method,
            norm_type,
            args.targeted,
            "learning_rate",
            args.learning_rate,
            0.01,
        )
        steps_cw = get_config_params(
            cfg_path, method, norm_type, args.targeted, "steps", args.steps, 500
        )
        # CW attack is typically L2. The --norm CLI argument is informational for CW in this demo.
        print(
            f"Initializing C&W (L2): c={c_val:.2f}, kappa={confidence:.2f}, steps={steps_cw}, lr={lr:.3f}. CLI --norm '{norm_type}' is noted but CW standard is L2."
        )
        return CW(
            model, c=c_val, kappa=confidence, steps=steps_cw, lr=lr
        )  # Removed norm argument
    elif method == "pgd":
        step_size = get_config_params(
            cfg_path,
            method,
            norm_type,
            args.targeted,
            "step_size",
            args.step_size,
            epsilon / 4.0,
        )
        steps_pgd = get_config_params(
            cfg_path, method, norm_type, args.targeted, "steps", args.steps, 40
        )
        loss_fn = get_config_params(
            cfg_path, method, norm_type, args.targeted, "loss_fn", None, "cross_entropy"
        )
        rand_init = get_config_params(
            cfg_path, method, norm_type, args.targeted, "rand_init", None, True
        )
        print(
            f"Initializing PGD: norm={norm_type}, eps={epsilon:.4f}, steps={steps_pgd}, step_size={step_size:.5f}"
        )
        return PGD(
            model,
            norm=norm_type.lower(),
            eps=epsilon,
            n_iterations=steps_pgd,
            step_size=step_size,
            loss_fn=loss_fn,
            rand_init=rand_init,
        )
    elif method == "cg":
        alpha_cg = get_config_params(
            cfg_path,
            method,
            norm_type,
            args.targeted,
            "alpha",
            args.alpha,
            epsilon / 4.0,
        )  # alpha for CG like step size
        steps_cg = get_config_params(
            cfg_path, method, norm_type, args.targeted, "steps", args.steps, 40
        )
        beta_method = get_config_params(
            cfg_path, method, norm_type, args.targeted, "beta_method", None, "PR"
        )
        rand_init = get_config_params(
            cfg_path, method, norm_type, args.targeted, "rand_init", None, False
        )
        print(
            f"Initializing CG: norm={norm_type}, eps={epsilon:.4f}, steps={steps_cg}, alpha={alpha_cg:.5f}, beta={beta_method}"
        )
        return CG(
            model,
            norm=norm_type.lower(),
            eps=epsilon,
            steps=steps_cg,
            alpha=alpha_cg,
            beta_method=beta_method,
            rand_init=rand_init,
        )
    else:
        raise ValueError(f"Unsupported attack method: {method}")


def compute_ssim_skimage(original_tensor_norm, perturbed_tensor_norm, mean, std):
    """Compute SSIM using skimage between original and perturbed normalized tensors."""
    orig_img_denorm = denormalize_tensor(original_tensor_norm.cpu().clone(), mean, std)
    pert_img_denorm = denormalize_tensor(perturbed_tensor_norm.cpu().clone(), mean, std)
    orig_img_np = orig_img_denorm.permute(1, 2, 0).numpy().astype(np.float32)
    pert_img_np = pert_img_denorm.permute(1, 2, 0).numpy().astype(np.float32)
    ssim_val = ssim_skimage(
        orig_img_np, pert_img_np, data_range=1.0, channel_axis=-1, win_size=7
    )
    return ssim_val


def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model: {args.model_name}...")
    model = get_model(model_name=args.model_name, pretrained=True).to(device)
    model.eval()
    # Get model-specific transforms (typically ImageNet ones)
    preprocess = get_imagenet_transforms(pretrained=True)

    # Load class names for display
    try:
        # Try loading from the data directory (data/imagenet/sample_images/)
        temp_data_dir = "data"  # Default or from config if available
        if os.path.exists(os.path.join(project_root, "config", "config.yaml")):
            try:
                with open(
                    os.path.join(project_root, "config", "config.yaml"), "r"
                ) as f:
                    cfg = yaml.safe_load(f)
                    temp_data_dir = cfg.get("dataset", {}).get("data_dir", "data")
            except:
                pass  # keep default temp_data_dir

        # Construct full path to dataset directory for class names
        dataset_base_path = os.path.join(project_root, temp_data_dir, "imagenet")
        if not os.path.exists(os.path.join(dataset_base_path, "imagenet_classes.txt")):
            print(
                f"Warning: imagenet_classes.txt not found at {os.path.join(dataset_base_path, 'imagenet_classes.txt')}. Class names will not be shown."
            )
            class_names = [f"ClassID {i}" for i in range(1000)]  # Fallback
        else:
            temp_dataset = ImageNetDataset(
                data_dir=dataset_base_path, transform=preprocess, max_samples=1
            )
            class_names = temp_dataset.class_names
            print(f"Loaded {len(class_names)} class names.")
    except Exception as e:
        print(f"Warning: Could not load class names: {e}. Displaying IDs only.")
        class_names = [f"ClassID {i}" for i in range(1000)]  # Fallback

    # Load and transform the single image
    print(f"Loading image from: {args.image_path}...")
    image_tensor = load_and_transform_image(args.image_path, preprocess)
    if image_tensor is None:
        sys.exit(1)
    image_tensor = image_tensor.to(device)

    # For demo, we need a label. If not provided, use model's top prediction as pseudo-true label for untargeted.
    # For targeted, a target_class must be provided or handled by attack's random/least-likely mode.
    true_label_idx = None
    if args.true_label is not None:
        true_label_idx = torch.tensor([args.true_label]).to(device)
        print(f"Using provided true label ID: {args.true_label}")

    with torch.no_grad():
        original_output = model(image_tensor)
        original_probs = torch.softmax(original_output, dim=1)
        original_conf, original_pred_idx = torch.max(original_probs, 1)

    # If true_label not given, use model's prediction as the reference for untargeted attacks
    if true_label_idx is None:
        true_label_idx = original_pred_idx.clone().detach()
        print(
            f"No true label provided. Using model's top prediction as reference: ID {true_label_idx.item()} (Confidence: {original_conf.item()*100:.2f}%)"
        )

    original_class_name = (
        class_names[original_pred_idx.item()]
        if original_pred_idx.item() < len(class_names)
        else f"ID {original_pred_idx.item()}"
    )
    original_pred_label_str = f"{original_class_name} (ID {original_pred_idx.item()})\n(Conf: {original_conf.item()*100:.2f}%)"

    # Instantiate attack
    try:
        attack = get_attack_instance(args, model)
        if args.targeted:
            if args.target_method == "specific":
                if args.target_class is None:
                    print(
                        "Error: For targeted attack with method 'specific', --target-class must be provided."
                    )
                    sys.exit(1)
                target_classes_tensor = torch.tensor([args.target_class]).to(device)
                attack.set_mode_targeted_fixed(target_classes_tensor)
                print(f"Attack mode: Targeted (Specific class: {args.target_class})")
            elif args.target_method == "random":
                if hasattr(attack, "set_mode_targeted_random"):
                    attack.set_mode_targeted_random()
                    print(
                        f"Attack mode: Targeted (Random target selection by attack class)"
                    )
                else:
                    print(
                        f"Error: Attack class {type(attack).__name__} does not support 'set_mode_targeted_random'."
                    )
                    sys.exit(1)
            elif args.target_method == "least-likely":
                if hasattr(attack, "set_mode_targeted_least_likely"):
                    attack.set_mode_targeted_least_likely()
                    print(
                        f"Attack mode: Targeted (Least-likely target selection by attack class)"
                    )
                else:
                    print(
                        f"Error: Attack class {type(attack).__name__} does not support 'set_mode_targeted_least_likely'."
                    )
                    sys.exit(1)
        else:
            attack.set_mode_default()  # Ensure it's in untargeted mode
            print("Attack mode: Untargeted")

    except ValueError as e:
        print(f"Error initializing attack: {e}")
        sys.exit(1)
    except AttributeError as e:
        print(
            f"AttributeError during attack setup (e.g. set_mode_...): {e}. Check attack class capabilities."
        )
        sys.exit(1)

    print(f"Running {args.attack_method.upper()} attack...")
    # Call the attack instance directly, assuming it implements __call__
    adv_image = attack(image_tensor, true_label_idx)

    with torch.no_grad():
        adv_output = model(adv_image)
        adv_probs = torch.softmax(adv_output, dim=1)
        adv_conf, adv_pred_idx = torch.max(adv_probs, 1)

    adv_class_name = (
        class_names[adv_pred_idx.item()]
        if adv_pred_idx.item() < len(class_names)
        else f"ID {adv_pred_idx.item()}"
    )
    adv_pred_label_str = f"{adv_class_name} (ID {adv_pred_idx.item()})\n(Conf: {adv_conf.item()*100:.2f}%)"

    perturbation = adv_image - image_tensor
    linf_norm = torch.norm(
        perturbation.view(perturbation.size(0), -1), p=float("inf"), dim=1
    ).item()
    l2_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=2, dim=1).item()

    # Calculate SSIM using the reinstated skimage-based function
    try:
        ssim_val = compute_ssim_skimage(
            image_tensor.squeeze(0), adv_image.squeeze(0), IMAGENET_MEAN, IMAGENET_STD
        )
    except Exception as e:
        print(f"Could not calculate SSIM with skimage: {e}. Using placeholder.")
        ssim_val = 0.0  # Placeholder

    print(f"Original prediction: {original_pred_label_str}")
    print(f"Adversarial prediction: {adv_pred_label_str}")
    print(f"Perturbation norms: L-inf: {linf_norm:.5f}, L2: {l2_norm:.5f}")
    print(f"SSIM: {ssim_val:.4f}")

    is_success = original_pred_idx.item() != adv_pred_idx.item()
    if args.targeted and args.target_class is not None:
        is_success = adv_pred_idx.item() == args.target_class
    elif args.targeted:  # Target was random/least-likely
        is_success = (
            original_pred_idx.item() != adv_pred_idx.item()
        )  # Basic check, specific target unknown here

    print(f"Attack {'successful' if is_success else 'failed'}.")

    # Plot and save
    if args.output_dir:
        img_fname = os.path.splitext(os.path.basename(args.image_path))[0]
        output_filename = f"{img_fname}_{args.attack_method}"
        if args.targeted:
            output_filename += f"_targeted_{args.target_method}"
            if args.target_method == "specific" and args.target_class is not None:
                output_filename += f"_{args.target_class}"
            # For random/least-likely, the actual target is determined by the attack and might not be easily available for filename
            # Consider logging the actual target if the attack class stores it.
        output_filename += ".png"

        plot_output_path = os.path.join(args.output_dir, output_filename)
        title_prefix = f"{args.model_name} - {args.attack_method.upper()}"
        plot_adv_images(
            image_tensor,
            adv_image,
            perturbation,
            original_pred_label_str,
            adv_pred_label_str,
            linf_norm,
            l2_norm,
            ssim_val,
            plot_output_path,
            title_prefix,
        )

    print(f"Demo for {args.attack_method.upper()} on {args.image_path} complete.")
    print(f"Results (if saved) are in {args.output_dir}")


if __name__ == "__main__":
    main()
