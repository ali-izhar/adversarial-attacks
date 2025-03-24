"""
Metrics for evaluating adversarial attack effectiveness.

This module provides functions for:
1. Calculating and comparing metrics for adversarial attacks
2. Evaluating attack success rates at different perturbation thresholds
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Callable
from sklearn.metrics import auc
import pandas as pd


def calculate_perturbation_metrics(
    original_images: torch.Tensor, adversarial_images: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Calculate various perturbation norms for comparison.

    Args:
        original_images: Clean images tensor (B, C, H, W)
        adversarial_images: Adversarial images tensor (B, C, H, W)

    Returns:
        Dictionary of metric tensors with batch dimension
    """
    # Calculate perturbation
    perturbation = adversarial_images - original_images

    # Reshape for per-image calculations
    batch_size = original_images.shape[0]
    flat_perturbation = perturbation.view(batch_size, -1)

    # Calculate different norms
    l0_norm = (flat_perturbation != 0).float().sum(dim=1)
    l0_percent = 100 * l0_norm / flat_perturbation.shape[1]
    l1_norm = flat_perturbation.abs().sum(dim=1)
    l2_norm = torch.norm(flat_perturbation, p=2, dim=1)
    l2_avg = l2_norm / np.sqrt(flat_perturbation.shape[1])  # Root mean square
    linf_norm = flat_perturbation.abs().max(dim=1)[0]

    # Mean Squared Error
    mse = (flat_perturbation**2).mean(dim=1)

    # Signal-to-Noise Ratio (SNR) in dB
    flat_original = original_images.view(batch_size, -1)
    signal_power = (flat_original**2).mean(dim=1)
    noise_power = (flat_perturbation**2).mean(dim=1)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))

    # Peak Signal-to-Noise Ratio (PSNR) in dB
    max_pixel_value = 1.0  # Assuming normalized images
    psnr = 10 * torch.log10((max_pixel_value**2) / (mse + 1e-10))

    return {
        "L0": l0_norm,
        "L0_percent": l0_percent,
        "L1": l1_norm,
        "L2": l2_norm,
        "L2_avg": l2_avg,
        "Linf": linf_norm,
        "MSE": mse,
        "SNR_dB": snr,
        "PSNR_dB": psnr,
    }


def evaluate_attack_success(
    model: torch.nn.Module,
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    true_labels: torch.Tensor,
    target_labels: Optional[torch.Tensor] = None,
    targeted: bool = False,
    confidence_threshold: float = 0.0,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Dict[str, Any]:
    """
    Evaluate attack success based on model predictions.

    Args:
        model: Model to evaluate against
        original_images: Clean images
        adversarial_images: Adversarial images
        true_labels: Original class labels
        target_labels: Target class labels (for targeted attacks)
        targeted: Whether attack is targeted
        confidence_threshold: Confidence threshold for success
        device: Device to run evaluation on

    Returns:
        Dictionary of success metrics
    """
    model.eval()

    with torch.no_grad():
        # Get predictions
        original_outputs = model(original_images.to(device))
        adversarial_outputs = model(adversarial_images.to(device))

        # Convert to probabilities
        original_probs = torch.nn.functional.softmax(original_outputs, dim=1)
        adversarial_probs = torch.nn.functional.softmax(adversarial_outputs, dim=1)

        # Get predictions
        original_preds = original_outputs.argmax(dim=1)
        adversarial_preds = adversarial_outputs.argmax(dim=1)

        # Calculate confidence scores
        original_conf = torch.gather(
            original_probs, 1, original_preds.unsqueeze(1)
        ).squeeze(1)
        adversarial_conf = torch.gather(
            adversarial_probs, 1, adversarial_preds.unsqueeze(1)
        ).squeeze(1)

        # For untargeted attacks, confidence in true class decreases
        if not targeted:
            true_class_conf_orig = torch.gather(
                original_probs, 1, true_labels.unsqueeze(1)
            ).squeeze(1)
            true_class_conf_adv = torch.gather(
                adversarial_probs, 1, true_labels.unsqueeze(1)
            ).squeeze(1)
            conf_decrease = true_class_conf_orig - true_class_conf_adv
        else:
            # For targeted attacks, confidence in target class increases
            target_class_conf_orig = torch.gather(
                original_probs, 1, target_labels.unsqueeze(1)
            ).squeeze(1)
            target_class_conf_adv = torch.gather(
                adversarial_probs, 1, target_labels.unsqueeze(1)
            ).squeeze(1)
            conf_increase = target_class_conf_adv - target_class_conf_orig

    # Calculate success metrics
    if targeted:
        # For targeted attacks, success means prediction matches target
        success_mask = adversarial_preds == target_labels
        # With confidence threshold
        if confidence_threshold > 0:
            success_mask = success_mask & (
                torch.gather(adversarial_probs, 1, target_labels.unsqueeze(1)).squeeze(
                    1
                )
                > confidence_threshold
            )
    else:
        # For untargeted attacks, success means prediction differs from original
        success_mask = adversarial_preds != true_labels
        # With confidence threshold
        if confidence_threshold > 0:
            top_class = adversarial_probs.argmax(dim=1)
            success_mask = success_mask & (
                torch.gather(adversarial_probs, 1, top_class.unsqueeze(1)).squeeze(1)
                > confidence_threshold
            )

    success_rate = success_mask.float().mean().item() * 100

    # Create results dictionary
    results = {
        "success_rate": success_rate,
        "success_mask": success_mask,
        "original_preds": original_preds,
        "adversarial_preds": adversarial_preds,
        "original_conf": original_conf,
        "adversarial_conf": adversarial_conf,
    }

    if not targeted:
        results["conf_decrease"] = conf_decrease
    else:
        results["conf_increase"] = conf_increase

    return results


def evaluate_across_thresholds(
    model: torch.nn.Module,
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    true_labels: torch.Tensor,
    target_labels: Optional[torch.Tensor] = None,
    targeted: bool = False,
    confidence_thresholds: List[float] = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Dict[str, List[float]]:
    """
    Evaluate attack success across different confidence thresholds.

    Args:
        model: Model to evaluate against
        original_images: Clean images
        adversarial_images: Adversarial images
        true_labels: Original class labels
        target_labels: Target class labels (for targeted attacks)
        targeted: Whether attack is targeted
        confidence_thresholds: List of confidence thresholds to evaluate
        device: Device to run evaluation on

    Returns:
        Dictionary of success rates at each threshold
    """
    threshold_results = []

    for threshold in confidence_thresholds:
        results = evaluate_attack_success(
            model,
            original_images,
            adversarial_images,
            true_labels,
            target_labels,
            targeted,
            threshold,
            device,
        )
        threshold_results.append(results["success_rate"])

    return {"thresholds": confidence_thresholds, "success_rates": threshold_results}


def evaluate_across_perturbation_budget(
    attack_fn: Callable,
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    target_labels: Optional[torch.Tensor] = None,
    targeted: bool = False,
    epsilon_values: List[float] = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
    norm: str = "L2",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Dict[str, List[Any]]:
    """
    Evaluate attack success across different perturbation budgets (epsilon).

    Args:
        attack_fn: Function that takes (model, images, labels, epsilon, norm, targeted) and returns adversarial examples
        model: Model to attack and evaluate
        images: Clean images
        labels: True labels
        target_labels: Target labels (for targeted attacks)
        targeted: Whether attack is targeted
        epsilon_values: List of epsilon values to test
        norm: Norm to use for the attack ("L2" or "Linf")
        device: Device to run on

    Returns:
        Dictionary containing lists of success rates, average perturbation norms, and other metrics
    """
    results = {
        "epsilon": epsilon_values,
        "success_rates": [],
        "avg_perturbation": [],
        "avg_l2_norm": [],
        "avg_linf_norm": [],
        "attack_time": [],
    }

    targets = target_labels if targeted else labels

    for eps in epsilon_values:
        print(f"Testing with epsilon = {eps}")

        # Apply the attack
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        adv_images = attack_fn(model, images, targets, eps, norm, targeted)
        end_time.record()

        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # convert to seconds

        # Evaluate success
        success_results = evaluate_attack_success(
            model, images, adv_images, labels, target_labels, targeted, device=device
        )

        # Calculate perturbation metrics
        pert_metrics = calculate_perturbation_metrics(images, adv_images)

        # Store results
        results["success_rates"].append(success_results["success_rate"])
        results["avg_perturbation"].append(
            torch.norm(adv_images - images, p=2).mean().item()
        )
        results["avg_l2_norm"].append(pert_metrics["L2"].mean().item())
        results["avg_linf_norm"].append(pert_metrics["Linf"].mean().item())
        results["attack_time"].append(elapsed_time)

    # Calculate attack ROC (success rate vs perturbation)
    if len(epsilon_values) >= 2:
        results["perturbation_auc"] = auc(
            results["avg_l2_norm"], [r / 100 for r in results["success_rates"]]
        )

    return results


def evaluate_transferability(
    source_model: torch.nn.Module,
    target_models: List[torch.nn.Module],
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    true_labels: torch.Tensor,
    target_labels: Optional[torch.Tensor] = None,
    targeted: bool = False,
    model_names: Optional[List[str]] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Dict[str, List[float]]:
    """
    Evaluate transferability of adversarial examples across different models.

    Args:
        source_model: Model used to generate adversarial examples
        target_models: List of models to evaluate transferability to
        original_images: Clean images
        adversarial_images: Adversarial images generated for source_model
        true_labels: Original class labels
        target_labels: Target class labels (for targeted attacks)
        targeted: Whether attack is targeted
        model_names: Optional list of model names for results
        device: Device to run evaluation on

    Returns:
        Dictionary of transferability success rates
    """
    # If model_names not provided, use generic names
    if model_names is None:
        model_names = [f"Model_{i}" for i in range(len(target_models) + 1)]
    else:
        assert (
            len(model_names) == len(target_models) + 1
        ), "Number of model names must match number of target models + 1"

    transfer_results = {"model_names": model_names, "success_rates": []}

    # First evaluate success on source model
    source_results = evaluate_attack_success(
        source_model,
        original_images,
        adversarial_images,
        true_labels,
        target_labels,
        targeted,
        device=device,
    )
    transfer_results["success_rates"].append(source_results["success_rate"])

    # Then evaluate on each target model
    for target_model in target_models:
        target_results = evaluate_attack_success(
            target_model,
            original_images,
            adversarial_images,
            true_labels,
            target_labels,
            targeted,
            device=device,
        )
        transfer_results["success_rates"].append(target_results["success_rate"])

    return transfer_results


# Plotting functions


def set_plot_style():
    """Set publication-quality plot style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 16
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def plot_success_rate_vs_confidence(
    threshold_results: Dict[str, List[float]],
    attack_name: str = "Attack",
    save_path: Optional[str] = None,
):
    """
    Plot attack success rate vs confidence threshold.

    Args:
        threshold_results: Results from evaluate_across_thresholds
        attack_name: Name of the attack method
        save_path: Path to save the figure (optional)
    """
    set_plot_style()

    thresholds = threshold_results["thresholds"]
    success_rates = threshold_results["success_rates"]

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, success_rates, "o-", linewidth=2, markersize=8)
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Attack Success Rate (%)")
    plt.title(f"{attack_name}: Success Rate vs. Confidence Threshold")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.ylim(0, 105)

    # Add value labels
    for i, (x, y) in enumerate(zip(thresholds, success_rates)):
        plt.annotate(
            f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
        )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_success_rate_vs_perturbation(
    perturbation_results: Dict[str, List[Any]],
    attack_name: str = "Attack",
    norm_type: str = "L2",
    save_path: Optional[str] = None,
):
    """
    Plot attack success rate vs perturbation size.

    Args:
        perturbation_results: Results from evaluate_across_perturbation_budget
        attack_name: Name of the attack method
        norm_type: Type of norm to display ("L2" or "Linf")
        save_path: Path to save the figure (optional)
    """
    set_plot_style()

    if norm_type == "L2":
        x_values = perturbation_results["avg_l2_norm"]
        x_label = "Average L2 Perturbation Norm"
    else:
        x_values = perturbation_results["avg_linf_norm"]
        x_label = "Average L∞ Perturbation Norm"

    success_rates = perturbation_results["success_rates"]

    # Create the figure
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, success_rates, "o-", linewidth=2, markersize=8)
    plt.xlabel(x_label)
    plt.ylabel("Attack Success Rate (%)")
    plt.title(f"{attack_name}: Success Rate vs. Perturbation Size")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Add AUC value if available
    if "perturbation_auc" in perturbation_results:
        auc_value = perturbation_results["perturbation_auc"]
        plt.text(
            0.05,
            0.05,
            f"AUC: {auc_value:.4f}",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
        )

    # Add epsilon values as secondary markers
    for i, (x, y, eps) in enumerate(
        zip(x_values, success_rates, perturbation_results["epsilon"])
    ):
        plt.annotate(
            f"ε={eps}", (x, y), textcoords="offset points", xytext=(5, 5), ha="left"
        )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_epsilon_vs_metrics(
    perturbation_results: Dict[str, List[Any]],
    attack_name: str = "Attack",
    save_path: Optional[str] = None,
):
    """
    Plot multiple metrics vs epsilon in a multi-panel figure.

    Args:
        perturbation_results: Results from evaluate_across_perturbation_budget
        attack_name: Name of the attack method
        save_path: Path to save the figure (optional)
    """
    set_plot_style()

    epsilon = perturbation_results["epsilon"]

    # Create a 2x2 panel of plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{attack_name}: Metrics vs. Perturbation Budget (ε)", fontsize=16)

    # Plot 1: Success Rate vs Epsilon
    axs[0, 0].plot(epsilon, perturbation_results["success_rates"], "o-", color="blue")
    axs[0, 0].set_xlabel("Perturbation Budget (ε)")
    axs[0, 0].set_ylabel("Success Rate (%)")
    axs[0, 0].set_title("Success Rate")
    axs[0, 0].grid(True, linestyle="--", alpha=0.7)

    # Plot 2: L2 Norm vs Epsilon
    axs[0, 1].plot(epsilon, perturbation_results["avg_l2_norm"], "o-", color="green")
    axs[0, 1].set_xlabel("Perturbation Budget (ε)")
    axs[0, 1].set_ylabel("Average L2 Norm")
    axs[0, 1].set_title("L2 Perturbation")
    axs[0, 1].grid(True, linestyle="--", alpha=0.7)

    # Plot 3: Linf Norm vs Epsilon
    axs[1, 0].plot(epsilon, perturbation_results["avg_linf_norm"], "o-", color="red")
    axs[1, 0].set_xlabel("Perturbation Budget (ε)")
    axs[1, 0].set_ylabel("Average L∞ Norm")
    axs[1, 0].set_title("L∞ Perturbation")
    axs[1, 0].grid(True, linestyle="--", alpha=0.7)

    # Plot 4: Attack Time vs Epsilon
    axs[1, 1].plot(epsilon, perturbation_results["attack_time"], "o-", color="purple")
    axs[1, 1].set_xlabel("Perturbation Budget (ε)")
    axs[1, 1].set_ylabel("Attack Time (seconds)")
    axs[1, 1].set_title("Computational Cost")
    axs[1, 1].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_transferability_heatmap(
    transfer_results: Dict[str, List[float]],
    attack_name: str = "Attack",
    save_path: Optional[str] = None,
):
    """
    Plot a heatmap of attack transferability across models.

    Args:
        transfer_results: Results from evaluate_transferability
        attack_name: Name of the attack method
        save_path: Path to save the figure (optional)
    """
    set_plot_style()

    model_names = transfer_results["model_names"]
    success_rates = transfer_results["success_rates"]

    # Create a dataframe for the heatmap
    # In transferability, we have only one row (the source model)
    data = np.array(success_rates).reshape(1, -1)
    df = pd.DataFrame(data, index=["Adversarial Examples"], columns=model_names)

    plt.figure(figsize=(12, 5))
    sns.heatmap(
        df,
        annot=True,
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "Success Rate (%)"},
    )

    plt.title(f"{attack_name}: Cross-Model Transferability")
    plt.ylabel("Source")
    plt.xlabel("Target Models")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")

    plt.show()


def compare_attacks(
    attack_results: Dict[str, Dict[str, List[Any]]],
    metric: str = "success_rates",
    x_axis: str = "epsilon",
    title: str = "Attack Comparison",
    xlabel: str = "Perturbation Budget (ε)",
    ylabel: str = "Success Rate (%)",
    save_path: Optional[str] = None,
):
    """
    Compare multiple attacks on the same plot.

    Args:
        attack_results: Dictionary of attack_name -> perturbation_results
        metric: Which metric to plot on y-axis
        x_axis: Which metric to plot on x-axis
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the figure (optional)
    """
    set_plot_style()

    plt.figure(figsize=(10, 6))

    # Color palette for different attacks
    colors = plt.cm.tab10(np.linspace(0, 1, len(attack_results)))

    for i, (attack_name, results) in enumerate(attack_results.items()):
        x_values = results[x_axis]
        y_values = results[metric]

        plt.plot(
            x_values,
            y_values,
            "o-",
            linewidth=2,
            markersize=8,
            label=attack_name,
            color=colors[i],
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_attack_efficiency(
    attack_results: Dict[str, Dict[str, List[Any]]], save_path: Optional[str] = None
):
    """
    Plot attack efficiency (success rate vs time) for multiple attacks.

    Args:
        attack_results: Dictionary of attack_name -> perturbation_results
        save_path: Path to save the figure (optional)
    """
    set_plot_style()

    plt.figure(figsize=(10, 6))

    # Color palette for different attacks
    colors = plt.cm.tab10(np.linspace(0, 1, len(attack_results)))

    for i, (attack_name, results) in enumerate(attack_results.items()):
        # Get average time per epsilon
        time_values = results["attack_time"]
        success_values = results["success_rates"]

        # Create scatter plot with size proportional to epsilon
        sizes = np.array(results["epsilon"]) * 50  # Scale for visibility

        plt.scatter(
            time_values,
            success_values,
            s=sizes,
            label=attack_name,
            color=colors[i],
            alpha=0.7,
        )

        # Connect points in epsilon order
        epsilon_order = np.argsort(results["epsilon"])
        plt.plot(
            np.array(time_values)[epsilon_order],
            np.array(success_values)[epsilon_order],
            "-",
            color=colors[i],
            alpha=0.5,
        )

    plt.xlabel("Attack Time (seconds)")
    plt.ylabel("Success Rate (%)")
    plt.title("Attack Efficiency: Success Rate vs. Computational Cost")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_confidence_distribution(
    original_conf: torch.Tensor,
    adversarial_conf: torch.Tensor,
    attack_name: str = "Attack",
    save_path: Optional[str] = None,
):
    """
    Plot the distribution of prediction confidences before and after attack.

    Args:
        original_conf: Confidence scores for original images
        adversarial_conf: Confidence scores for adversarial images
        attack_name: Name of the attack method
        save_path: Path to save the figure (optional)
    """
    set_plot_style()

    plt.figure(figsize=(10, 6))

    # Convert to numpy for plotting
    orig_conf = original_conf.cpu().numpy()
    adv_conf = adversarial_conf.cpu().numpy()

    # Plot histograms
    plt.hist(orig_conf, bins=20, alpha=0.5, label="Original Images", color="blue")
    plt.hist(adv_conf, bins=20, alpha=0.5, label="Adversarial Images", color="red")

    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.title(f"{attack_name}: Distribution of Prediction Confidence")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Add summary statistics
    plt.text(
        0.02,
        0.95,
        f"Original Mean: {orig_conf.mean():.3f}",
        transform=plt.gca().transAxes,
        color="blue",
    )
    plt.text(
        0.02,
        0.9,
        f"Adversarial Mean: {adv_conf.mean():.3f}",
        transform=plt.gca().transAxes,
        color="red",
    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")

    plt.show()


def save_metrics_to_csv(
    attack_results: Dict[str, Dict[str, List[Any]]], output_file: str
):
    """
    Save attack comparison metrics to a CSV file.

    Args:
        attack_results: Dictionary of attack_name -> perturbation_results
        output_file: Path to save the CSV file
    """
    # Prepare data for CSV
    data = []

    for attack_name, results in attack_results.items():
        for i, eps in enumerate(results["epsilon"]):
            row = {
                "Attack": attack_name,
                "Epsilon": eps,
                "Success_Rate": results["success_rates"][i],
                "Avg_L2_Norm": results["avg_l2_norm"][i],
                "Avg_Linf_Norm": results["avg_linf_norm"][i],
                "Attack_Time": results["attack_time"][i],
            }
            data.append(row)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Metrics saved to {output_file}")


def generate_latex_table(
    attack_results: Dict[str, Dict[str, List[Any]]], output_file: Optional[str] = None
) -> str:
    """
    Generate a LaTeX table for paper inclusion.

    Args:
        attack_results: Dictionary of attack_name -> perturbation_results
        output_file: Path to save the LaTeX code (optional)

    Returns:
        String containing LaTeX table code
    """
    # Get all unique epsilon values
    all_epsilons = set()
    for results in attack_results.values():
        all_epsilons.update(results["epsilon"])
    all_epsilons = sorted(list(all_epsilons))

    # Start LaTeX table
    latex_code = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Comparison of Attack Methods}",
        "\\label{tab:attack_comparison}",
        "\\begin{tabular}{l|c|ccc}",
        "\\hline",
        "Attack Method & $\\epsilon$ & Success Rate (\\%) & Avg. $L_2$ Norm & Time (s) \\\\",
        "\\hline",
    ]

    # Add data rows
    for attack_name, results in attack_results.items():
        first_row = True
        for i, eps in enumerate(results["epsilon"]):
            if eps in all_epsilons:  # Only include common epsilon values
                if first_row:
                    row = f"{attack_name} & {eps:.2f} & {results['success_rates'][i]:.1f} & {results['avg_l2_norm'][i]:.4f} & {results['attack_time'][i]:.2f} \\\\"
                    first_row = False
                else:
                    row = f"& {eps:.2f} & {results['success_rates'][i]:.1f} & {results['avg_l2_norm'][i]:.4f} & {results['attack_time'][i]:.2f} \\\\"
                latex_code.append(row)
        latex_code.append("\\hline")

    # End table
    latex_code.extend(["\\end{tabular}", "\\end{table}"])

    full_latex = "\n".join(latex_code)

    if output_file:
        with open(output_file, "w") as f:
            f.write(full_latex)
        print(f"LaTeX table saved to {output_file}")

    return full_latex


# Example helper function for running full metric analysis
def run_comprehensive_analysis(
    attack_methods: Dict[str, Callable],
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    output_dir: str = "results",
    epsilon_values: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
    norm: str = "L2",
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None,
    transfer_models: Optional[List[torch.nn.Module]] = None,
    transfer_model_names: Optional[List[str]] = None,
):
    """
    Run a comprehensive analysis of multiple attack methods.

    Args:
        attack_methods: Dictionary of attack_name -> attack_function
        model: Model to attack
        images: Clean images
        labels: True labels
        output_dir: Directory to save results
        epsilon_values: List of epsilon values to test
        norm: Norm to use for attacks
        targeted: Whether attacks are targeted
        target_labels: Target labels (for targeted attacks)
        transfer_models: List of models for transferability testing
        transfer_model_names: Names of transfer models
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to collect all results
    all_results = {}

    # Run each attack method
    for attack_name, attack_fn in attack_methods.items():
        print(f"\nRunning {attack_name}...")

        # Evaluate across perturbation budgets
        pert_results = evaluate_across_perturbation_budget(
            attack_fn,
            model,
            images,
            labels,
            target_labels,
            targeted,
            epsilon_values,
            norm,
        )
        all_results[attack_name] = pert_results

        # Plot individual results
        plot_success_rate_vs_perturbation(
            pert_results,
            attack_name,
            norm_type="L2",
            save_path=os.path.join(output_dir, f"{attack_name}_success_vs_l2.png"),
        )

        plot_epsilon_vs_metrics(
            pert_results,
            attack_name,
            save_path=os.path.join(output_dir, f"{attack_name}_metrics.png"),
        )

        # If transfer models are provided, evaluate transferability
        if transfer_models and len(transfer_models) > 0:
            # Get adversarial examples at middle epsilon value
            mid_idx = len(epsilon_values) // 2
            mid_eps = epsilon_values[mid_idx]
            print(f"Evaluating transferability with ε={mid_eps}...")

            adv_images = attack_fn(
                model,
                images,
                target_labels if targeted else labels,
                mid_eps,
                norm,
                targeted,
            )

            transfer_results = evaluate_transferability(
                model,
                transfer_models,
                images,
                adv_images,
                labels,
                target_labels,
                targeted,
                transfer_model_names,
            )

            plot_transferability_heatmap(
                transfer_results,
                attack_name,
                save_path=os.path.join(
                    output_dir, f"{attack_name}_transferability.png"
                ),
            )

    # Comparison plots
    compare_attacks(
        all_results,
        metric="success_rates",
        x_axis="avg_l2_norm",
        title="Attack Comparison: Success Rate vs. L2 Perturbation",
        xlabel="Average L2 Perturbation Norm",
        ylabel="Success Rate (%)",
        save_path=os.path.join(output_dir, "attack_comparison_l2.png"),
    )

    compare_attacks(
        all_results,
        metric="success_rates",
        x_axis="epsilon",
        title="Attack Comparison: Success Rate vs. Epsilon",
        xlabel="Perturbation Budget (ε)",
        ylabel="Success Rate (%)",
        save_path=os.path.join(output_dir, "attack_comparison_epsilon.png"),
    )

    plot_attack_efficiency(
        all_results, save_path=os.path.join(output_dir, "attack_efficiency.png")
    )

    # Save numerical results
    save_metrics_to_csv(all_results, os.path.join(output_dir, "attack_metrics.csv"))

    # Generate LaTeX table
    generate_latex_table(all_results, os.path.join(output_dir, "attack_table.tex"))

    return all_results
