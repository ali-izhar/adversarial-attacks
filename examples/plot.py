"""Adversarial attack visualization utility."""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better aesthetics
sns.set_theme(style="whitegrid")
PALETTE = "viridis"

# CIFAR-10 class names
CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def compare_norms(images, adv_images):
    """
    Compare different norms of the perturbation.

    Args:
        images: Original images
        adv_images: Adversarial images

    Returns:
        Dictionary with L0, L1, L2, and Linf norms
    """
    perturbation = adv_images - images
    batch_size = perturbation.shape[0]

    # Reshape to (batch_size, -1) for norm calculations
    perturbation_flat = perturbation.reshape(batch_size, -1)

    # Calculate L0 norm (number of changed pixels)
    l0_norm = (perturbation_flat.abs() > 1e-5).float().sum(dim=1)

    # Calculate L1 norm (sum of absolute values)
    l1_norm = perturbation_flat.abs().sum(dim=1)

    # Calculate L2 norm (Euclidean distance)
    l2_norm = torch.norm(perturbation_flat, dim=1, p=2)

    # Calculate Linf norm (maximum absolute value)
    linf_norm = perturbation_flat.abs().max(dim=1)[0]

    return {"L0": l0_norm, "L1": l1_norm, "L2": l2_norm, "Linf": linf_norm}


def compute_perturbation_visualization(images, adv_images, factor=5):
    """
    Compute enhanced perturbation visualization.

    Args:
        images: Original images
        adv_images: Adversarial images
        factor: Enhancement factor to make perturbations more visible

    Returns:
        The enhanced perturbation visualization
    """
    perturbation = adv_images - images

    # Enhance the perturbation to make it more visible
    enhanced_perturbation = perturbation * factor

    # Ensure the values are in a valid range
    enhanced_perturbation = torch.clamp(enhanced_perturbation + 0.5, 0, 1)

    return enhanced_perturbation


def visualize_results(
    images,
    adv_images,
    labels,
    targets,
    predictions,
    adv_predictions,
    attack_params,
    metrics,
    method,
    targeted,
    output_dir=None,
):
    """
    Create an enhanced side-by-side visualization of original and adversarial images.

    Args:
        images: Original images (tensor)
        adv_images: Adversarial images (tensor)
        labels: True labels
        targets: Target labels (for targeted attacks)
        predictions: Original predictions
        adv_predictions: Adversarial predictions
        attack_params: Dictionary of attack parameters
        metrics: Dictionary of attack metrics (success rate, iterations, etc.)
        method: Attack method name ("pgd", "cg", or "lbfgs")
        targeted: Whether the attack is targeted
        output_dir: Directory to save visualization (optional)
    """
    n = len(images)

    # Set up a nice color palette
    colors = sns.color_palette(PALETTE, n_colors=3)

    # Set figure aesthetics
    with plt.style.context("seaborn-v0_8-white"):
        # Create a figure with n rows and 2 columns
        fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))

        # Format the attack name for the title
        method_name = method.upper()
        if method == "cg":
            method_name = "Conjugate Gradient"
        elif method == "lbfgs":
            method_name = "L-BFGS"

        # Add a title for the entire figure with enhanced styling
        attack_type = "Targeted" if targeted else "Untargeted"
        title = f"{method_name} {attack_type} Attack: Original vs Adversarial Images"
        fig.suptitle(title, fontsize=16, fontweight="bold", color=colors[0])

        # Add a styled legend with attack parameters
        param_text = "Attack Parameters:\n"
        for key, value in attack_params.items():
            if key not in ["model", "verbose"]:
                param_text += f"• {key}: {value}\n"

        # Add metrics with styled text
        param_text += f"\nResults:\n• Success Rate: {metrics['success_rate']:.1f}%\n"
        param_text += f"• Iterations: {metrics['iterations']:.1f}\n"
        param_text += f"• Time: {metrics['time']:.2f}s"

        # Add styled parameter box
        props = dict(
            boxstyle="round", facecolor="white", alpha=0.7, edgecolor=colors[0]
        )
        fig.text(
            0.1,
            0.01,
            param_text,
            fontsize=10,
            verticalalignment="bottom",
            bbox=props,
            fontfamily="monospace",
        )

        # Function to denormalize images
        def denormalize(img):
            return img.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)

        # Plot each image with enhanced styling
        for i in range(n):
            # Handle different cases for single image vs multiple images
            if n > 1:
                ax1 = axes[i, 0]
                ax2 = axes[i, 1]
            else:
                ax1 = axes[0]
                ax2 = axes[1]

            # Original image
            ax1.imshow(denormalize(images[i]))
            ax1.set_title(
                f"Original: {CLASSES[labels[i]]}\nPredicted: {CLASSES[predictions[i]]}",
                fontweight="bold",
                color=colors[0],
            )
            ax1.axis("off")

            # Add a thin border around the original image
            for spine in ax1.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(colors[0])
                spine.set_linewidth(2)

            # Adversarial image
            ax2.imshow(denormalize(adv_images[i]))
            if targeted:
                ax2.set_title(
                    f"Adversarial (Target: {CLASSES[targets[i]]})\nPredicted: {CLASSES[adv_predictions[i]]}",
                    fontweight="bold",
                    color=colors[2],
                )
            else:
                ax2.set_title(
                    f"Adversarial\nPredicted: {CLASSES[adv_predictions[i]]}",
                    fontweight="bold",
                    color=colors[2],
                )
            ax2.axis("off")

            # Add a thin border around the adversarial image
            for spine in ax2.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(colors[2])
                spine.set_linewidth(2)

            # Add a visual indicator for attack success
            if targeted:
                success = adv_predictions[i] == targets[i]
            else:
                success = predictions[i] != adv_predictions[i]

            success_text = "Success" if success else "Failure"
            success_color = "green" if success else "red"

            # Add success indicator
            ax2.annotate(
                success_text,
                xy=(1, 1),
                xycoords="axes fraction",
                fontsize=12,
                fontweight="bold",
                color=success_color,
                xytext=(-10, -10),
                textcoords="offset points",
                ha="right",
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="white",
                    alpha=0.7,
                    edgecolor=success_color,
                ),
            )

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Make room for the text

        # Save the figure if output directory is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{method}_{attack_type.lower()}_visualization.png"
            plt.savefig(
                os.path.join(output_dir, filename), dpi=150, bbox_inches="tight"
            )

        plt.show()


def visualize_perturbations(
    images,
    adv_images,
    labels,
    targets,
    predictions,
    adv_predictions,
    method,
    targeted,
    enhancement_factor=5,
    output_dir=None,
    num_images=5,
):
    """
    Create an enhanced visualization showing original, perturbation, and adversarial images.

    Args:
        images: Original images
        adv_images: Adversarial images
        labels: True labels
        targets: Target labels (for targeted attacks)
        predictions: Original predictions
        adv_predictions: Adversarial predictions
        method: Attack method name
        targeted: Whether the attack is targeted
        enhancement_factor: Factor to enhance perturbation visibility
        output_dir: Directory to save visualization (optional)
        num_images: Number of images to display
    """
    # Create an enhanced visualization with perturbations
    perturbation_vis = compute_perturbation_visualization(
        images, adv_images, enhancement_factor
    )

    # Calculate raw perturbation magnitude (for heatmap)
    raw_perturbation = (adv_images - images).abs()

    # Set up custom color maps
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    # Set up a nice color palette
    colors = sns.color_palette(PALETTE, n_colors=4)

    with plt.style.context("seaborn-v0_8-white"):
        # Create a figure showing original, perturbation magnitude, enhanced perturbation, and adversarial
        fig, axes = plt.subplots(num_images, 4, figsize=(16, 3 * num_images))

        # Format the attack name for the title
        method_name = method.upper()
        if method == "cg":
            method_name = "Conjugate Gradient"
        elif method == "lbfgs":
            method_name = "L-BFGS"

        attack_type = "Targeted" if targeted else "Untargeted"
        title = f"{method_name} {attack_type} Attack Analysis"
        fig.suptitle(title, fontsize=18, fontweight="bold", color=colors[0], y=0.98)

        # Add column titles at the top
        cols = [
            "Original Image",
            "Perturbation Magnitude",
            f"Enhanced Perturbation ({enhancement_factor}x)",
            "Adversarial Image",
        ]

        for ax, col in zip(axes[0], cols):
            idx = cols.index(col)
            ax.set_title(
                col, fontsize=14, fontweight="bold", color=colors[idx % len(colors)]
            )

        for i in range(num_images):
            # Original image
            axes[i, 0].imshow(images[i].permute(1, 2, 0).detach().cpu().numpy())
            axes[i, 0].set_title(
                f"Original: {CLASSES[labels[i]]}", fontsize=12, color=colors[0]
            )
            axes[i, 0].axis("off")
            for spine in axes[i, 0].spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(colors[0])
                spine.set_linewidth(1.5)

            # Perturbation magnitude (heatmap)
            perturbation_magnitude = (
                raw_perturbation[i].mean(dim=0).detach().cpu().numpy()
            )
            axes[i, 1].imshow(perturbation_magnitude, cmap=cmap)
            axes[i, 1].set_title(
                f"Max change: {raw_perturbation[i].max().item():.4f}",
                fontsize=12,
                color=colors[1],
            )
            axes[i, 1].axis("off")
            for spine in axes[i, 1].spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(colors[1])
                spine.set_linewidth(1.5)

            # Enhanced perturbation visualization
            axes[i, 2].imshow(
                perturbation_vis[i].permute(1, 2, 0).detach().cpu().numpy()
            )
            axes[i, 2].set_title(
                f"Enhanced for visibility", fontsize=12, color=colors[2]
            )
            axes[i, 2].axis("off")
            for spine in axes[i, 2].spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(colors[2])
                spine.set_linewidth(1.5)

            # Adversarial image
            axes[i, 3].imshow(adv_images[i].permute(1, 2, 0).detach().cpu().numpy())
            if targeted:
                axes[i, 3].set_title(
                    f"Target: {CLASSES[targets[i]]}", fontsize=12, color=colors[3]
                )
            else:
                axes[i, 3].set_title(
                    f"Classified as: {CLASSES[adv_predictions[i]]}",
                    fontsize=12,
                    color=colors[3],
                )
            axes[i, 3].axis("off")
            for spine in axes[i, 3].spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(colors[3])
                spine.set_linewidth(1.5)

            # Add a visual indicator for attack success
            if targeted:
                success = adv_predictions[i] == targets[i]
            else:
                success = predictions[i] != adv_predictions[i]

            success_text = "Success" if success else "Failure"
            success_color = "green" if success else "red"

            # Add success indicator to the last image
            axes[i, 3].annotate(
                success_text,
                xy=(1, 1),
                xycoords="axes fraction",
                fontsize=12,
                fontweight="bold",
                color=success_color,
                xytext=(-10, -10),
                textcoords="offset points",
                ha="right",
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="white",
                    alpha=0.7,
                    edgecolor=success_color,
                ),
            )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the figure if output directory is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{method}_{attack_type.lower()}_analysis.png"
            plt.savefig(
                os.path.join(output_dir, filename), dpi=150, bbox_inches="tight"
            )

        plt.show()


def visualize_convergence(metrics, method, targeted, output_dir=None):
    """
    Create an enhanced visualization of optimization convergence.

    Args:
        metrics: Dictionary containing optimization metrics including loss_trajectory
        method: Attack method name
        targeted: Whether the attack is targeted
        output_dir: Directory to save visualization (optional)
    """
    if "loss_trajectory" not in metrics or not metrics["loss_trajectory"]:
        print("Loss trajectory not available in metrics")
        return

    loss_values = metrics["loss_trajectory"]
    iterations = range(1, len(loss_values) + 1)

    # Set up color palette
    colors = sns.color_palette(PALETTE, n_colors=3)

    with sns.axes_style("whitegrid"):
        plt.figure(figsize=(10, 6))

        # Create enhanced line plot
        sns.lineplot(x=iterations, y=loss_values, linewidth=3, color=colors[0])

        # Add markers at key points
        plt.scatter(
            iterations[::5],
            [loss_values[i] for i in range(0, len(loss_values), 5)],
            s=80,
            color=colors[1],
            zorder=3,
            alpha=0.8,
        )

        # Format the attack name for the title
        method_name = method.upper()
        if method == "cg":
            method_name = "Conjugate Gradient"
        elif method == "lbfgs":
            method_name = "L-BFGS"

        attack_type = "Targeted" if targeted else "Untargeted"

        # Enhanced title and labels
        plt.title(
            f"{method_name} {attack_type} Attack Optimization Convergence",
            fontsize=16,
            fontweight="bold",
            color=colors[0],
        )
        plt.xlabel("Iteration", fontsize=14, fontweight="bold")
        plt.ylabel("Loss Value", fontsize=14, fontweight="bold")

        # Use log scale with better formatting
        plt.yscale("log")

        # Add grid
        plt.grid(True, linestyle="--", alpha=0.7)

        # Add annotations for initial and final loss
        plt.annotate(
            f"Initial: {loss_values[0]:.4f}",
            xy=(1, loss_values[0]),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=12,
            arrowprops=dict(arrowstyle="->", color=colors[2]),
        )

        plt.annotate(
            f"Final: {loss_values[-1]:.4f}",
            xy=(len(loss_values), loss_values[-1]),
            xytext=(-80, -20),
            textcoords="offset points",
            fontsize=12,
            arrowprops=dict(arrowstyle="->", color=colors[2]),
        )

        # Add text showing convergence rate
        if len(loss_values) > 1:
            reduction_rate = (loss_values[0] - loss_values[-1]) / loss_values[0]
            reduction_text = f"Loss reduced by {reduction_rate*100:.1f}%"
            plt.text(
                0.5,
                0.05,
                reduction_text,
                transform=plt.gca().transAxes,
                fontsize=14,
                ha="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
            )

        plt.tight_layout()

        # Save the figure if output directory is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{method}_{attack_type.lower()}_convergence.png"
            plt.savefig(
                os.path.join(output_dir, filename), dpi=150, bbox_inches="tight"
            )

        plt.show()


def visualize_norm_comparison(norms, method, targeted, output_dir=None):
    """
    Create a visualization comparing different perturbation norms.

    Args:
        norms: Dictionary with L0, L1, L2, and Linf norms
        method: Attack method name
        targeted: Whether the attack is targeted
        output_dir: Directory to save visualization (optional)
    """
    # Convert tensor values to numpy arrays
    norm_values = {
        "L0 (pixels changed)": norms["L0"].detach().cpu().numpy(),
        "L1 (total change)": norms["L1"].detach().cpu().numpy(),
        "L2 (Euclidean)": norms["L2"].detach().cpu().numpy(),
        "Linf (maximum)": norms["Linf"].detach().cpu().numpy(),
    }

    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # Format the attack name for the title
        method_name = method.upper()
        if method == "cg":
            method_name = "Conjugate Gradient"
        elif method == "lbfgs":
            method_name = "L-BFGS"

        attack_type = "Targeted" if targeted else "Untargeted"
        fig.suptitle(
            f"{method_name} {attack_type} Attack: Perturbation Norm Analysis",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        # Plot each norm
        for i, (norm_name, values) in enumerate(norm_values.items()):
            # Create the distribution plot
            sns.histplot(values, kde=True, ax=axes[i], palette=PALETTE)

            # Enhance the plot
            axes[i].set_title(norm_name, fontsize=14, fontweight="bold")
            axes[i].set_xlabel("Value", fontsize=12)
            axes[i].set_ylabel("Frequency", fontsize=12)

            # Add mean and max annotations
            mean_val = np.mean(values)
            max_val = np.max(values)

            axes[i].axvline(mean_val, color="red", linestyle="--", linewidth=2)
            axes[i].axvline(max_val, color="green", linestyle="--", linewidth=2)

            # Add text annotations
            axes[i].text(
                0.05,
                0.95,
                f"Mean: {mean_val:.4f}",
                transform=axes[i].transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

            axes[i].text(
                0.05,
                0.85,
                f"Max: {max_val:.4f}",
                transform=axes[i].transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the figure if output directory is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{method}_{attack_type.lower()}_norm_analysis.png"
            plt.savefig(
                os.path.join(output_dir, filename), dpi=150, bbox_inches="tight"
            )

        plt.show()
