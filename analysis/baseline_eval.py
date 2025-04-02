"""
This script evaluates baseline adversarial attacks on pretrained models
and collects metrics for the paper tables, including success rates,
perturbation metrics (L2, L-inf, SSIM), and computational metrics
(iterations, gradient calls, runtime).
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class AttackEvaluator:
    """Framework for evaluating adversarial attacks across multiple models and metrics."""

    def __init__(self, models, dataset, device=None):
        """
        Args:
            models (dict): Dictionary mapping model names to model instances
            dataset: Dataset containing clean images and labels
            device: Device to run evaluation on (default: auto-detect)
        """
        self.models = models
        self.dataset = dataset
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.results = {}

    def evaluate_attack(self, attack_name, attack_fn, targeted=False, **attack_kwargs):
        """Evaluate an attack across all models and metrics.

        Args:
            attack_name (str): Name of the attack
            attack_fn (callable): Function that returns an attack instance
            targeted (bool): Whether to run as targeted attack
            **attack_kwargs: Additional arguments for the attack

        Returns:
            dict: Evaluation results
        """
        # Initialize results structure for this attack
        attack_type = "targeted" if targeted else "untargeted"
        if attack_name not in self.results:
            self.results[attack_name] = {}

        self.results[attack_name][attack_type] = {
            model_name: {} for model_name in self.models
        }

        # Evaluate for each model
        for model_name, model in self.models.items():
            model.to(self.device)
            model.eval()

            # Create dataloader
            dataloader = torch.utils.data.DataLoader(
                self.dataset, batch_size=16, shuffle=False
            )

            # Create attack instance
            attack = attack_fn(model, **attack_kwargs)
            print(f"\nEvaluating {attack_name} ({attack_type}) on {model_name}")

            # Setup targeted mode if needed
            if targeted:
                attack.set_mode_targeted_least_likely()
            else:
                attack.set_mode_default()

            # Run the evaluation using our new method in the Attack base class
            attack_results = attack.evaluate(dataloader, verbose=True)

            # Store results - we don't store the attack instance itself
            self.results[attack_name][attack_type][model_name] = attack_results

        return self.results[attack_name]

    def export_results_to_tables(self, output_dir):
        """Export results in a format suitable for the paper tables."""
        os.makedirs(output_dir, exist_ok=True)

        # Table 2: Attack Effectiveness (Success Rates)
        effectiveness_data = []

        for attack_name in self.results:
            # Determine category (baseline or optimization)
            category = (
                "Baseline"
                if attack_name in ["FGSM", "FFGSM", "DeepFool", "CW"]
                else "Optimization"
            )

            row = {"Category": category, "Method": attack_name}

            # Add untargeted success rates
            if "untargeted" in self.results[attack_name]:
                for model_name in self.models:
                    model_results = self.results[attack_name]["untargeted"][model_name]
                    row[f"{model_name} (Untargeted)"] = (
                        f"{model_results['success_rate']:.1f}"
                    )
            else:
                for model_name in self.models:
                    row[f"{model_name} (Untargeted)"] = "N/A"

            # Add targeted success rates
            if "targeted" in self.results[attack_name]:
                for model_name in self.models:
                    model_results = self.results[attack_name]["targeted"][model_name]
                    row[f"{model_name} (Targeted)"] = (
                        f"{model_results['success_rate']:.1f}"
                    )
            else:
                for model_name in self.models:
                    row[f"{model_name} (Targeted)"] = "N/A"

            effectiveness_data.append(row)

        # Convert to DataFrame and save
        effectiveness_df = pd.DataFrame(effectiveness_data)
        effectiveness_df.to_csv(
            os.path.join(output_dir, "table2_attack_effectiveness.csv"), index=False
        )

        # Table 3: Perturbation Efficiency
        perturbation_data = []

        for attack_name in self.results:
            # Determine category
            category = (
                "Baseline"
                if attack_name in ["FGSM", "FFGSM", "DeepFool", "CW"]
                else "Optimization"
            )

            row = {"Category": category, "Method": attack_name}

            # Average across all models for each metric (L2, L∞, SSIM)
            for attack_type in ["untargeted", "targeted"]:
                if attack_type in self.results[attack_name]:
                    l2_values = []
                    linf_values = []
                    ssim_values = []

                    for model_name in self.models:
                        if model_name in self.results[attack_name][attack_type]:
                            model_results = self.results[attack_name][attack_type][
                                model_name
                            ]
                            l2_values.append(model_results["l2_norm"])
                            linf_values.append(model_results["linf_norm"])
                            ssim_values.append(model_results["ssim"])

                    # Calculate averages
                    if l2_values:
                        row[f"L2 ({attack_type})"] = f"{np.mean(l2_values):.4f}"
                        row[f"L∞ ({attack_type})"] = f"{np.mean(linf_values):.4f}"
                        row[f"SSIM ({attack_type})"] = f"{np.mean(ssim_values):.4f}"
                    else:
                        row[f"L2 ({attack_type})"] = "N/A"
                        row[f"L∞ ({attack_type})"] = "N/A"
                        row[f"SSIM ({attack_type})"] = "N/A"
                else:
                    row[f"L2 ({attack_type})"] = "N/A"
                    row[f"L∞ ({attack_type})"] = "N/A"
                    row[f"SSIM ({attack_type})"] = "N/A"

            perturbation_data.append(row)

        # Convert to DataFrame and save
        perturbation_df = pd.DataFrame(perturbation_data)
        perturbation_df.to_csv(
            os.path.join(output_dir, "table3_perturbation_efficiency.csv"), index=False
        )

        # Table 4: Computational Efficiency
        computational_data = []

        for attack_name in self.results:
            row = {"Attack Method": attack_name}

            # Process untargeted metrics
            if "untargeted" in self.results[attack_name]:
                iterations = []
                gradient_calls = []
                runtime = []

                for model_name in self.models:
                    if model_name in self.results[attack_name]["untargeted"]:
                        model_results = self.results[attack_name]["untargeted"][
                            model_name
                        ]
                        iterations.append(model_results["iterations"])
                        gradient_calls.append(model_results["gradient_calls"])
                        runtime.append(model_results["time_per_sample"])

                if iterations:
                    row["Iterations (Untargeted)"] = f"{np.mean(iterations):.1f}"
                    row["Gradient Calls (Untargeted)"] = (
                        f"{np.mean(gradient_calls):.1f}"
                    )
                    row["Runtime (s) (Untargeted)"] = f"{np.mean(runtime):.4f}"
                else:
                    row["Iterations (Untargeted)"] = "N/A"
                    row["Gradient Calls (Untargeted)"] = "N/A"
                    row["Runtime (s) (Untargeted)"] = "N/A"
            else:
                row["Iterations (Untargeted)"] = "N/A"
                row["Gradient Calls (Untargeted)"] = "N/A"
                row["Runtime (s) (Untargeted)"] = "N/A"

            # Process targeted metrics
            if "targeted" in self.results[attack_name]:
                iterations = []
                gradient_calls = []
                runtime = []

                for model_name in self.models:
                    if model_name in self.results[attack_name]["targeted"]:
                        model_results = self.results[attack_name]["targeted"][
                            model_name
                        ]
                        iterations.append(model_results["iterations"])
                        gradient_calls.append(model_results["gradient_calls"])
                        runtime.append(model_results["time_per_sample"])

                if iterations:
                    row["Iterations (Targeted)"] = f"{np.mean(iterations):.1f}"
                    row["Gradient Calls (Targeted)"] = f"{np.mean(gradient_calls):.1f}"
                    row["Runtime (s) (Targeted)"] = f"{np.mean(runtime):.4f}"
                else:
                    row["Iterations (Targeted)"] = "N/A"
                    row["Gradient Calls (Targeted)"] = "N/A"
                    row["Runtime (s) (Targeted)"] = "N/A"
            else:
                row["Iterations (Targeted)"] = "N/A"
                row["Gradient Calls (Targeted)"] = "N/A"
                row["Runtime (s) (Targeted)"] = "N/A"

            computational_data.append(row)

        # Convert to DataFrame and save
        computational_df = pd.DataFrame(computational_data)
        computational_df.to_csv(
            os.path.join(output_dir, "table4_computational_efficiency.csv"), index=False
        )

        print(f"Results exported to {output_dir}")

    def visualize_perturbations(
        self, attack_name, model_name, num_samples=5, output_dir=None
    ):
        """Visualize original images, perturbations, and adversarial examples for a specific attack and model."""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Create dataloader with a small batch size
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=num_samples, shuffle=True
        )

        # Get a batch of images and labels
        images, labels = next(iter(dataloader))
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Ensure the model is on the correct device
        model = self.models[model_name].to(self.device)
        model.eval()

        # Access the base model directly to avoid double normalization
        if hasattr(model, "model"):
            # This is our DirectModelAttackWrapper
            base_model = model.model
        elif hasattr(model, "_model"):
            # This is a regular model wrapper
            base_model = model._model
        else:
            # This is already a base model
            base_model = model

        # Print some info about the labels being used
        print(f"Visualizing attack: {attack_name} on {model_name}")
        print(f"Using labels: {labels.tolist()}")

        # Get original model predictions for these images
        with torch.no_grad():
            outputs = base_model(images)  # Use base model directly
            _, init_preds = torch.max(outputs, 1)
            # Count how many are already correctly classified
            init_correct = (init_preds == labels).sum().item()
            print(f"Initial model accuracy: {100 * init_correct / len(labels):.2f}%")
            print(f"Initial predictions: {init_preds.tolist()}")

        # Import attacks dynamically based on attack name
        if attack_name.startswith("FGSM-"):
            from src.attacks.baseline.attack_fgsm import FGSM

            eps = float(attack_name.split("-")[1])
            attack_fn = lambda model: FGSM(model, eps=eps)
        elif attack_name.startswith("FFGSM-"):
            from src.attacks.baseline.attack_ffgsm import FFGSM

            eps = float(attack_name.split("-")[1])
            attack_fn = lambda model: FFGSM(model, eps=eps, alpha=0.02)
        elif attack_name == "DeepFool":
            from src.attacks.baseline.attack_deepfool import DeepFool

            attack_fn = lambda model: DeepFool(model, steps=100, overshoot=0.05)
        elif attack_name.startswith("CW-"):
            from src.attacks.baseline.attack_cw import CW

            kappa = float(attack_name.split("kappa")[1])
            attack_fn = lambda model: CW(model, c=1.0, kappa=kappa, steps=100, lr=0.01)
        else:
            print(f"Attack {attack_name} not recognized")
            return

        # For both untargeted and targeted attacks
        for attack_type in ["untargeted", "targeted"]:
            if attack_type == "targeted" and attack_name == "DeepFool":
                print(f"DeepFool doesn't support targeted attacks, skipping")
                continue

            print(f"Generating {attack_type} adversarial examples...")

            # Create a new attack instance with the same parameters
            attack_instance = attack_fn(model)

            # Set attack mode
            if attack_type == "targeted":
                attack_instance.set_mode_targeted_least_likely()
            else:
                attack_instance.set_mode_default()

            # Generate adversarial examples
            try:
                adv_images = attack_instance(images, labels)

                # Compute perturbations
                perturbations = adv_images - images

                # Convert to numpy for visualization
                images_np = (
                    images.cpu().detach().numpy().transpose(0, 2, 3, 1)
                )  # NCHW -> NHWC
                adv_images_np = adv_images.cpu().detach().numpy().transpose(0, 2, 3, 1)
                perturbations_np = (
                    perturbations.cpu().detach().numpy().transpose(0, 2, 3, 1)
                )

                # Normalize perturbations for better visualization
                perturbations_np = (perturbations_np - perturbations_np.min()) / (
                    perturbations_np.max() - perturbations_np.min() + 1e-8
                )

                # Get predictions for clean and adversarial images
                with torch.no_grad():
                    clean_outputs = base_model(images)  # Use base model directly
                    clean_preds = clean_outputs.argmax(dim=1).cpu().numpy()

                    adv_outputs = base_model(adv_images)  # Use base model directly
                    adv_preds = adv_outputs.argmax(dim=1).cpu().numpy()

                # Plot images
                fig, axs = plt.subplots(num_samples, 3, figsize=(15, 3 * num_samples))
                if num_samples == 1:
                    axs = axs.reshape(1, -1)  # Handle case with single sample

                success_count = 0
                for i in range(num_samples):
                    # Original image
                    clean_pred = clean_preds[i]
                    true_label = labels[i].item()

                    # Original image
                    img_title = f"Original: {clean_pred}"
                    if true_label != clean_pred:
                        img_title += f" (True: {true_label})"
                    axs[i, 0].imshow(images_np[i])
                    axs[i, 0].set_title(img_title)
                    axs[i, 0].axis("off")

                    # Perturbation
                    axs[i, 1].imshow(perturbations_np[i])
                    axs[i, 1].set_title("Perturbation")
                    axs[i, 1].axis("off")

                    # Adversarial image
                    adv_pred = adv_preds[i]
                    success = (
                        clean_pred != adv_pred
                        if attack_type == "untargeted"
                        else adv_pred
                        == attack_instance._target_map_function(
                            images[i : i + 1], labels[i : i + 1]
                        ).item()
                    )
                    success_count += int(success)

                    adv_title = f"Adversarial: {adv_pred}"
                    if success:
                        adv_title += " ✓"
                    else:
                        adv_title += " ✗"
                    axs[i, 2].imshow(adv_images_np[i])
                    axs[i, 2].set_title(adv_title)
                    axs[i, 2].axis("off")

                # Add success rate to title
                success_rate = 100 * success_count / num_samples
                plt.tight_layout()
                fig.suptitle(
                    f"{attack_name} ({attack_type}) on {model_name} - Success Rate: {success_rate:.1f}%",
                    fontsize=16,
                )
                plt.subplots_adjust(top=0.95)

                if output_dir:
                    save_path = os.path.join(
                        output_dir, f"{attack_name}_{attack_type}_{model_name}.png"
                    )
                    plt.savefig(save_path)
                    print(f"Saved visualization to {save_path}")
                else:
                    plt.show()

                plt.close()

            except Exception as e:
                print(f"Error generating {attack_type} examples: {e}")
                continue
