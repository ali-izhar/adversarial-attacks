"""
This script evaluates baseline adversarial attacks on pretrained models
and collects metrics for the paper tables, including success rates,
perturbation metrics (L2, L-inf, SSIM), and computational metrics
(iterations, gradient calls, runtime).
"""

import os
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class AttackEvaluator:
    """Framework for evaluating adversarial attacks across multiple models and metrics."""

    def __init__(
        self,
        models,
        dataset,
        device=None,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
    ):
        """
        Args:
            models (dict): Dictionary mapping model names to model instances
            dataset: Dataset containing clean images and labels
            device: Device to run evaluation on (default: auto-detect)
            batch_size: Batch size for evaluation (default: 32, will be auto-adjusted for GPU)
            num_workers: Number of data loading workers (default: 8)
            pin_memory: Whether to use pinned memory for faster data transfer to GPU
        """
        self.models = models
        self.dataset = dataset
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize batch size based on GPU model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Auto-tune batch size for high-end GPUs
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (
                1024**3
            )  # in GB

            # RTX 4090 has ~24GB of VRAM, optimize for large batches
            if "4090" in gpu_name or "3090" in gpu_name or gpu_mem > 20:
                self.batch_size = min(128, len(dataset))
                self.num_workers = min(16, os.cpu_count() or 8)
                print(f"Optimizing for high-end GPU ({gpu_name}, {gpu_mem:.1f}GB VRAM)")
                print(f"Using batch_size={self.batch_size}, workers={self.num_workers}")
            # RTX 3080/3070 or similar has ~10GB of VRAM
            elif "3080" in gpu_name or "3070" in gpu_name or gpu_mem > 8:
                self.batch_size = min(64, len(dataset))
                self.num_workers = min(12, os.cpu_count() or 8)

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

            # Create dataloader with optimized settings for GPU
            dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
                prefetch_factor=2 if self.num_workers > 0 else None,
            )

            # Create attack instance
            attack = attack_fn(model, **attack_kwargs)
            print(f"\nEvaluating {attack_name} ({attack_type}) on {model_name}")

            # Reset metrics before starting the evaluation
            attack.reset_metrics()

            # Setup targeted mode if needed
            if targeted:
                attack.set_mode_targeted_least_likely()
            else:
                attack.set_mode_default()

            # Process each batch
            total_samples = 0
            successful_examples = []
            total_attacked = 0
            total_success = 0

            # Initialize aggregate metrics
            total_l2 = 0.0
            total_linf = 0.0
            total_ssim = 0.0
            total_time = 0.0
            total_iterations = 0.0
            total_grad_calls = 0.0

            # Use CUDA profiler to optimize performance
            with torch.autograd.profiler.emit_nvtx():
                for batch_idx, (inputs, labels) in enumerate(
                    tqdm(dataloader, desc=f"{attack_name}")
                ):
                    # Ensure inputs and labels are the right type and on the right device
                    # Use non-blocking transfers for better CUDA performance
                    inputs = inputs.to(self.device, non_blocking=True).float()
                    labels = labels.to(self.device, non_blocking=True)

                    total_samples += inputs.size(0)

                    # Pre-emptively clear GPU cache if running large batch sizes
                    if batch_idx % 10 == 0 and self.batch_size >= 64:
                        torch.cuda.empty_cache()

                    # Get original predictions
                    with torch.no_grad():
                        outputs = model(inputs)
                        _, original_preds = torch.max(outputs, 1)

                        # Skip samples that are already misclassified
                        correct_mask = original_preds == labels
                        if not correct_mask.any():
                            print(
                                f"  All samples in this batch are already misclassified, skipping"
                            )
                            continue

                        # Only attack correctly classified samples
                        correct_inputs = inputs[correct_mask]
                        correct_labels = labels[correct_mask]

                        if len(correct_inputs) == 0:
                            continue

                    # Attack only the correctly classified samples
                    total_attacked += len(correct_inputs)
                    attack_start = time.time()

                    # Use CUDA events for more accurate timing
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    start_event.record()
                    adversarial = attack(correct_inputs, correct_labels)
                    end_event.record()

                    # Synchronize CUDA to wait for attack to complete
                    torch.cuda.synchronize()
                    attack_time_ms = start_event.elapsed_time(end_event)
                    total_time += (
                        attack_time_ms / 1000.0 * len(correct_inputs)
                    )  # convert to seconds per sample

                    # Explicitly evaluate attack success and update metrics
                    batch_success_rate, success_mask, (orig_preds, adv_preds) = (
                        attack.evaluate_attack_success(
                            correct_inputs, adversarial, correct_labels
                        )
                    )

                    batch_success_count = success_mask.sum().item()
                    total_success += batch_success_count

                    # Explicitly compute perturbation metrics
                    perturbation_metrics = attack.compute_perturbation_metrics(
                        correct_inputs, adversarial
                    )

                    # Update aggregate metrics
                    if batch_success_count > 0:
                        total_l2 += (
                            perturbation_metrics["l2_norm"] * batch_success_count
                        )
                        total_linf += (
                            perturbation_metrics["linf_norm"] * batch_success_count
                        )
                        total_ssim += perturbation_metrics["ssim"] * batch_success_count

                    # Get iteration and gradient call counts from attack
                    if hasattr(attack, "iterations"):
                        total_iterations += attack.iterations * len(correct_inputs)
                    if hasattr(attack, "gradient_calls"):
                        total_grad_calls += attack.gradient_calls * len(correct_inputs)

                    # Display batch results
                    print(
                        f"  Batch {batch_idx+1}: Success Rate = {batch_success_rate:.2f}%, "
                        f"L2: {perturbation_metrics['l2_norm']:.4f}, "
                        f"L∞: {perturbation_metrics['linf_norm']:.4f}, "
                        f"SSIM: {perturbation_metrics['ssim']:.4f}, "
                        f"Time: {attack_time_ms:.1f}ms"
                    )

                    # Store successful examples for visualization
                    if success_mask.any():
                        for i in range(min(3, success_mask.sum().item())):
                            success_idx = torch.where(success_mask)[0][i].item()

                            successful_examples.append(
                                {
                                    "original": correct_inputs[success_idx].cpu(),
                                    "adversarial": adversarial[success_idx].cpu(),
                                    "original_label": correct_labels[
                                        success_idx
                                    ].item(),
                                    "adversarial_label": adv_preds[success_idx].item(),
                                }
                            )

                            if len(successful_examples) >= 3:  # Limit to 3 examples
                                break

            # Calculate final metrics
            # Use attack's metrics if available, otherwise use our own calculations
            attack_metrics = attack.get_metrics()

            # If no successful attacks, set metrics to reasonable defaults
            if total_attacked == 0:
                print("  Warning: No samples were attacked!")
                success_rate = 0.0
                l2_norm = 0.0
                linf_norm = 0.0
                ssim = 0.0
                time_per_sample = 0.0
                iterations = 0.0
                gradient_calls = 0.0
            else:
                # Calculate success rate as percentage
                success_rate = (total_success / total_attacked) * 100.0

                # Average perturbation metrics across successful examples only
                if total_success > 0:
                    l2_norm = total_l2 / total_success
                    linf_norm = total_linf / total_success
                    ssim = total_ssim / total_success
                else:
                    l2_norm = 0.0
                    linf_norm = 0.0
                    ssim = 0.0

                # Calculate per-sample time and complexity metrics
                time_per_sample = total_time / total_attacked
                iterations = total_iterations / total_attacked
                gradient_calls = total_grad_calls / total_attacked

            # Replace attack metrics with our calculations
            final_metrics = {
                "success_rate": success_rate,
                "l2_norm": l2_norm,
                "linf_norm": linf_norm,
                "ssim": ssim,
                "time_per_sample": time_per_sample,
                "iterations": iterations,
                "gradient_calls": gradient_calls,
                "total_samples": total_attacked,
                "successful_samples": total_success,
            }

            # Store the metrics and examples
            self.results[attack_name][attack_type][model_name] = final_metrics
            self.results[attack_name][attack_type][model_name][
                "examples"
            ] = successful_examples

            # Print summary metrics
            print(
                f"\nSummary metrics for {attack_name} ({attack_type}) on {model_name}:"
            )
            print(f"  Success Rate: {final_metrics['success_rate']:.2f}%")
            print(f"  L2 Norm: {final_metrics['l2_norm']:.4f}")
            print(f"  L∞ Norm: {final_metrics['linf_norm']:.4f}")
            print(f"  SSIM: {final_metrics['ssim']:.4f}")
            print(f"  Time per sample: {final_metrics['time_per_sample']*1000:.2f} ms")
            print(f"  Average iterations: {final_metrics['iterations']:.2f}")
            print(f"  Average gradient calls: {final_metrics['gradient_calls']:.2f}")

            # Explicitly free memory after each model evaluation
            torch.cuda.empty_cache()

        # Debug: print the results structure
        print(f"\nResults for {attack_name}:")
        for attack_type in self.results[attack_name]:
            print(f"  {attack_type}:")
            for model_name in self.results[attack_name][attack_type]:
                metrics = self.results[attack_name][attack_type][model_name]
                if isinstance(metrics, dict):
                    print(
                        f"    {model_name}: Success Rate={metrics.get('success_rate', 'N/A')}%"
                    )
                else:
                    print(f"    {model_name}: {metrics}")

        return self.results[attack_name]

    def export_results_to_tables(self, output_dir):
        """Export results in a format suitable for the paper tables."""
        os.makedirs(output_dir, exist_ok=True)

        if not self.results:
            print("Warning: No results to export!")
            # Create empty placeholder files
            pd.DataFrame().to_csv(
                os.path.join(output_dir, "table2_attack_effectiveness.csv")
            )
            pd.DataFrame().to_csv(
                os.path.join(output_dir, "table3_perturbation_efficiency.csv")
            )
            pd.DataFrame().to_csv(
                os.path.join(output_dir, "table4_computational_efficiency.csv")
            )
            return

        # Table 2: Attack Effectiveness (Success Rates)
        effectiveness_data = []

        for attack_name in self.results:
            # Determine category (baseline or optimization)
            category = (
                "Baseline"
                if attack_name.startswith(("FGSM", "FFGSM", "DeepFool", "CW"))
                else "Optimization"
            )

            row = {"Category": category, "Method": attack_name}

            # Add untargeted success rates
            if "untargeted" in self.results[attack_name]:
                for model_name in self.models:
                    if model_name in self.results[attack_name]["untargeted"]:
                        model_results = self.results[attack_name]["untargeted"][
                            model_name
                        ]
                        if (
                            isinstance(model_results, dict)
                            and "success_rate" in model_results
                        ):
                            row[f"{model_name}"] = (
                                f"{model_results['success_rate']:.1f}"
                            )
                        else:
                            row[f"{model_name}"] = "N/A"
                    else:
                        row[f"{model_name}"] = "N/A"
            else:
                for model_name in self.models:
                    row[f"{model_name}"] = "N/A"

            effectiveness_data.append(row)

        # Add another table for targeted attacks if available
        targeted_data = []

        for attack_name in self.results:
            if "targeted" not in self.results[attack_name]:
                continue

            category = (
                "Baseline"
                if attack_name.startswith(("FGSM", "FFGSM", "DeepFool", "CW"))
                else "Optimization"
            )

            row = {"Category": category, "Method": attack_name}

            for model_name in self.models:
                if model_name in self.results[attack_name]["targeted"]:
                    model_results = self.results[attack_name]["targeted"][model_name]
                    if (
                        isinstance(model_results, dict)
                        and "success_rate" in model_results
                    ):
                        row[f"{model_name}"] = f"{model_results['success_rate']:.1f}"
                    else:
                        row[f"{model_name}"] = "N/A"
                else:
                    row[f"{model_name}"] = "N/A"

            targeted_data.append(row)

        # Convert to DataFrame and save
        effectiveness_df = pd.DataFrame(effectiveness_data)
        if not effectiveness_df.empty:
            effectiveness_df.to_csv(
                os.path.join(output_dir, "table2_attack_effectiveness.csv"), index=False
            )
            print(f"Saved effectiveness table with {len(effectiveness_df)} rows")
        else:
            print("Warning: Effectiveness table is empty!")
            # Create empty placeholder file
            pd.DataFrame(
                columns=["Category", "Method"] + list(self.models.keys())
            ).to_csv(
                os.path.join(output_dir, "table2_attack_effectiveness.csv"), index=False
            )

        if targeted_data:
            targeted_df = pd.DataFrame(targeted_data)
            targeted_df.to_csv(
                os.path.join(output_dir, "table2b_targeted_effectiveness.csv"),
                index=False,
            )
            print(f"Saved targeted effectiveness table with {len(targeted_df)} rows")
        else:
            print("No targeted attack data to save")

        # Table 3: Perturbation Efficiency
        perturbation_data = []

        for attack_name in self.results:
            # Determine category
            category = (
                "Baseline"
                if attack_name.startswith(("FGSM", "FFGSM", "DeepFool", "CW"))
                else "Optimization"
            )

            row = {"Category": category, "Method": attack_name}

            # Average perturbation metrics across models
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
                            if isinstance(model_results, dict):
                                if "l2_norm" in model_results:
                                    l2_values.append(model_results["l2_norm"])
                                if "linf_norm" in model_results:
                                    linf_values.append(model_results["linf_norm"])
                                if "ssim" in model_results:
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
        if not perturbation_df.empty:
            perturbation_df.to_csv(
                os.path.join(output_dir, "table3_perturbation_efficiency.csv"),
                index=False,
            )
            print(f"Saved perturbation table with {len(perturbation_df)} rows")
        else:
            print("Warning: Perturbation table is empty!")
            # Create empty placeholder file
            pd.DataFrame(columns=["Category", "Method"]).to_csv(
                os.path.join(output_dir, "table3_perturbation_efficiency.csv"),
                index=False,
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
                        if isinstance(model_results, dict):
                            if "iterations" in model_results:
                                iterations.append(model_results["iterations"])
                            if "gradient_calls" in model_results:
                                gradient_calls.append(model_results["gradient_calls"])
                            if "time_per_sample" in model_results:
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
                        if isinstance(model_results, dict):
                            if "iterations" in model_results:
                                iterations.append(model_results["iterations"])
                            if "gradient_calls" in model_results:
                                gradient_calls.append(model_results["gradient_calls"])
                            if "time_per_sample" in model_results:
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
        if not computational_df.empty:
            computational_df.to_csv(
                os.path.join(output_dir, "table4_computational_efficiency.csv"),
                index=False,
            )
            print(f"Saved computational table with {len(computational_df)} rows")
        else:
            print("Warning: Computational table is empty!")
            # Create empty placeholder file
            pd.DataFrame(columns=["Attack Method"]).to_csv(
                os.path.join(output_dir, "table4_computational_efficiency.csv"),
                index=False,
            )

        print(f"Results exported to {output_dir}")

    def visualize_perturbations(
        self, attack_name, model_name, num_samples=5, output_dir=None
    ):
        """Visualize original images, perturbations, and adversarial examples for a specific attack and model."""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Check if results are available for this attack/model
        if (
            attack_name not in self.results
            or model_name not in self.results[attack_name].get("untargeted", {})
            and model_name not in self.results[attack_name].get("targeted", {})
        ):
            print(f"No results found for {attack_name} on {model_name}")
            return

        # Try to get successful examples from either untargeted or targeted results
        examples = []
        for attack_type in ["untargeted", "targeted"]:
            if attack_type in self.results[attack_name]:
                if model_name in self.results[attack_name][attack_type]:
                    if "examples" in self.results[attack_name][attack_type][model_name]:
                        examples = self.results[attack_name][attack_type][model_name][
                            "examples"
                        ]
                        if examples:
                            break

        if not examples:
            print(f"No successful examples found for {attack_name} on {model_name}")
            return

        # Use up to num_samples examples
        examples = examples[:num_samples]

        # Get class names if available
        class_names = (
            self.dataset.class_names if hasattr(self.dataset, "class_names") else None
        )

        # Visualize each example
        for i, example in enumerate(examples):
            original = example["original"]
            adversarial = example["adversarial"]
            original_label = example["original_label"]
            adversarial_label = example["adversarial_label"]

            # Get label text if class names are available
            original_label_text = (
                class_names[original_label] if class_names else str(original_label)
            )
            adversarial_label_text = (
                class_names[adversarial_label]
                if class_names
                else str(adversarial_label)
            )

            # Plot the images with improved visualization
            self.plot_adversarial_comparison(
                original=original,
                adversarial=adversarial,
                attack_name=attack_name,
                original_label=f"{original_label_text} ({original_label})",
                adv_prediction=f"{adversarial_label_text} ({adversarial_label})",
                save_path=(
                    f"{output_dir}/{attack_name}_{attack_type}_{model_name}_example_{i+1}.png"
                    if output_dir
                    else None
                ),
            )

    def plot_adversarial_comparison(
        self,
        original,
        adversarial,
        attack_name,
        original_label,
        adv_prediction,
        save_path=None,
    ):
        """
        Create an improved comparison plot between original and adversarial images.

        Args:
            original: Original image tensor
            adversarial: Adversarial image tensor
            attack_name: Name of the attack
            original_label: Original label
            adv_prediction: Prediction on adversarial example
            save_path: Path to save the figure
        """
        # Denormalize images for visualization
        mean = (
            torch.tensor([0.485, 0.456, 0.406], dtype=original.dtype)
            .view(3, 1, 1)
            .to(original.device)
        )
        std = (
            torch.tensor([0.229, 0.224, 0.225], dtype=original.dtype)
            .view(3, 1, 1)
            .to(original.device)
        )

        def denormalize(x):
            """Convert from normalized to [0,1] range for visualization"""
            img = x.cpu().clone()
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            return img

        # Get numpy versions for plotting
        original_np = denormalize(original).permute(1, 2, 0).numpy()
        adversarial_np = denormalize(adversarial).permute(1, 2, 0).numpy()

        # Calculate perturbation for visualization
        perturbation = adversarial - original

        # Sign/direction view (shows how values change: increase=red, decrease=blue)
        # Create a diverging colormap visualization
        perturbation_sign = perturbation.cpu()
        # Scale the perturbation for better visualization, with diverging colors
        max_pert = (
            perturbation_sign.abs().max().item()
            if perturbation_sign.abs().max().item() > 0
            else 1.0
        )
        perturbation_sign = perturbation_sign / (
            max_pert * 0.1
        )  # Normalize and enhance
        perturbation_sign = torch.clamp(perturbation_sign, -1, 1)

        # Convert to a diverging colormap (red-white-blue)
        # Red for positive changes, blue for negative
        pert_rgb = torch.zeros(
            (3, perturbation_sign.shape[1], perturbation_sign.shape[2]),
            device=perturbation_sign.device,
        )
        pert_rgb[0] = torch.clamp(
            perturbation_sign.mean(dim=0) * 0.5 + 0.5, 0, 1
        )  # Red channel
        pert_rgb[2] = torch.clamp(
            -perturbation_sign.mean(dim=0) * 0.5 + 0.5, 0, 1
        )  # Blue channel
        pert_rgb[1] = torch.clamp(
            1.0 - perturbation_sign.mean(dim=0).abs() * 0.5, 0, 1
        )  # Green channel

        # Convert to numpy
        diverging_pert = pert_rgb.permute(1, 2, 0).numpy()

        # Create the figure
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Plot original image
        ax[0].imshow(original_np)
        ax[0].set_title(f"Original\nLabel: {original_label}")
        ax[0].axis("off")

        # Plot perturbation (sign)
        ax[1].imshow(diverging_pert)
        ax[1].set_title(f"Perturbation (Direction)\nRed=Increase, Blue=Decrease")
        ax[1].axis("off")

        # Plot adversarial image
        ax[2].imshow(adversarial_np)
        ax[2].set_title(f"Adversarial\nPredicted: {adv_prediction}")
        ax[2].axis("off")

        # Add attack information
        plt.suptitle(f"Adversarial Example - {attack_name}", fontsize=16)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()
