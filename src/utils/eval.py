"""Adversarial attack evaluation framework."""

import os
import torch
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Callable

from .metrics import (
    calculate_perturbation_metrics,
    evaluate_attack_success,
    evaluate_across_thresholds,
    evaluate_across_perturbation_budget,
    evaluate_transferability,
    plot_success_rate_vs_confidence,
    plot_success_rate_vs_perturbation,
    plot_epsilon_vs_metrics,
    plot_transferability_heatmap,
    compare_attacks,
    plot_attack_efficiency,
    plot_confidence_distribution,
    save_metrics_to_csv,
    generate_latex_table,
    run_comprehensive_analysis,
)


class AdversarialAttackEvaluator:
    """
    Framework for evaluating and comparing multiple adversarial attack methods.

    This class provides a structured approach to evaluate multiple attack methods
    against the same model and dataset, generating standardized metrics and
    visualizations suitable for research papers.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        correctly_classified_images: List[Tuple[torch.Tensor, torch.Tensor]],
        class_names: List[str],
        output_dir: str = "results",
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """
        Initialize the evaluator.

        Args:
            model: The model to evaluate against
            correctly_classified_images: List of (image, label) tuples for correctly classified images
            class_names: List of class names for display
            output_dir: Directory to save evaluation results
            device: Device to run evaluation on
        """
        self.model = model
        self.correctly_classified_images = correctly_classified_images
        self.class_names = class_names
        self.output_dir = output_dir
        self.device = device

        # Create the output directory
        os.makedirs(output_dir, exist_ok=True)

        # Extract images and labels
        self.images = torch.cat([img for img, _ in correctly_classified_images], dim=0)
        self.labels = torch.cat([lbl for _, lbl in correctly_classified_images], dim=0)

        # Dictionary to store attack methods
        self.attack_methods = {}

        # Dictionary to store evaluation results
        self.results = {}

        # Metadata for the evaluation
        self.metadata = {
            "model_name": model.__class__.__name__,
            "num_images": len(self.correctly_classified_images),
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
        }

        print(
            f"Initialized AdversarialAttackEvaluator with {len(self.correctly_classified_images)} images"
        )
        print(f"Results will be saved to: {output_dir}")

    def add_attack(self, name: str, attack_fn: Callable, description: str = ""):
        """
        Add an attack method to evaluate.

        Args:
            name: Name of the attack method
            attack_fn: Function that takes (model, images, labels, epsilon, norm, targeted)
                      and returns adversarial examples
            description: Optional description of the attack method
        """
        self.attack_methods[name] = {"function": attack_fn, "description": description}
        print(f"Added attack method: {name}")

    def evaluate_attack(
        self,
        attack_name: str,
        epsilon_values: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
        norm: str = "L2",
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None,
        confidence_thresholds: List[float] = [0.0, 0.5, 0.9],
        save_adv_examples: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate a single attack method across multiple configurations.

        Args:
            attack_name: Name of the attack to evaluate
            epsilon_values: List of epsilon values to test
            norm: Norm to use for the attack ("L2" or "Linf")
            targeted: Whether the attack is targeted
            target_labels: Target labels for targeted attacks
            confidence_thresholds: List of confidence thresholds to evaluate
            save_adv_examples: Whether to save generated adversarial examples

        Returns:
            Dictionary of evaluation results
        """
        if attack_name not in self.attack_methods:
            raise ValueError(f"Attack method '{attack_name}' not found")

        attack_fn = self.attack_methods[attack_name]["function"]
        attack_dir = os.path.join(self.output_dir, attack_name)
        os.makedirs(attack_dir, exist_ok=True)

        print(f"\n{'-'*50}")
        print(f"Evaluating {attack_name} attack")
        print(f"{'-'*50}")

        # Create attack-specific target labels if needed
        targets = target_labels if targeted else self.labels
        if targeted and target_labels is None:
            # Generate random target labels different from true labels
            num_classes = len(self.class_names)
            targets = (
                self.labels
                + torch.randint(
                    1, num_classes - 1, self.labels.shape, device=self.device
                )
            ) % num_classes

        # Evaluate across perturbation budgets
        print(
            f"Evaluating across {len(epsilon_values)} epsilon values: {epsilon_values}"
        )
        perturbation_results = evaluate_across_perturbation_budget(
            attack_fn,
            self.model,
            self.images,
            self.labels,
            targets if targeted else None,
            targeted,
            epsilon_values,
            norm,
            self.device,
        )

        # Set aside the middle epsilon value for other evaluations
        mid_idx = len(epsilon_values) // 2
        mid_eps = epsilon_values[mid_idx]
        print(f"Using ε={mid_eps} for additional evaluations")

        # Generate adversarial examples with middle epsilon
        adv_images = attack_fn(
            self.model, self.images, targets, mid_eps, norm, targeted
        )

        # Evaluate across confidence thresholds
        print(
            f"Evaluating across {len(confidence_thresholds)} confidence thresholds: {confidence_thresholds}"
        )
        threshold_results = evaluate_across_thresholds(
            self.model,
            self.images,
            adv_images,
            self.labels,
            targets if targeted else None,
            targeted,
            confidence_thresholds,
            self.device,
        )

        # Calculate detailed metrics for this epsilon
        detailed_metrics = {}

        # Get predictions for original and adversarial images
        with torch.no_grad():
            original_outputs = self.model(self.images.to(self.device))
            original_preds = original_outputs.argmax(dim=1)

            adv_outputs = self.model(adv_images.to(self.device))
            adv_preds = adv_outputs.argmax(dim=1)

        # Calculate perturbation metrics
        pert_metrics = calculate_perturbation_metrics(self.images, adv_images)
        for metric_name, value in pert_metrics.items():
            detailed_metrics[metric_name] = value.mean().item()

        # Calculate success rate
        if targeted:
            success = (adv_preds == targets).float().mean().item() * 100
        else:
            success = (adv_preds != self.labels).float().mean().item() * 100

        detailed_metrics["success_rate"] = success
        detailed_metrics["epsilon"] = mid_eps

        # If requested, save adversarial examples
        if save_adv_examples:
            print("Saving adversarial examples")
            adv_dir = os.path.join(attack_dir, "adversarial_examples")
            os.makedirs(adv_dir, exist_ok=True)

            # Save tensor
            torch.save(
                adv_images, os.path.join(adv_dir, f"adv_images_eps_{mid_eps}.pt")
            )

            # Save examples as individual images
            try:
                from torchvision.utils import save_image

                for i in range(min(10, len(adv_images))):
                    save_image(
                        adv_images[i],
                        os.path.join(adv_dir, f"adv_image_{i}_eps_{mid_eps}.png"),
                    )
            except ImportError:
                print("Skipping saving as images (torchvision not available)")

        # Generate plots
        print("Generating plots")

        # Success rate vs confidence threshold
        plot_success_rate_vs_confidence(
            threshold_results,
            attack_name=attack_name,
            save_path=os.path.join(attack_dir, "success_vs_confidence.png"),
        )

        # Success rate vs epsilon (perturbation budget)
        plot_success_rate_vs_perturbation(
            perturbation_results,
            attack_name=attack_name,
            save_path=os.path.join(attack_dir, "success_vs_perturbation.png"),
        )

        # Multiple metrics vs epsilon
        plot_epsilon_vs_metrics(
            perturbation_results,
            attack_name=attack_name,
            save_path=os.path.join(attack_dir, "metrics_vs_epsilon.png"),
        )

        # Save numerical results
        results = {
            "perturbation_results": perturbation_results,
            "threshold_results": threshold_results,
            "detailed_metrics": detailed_metrics,
        }

        # Save results to JSON
        with open(os.path.join(attack_dir, "results.json"), "w") as f:
            # Convert some values for JSON serialization
            json_results = {
                "perturbation_results": {
                    k: (
                        v
                        if not isinstance(v, (np.ndarray, torch.Tensor))
                        else v.tolist() if hasattr(v, "tolist") else list(v)
                    )
                    for k, v in perturbation_results.items()
                },
                "threshold_results": {
                    k: (
                        v
                        if not isinstance(v, (np.ndarray, torch.Tensor))
                        else v.tolist() if hasattr(v, "tolist") else list(v)
                    )
                    for k, v in threshold_results.items()
                },
                "detailed_metrics": detailed_metrics,
            }
            json.dump(json_results, f, indent=2)

        # Store results
        self.results[attack_name] = results

        print(f"Evaluation of {attack_name} completed")

        return results

    def evaluate_all(
        self,
        epsilon_values: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
        norm: str = "L2",
        targeted: bool = False,
        confidence_thresholds: List[float] = [0.0, 0.5, 0.9],
        save_adv_examples: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all added attack methods.

        Args:
            epsilon_values: List of epsilon values to test
            norm: Norm to use for the attack ("L2" or "Linf")
            targeted: Whether the attack is targeted
            confidence_thresholds: List of confidence thresholds to evaluate
            save_adv_examples: Whether to save generated adversarial examples

        Returns:
            Dictionary of evaluation results for all attack methods
        """
        if not self.attack_methods:
            raise ValueError("No attack methods added. Use add_attack() first.")

        # Generate random target labels for targeted attacks
        target_labels = None
        if targeted:
            num_classes = len(self.class_names)
            target_labels = (
                self.labels
                + torch.randint(
                    1, num_classes - 1, self.labels.shape, device=self.device
                )
            ) % num_classes

        # Evaluate each attack method
        for attack_name in self.attack_methods:
            self.evaluate_attack(
                attack_name,
                epsilon_values,
                norm,
                targeted,
                target_labels,
                confidence_thresholds,
                save_adv_examples,
            )

        # Generate comparison results
        self.generate_comparison()

        return self.results

    def generate_comparison(self):
        """Generate comparison results between all evaluated attack methods."""
        if not self.results:
            print("No results to compare. Run evaluate_all() first.")
            return

        print("\nGenerating attack comparison")

        # Create comparison directory
        comparison_dir = os.path.join(self.output_dir, "comparison")
        os.makedirs(comparison_dir, exist_ok=True)

        # Prepare attack results for comparison functions
        attack_results = {}
        for attack_name, results in self.results.items():
            attack_results[attack_name] = results["perturbation_results"]

        # Generate comparison plots
        compare_attacks(
            attack_results,
            metric="success_rates",
            x_axis="avg_l2_norm",
            title="Attack Comparison: Success Rate vs. L2 Perturbation",
            xlabel="Average L2 Perturbation Norm",
            ylabel="Success Rate (%)",
            save_path=os.path.join(comparison_dir, "comparison_l2.png"),
        )

        compare_attacks(
            attack_results,
            metric="success_rates",
            x_axis="epsilon",
            title="Attack Comparison: Success Rate vs. Epsilon",
            xlabel="Perturbation Budget (ε)",
            ylabel="Success Rate (%)",
            save_path=os.path.join(comparison_dir, "comparison_epsilon.png"),
        )

        plot_attack_efficiency(
            attack_results,
            save_path=os.path.join(comparison_dir, "attack_efficiency.png"),
        )

        # Save numerical results to CSV
        save_metrics_to_csv(
            attack_results, os.path.join(comparison_dir, "attack_metrics.csv")
        )

        # Generate LaTeX table
        generate_latex_table(
            attack_results, os.path.join(comparison_dir, "attack_table.tex")
        )

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive markdown report of evaluation results.

        Args:
            output_path: Path to save the report (optional)

        Returns:
            Markdown string containing the report
        """
        if not self.results:
            print("No results to report. Run evaluate_all() first.")
            return ""

        # Default output path
        if output_path is None:
            output_path = os.path.join(self.output_dir, "evaluation_report.md")

        # Create report sections
        sections = []

        # Add header
        sections.append("# Adversarial Attack Evaluation Report\n")
        sections.append(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        # Add metadata
        sections.append("## Evaluation Setup\n")
        sections.append(f"- **Model:** {self.metadata['model_name']}")
        sections.append(f"- **Number of Images:** {self.metadata['num_images']}")
        sections.append(f"- **Device:** {self.metadata['device']}")
        sections.append(
            f"- **Attacks Evaluated:** {', '.join(self.attack_methods.keys())}\n"
        )

        # Add comparison section
        sections.append("## Attack Comparison\n")
        sections.append("### Success Rate vs. Perturbation Size\n")
        sections.append(
            f"![Success Rate vs. L2 Perturbation](comparison/comparison_l2.png)\n"
        )

        sections.append("### Success Rate vs. Epsilon\n")
        sections.append(
            f"![Success Rate vs. Epsilon](comparison/comparison_epsilon.png)\n"
        )

        sections.append("### Attack Efficiency\n")
        sections.append(f"![Attack Efficiency](comparison/attack_efficiency.png)\n")

        # Add metrics table
        sections.append("### Attack Metrics\n")

        # Create metrics table
        table_rows = []
        table_rows.append(
            "| Attack | Epsilon | Success Rate (%) | Avg. L2 Norm | Time (s) |"
        )
        table_rows.append(
            "| ------ | ------- | --------------- | ------------ | -------- |"
        )

        for attack_name, results in self.results.items():
            pert_results = results["perturbation_results"]
            for i, eps in enumerate(pert_results["epsilon"]):
                if isinstance(eps, (list, np.ndarray, torch.Tensor)):
                    eps_val = eps[0] if len(eps) > 0 else 0
                else:
                    eps_val = eps

                row = f"| {attack_name} | {eps_val:.2f} | {pert_results['success_rates'][i]:.1f} | {pert_results['avg_l2_norm'][i]:.4f} | {pert_results['attack_time'][i]:.2f} |"
                table_rows.append(row)

        sections.append("\n".join(table_rows) + "\n")

        # Add individual attack sections
        sections.append("## Individual Attack Results\n")

        for attack_name, results in self.results.items():
            sections.append(f"### {attack_name}\n")

            # Add description if available
            if self.attack_methods[attack_name]["description"]:
                sections.append(f"{self.attack_methods[attack_name]['description']}\n")

            # Add detailed metrics
            metrics = results["detailed_metrics"]
            sections.append("#### Metrics at ε=" + f"{metrics['epsilon']:.2f}\n")
            sections.append(f"- **Success Rate:** {metrics['success_rate']:.2f}%")
            sections.append(f"- **Average L2 Norm:** {metrics['L2']:.4f}")
            sections.append(f"- **Average L∞ Norm:** {metrics['Linf']:.4f}")
            sections.append(f"- **PSNR:** {metrics['PSNR_dB']:.2f} dB\n")

            # Add plots
            sections.append("#### Success Rate vs. Confidence Threshold\n")
            sections.append(
                f"![Success vs Confidence]({attack_name}/success_vs_confidence.png)\n"
            )

            sections.append("#### Success Rate vs. Perturbation Size\n")
            sections.append(
                f"![Success vs Perturbation]({attack_name}/success_vs_perturbation.png)\n"
            )

            sections.append("#### Metrics vs. Epsilon\n")
            sections.append(f"![Metrics]({attack_name}/metrics_vs_epsilon.png)\n")

        # Combine all sections
        report = "\n".join(sections)

        # Save report if output_path is provided
        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            print(f"Report saved to {output_path}")

        return report


def adaptive_attack_wrapper(
    attack_fn: Callable, parameters: Dict[str, Any]
) -> Callable:
    """
    Wrap an attack function to use specific parameters.

    Args:
        attack_fn: The attack function to wrap
        parameters: Dictionary of parameters to pass to the attack function

    Returns:
        Wrapped attack function with standardized interface
    """

    def wrapped_attack(
        model: torch.nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        epsilon: float,
        norm: str,
        targeted: bool,
    ) -> torch.Tensor:
        # Create a copy of parameters to avoid modifying the original
        attack_params = parameters.copy()

        # Override parameters based on the inputs
        attack_params["model"] = model
        attack_params["norm"] = norm
        attack_params["eps"] = epsilon
        attack_params["targeted"] = targeted

        # Create attack instance
        attack = attack_fn(**attack_params)

        # Generate adversarial examples
        adv_images, _ = attack.generate(images, labels)

        return adv_images

    return wrapped_attack


def evaluate_attack_on_imagenet(
    model_name: str,
    attack_name: str,
    attack_fn: Callable,
    attack_params: Dict[str, Any],
    num_images: int = 10,
    epsilon_values: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
    norm: str = "L2",
    targeted: bool = False,
    output_dir: str = "results",
) -> AdversarialAttackEvaluator:
    """
    Convenience function to evaluate an attack on ImageNet.

    This function handles all the boilerplate code to:
    1. Load a model
    2. Load ImageNet data
    3. Find correctly classified images
    4. Initialize the evaluator
    5. Run the evaluation
    6. Generate reports

    Args:
        model_name: Name of the model to use
        attack_name: Name of the attack to evaluate
        attack_fn: The attack function constructor
        attack_params: Parameters for the attack function
        num_images: Number of correctly classified images to test
        epsilon_values: List of epsilon values to test
        norm: Norm to use for the attack ("L2" or "Linf")
        targeted: Whether the attack is targeted
        output_dir: Directory to save results

    Returns:
        AdversarialAttackEvaluator with results
    """
    import torch
    import os
    from src.models.wrappers import get_model
    from src.datasets.imagenet import get_dataset, get_dataloader

    # Load model
    print(f"Loading {model_name} model...")
    model = get_model(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load data
    print("Loading ImageNet data...")
    dataloader, class_names = get_dataloader(
        get_dataset("imagenet", "data"), batch_size=1, shuffle=True
    )

    # Find correctly classified images
    print(f"Finding {num_images} correctly classified images...")
    correctly_classified = []
    tested_images = 0

    for images, labels in dataloader:
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            if predictions[0] == labels[0]:
                correctly_classified.append((images, labels))
                print(
                    f"Found correctly classified image {len(correctly_classified)}: {class_names[labels[0]]}"
                )

                if len(correctly_classified) >= num_images:
                    break

        tested_images += 1
        if tested_images % 10 == 0:
            print(
                f"Tested {tested_images} images, found {len(correctly_classified)} correctly classified"
            )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize evaluator
    evaluator = AdversarialAttackEvaluator(
        model,
        correctly_classified,
        class_names,
        output_dir=os.path.join(output_dir, model_name),
        device=device,
    )

    # Create wrapped attack function
    wrapped_attack = adaptive_attack_wrapper(attack_fn, attack_params)

    # Add attack
    evaluator.add_attack(
        attack_name,
        wrapped_attack,
        description=f"{attack_name} attack with {norm} norm",
    )

    # Run evaluation
    evaluator.evaluate_all(
        epsilon_values=epsilon_values,
        norm=norm,
        targeted=targeted,
        save_adv_examples=True,
    )

    # Generate report
    evaluator.generate_report()

    return evaluator
