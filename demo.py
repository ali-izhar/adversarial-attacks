"""
Script to test a single adversarial attack method at a time.
Supports multiple adversarial attack methods with visualization.
"""

import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import argparse

from src.attacks.attack_cg import ConjugateGradient
from src.attacks.attack_cw import CW
from src.attacks.attack_deepfool import DeepFool
from src.attacks.attack_ffgsm import FFGSM
from src.attacks.attack_fgsm import FGSM
from src.attacks.attack_lbfgs import LBFGS
from src.attacks.attack_pgd import PGD

from src.plot.compare import visualize_attack


class AdversarialAttackDemo:
    """Class to demonstrate various adversarial attacks with visualization."""

    def __init__(self, model_name="resnet50", device=None, verbose=False):
        """Initialize with a pretrained model."""
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.verbose = verbose

        print(f"Using device: {self.device}")

        # Load pretrained model
        if model_name == "resnet50":
            self.model = models.resnet50(weights="IMAGENET1K_V1")
        elif model_name == "vgg16":
            self.model = models.vgg16(weights="IMAGENET1K_V1")
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model.to(self.device)
        self.model.eval()

        # Set up normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

        # Add mean and std as attributes to the model (needed for our implementation)
        self.model.mean = [0.485, 0.456, 0.406]
        self.model.std = [0.229, 0.224, 0.225]

        # Load ImageNet class names
        try:
            with open("data/imagenet/imagenet_classes.txt") as f:
                self.classes = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            # Fallback to alternate location
            print(
                "Warning: data/imagenet/imagenet_classes.txt not found, using default classes"
            )
            self.classes = [f"class_{i}" for i in range(1000)]

    def load_image(self, img_path):
        """Load and preprocess an image."""
        image = Image.open(img_path).convert("RGB")

        # Define transforms
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Apply transforms and add batch dimension
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        return input_batch, image

    def denormalize(self, tensor):
        """Convert normalized image tensor to displayable image."""
        # Clone the tensor to avoid modifying the original
        img = tensor.clone().detach()

        # Denormalize
        for t, m, s in zip(img, self.mean, self.std):
            t.mul_(s).add_(m)

        # Clamp to ensure valid image range [0, 1]
        img = torch.clamp(img, 0, 1)

        return img

    def run_attack(self, method, image, **kwargs):
        """Run the specified attack on the input image."""
        # Extract common parameters
        epsilon = kwargs.get("epsilon", 0.01)
        targeted = kwargs.get("targeted", False)
        target = kwargs.get("target", None)
        norm = kwargs.get("norm", "Linf")
        verbose = kwargs.get("verbose", self.verbose)

        if verbose:
            print(f"Running {method.upper()} attack with parameters:")
            print(f"  - epsilon: {epsilon}")
            print(f"  - norm: {norm}")
            print(f"  - targeted: {targeted}")
            if targeted:
                print(f"  - target: {target}")
            for key, value in kwargs.items():
                if key not in ["epsilon", "targeted", "target", "norm", "verbose"]:
                    print(f"  - {key}: {value}")

        # Get original prediction
        with torch.no_grad():
            output = self.model(image)

        # Get prediction class and confidence
        _, pred_class = torch.max(output, 1)
        confidence = torch.nn.functional.softmax(output, dim=1)[
            0, pred_class.item()
        ].item()

        # Print original prediction
        print(
            f"Original prediction: {self.classes[pred_class.item()]} ({confidence*100:.1f}% confidence)"
        )

        # If no target is provided for targeted attack, choose a random different class
        if targeted and target is None:
            target = (pred_class + 1) % 1000  # Just pick the next class

        # Set target for the attack
        target_class = target if target is not None else pred_class

        # Create and run the specified attack
        if method == "fgsm":
            attack = FGSM(
                model=self.model,
                norm=norm,
                eps=epsilon,
                targeted=targeted,
                loss_fn="cross_entropy",
                device=self.device,
            )
            method_name = "FGSM"

        elif method == "ffgsm":
            alpha = kwargs.get("alpha", 0.2)
            attack = FFGSM(
                model=self.model,
                norm=norm,
                eps=epsilon,
                alpha=alpha,
                targeted=targeted,
                loss_fn="cross_entropy",
                device=self.device,
            )
            method_name = f"FFGSM (α={alpha})"

        elif method == "pgd":
            steps = kwargs.get("steps", 10)
            alpha_init = kwargs.get("alpha_init", epsilon / 4)
            alpha_type = kwargs.get("alpha_type", "constant")
            rand_init = kwargs.get("rand_init", True)
            attack = PGD(
                model=self.model,
                norm=norm,
                eps=epsilon,
                targeted=targeted,
                loss_fn="cross_entropy",
                n_iterations=steps,
                alpha_init=alpha_init,
                alpha_type=alpha_type,
                rand_init=rand_init,
                device=self.device,
            )
            method_name = f"PGD ({steps} steps)"

        elif method == "deepfool":
            steps = kwargs.get("steps", 50)
            # Ensure we have enough classes to handle whatever the model might predict
            # The default ImageNet model has 1000 classes, but DeepFool works better with fewer
            num_classes = kwargs.get(
                "num_classes", 20
            )  # Increased from 10 for better coverage
            overshoot = kwargs.get("overshoot", 0.02)

            # For DeepFool, it's important to first check what class the model predicts
            with torch.no_grad():
                original_output = self.model(image)
                original_pred_class = original_output.argmax(dim=1).item()

            # Debug info
            if verbose:
                print(f"Original prediction class index: {original_pred_class}")

            attack = DeepFool(
                model=self.model,
                norm=norm,
                num_classes=num_classes,
                overshoot=overshoot,
                max_iter=steps,
                device=self.device,
            )
            method_name = f"DeepFool (overshoot={overshoot})"

            try:
                # Run DeepFool attack
                adv_image, metrics = attack.generate(image)
                # For DeepFool, we don't pass target_class as it doesn't use it
            except ValueError as e:
                # Handle the specific error where orig_pred isn't in other_classes
                if "list.remove" in str(e):
                    print(f"Warning: DeepFool error - {e}")
                    print("Trying with full class set...")
                    # Try again with full class set
                    attack = DeepFool(
                        model=self.model,
                        norm=norm,
                        num_classes=1000,  # Use all classes
                        overshoot=overshoot,
                        max_iter=steps,
                        device=self.device,
                    )
                    adv_image, metrics = attack.generate(image)
                else:
                    # Re-raise if it's a different error
                    raise

            # The rest of the method can proceed as before...
            # Get prediction for adversarial example
            with torch.no_grad():
                adv_output = self.model(adv_image)

            # Get adversarial prediction class and confidence
            _, adv_pred_class = torch.max(adv_output, 1)
            adv_confidence = torch.nn.functional.softmax(adv_output, dim=1)[
                0, adv_pred_class.item()
            ].item()

            # Print adversarial prediction
            print(
                f"Adversarial prediction: {self.classes[adv_pred_class.item()]} ({adv_confidence*100:.1f}% confidence)"
            )
            print(f"Attack success: {metrics['success_rate']:.1f}%")

            # Calculate perturbation
            perturbation = adv_image - image

            return (
                image,
                perturbation,
                adv_image,
                {
                    "original_class": self.classes[pred_class.item()],
                    "original_confidence": confidence,
                    "adversarial_class": self.classes[adv_pred_class.item()],
                    "adversarial_confidence": adv_confidence,
                    "success_rate": metrics["success_rate"],
                    "iterations": metrics.get("iterations", 1),
                    "time": metrics.get("time", 0),
                },
                method_name,
            )

        elif method == "cw":
            confidence = kwargs.get("confidence", 0.0)
            c_init = kwargs.get("c_init", 0.01)
            max_iter = kwargs.get("max_iter", 100)
            binary_search_steps = kwargs.get("binary_search_steps", 5)
            learning_rate = kwargs.get("learning_rate", 0.01)
            attack = CW(
                model=self.model,
                confidence=confidence,
                c_init=c_init,
                max_iter=max_iter,
                binary_search_steps=binary_search_steps,
                learning_rate=learning_rate,
                targeted=targeted,
                device=self.device,
            )
            method_name = f"C&W (conf={confidence}, steps={max_iter})"

        elif method == "cg":
            # ConjugateGradient parameters
            n_iterations = kwargs.get("steps", 50)
            line_search_max_iter = kwargs.get("max_line_search", 10)
            attack = ConjugateGradient(
                model=self.model,
                norm=norm,
                eps=epsilon,
                targeted=targeted,
                loss_fn="cross_entropy",
                n_iterations=n_iterations,  # Use n_iterations instead of max_iter
                line_search_max_iter=line_search_max_iter,  # Use line_search_max_iter
                device=self.device,
                verbose=verbose,
            )
            method_name = f"Conjugate Gradient ({n_iterations} steps)"

        elif method == "lbfgs":
            # LBFGS parameters
            n_iterations = kwargs.get("steps", 50)
            history_size = kwargs.get("history_size", 100)
            max_line_search = kwargs.get("max_line_search", 10)
            attack = LBFGS(
                model=self.model,
                norm=norm,
                eps=epsilon,
                targeted=targeted,
                loss_fn="cross_entropy",
                n_iterations=n_iterations,  # Use n_iterations instead of max_iter
                history_size=history_size,
                max_line_search=max_line_search,
                device=self.device,
                verbose=verbose,
            )
            method_name = f"L-BFGS ({n_iterations} steps)"

        else:
            raise ValueError(f"Unsupported attack method: {method}")

        # Generate adversarial example
        adv_image, metrics = attack.generate(image, target_class)

        # Get prediction for adversarial example
        with torch.no_grad():
            adv_output = self.model(adv_image)

        # Get adversarial prediction class and confidence
        _, adv_pred_class = torch.max(adv_output, 1)
        adv_confidence = torch.nn.functional.softmax(adv_output, dim=1)[
            0, adv_pred_class.item()
        ].item()

        # Print adversarial prediction
        print(
            f"Adversarial prediction: {self.classes[adv_pred_class.item()]} ({adv_confidence*100:.1f}% confidence)"
        )
        print(f"Attack success: {metrics['success_rate']:.1f}%")

        # Calculate perturbation
        perturbation = adv_image - image

        return (
            image,
            perturbation,
            adv_image,
            {
                "original_class": self.classes[pred_class.item()],
                "original_confidence": confidence,
                "adversarial_class": self.classes[adv_pred_class.item()],
                "adversarial_confidence": adv_confidence,
                "success_rate": metrics["success_rate"],
                "iterations": metrics.get("iterations", 1),
                "time": metrics.get("time", 0),
            },
            method_name,
        )


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="Adversarial Attack Demonstration")
    parser.add_argument(
        "--image",
        type=str,
        default="panda.jpg",
        help="Path to the input image",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "vgg16"],
        help="Model to use for the attack",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="fgsm",
        choices=["fgsm", "ffgsm", "pgd", "deepfool", "cw", "cg", "lbfgs"],
        help="Attack method to use",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.03,  # Increased from 0.01 for better attack success
        help="Epsilon value for the attack (perturbation magnitude)",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="Linf",
        choices=["L2", "Linf"],
        help="Norm to use for the attack",
    )
    parser.add_argument(
        "--targeted", action="store_true", help="Whether to perform a targeted attack"
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Target class for targeted attack (0-999 for ImageNet)",
    )

    # Method-specific parameters
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of iterations for iterative attacks",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Alpha value for FFGSM"
    )
    parser.add_argument(
        "--alpha-init", type=float, default=None, help="Initial step size for PGD"
    )
    parser.add_argument(
        "--alpha-type",
        type=str,
        default="constant",
        choices=["constant", "diminishing"],
        help="Step size schedule for PGD",
    )
    parser.add_argument(
        "--rand-init",
        action="store_true",
        default=True,
        help="Whether to use random initialization for PGD",
    )
    parser.add_argument(
        "--overshoot", type=float, default=0.02, help="Overshoot parameter for DeepFool"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.0,
        help="Confidence parameter for C&W attack",
    )
    parser.add_argument(
        "--c-init", type=float, default=0.01, help="Initial c value for C&W attack"
    )
    parser.add_argument(
        "--binary-search-steps",
        type=int,
        default=5,
        help="Number of binary search steps for C&W",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for optimization-based attacks",
    )
    parser.add_argument(
        "--max-line-search",
        type=int,
        default=10,
        help="Maximum line search iterations for CG and L-BFGS",
    )
    parser.add_argument(
        "--history-size", type=int, default=100, help="History size for L-BFGS"
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save the visualization",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug output",
    )

    args = parser.parse_args()

    # Ensure image file exists
    if not os.path.exists(args.image):
        print(
            f"Warning: Image file {args.image} does not exist. Looking in data/images/..."
        )
        # Try alternate paths
        alt_paths = [
            f"data/images/{args.image}",
            f"data/{args.image}",
            "data/images/panda.jpg",
            "panda.jpg",
        ]

        for path in alt_paths:
            if os.path.exists(path):
                args.image = path
                print(f"Found image at: {path}")
                break
        else:
            print("Error: Could not find a valid image file.")
            return

    # Create demo object
    demo = AdversarialAttackDemo(model_name=args.model, verbose=args.verbose)

    # Load image
    image, original_pil = demo.load_image(args.image)

    # Set alpha_init for PGD if not specified
    if args.alpha_init is None:
        args.alpha_init = args.epsilon / 4

    print(
        f"Running {args.method.upper()} attack with epsilon={args.epsilon}, norm={args.norm}"
    )

    # Run attack with all parameters
    if args.method == "deepfool":
        # DeepFool has a special implementation and doesn't need target_class
        original, perturbation, adversarial, results, method_name = demo.run_attack(
            method=args.method,
            image=image,
            epsilon=args.epsilon,
            norm=args.norm,
            steps=args.steps,  # This is passed as max_iter to DeepFool
            num_classes=1000,  # Use all classes by default
            overshoot=args.overshoot,
            verbose=args.verbose,  # Add verbose flag for debugging
        )
    elif args.method == "cg":
        # CG with correct parameter names
        original, perturbation, adversarial, results, method_name = demo.run_attack(
            method=args.method,
            image=image,
            epsilon=args.epsilon,
            targeted=args.targeted,
            target=args.target_class,
            norm=args.norm,
            steps=args.steps,  # Will be mapped to n_iterations
            max_line_search=args.max_line_search,
            verbose=args.verbose,
        )
    elif args.method == "lbfgs":
        # LBFGS with correct parameter names
        original, perturbation, adversarial, results, method_name = demo.run_attack(
            method=args.method,
            image=image,
            epsilon=args.epsilon,
            targeted=args.targeted,
            target=args.target_class,
            norm=args.norm,
            steps=args.steps,  # Will be mapped to n_iterations
            history_size=args.history_size,
            max_line_search=args.max_line_search,
            verbose=args.verbose,
        )
    else:
        # For other attack methods (FGSM, FFGSM, PGD, CW)
        original, perturbation, adversarial, results, method_name = demo.run_attack(
            method=args.method,
            image=image,
            epsilon=args.epsilon,
            targeted=args.targeted,
            target=args.target_class,
            norm=args.norm,
            steps=args.steps,
            alpha=args.alpha,
            alpha_init=args.alpha_init,
            alpha_type=args.alpha_type,
            rand_init=args.rand_init,
            overshoot=args.overshoot,
            confidence=args.confidence,
            c_init=args.c_init,
            binary_search_steps=args.binary_search_steps,
            learning_rate=args.learning_rate,
            verbose=args.verbose,
        )

    # Add attack parameters to results
    results["method"] = method_name
    results["epsilon"] = args.epsilon
    results["norm"] = args.norm

    # Use our visualization module to display the results
    visualize_attack(
        original=original[0],  # Remove batch dimension
        perturbation=perturbation[0],
        adversarial=adversarial[0],
        results=results,
        epsilon=args.epsilon,
        method_name=method_name,
        mean=demo.mean.squeeze(),  # Remove extra dimensions
        std=demo.std.squeeze(),
        save_path=args.save,
        show=True,
    )


if __name__ == "__main__":
    main()
