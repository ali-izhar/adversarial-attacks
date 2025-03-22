"""Tests for the PGD optimizer implementation."""

import os
import sys
import pytest
import torch
import numpy as np
from typing import Callable, Tuple

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.attacks.optimize.pgd import PGDOptimizer


# Create a simplified quadratic function that works with 4D tensors
class SimpleImageQuadratic:
    """A simple quadratic function for testing optimization with 4D tensors."""

    def __init__(self, device="cpu"):
        self.device = device
        # Simple scalar multipliers for quadratic components
        self.a = 1.0
        self.b = 2.0
        self.c = 0.0

    def __call__(self, x):
        """Evaluate the function f(x) = a * ||x||^2 - b * sum(x) + c."""
        batch_size = x.shape[0]
        # Compute simple quadratic function
        x_flat = x.reshape(batch_size, -1)
        term1 = self.a * torch.sum(x_flat**2, dim=1)
        term2 = self.b * torch.sum(x_flat, dim=1)
        return term1 - term2 + self.c

    def gradient(self, x):
        """Compute the gradient of the quadratic function."""
        # Gradient is 2a*x - b
        return 2 * self.a * x - self.b


class TestPGDOptimizer:
    """Tests for the PGD optimizer."""

    @pytest.fixture
    def device(self):
        """Return the device to use for testing."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture
    def batch_size(self):
        """Return the batch size to use for testing."""
        return 2

    @pytest.fixture
    def test_images(self, batch_size, device):
        """Create test images with consistent 4D shape that the PGD optimizer expects."""
        # IMPORTANT: All images must have identical shape (C,H,W)
        channels, height, width = 3, 4, 4
        torch.manual_seed(42)  # For reproducibility
        images = torch.rand((batch_size, channels, height, width), device=device)
        return images

    @pytest.fixture
    def simple_quad_objective(self, device) -> Tuple[Callable, Callable]:
        """Create a simple quadratic objective compatible with 4D tensors."""
        quad = SimpleImageQuadratic(device=device)

        def loss_fn(x):
            return quad(x)

        def grad_fn(x):
            return quad.gradient(x)

        return loss_fn, grad_fn

    def test_initialization(self):
        """Test that the optimizer initializes with different parameters."""
        # Default initialization
        opt1 = PGDOptimizer()
        assert opt1.norm == "L2"
        assert opt1.eps == 0.5
        assert opt1.n_iterations == 100
        assert opt1.alpha_init == 0.1
        assert opt1.alpha_type == "diminishing"
        assert opt1.rand_init is True
        assert opt1.init_std == 0.01
        assert opt1.early_stopping is True
        assert opt1.verbose is False

        # Custom initialization
        opt2 = PGDOptimizer(
            norm="Linf",
            eps=0.1,
            n_iterations=50,
            alpha_init=0.05,
            alpha_type="constant",
            rand_init=False,
            init_std=0.05,
            early_stopping=False,
            verbose=True,
        )

        assert opt2.norm == "Linf"
        assert opt2.eps == 0.1
        assert opt2.n_iterations == 50
        assert opt2.alpha_init == 0.05
        assert opt2.alpha_type == "constant"
        assert opt2.rand_init is False
        assert opt2.init_std == 0.05
        assert opt2.early_stopping is False
        assert opt2.verbose is True

    def test_step_size_scheduling(self):
        """Test step size scheduling strategies."""
        # Test constant step size
        opt_constant = PGDOptimizer(alpha_type="constant", alpha_init=0.2)
        assert opt_constant._get_step_size(0) == 0.2
        assert opt_constant._get_step_size(10) == 0.2
        assert opt_constant._get_step_size(100) == 0.2

        # Test diminishing step size
        opt_diminishing = PGDOptimizer(alpha_type="diminishing", alpha_init=0.2)
        assert opt_diminishing._get_step_size(0) == 0.2
        assert opt_diminishing._get_step_size(3) == 0.2 / np.sqrt(4)
        assert opt_diminishing._get_step_size(99) == 0.2 / np.sqrt(100)

        # Test invalid step size
        opt_invalid = PGDOptimizer(alpha_type="invalid")
        with pytest.raises(ValueError):
            opt_invalid._get_step_size(0)

    def test_random_initialization(self, test_images):
        """Test that random initialization works as expected."""
        # Create optimizer with random initialization
        opt_rand = PGDOptimizer(
            rand_init=True, init_std=0.1, n_iterations=1, early_stopping=False
        )

        # Create optimizer without random initialization
        opt_no_rand = PGDOptimizer(
            rand_init=False, init_std=0.1, n_iterations=1, early_stopping=False
        )

        # Simple loss and gradient functions (identity)
        def loss_fn(x):
            return torch.sum(x, dim=(1, 2, 3))

        def grad_fn(x):
            return torch.ones_like(x)

        # Optimize with both initializations
        x_init = test_images.clone()
        result_rand, _ = opt_rand.optimize(x_init, grad_fn, loss_fn)
        result_no_rand, _ = opt_no_rand.optimize(x_init, grad_fn, loss_fn)

        # Random init should produce different result than the one without random init
        assert not torch.allclose(result_rand, result_no_rand, atol=1e-5)

    def test_quadratic_optimization(self, test_images, simple_quad_objective):
        """Test optimization of a simple quadratic function."""
        loss_fn, grad_fn = simple_quad_objective

        # Create optimizer
        opt = PGDOptimizer(
            n_iterations=30,
            alpha_type="constant",
            alpha_init=0.1,
            rand_init=False,
            maximize=False,  # Minimize the quadratic function
        )

        # Run optimizer
        result, metrics = opt.optimize(test_images.clone(), grad_fn, loss_fn)

        # Verify loss decreased
        initial_loss = loss_fn(test_images)
        final_loss = loss_fn(result)
        assert torch.all(final_loss < initial_loss)

    def test_batch_processing(self, device):
        """Test batch processing with minimal complexity."""
        batch_size = 2
        channels, height, width = 3, 4, 4

        # Create input and target - ensure consistent shapes
        torch.manual_seed(0)  # Set seed for reproducibility
        inputs = torch.rand((batch_size, channels, height, width), device=device)
        target = torch.ones_like(inputs) * 0.5

        def loss_fn(x):
            losses = torch.sum((x - target) ** 2, dim=(1, 2, 3))
            return losses

        def grad_fn(x):
            return 2 * (x - target)

        # Create optimizer
        opt = PGDOptimizer(
            n_iterations=20,
            alpha_type="constant",
            alpha_init=0.1,
            rand_init=False,
            maximize=False,  # Minimize MSE loss
        )

        # Run optimization
        result, _ = opt.optimize(inputs.clone(), grad_fn, loss_fn)

        # Check shape matches
        assert result.shape == inputs.shape

        # Check that loss decreased
        initial_loss = loss_fn(inputs)
        final_loss = loss_fn(result)
        assert torch.all(final_loss < initial_loss)

    def test_l2_constraint(self, test_images, device):
        """Test that L2 constraint is respected."""

        # Create a constant gradient pointing toward 1.0
        def loss_fn(x):
            return -torch.sum(x, dim=(1, 2, 3))  # Move toward 1.0

        def grad_fn(x):
            return -torch.ones_like(x)

        # Set a small epsilon
        eps = 0.05
        opt = PGDOptimizer(
            norm="L2",
            eps=eps,
            n_iterations=10,
            alpha_type="constant",
            alpha_init=0.1,
            rand_init=False,
            maximize=True,  # Maximize the negative loss to move toward 1.0
        )

        # Run optimizer with constraint
        result, _ = opt.optimize(
            test_images.clone(), grad_fn, loss_fn, x_original=test_images
        )

        # Calculate perturbation and check norm
        delta = result - test_images
        delta_flat = delta.reshape(delta.shape[0], -1)
        l2_norms = torch.norm(delta_flat, dim=1)

        # Check if norms are within epsilon bound (allow slight tolerance)
        assert torch.all(l2_norms <= eps * 1.01)

    def test_linf_constraint(self, test_images, device):
        """Test that Linf constraint is respected."""

        # Create a constant gradient pointing toward 1.0
        def loss_fn(x):
            return -torch.sum(x, dim=(1, 2, 3))  # Move toward 1.0

        def grad_fn(x):
            return -torch.ones_like(x)

        # Set a small epsilon
        eps = 0.03
        opt = PGDOptimizer(
            norm="Linf",
            eps=eps,
            n_iterations=10,
            alpha_type="constant",
            alpha_init=0.1,
            rand_init=False,
            maximize=True,  # Maximize the negative loss to move toward 1.0
        )

        # Run optimizer with constraint
        result, _ = opt.optimize(
            test_images.clone(), grad_fn, loss_fn, x_original=test_images
        )

        # Calculate perturbation and check max absolute value
        delta = result - test_images
        max_perturbation = torch.max(torch.abs(delta))

        # Check constraint with slight tolerance
        assert max_perturbation <= eps * 1.01

    def test_early_stopping_basic(self, test_images, device):
        """Test simplest early stopping behavior."""
        # Ensure we have exactly 2 examples for this test
        if test_images.shape[0] == 1:
            # If there's only one example, duplicate it
            test_images = torch.cat([test_images, test_images], dim=0)
        elif test_images.shape[0] > 2:
            # If there are more than 2 examples, take just the first 2
            test_images = test_images[:2]

        batch_size = test_images.shape[0]
        assert batch_size == 2, "This test requires exactly 2 examples"

        # Create a target with some ones and zeros
        target = torch.zeros_like(test_images)
        target[:, 0, :, :] = 1.0  # Set first channel to 1.0

        # Define loss function
        def loss_fn(x):
            return torch.sum((x - target) ** 2, dim=(1, 2, 3))

        # Modified gradient function that works with any batch size
        def grad_fn(x):
            # Calculate the targets for this batch
            current_target = torch.zeros_like(x)
            current_target[:, 0, :, :] = 1.0
            return 2 * (x - current_target)

        # Define a very clear success criterion for testing
        def success_fn(x):
            # Example is successful if at least 3 pixels in first channel are close to 1.0
            close_pixels = (torch.abs(x[:, 0, :, :] - 1.0) < 0.05).sum(dim=(1, 2))
            return close_pixels >= 3

        # Reset and prepare test images
        x_init = test_images.clone()

        # First example will succeed immediately - set many pixels very close to 1.0
        # to ensure it stays successful throughout optimization
        x_init[0, 0, :, :] = 0.0  # Reset first
        x_init[0, 0, 0, 0] = 0.98  # Pixel 1
        x_init[0, 0, 0, 1] = 0.98  # Pixel 2
        x_init[0, 0, 1, 0] = 0.98  # Pixel 3
        x_init[0, 0, 1, 1] = 0.98  # Pixel 4 (extra to ensure it stays successful)

        # Second example starts far from success
        x_init[1, 0, :, :] = 0.0  # Set all pixels in first channel away from target

        # Verify initial conditions
        assert success_fn(x_init)[0].item() == True  # First example starts successful
        assert (
            success_fn(x_init)[1].item() == False
        )  # Second example starts unsuccessful

        # Create optimizer with early stopping - use more iterations
        opt_with_stopping = PGDOptimizer(
            early_stopping=True,
            n_iterations=50,  # More iterations to clearly show the difference
            alpha_type="constant",
            alpha_init=0.4,  # Increase step size to make second example successful
            rand_init=False,
            maximize=False,  # Minimize MSE loss
        )

        # Create optimizer without early stopping
        opt_without_stopping = PGDOptimizer(
            early_stopping=False,
            n_iterations=50,  # Match the iterations count
            alpha_type="constant",
            alpha_init=0.4,  # Increase step size to make second example successful
            rand_init=False,
            maximize=False,  # Minimize MSE loss
        )

        # Run optimizer with early stopping
        result_with, metrics_with = opt_with_stopping.optimize(
            x_init.clone(), grad_fn, loss_fn, success_fn=success_fn
        )

        # Run optimizer without early stopping
        result_without, metrics_without = opt_without_stopping.optimize(
            x_init.clone(), grad_fn, loss_fn, success_fn=success_fn
        )

        # Verify that both optimizers produced successful examples
        assert success_fn(result_with)[0].item() == True
        assert success_fn(result_without)[0].item() == True

        # Check if the second example became successful
        second_example_with_success = success_fn(result_with)[1].item()
        second_example_without_success = success_fn(result_without)[1].item()

        print(
            f"Second example successful (with early stopping): {second_example_with_success}"
        )
        print(
            f"Second example successful (without early stopping): {second_example_without_success}"
        )

        # If the second example became successful, success rate should be 50%
        # If not, then we should modify our assertion accordingly
        expected_success_rate = 50.0 if second_example_with_success else 0.0
        assert metrics_with["success_rate"] == expected_success_rate

        expected_success_rate_without = 50.0 if second_example_without_success else 0.0
        assert metrics_without["success_rate"] == expected_success_rate_without

        # The optimizer with early stopping should compute significantly fewer gradient calls
        # since it doesn't process the first example after the first iteration
        assert metrics_with["gradient_calls"] < metrics_without["gradient_calls"]

        # Print detailed information when the test runs
        print(f"\nEarly stopping metrics:")
        print(f"- With early stopping: {metrics_with['gradient_calls']} gradient calls")
        print(
            f"- Without early stopping: {metrics_without['gradient_calls']} gradient calls"
        )
        print(
            f"- Ratio: {metrics_with['gradient_calls'] / metrics_without['gradient_calls']:.2f}"
        )

    def test_large_batch(self, device):
        """Test with a larger batch size to ensure scalability."""
        batch_size = 8  # Larger batch
        channels, height, width = 3, 4, 4

        torch.manual_seed(123)
        x_init = torch.rand((batch_size, channels, height, width), device=device)
        target = torch.ones_like(x_init) * 0.5

        def loss_fn(x):
            losses = torch.sum((x - target) ** 2, dim=(1, 2, 3))
            return losses

        def grad_fn(x):
            return 2 * (x - target)

        # Create optimizer
        opt = PGDOptimizer(
            n_iterations=10,
            alpha_type="constant",
            alpha_init=0.1,
            rand_init=False,
            maximize=False,  # Minimize MSE loss
        )

        # Run optimizer
        result, metrics = opt.optimize(x_init, grad_fn, loss_fn)

        # Check batch dimension is preserved
        assert result.shape[0] == batch_size

        # Check loss decreased for all examples
        initial_loss = loss_fn(x_init)
        final_loss = loss_fn(result)
        assert torch.all(final_loss < initial_loss)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Skipping integration test on CPU"
    )
    def test_network_integration(self, device):
        """Integration test with a simple neural network for adversarial attack."""
        # Skip if not on GPU to avoid slow tests
        if device != "cuda":
            pytest.skip("Skipping integration test on CPU")

        batch_size = 2
        channels, height, width = 3, 32, 32
        num_classes = 10

        # Create a simple ConvNet
        class SimpleConvNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(channels, 16, kernel_size=3, padding=1)
                self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.pool = torch.nn.MaxPool2d(2, 2)
                self.fc1 = torch.nn.Linear(32 * (height // 4) * (width // 4), 128)
                self.fc2 = torch.nn.Linear(128, num_classes)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        # Create model and move to device
        model = SimpleConvNet().to(device)
        model.eval()  # Set to evaluation mode

        # Create fake images and targets
        torch.manual_seed(42)
        images = torch.rand((batch_size, channels, height, width), device=device)
        true_labels = torch.randint(0, num_classes, (batch_size,), device=device)
        target_labels = (true_labels + 1) % num_classes  # Different from true labels

        # Define cross-entropy loss
        criterion = torch.nn.CrossEntropyLoss(reduction="none")

        # Define the adversarial loss function - untargeted attack (maximize loss for true label)
        def adversarial_loss(x):
            logits = model(x)
            # For untargeted attack, maximize loss for true label
            return -criterion(logits, true_labels)  # Negative because we'll maximize

        # Define the gradient function
        def adversarial_grad(x):
            x_tensor = x.clone().detach().requires_grad_(True)
            logits = model(x_tensor)
            # For untargeted attack, maximize loss for true label
            loss = -criterion(logits, true_labels)  # Negative because we'll maximize
            grad = torch.autograd.grad(loss.sum(), x_tensor)[0]
            return grad

        # Define success function (check if prediction is different from true label)
        def success_fn(x):
            logits = model(x)
            pred = logits.argmax(dim=1)
            # Success if prediction is not the true label
            return pred != true_labels

        # Create optimizer for adversarial attack
        opt = PGDOptimizer(
            norm="L2",
            eps=5.0,  # Larger epsilon for easier success
            n_iterations=100,  # More iterations
            alpha_type="constant",
            alpha_init=0.2,  # Larger step size
            rand_init=True,
            early_stopping=True,
            maximize=True,  # Maximize loss (untargeted attack)
        )

        # Run optimization to generate adversarial examples
        adv_images, metrics = opt.optimize(
            images.clone(),
            adversarial_grad,
            adversarial_loss,
            success_fn=success_fn,
            x_original=images,
        )

        # Check that the adversarial examples are within epsilon bound
        diff = adv_images - images
        diff_flat = diff.reshape(batch_size, -1)
        l2_norms = torch.norm(diff_flat, dim=1)
        assert torch.all(l2_norms <= opt.eps * 1.01)  # Allow slight tolerance

        # Evaluate adversarial success
        with torch.no_grad():
            clean_logits = model(images)
            adv_logits = model(adv_images)

            clean_pred = clean_logits.argmax(dim=1)
            adv_pred = adv_logits.argmax(dim=1)

        # Check if at least one example was successfully attacked
        assert torch.any(adv_pred != true_labels)

        # Get initial success before attack
        with torch.no_grad():
            initial_pred = clean_logits.argmax(dim=1)
            initial_success = initial_pred != true_labels

        # Now we check only new successes (examples that weren't successful initially)
        new_success = (adv_pred != true_labels) & ~initial_success
        actual_new_success_rate = torch.mean(new_success.float()).item() * 100

        # The predicted success rate should match the actual new success rate
        assert abs(metrics["success_rate"] - actual_new_success_rate) < 1e-5
