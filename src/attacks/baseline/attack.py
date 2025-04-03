"""Base class for all baseline attacks.

Code is adapted from https://github.com/Harry24k/adversarial-attacks-pytorch

NORMALIZATION WARNING:
When using these attacks with our framework:
1. Our model wrappers now expect ALREADY NORMALIZED inputs from ImageNetDataset
2. No additional normalization should be needed or applied
3. Perturbations are applied in the normalized space [-2.64, 2.64]
4. For reporting and visualization, we convert back to [0,1] space
"""

import time
from collections import OrderedDict
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from skimage.metrics import structural_similarity

    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("Warning: scikit-image not available. SSIM calculation will be disabled.")

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def wrapper_method(func):
    """Decorator that applies a method to all sub-attacks in the attack chain.
    This allows composite attacks to propagate method calls to their sub-attacks."""

    def wrapper_func(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        for atk in self.__dict__.get("_attacks").values():
            eval("atk." + func.__name__ + "(*args, **kwargs)")
        return result

    return wrapper_func


class Attack(object):
    r"""
    Base class for all attacks.

    .. note::
        It automatically set device to the device where given model is.
        It basically changes training mode to eval during attack process.
        To change this, please see `set_model_training_mode`.

    NORMALIZATION EXPECTATIONS:
    - The attack methods work with normalized image inputs (ImageNet normalization)
    - This means inputs are in range [-2.64, 2.64] with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    - Perturbations are computed and applied in this normalized space
    - When clamping or visualizing, we denormalize to [0,1] range
    """

    def __init__(self, name, model):
        r"""
        Initializes internal attack state.

        Arguments:
            name (str): name of attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name
        # Dictionary to store sub-attacks for composite attacks
        self._attacks = OrderedDict()

        self.set_model(model)
        # Automatically detect and set the device (CPU/GPU) where the model is
        try:
            self.device = next(model.parameters()).device
            # Get the model's dtype for consistency
            self.model_dtype = next(model.parameters()).dtype
        except Exception:
            self.device = None
            self.model_dtype = torch.float32  # Default to float32
            print("Failed to set device automatically, please try set_device() manual.")

        # Attack mode configuration
        self.attack_mode = "default"  # Can be "default" or "targeted"
        self.supported_mode = ["default"]
        self.targeted = False  # Flag for targeted vs untargeted attacks
        self._target_map_function = (
            None  # Function to map input labels to target labels
        )

        # Normalization settings for input preprocessing
        # For our current framework, this should usually be None since
        # inputs are already normalized and model wrappers expect normalized inputs
        self.normalization_used = None
        self._normalization_applied = None

        # Store ImageNet normalization parameters for conversion
        # CRITICAL: Ensure these tensors use the same float type as the model
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=self.model_dtype).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=self.model_dtype).view(1, 3, 1, 1)

        # Model mode control during attack
        self._model_training = False
        self._batchnorm_training = False
        self._dropout_training = False

        # Metrics tracking for paper evaluation
        self.reset_metrics()

    def reset_metrics(self):
        """Reset all tracked metrics for a new evaluation."""
        self.total_iterations = 0
        self.total_gradient_calls = 0
        self.start_time = None
        self.end_time = None
        self.total_time = 0
        self.attack_success_count = 0
        self.total_samples = 0
        self.l2_norms = []
        self.linf_norms = []
        self.ssim_values = []

    def denormalize(self, images):
        """
        Convert normalized images back to [0,1] range for visualization
        and accurate perturbation magnitude reporting.

        Args:
            images: Normalized input images (with ImageNet normalization)

        Returns:
            Images in [0,1] range
        """
        # Move normalization tensors to the correct device and ensure correct dtype
        mean = self.mean.to(device=images.device, dtype=images.dtype)
        std = self.std.to(device=images.device, dtype=images.dtype)

        # Denormalize: x * std + mean
        denorm_images = images * std + mean

        # Clamp to [0,1] range to handle any out-of-bounds values
        denorm_images = torch.clamp(denorm_images, 0, 1)

        return denorm_images

    def normalize(self, images):
        """
        Convert images from [0,1] range to normalized range with ImageNet stats.

        Args:
            images: Images in [0,1] range

        Returns:
            Normalized images (with ImageNet normalization)
        """
        # Move normalization tensors to the correct device and ensure correct dtype
        mean = self.mean.to(device=images.device, dtype=images.dtype)
        std = self.std.to(device=images.device, dtype=images.dtype)

        # Normalize: (x - mean) / std
        norm_images = (images - mean) / std

        return norm_images

    def get_metrics(self):
        """Return dictionary of metrics for paper evaluation."""
        # Handle case where no samples were evaluated
        if self.total_samples == 0:
            success_rate = 0
            avg_iterations = 0
            avg_grad_calls = 0
            avg_time = 0
            avg_l2 = 0.0
            avg_linf = 0.0
            avg_ssim = 1.0
        else:
            success_rate = 100 * self.attack_success_count / self.total_samples
            avg_iterations = self.total_iterations / self.total_samples
            avg_grad_calls = self.total_gradient_calls / self.total_samples
            avg_time = self.total_time / self.total_samples
            avg_l2 = np.mean(self.l2_norms) if self.l2_norms else 0.0
            avg_linf = np.mean(self.linf_norms) if self.linf_norms else 0.0
            avg_ssim = np.mean(self.ssim_values) if self.ssim_values else 1.0

        metrics = {
            "attack_name": self.attack,
            "attack_mode": self.attack_mode,
            "model_name": self.model_name,
            "success_rate": success_rate,
            "iterations": avg_iterations,
            "gradient_calls": avg_grad_calls,
            "time_per_sample": avg_time,
            "l2_norm": avg_l2,
            "linf_norm": avg_linf,
            "ssim": avg_ssim,
        }
        return metrics

    def forward(self, inputs, labels=None, *args, **kwargs):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    @wrapper_method
    def set_model(self, model):
        """Set the target model and store its name."""
        self.model = model
        self.model_name = model.__class__.__name__

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        """Get model predictions (logits) for given inputs."""
        # Track gradient call
        self.total_gradient_calls += inputs.size(0)

        # Forward directly - normalization handled by model or dataset
        logits = self.model(inputs)
        return logits

    def increment_iteration(self):
        """Increment iteration counter.
        This should be called once per sample per main iteration."""
        self.total_iterations += 1

    def compute_perturbation_metrics(self, original_images, adversarial_images):
        """Compute perturbation metrics between original and adversarial images.

        Args:
            original_images: Clean input images (normalized)
            adversarial_images: Adversarial examples (normalized)

        Returns:
            dict: Dictionary with L2 norm, L-inf norm, and SSIM

        Note:
            For accurate reporting, we denormalize images to [0,1] range
            before computing perturbation metrics.
        """
        # Denormalize images for more intuitive metrics in [0,1] space
        original_denorm = self.denormalize(original_images)
        adversarial_denorm = self.denormalize(adversarial_images)

        # Calculate perturbation in denormalized space [0,1]
        perturbation = adversarial_denorm - original_denorm

        # L2 norm (Euclidean distance) - normalize by image dimensions
        # This gives the average per-pixel, per-channel distortion
        pixel_count = (
            original_images.size(1) * original_images.size(2) * original_images.size(3)
        )  # C * H * W
        l2_norm = torch.norm(
            perturbation.view(original_images.size(0), -1), p=2, dim=1
        ) / torch.sqrt(torch.tensor(pixel_count).float())
        l2_norm_mean = l2_norm.mean().item()
        self.l2_norms.extend(l2_norm.detach().cpu().numpy())

        # L-inf norm (maximum absolute pixel difference)
        linf_norm = torch.norm(
            perturbation.view(original_images.size(0), -1), p=float("inf"), dim=1
        )
        linf_norm_mean = linf_norm.mean().item()
        self.linf_norms.extend(linf_norm.detach().cpu().numpy())

        # SSIM (structural similarity) - computed in denormalized [0,1] space
        if SSIM_AVAILABLE:
            batch_size = original_images.size(0)
            ssim_vals = []

            # Process each image in the batch individually
            for i in range(batch_size):
                # Get individual images in denormalized [0,1] range
                orig_img = original_denorm[i].detach().cpu().permute(1, 2, 0).numpy()
                adv_img = adversarial_denorm[i].detach().cpu().permute(1, 2, 0).numpy()

                # Clamp images to [0, 1] range for SSIM calculation
                orig_img = np.clip(orig_img, 0, 1)
                adv_img = np.clip(adv_img, 0, 1)

                # Calculate SSIM using skimage
                try:
                    # Try newer scikit-image API (>=0.19.0) with channel_axis
                    ssim_val = structural_similarity(
                        orig_img, adv_img, channel_axis=2, data_range=1.0
                    )
                except TypeError:
                    # Fall back to older API (<0.19.0) with multichannel
                    ssim_val = structural_similarity(
                        orig_img, adv_img, multichannel=True, data_range=1.0
                    )

                # Ensure SSIM is in [0, 1] range
                ssim_val = max(0.0, min(1.0, ssim_val))
                ssim_vals.append(ssim_val)

            # Average SSIM across batch
            ssim_val = np.mean(ssim_vals)
            self.ssim_values.extend(ssim_vals)
        else:
            ssim_val = 1.0  # Default value when SSIM is not available

        return {"l2_norm": l2_norm_mean, "linf_norm": linf_norm_mean, "ssim": ssim_val}

    def evaluate_attack_success(self, original_images, adversarial_images, true_labels):
        """Evaluate if attack was successful (caused misclassification).

        Args:
            original_images: Clean input images (normalized)
            adversarial_images: Adversarial examples (normalized)
            true_labels: True class labels

        Returns:
            tuple: (success_rate, success_mask) where success_mask is a boolean tensor
                  indicating which samples were successfully attacked
        """
        with torch.no_grad():
            # Get original predictions
            orig_outputs = self.get_output_with_eval_nograd(original_images)
            orig_predictions = torch.argmax(orig_outputs, dim=1)

            # Get adversarial predictions
            adv_outputs = self.get_output_with_eval_nograd(adversarial_images)
            adv_predictions = torch.argmax(adv_outputs, dim=1)

            # Calculate success based on attack mode
            if self.targeted:
                target_labels = self.get_target_label(original_images, true_labels)
                success_mask = adv_predictions == target_labels
            else:
                success_mask = adv_predictions != true_labels

            success_count = success_mask.sum().item()
            total = success_mask.size(0)
            success_rate = 100 * success_count / total

            # Update attack metrics
            self.attack_success_count += success_count
            self.total_samples += total

            return success_rate, success_mask, (orig_predictions, adv_predictions)

    @wrapper_method
    def _set_normalization_applied(self, flag):
        """Track whether normalization has been applied to inputs."""
        self._normalization_applied = flag

    @wrapper_method
    def set_device(self, device):
        """Set the device (CPU/GPU) for the attack."""
        self.device = device

    @wrapper_method
    def _set_rmodel_normalization_used(self, model):
        r"""
        Set attack normalization for MAIR [https://github.com/Harry24k/MAIR].
        Extracts mean and std from the model if available.
        """
        mean = getattr(model, "mean", None)
        std = getattr(model, "std", None)
        if (mean is not None) and (std is not None):
            if isinstance(mean, torch.Tensor):
                mean = mean.cpu().numpy()
            if isinstance(std, torch.Tensor):
                std = std.cpu().numpy()
            if (mean != 0).all() or (std != 1).all():
                self.set_normalization_used(mean, std)

    @wrapper_method
    def set_normalization_used(self, mean, std):
        """
        Configure normalization parameters for input preprocessing.

        Note: For our current framework, this should not be needed since
        inputs are already normalized from the dataset and models expect
        normalized inputs directly.
        """
        self.normalization_used = {}
        n_channels = len(mean)
        # Reshape mean and std for broadcasting across image dimensions
        mean = torch.tensor(mean).reshape(1, n_channels, 1, 1)
        std = torch.tensor(std).reshape(1, n_channels, 1, 1)
        self.normalization_used["mean"] = mean
        self.normalization_used["std"] = std
        self._set_normalization_applied(True)

    def normalize(self, inputs):
        """
        Apply normalization to inputs using stored mean and std.

        Note: For our current framework, this should rarely be needed since
        inputs from the dataset are already normalized.
        """
        mean = self.normalization_used["mean"].to(inputs.device)
        std = self.normalization_used["std"].to(inputs.device)
        return (inputs - mean) / std

    def inverse_normalize(self, inputs):
        """
        Reverse normalization to get back to original input space.

        This is still needed when we want to visualize or save images
        for humans to view.
        """
        mean = self.normalization_used["mean"].to(inputs.device)
        std = self.normalization_used["std"].to(inputs.device)
        return inputs * std + mean

    def get_mode(self):
        r"""
        Get attack mode.
        """
        return self.attack_mode

    @wrapper_method
    def set_mode_default(self):
        r"""
        Set attack mode as default mode (untargeted attack).
        """
        self.attack_mode = "default"
        self.targeted = False
        print("Attack mode is changed to 'default.'")

    @wrapper_method
    def _set_mode_targeted(self, mode, quiet):
        """Set up targeted attack mode with specified configuration."""
        if "targeted" not in self.supported_mode:
            raise ValueError("Targeted mode is not supported.")
        self.targeted = True
        self.attack_mode = mode
        if not quiet:
            print("Attack mode is changed to '%s'." % mode)

    @wrapper_method
    def set_mode_targeted_by_function(self, target_map_function, quiet=False):
        r"""
        Set attack mode as targeted with custom label mapping function.

        Arguments:
            target_map_function (function): Label mapping function.
                e.g. lambda inputs, labels:(labels+1)%10.
                None for using input labels as targeted labels. (Default)
            quiet (bool): Display information message or not. (Default: False)
        """
        self._set_mode_targeted("targeted(custom)", quiet)
        self._target_map_function = target_map_function

    @wrapper_method
    def set_mode_targeted_random(self, quiet=False):
        r"""
        Set attack mode as targeted with random labels.

        Arguments:
            quiet (bool): Display information message or not. (Default: False)
        """
        self._set_mode_targeted("targeted(random)", quiet)
        self._target_map_function = self.get_random_target_label

    @wrapper_method
    def set_mode_targeted_least_likely(self, kth_min=1, quiet=False):
        r"""
        Set attack mode as targeted with least likely labels.

        Arguments:
            kth_min (str): label with the k-th smallest probability used as target labels. (Default: 1)
            num_classses (str): number of classes. (Default: False)
        """
        self._set_mode_targeted("targeted(least-likely)", quiet)
        assert kth_min > 0
        self._kth_min = kth_min
        self._target_map_function = self.get_least_likely_label

    @wrapper_method
    def set_mode_targeted_by_label(self, quiet=False):
        r"""
        Set attack mode as targeted using user-supplied labels.

        Arguments:
            quiet (bool): Display information message or not. (Default: False)
        """
        self._set_mode_targeted("targeted(label)", quiet)
        self._target_map_function = "function is a string"

    @wrapper_method
    def set_model_training_mode(
        self, model_training=False, batchnorm_training=False, dropout_training=False
    ):
        r"""
        Set training mode during attack process.

        Arguments:
            model_training (bool): True for using training mode for the entire model during attack process.
            batchnorm_training (bool): True for using training mode for batchnorms during attack process.
            dropout_training (bool): True for using training mode for dropouts during attack process.

        .. note::
            For RNN-based models, we cannot calculate gradients with eval mode.
            Thus, it should be changed to the training mode during the attack.
        """
        self._model_training = model_training
        self._batchnorm_training = batchnorm_training
        self._dropout_training = dropout_training

    @wrapper_method
    def _change_model_mode(self, given_training):
        """Change model mode based on training settings."""
        if self._model_training:
            self.model.train()
            for _, m in self.model.named_modules():
                if not self._batchnorm_training:
                    if "BatchNorm" in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if "Dropout" in m.__class__.__name__:
                        m = m.eval()
        else:
            self.model.eval()

    @wrapper_method
    def _recover_model_mode(self, given_training):
        """Restore original model training mode after attack."""
        if given_training:
            self.model.train()

    def evaluate_on_batch(self, inputs, labels, verbose=False):
        """Evaluate attack on a batch of samples and collect metrics.

        Args:
            inputs: Clean input images
            labels: True class labels
            verbose: Whether to print progress

        Returns:
            dict: Dictionary of metrics for the batch
        """
        # Reset iteration/gradient counters per batch
        if self.start_time is None:
            self.start_time = time.time()

        # Generate adversarial examples
        adv_inputs = self.__call__(inputs, labels)

        # Calculate metrics
        batch_size = inputs.size(0)

        # Evaluate attack success (misclassification)
        success_rate, success_mask, predictions = self.evaluate_attack_success(
            inputs, adv_inputs, labels
        )

        # Calculate perturbation metrics
        perturbation_metrics = self.compute_perturbation_metrics(
            inputs.to(self.device), adv_inputs
        )

        # Update timing information
        self.end_time = time.time()
        batch_time = self.end_time - self.start_time
        self.total_time += batch_time
        self.start_time = self.end_time

        # Compile batch results
        batch_metrics = {
            "batch_size": batch_size,
            "success_rate": success_rate,
            "time_taken": batch_time,
            **perturbation_metrics,
        }

        if verbose:
            print(
                f"Attack: {self.attack}, "
                f"Mode: {self.attack_mode}, "
                f"Success: {success_rate:.2f}%, "
                f"L2: {perturbation_metrics['l2_norm']:.4f}, "
                f"L∞: {perturbation_metrics['linf_norm']:.4f}, "
                f"SSIM: {perturbation_metrics['ssim']:.4f}, "
                f"Time: {batch_time:.3f}s"
            )

        return batch_metrics

    def evaluate(self, data_loader, verbose=True):
        """Evaluate attack on the entire dataset.

        Args:
            data_loader: DataLoader with clean samples
            verbose: Whether to print progress

        Returns:
            dict: Dictionary of overall metrics
        """
        # Reset metrics
        self.reset_metrics()
        all_batch_metrics = []

        total_batches = len(data_loader)

        for batch_idx, (inputs, labels) in enumerate(data_loader):
            # Evaluate on batch
            batch_metrics = self.evaluate_on_batch(
                inputs.to(self.device), labels.to(self.device), verbose=False
            )
            all_batch_metrics.append(batch_metrics)

            # Print progress
            if verbose and (batch_idx % max(total_batches // 10, 1) == 0):
                print(
                    f"Batch {batch_idx+1}/{total_batches} - "
                    f"Success: {batch_metrics['success_rate']:.2f}%, "
                    f"L2: {batch_metrics['l2_norm']:.4f}, "
                    f"L∞: {batch_metrics['linf_norm']:.4f}"
                )

        # Compile overall results
        overall_metrics = self.get_metrics()

        if verbose:
            print("\nOverall Attack Results:")
            print(
                f"Attack: {self.attack}, Mode: {self.attack_mode}, Model: {self.model_name}"
            )
            print(f"Success Rate: {overall_metrics['success_rate']:.2f}%")
            print(f"Average Iterations: {overall_metrics['iterations']:.2f}")
            print(f"Average Gradient Calls: {overall_metrics['gradient_calls']:.2f}")
            print(f"Average Time: {overall_metrics['time_per_sample']:.4f}s")
            print(f"Average L2 Norm: {overall_metrics['l2_norm']:.4f}")
            print(f"Average L∞ Norm: {overall_metrics['linf_norm']:.4f}")
            print(f"Average SSIM: {overall_metrics['ssim']:.4f}")

        return overall_metrics

    def save(
        self,
        data_loader,
        save_path=None,
        verbose=True,
        return_verbose=False,
        save_predictions=False,
        save_clean_inputs=False,
        save_type="float",
    ):
        r"""
        Save adversarial inputs as torch.tensor from given torch.utils.data.DataLoader.

        Arguments:
            save_path (str): save_path.
            data_loader (torch.utils.data.DataLoader): data loader.
            verbose (bool): True for displaying detailed information. (Default: True)
            return_verbose (bool): True for returning detailed information. (Default: False)
            save_predictions (bool): True for saving predicted labels (Default: False)
            save_clean_inputs (bool): True for saving clean inputs (Default: False)

        """
        if save_path is not None:
            # Initialize lists to store results
            adv_input_list = []
            label_list = []
            if save_predictions:
                pred_list = []
            if save_clean_inputs:
                input_list = []

        correct = 0
        total = 0
        l2_distance = []

        total_batch = len(data_loader)
        given_training = self.model.training

        # Process each batch in the data loader
        for step, (inputs, labels) in enumerate(data_loader):
            start = time.time()
            # Generate adversarial examples
            adv_inputs = self.__call__(inputs, labels)
            batch_size = len(inputs)

            if verbose or return_verbose:
                with torch.no_grad():
                    outputs = self.get_output_with_eval_nograd(adv_inputs)

                    # Calculate robust accuracy
                    _, pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = pred == labels.to(self.device)
                    correct += right_idx.sum()
                    rob_acc = 100 * float(correct) / total

                    # Calculate l2 distance between clean and adversarial inputs
                    delta = (adv_inputs - inputs.to(self.device)).view(
                        batch_size, -1
                    )  # nopep8
                    l2_distance.append(
                        torch.norm(delta[~right_idx], p=2, dim=1)
                    )  # nopep8
                    l2 = torch.cat(l2_distance).mean().item()

                    # Calculate time computation
                    progress = (step + 1) / total_batch * 100
                    end = time.time()
                    elapsed_time = end - start

                    if verbose:
                        self._save_print(
                            progress, rob_acc, l2, elapsed_time, end="\r"
                        )  # nopep8

            if save_path is not None:
                # Store results
                adv_input_list.append(adv_inputs.detach().cpu())
                label_list.append(labels.detach().cpu())

                adv_input_list_cat = torch.cat(adv_input_list, 0)
                label_list_cat = torch.cat(label_list, 0)

                save_dict = {
                    "adv_inputs": adv_input_list_cat,
                    "labels": label_list_cat,
                }  # nopep8

                if save_predictions:
                    pred_list.append(pred.detach().cpu())
                    pred_list_cat = torch.cat(pred_list, 0)
                    save_dict["preds"] = pred_list_cat

                if save_clean_inputs:
                    input_list.append(inputs.detach().cpu())
                    input_list_cat = torch.cat(input_list, 0)
                    save_dict["clean_inputs"] = input_list_cat

                # Handle normalization if needed
                if self.normalization_used is not None:
                    save_dict["adv_inputs"] = self.inverse_normalize(
                        save_dict["adv_inputs"]
                    )  # nopep8
                    if save_clean_inputs:
                        save_dict["clean_inputs"] = self.inverse_normalize(
                            save_dict["clean_inputs"]
                        )  # nopep8

                # Convert to integer type if requested
                if save_type == "int":
                    save_dict["adv_inputs"] = self.to_type(
                        save_dict["adv_inputs"], "int"
                    )  # nopep8
                    if save_clean_inputs:
                        save_dict["clean_inputs"] = self.to_type(
                            save_dict["clean_inputs"], "int"
                        )  # nopep8

                save_dict["save_type"] = save_type
                torch.save(save_dict, save_path)

        # To avoid erasing the printed information.
        if verbose:
            self._save_print(progress, rob_acc, l2, elapsed_time, end="\n")

        if given_training:
            self.model.train()

        if return_verbose:
            return rob_acc, l2, elapsed_time

    @staticmethod
    def to_type(inputs, type):
        r"""
        Convert inputs between float and int types.
        """
        if type == "int":
            if isinstance(inputs, torch.FloatTensor) or isinstance(
                inputs, torch.cuda.FloatTensor
            ):
                return (inputs * 255).type(torch.uint8)
        elif type == "float":
            if isinstance(inputs, torch.ByteTensor) or isinstance(
                inputs, torch.cuda.ByteTensor
            ):
                return inputs.float() / 255
        else:
            raise ValueError(type + " is not a valid type. [Options: float, int]")
        return inputs

    @staticmethod
    def _save_print(progress, rob_acc, l2, elapsed_time, end):
        """Print progress information during saving."""
        print(
            "- Save progress: %2.2f %% / Robust accuracy: %2.2f %% / L2: %1.5f (%2.3f it/s) \t"
            % (progress, rob_acc, l2, elapsed_time),
            end=end,
        )

    @staticmethod
    def load(
        load_path,
        batch_size=128,
        shuffle=False,
        normalize=None,
        load_predictions=False,
        load_clean_inputs=False,
    ):
        """Load saved adversarial examples and create a DataLoader."""
        save_dict = torch.load(load_path)
        keys = ["adv_inputs", "labels"]

        if load_predictions:
            keys.append("preds")
        if load_clean_inputs:
            keys.append("clean_inputs")

        # Convert from int to float if needed
        if save_dict["save_type"] == "int":
            save_dict["adv_inputs"] = save_dict["adv_inputs"].float() / 255
            if load_clean_inputs:
                save_dict["clean_inputs"] = (
                    save_dict["clean_inputs"].float() / 255
                )  # nopep8

        # Apply normalization if specified
        if normalize is not None:
            n_channels = len(normalize["mean"])
            mean = torch.tensor(normalize["mean"]).reshape(1, n_channels, 1, 1)
            std = torch.tensor(normalize["std"]).reshape(1, n_channels, 1, 1)
            save_dict["adv_inputs"] = (save_dict["adv_inputs"] - mean) / std
            if load_clean_inputs:
                save_dict["clean_inputs"] = (
                    save_dict["clean_inputs"] - mean
                ) / std  # nopep8

        # Create DataLoader from saved data
        adv_data = TensorDataset(*[save_dict[key] for key in keys])
        adv_loader = DataLoader(adv_data, batch_size=batch_size, shuffle=shuffle)
        print(
            "Data is loaded in the following order: [%s]" % (", ".join(keys))
        )  # nopep8
        return adv_loader

    @torch.no_grad()
    def get_output_with_eval_nograd(self, inputs):
        """Get model predictions without gradients and in eval mode."""
        given_training = self.model.training
        if given_training:
            self.model.eval()
        outputs = self.get_logits(inputs)
        if given_training:
            self.model.train()
        return outputs

    def get_target_label(self, inputs, labels=None):
        r"""
        Get target labels for targeted attacks.
        """
        if self._target_map_function is None:
            raise ValueError(
                "target_map_function is not initialized by set_mode_targeted."
            )
        if self.attack_mode == "targeted(label)":
            target_labels = labels
        else:
            target_labels = self._target_map_function(inputs, labels)
        return target_labels

    @torch.no_grad()
    def get_least_likely_label(self, inputs, labels=None):
        """Get the least likely predicted label as target for targeted attacks."""
        outputs = self.get_output_with_eval_nograd(inputs)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            _, t = torch.kthvalue(outputs[counter][l], self._kth_min)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    @torch.no_grad()
    def get_random_target_label(self, inputs, labels=None):
        """Get random labels as targets for targeted attacks."""
        outputs = self.get_output_with_eval_nograd(inputs)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            t = (len(l) * torch.rand([1])).long().to(self.device)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    def __call__(self, inputs, labels=None, *args, **kwargs):
        """Main entry point for the attack. Handles model mode."""
        given_training = self.model.training
        self._change_model_mode(given_training)

        # Our models now expect normalized inputs directly
        # Only apply normalization/inverse_normalization if explicitly set
        if self._normalization_applied is True and self.normalization_used is not None:
            inputs = self.inverse_normalize(inputs)
            self._set_normalization_applied(False)

            adv_inputs = self.forward(inputs, labels, *args, **kwargs)

            adv_inputs = self.normalize(adv_inputs)
            self._set_normalization_applied(True)
        else:
            adv_inputs = self.forward(inputs, labels, *args, **kwargs)

        self._recover_model_mode(given_training)

        return adv_inputs

    def __repr__(self):
        """String representation of the attack configuration."""
        info = self.__dict__.copy()

        del_keys = ["model", "attack", "supported_mode"]

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        info["attack_mode"] = self.attack_mode
        info["normalization_used"] = (
            True if self.normalization_used is not None else False
        )

        return (
            self.attack
            + "("
            + ", ".join("{}={}".format(key, val) for key, val in info.items())
            + ")"
        )

    def __setattr__(self, name, value):
        """Handle attribute setting for composite attacks."""
        object.__setattr__(self, name, value)

        attacks = self.__dict__.get("_attacks")

        # Get all items in iterable items.
        def get_all_values(items, stack=[]):
            if items not in stack:
                stack.append(items)
                if isinstance(items, list) or isinstance(items, dict):
                    if isinstance(items, dict):
                        items = list(items.keys()) + list(items.values())
                    for item in items:
                        yield from get_all_values(item, stack)
                else:
                    if isinstance(items, Attack):
                        yield items
            else:
                if isinstance(items, Attack):
                    yield items

        for num, value in enumerate(get_all_values(value)):
            attacks[name + "." + str(num)] = value
            for subname, subvalue in value.__dict__.get("_attacks").items():
                attacks[name + "." + subname] = subvalue
