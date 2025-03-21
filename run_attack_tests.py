#!/usr/bin/env python
"""Runner script for all adversarial attack tests."""

import os
import sys
import argparse
import subprocess
from datetime import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run adversarial attack tests")
    parser.add_argument(
        "--test-dir",
        type=str,
        default="tests/test_attacks",
        help="Directory containing the test files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_results",
        help="Directory to save test results",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "comparison", "targeted", "perturbation"],
        help="Which test to run",
    )
    parser.add_argument(
        "--eps", type=float, default=None, help="Override epsilon value for tests"
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Override target class for targeted attacks",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override number of samples to test",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override device (cuda or cpu)"
    )
    return parser.parse_args()


def run_test(test_file, output_dir, params=None):
    """
    Run a specific test with the given parameters.

    Args:
        test_file: The test file to run
        output_dir: Directory to save results
        params: Dictionary of parameters to override
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create a copy of the current environment
    env = os.environ.copy()

    # Add parameters as environment variables
    if params:
        for key, value in params.items():
            if value is not None:
                env[f"TEST_{key.upper()}"] = str(value)

    # Set the output directory as an environment variable
    env["TEST_OUTPUT_DIR"] = output_dir

    # Construct command
    cmd = ["pytest", test_file, "-v"]

    # Redirect output to file
    output_file = os.path.join(
        output_dir, f"{os.path.basename(test_file).replace('.py', '')}_log.txt"
    )

    print(f"Running {test_file}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output will be saved to {output_file}")
    print(
        f"Using environment variables: {', '.join([f'{k}={v}' for k, v in params.items() if v is not None])}"
    )

    # Run the command with modified environment
    with open(output_file, "w") as f:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)

    print(f"Test completed. Log saved to {output_file}")


def main():
    """Main function."""
    args = parse_args()

    # Create timestamp for the output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)

    # Get list of test files
    if args.test == "all":
        test_files = [
            os.path.join(args.test_dir, "test_attack_comparison.py"),
            os.path.join(args.test_dir, "test_perturbation_analysis.py"),
            os.path.join(args.test_dir, "test_targeted_attacks.py"),
        ]
    elif args.test == "comparison":
        test_files = [os.path.join(args.test_dir, "test_attack_comparison.py")]
    elif args.test == "targeted":
        test_files = [os.path.join(args.test_dir, "test_targeted_attacks.py")]
    elif args.test == "perturbation":
        test_files = [os.path.join(args.test_dir, "test_perturbation_analysis.py")]

    # Define parameters to override
    params = {
        "output_dir": output_dir,
        "eps": args.eps,
        "target_class": args.target_class,
        "num_samples": args.num_samples,
        "device": args.device,
    }

    # Run each test
    for test_file in test_files:
        if os.path.exists(test_file):
            run_test(test_file, output_dir, params)
        else:
            print(f"Test file not found: {test_file}")

    print("All tests completed.")


if __name__ == "__main__":
    main()


"""
USAGE EXAMPLES:

# Run all tests with default parameters
python run_attack_tests.py

# Run only targeted attack tests
python run_attack_tests.py --test targeted

# Run with custom parameters
python run_attack_tests.py --test comparison --eps 0.1 --num-samples 10

# Run on CPU explicitly
python run_attack_tests.py --device cpu

# Run targeted attacks with specific target class
python run_attack_tests.py --test targeted --target-class 10 --eps 0.2

The script sets environment variables that are read by the test files:
- TEST_OUTPUT_DIR: Directory to save results
- TEST_EPS: Epsilon value for perturbation constraints
- TEST_TARGET_CLASS: Target class for targeted attacks
- TEST_NUM_SAMPLES: Number of samples to test
- TEST_DEVICE: Device to use (cuda or cpu)

Results will be saved in a timestamped directory under the specified output directory.
Each test will produce:
- Visualization images of attacks
- Text files with metrics results
- Log files with test output
"""
