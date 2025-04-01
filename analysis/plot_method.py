#!/usr/bin/env python3
"""
Visualization tool for adversarial attack method evaluation results.

This script creates professional publication-quality plots from the results
of adversarial attack evaluations. It supports visualizing data from both CSV and JSON
result files, and can generate comparison plots across different attack methods.

Usage:
    python plot_method.py --type success_rates --methods FGSM FFGSM --norm Linf
    python plot_method.py --type perturbation --method FGSM --norm L2
    python plot_method.py --type compare_all --methods FGSM FFGSM DeepFool CW
"""

import os
import re
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import ScalarFormatter
from matplotlib.gridspec import GridSpec

# Set seaborn style for professional publication-quality plots
sns.set_theme(style="whitegrid", context="paper", 
              rc={"font.family": "serif", "font.serif": ["Times New Roman"],
                  "text.usetex": False, "axes.grid": True, "grid.linestyle": ":",
                  "grid.linewidth": 0.5, "grid.alpha": 0.7})

# Define color palettes for consistent colors across plots
METHOD_COLORS = {
    "FGSM": "#1f77b4",    # blue
    "FFGSM": "#ff7f0e",   # orange
    "DeepFool": "#2ca02c", # green
    "CW": "#d62728",      # red
    "PGD": "#9467bd",     # purple
    "CG": "#8c564b",      # brown
    "L-BFGS": "#e377c2"   # pink
}

MODEL_COLORS = {
    "resnet18": "#1f77b4",
    "resnet50": "#ff7f0e",
    "vgg16": "#2ca02c", 
    "efficientnet": "#d62728",
    "mobilenet": "#9467bd"
}

# Epsilon symbol for plot labels
EPSILON = "ε"


def load_results(methods=None, norm_types=None, targeted=False):
    """
    Load results from JSON and CSV files in the results directory.
    
    Args:
        methods: List of method names to load (e.g., ["FGSM", "FFGSM"])
        norm_types: List of norm types to load (e.g., ["Linf", "L2"])
        targeted: Boolean indicating whether to load targeted or untargeted attack results
    
    Returns:
        tuple: (csv_data, json_data) - DataFrames with raw data and processed tables
    """
    results_dir = Path("results")
    
    # Default to all methods and norm types if not specified
    if methods is None:
        methods = ["FGSM", "FFGSM", "DeepFool", "CW"]
    if norm_types is None:
        norm_types = ["Linf", "L2"]
    
    target_type = "targeted" if targeted else "untargeted"
    
    print(f"Searching for results for methods: {methods}")
    print(f"Norm types: {norm_types}")
    print(f"Target type: {target_type}")
    
    # Find and load CSV files
    csv_files = []
    for method in methods:
        for norm in norm_types:
            pattern = f"{method}_{norm}_{target_type}_*.csv"
            print(f"Looking for CSV files matching: {pattern}")
            matched_files = list(results_dir.glob(pattern))
            if matched_files:
                # Sort by date (newest first) and take the latest
                matched_files.sort(reverse=True)
                csv_files.append(matched_files[0])
                print(f"Found CSV file: {matched_files[0]}")
            else:
                print(f"No CSV files found matching: {pattern}")
    
    # Find and load JSON files
    json_files = []
    for method in methods:
        for norm in norm_types:
            pattern = f"{method}_{norm}_tables_{target_type}_*.json"
            print(f"Looking for JSON files matching: {pattern}")
            matched_files = list(results_dir.glob(pattern))
            if matched_files:
                # Sort by date (newest first) and take the latest
                matched_files.sort(reverse=True)
                json_files.append(matched_files[0])
                print(f"Found JSON file: {matched_files[0]}")
            else:
                print(f"No JSON files found matching: {pattern}")
    
    # Load CSV data
    csv_data = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Extract method and norm from filename
            file_str = str(file)
            if "FGSM_Linf" in file_str:
                method, norm = "FGSM", "Linf"
            elif "FGSM_L2" in file_str:
                method, norm = "FGSM", "L2"
            elif "FFGSM_Linf" in file_str:
                method, norm = "FFGSM", "Linf"
            elif "FFGSM_L2" in file_str:
                method, norm = "FFGSM", "L2"
            elif "DeepFool_Linf" in file_str:
                method, norm = "DeepFool", "Linf"
            elif "DeepFool_L2" in file_str:
                method, norm = "DeepFool", "L2"
            elif "CW_Linf" in file_str:
                method, norm = "CW", "Linf"
            elif "CW_L2" in file_str:
                method, norm = "CW", "L2"
            else:
                # Try generic pattern matching as fallback
                match = re.search(r"(.*?)_(Linf|L2)_.*?\.csv", file_str)
                if match:
                    method, norm = match.groups()
                else:
                    print(f"Warning: Could not extract method and norm from {file}")
                    continue
                    
            # Add method and norm_type columns
            df["method"] = method
            df["norm_type"] = norm
            csv_data.append(df)
            print(f"Loaded {len(df)} rows from {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if csv_data:
        csv_data = pd.concat(csv_data, ignore_index=True)
        print(f"Total CSV data loaded: {len(csv_data)} rows")
    else:
        csv_data = pd.DataFrame()
        print("No CSV data loaded")
    
    # Load JSON data
    json_data = {}
    for file in json_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            # Extract method and norm from filename
            file_str = str(file)
            if "FGSM_Linf" in file_str:
                method, norm = "FGSM", "Linf"
            elif "FGSM_L2" in file_str:
                method, norm = "FGSM", "L2"
            elif "FFGSM_Linf" in file_str:
                method, norm = "FFGSM", "Linf"
            elif "FFGSM_L2" in file_str:
                method, norm = "FFGSM", "L2"
            elif "DeepFool_Linf" in file_str:
                method, norm = "DeepFool", "Linf"
            elif "DeepFool_L2" in file_str:
                method, norm = "DeepFool", "L2"
            elif "CW_Linf" in file_str:
                method, norm = "CW", "Linf"
            elif "CW_L2" in file_str:
                method, norm = "CW", "L2"
            else:
                # Try generic pattern matching as fallback
                match = re.search(r"(.*?)_(Linf|L2)_tables_.*?\.json", file_str)
                if match:
                    method, norm = match.groups()
                else:
                    print(f"Warning: Could not extract method and norm from {file}")
                    continue
                    
            json_data[(method, norm)] = data
            print(f"Loaded JSON data from {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    print(f"Methods found in JSON data: {[k[0] for k in json_data.keys()]}")
    
    return csv_data, json_data


def plot_success_rates(json_data, methods, norm_type, output_dir="plots"):
    """
    Plot success rates across different models and epsilon values.
    
    Args:
        json_data: Dictionary of JSON data indexed by (method, norm)
        methods: List of methods to plot
        norm_type: Norm type to plot ("Linf" or "L2")
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Determine all epsilon values and models across methods
    all_eps = set()
    all_models = set()
    
    # Check if we have any data for the specified methods and norm
    available_methods = [method for method, norm in json_data.keys() if norm == norm_type]
    print(f"Available methods for {norm_type} norm: {available_methods}")
    
    # Filter methods to only include those we have data for
    valid_methods = [m for m in methods if m in available_methods]
    print(f"Using valid methods: {valid_methods}")
    
    if not valid_methods:
        print(f"No valid methods found for {norm_type} norm")
        return
    
    for method in valid_methods:
        if (method, norm_type) in json_data:
            table1 = json_data[(method, norm_type)]["table1"]
            for model, eps_dict in table1.items():
                all_models.add(model)
                all_eps.update(eps_dict.keys())
    
    all_eps = sorted([float(eps) for eps in all_eps])
    all_models = sorted(list(all_models))
    
    print(f"Found epsilon values: {all_eps}")
    print(f"Found models: {all_models}")
    
    # Track if any plots were created
    plot_added = False
    
    # Plot success rates for each method
    for method in valid_methods:
        if (method, norm_type) in json_data:
            table1 = json_data[(method, norm_type)]["table1"]
            
            # Plot success rate for each model
            for model in all_models:
                if model in table1:
                    eps_values = []
                    success_rates = []
                    
                    for eps in all_eps:
                        eps_str = str(eps)
                        if eps_str in table1[model]:
                            eps_values.append(eps)
                            success_rates.append(table1[model][eps_str])
                    
                    if eps_values:
                        label = f"{method} - {model}"
                        plt.plot(eps_values, success_rates, 'o-', 
                                 label=label,
                                 color=METHOD_COLORS.get(method, None),
                                 linestyle=['-', '--', ':', '-.'][all_models.index(model) % 4],
                                 linewidth=2, markersize=8)
                        print(f"Added plot for {label}")
                        plot_added = True
    
    if not plot_added:
        print("No plots created, skipping legend and save")
        return
    
    plt.xlabel(f"{EPSILON} ({norm_type} norm)", fontsize=12)
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.title(f"Attack Success Rates by Model ({norm_type} norm)", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Only add legend if we have plots
    if plot_added:
        plt.legend(loc='lower right', fontsize=10, frameon=True, framealpha=0.9)
    
    plt.ylim(0, 105)  # Leave room for error bars in the future
    
    if norm_type == "Linf":
        plt.xscale('log')
        plt.xticks(all_eps)
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/success_rates_{norm_type}.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_dir}/success_rates_{norm_type}.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved success_rates_{norm_type} plot")


def plot_perturbation_metrics(json_data, methods, norm_type, output_dir="plots"):
    """
    Plot perturbation metrics (L2 norm, Linf norm, SSIM) vs epsilon.
    
    Args:
        json_data: Dictionary of JSON data indexed by (method, norm)
        methods: List of methods to plot
        norm_type: Norm type to plot ("Linf" or "L2")
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ["l2_norm", "linf_norm", "ssim"]
    titles = ["L2 Norm", "L∞ Norm", "SSIM"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Determine all epsilon values across methods
    all_eps = set()
    
    for method in methods:
        if (method, norm_type) in json_data:
            table2 = json_data[(method, norm_type)]["table2"]
            all_eps.update(table2.keys())
    
    all_eps = sorted([float(eps) for eps in all_eps])
    
    # Plot metrics for each method
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        for method in methods:
            if (method, norm_type) in json_data:
                table2 = json_data[(method, norm_type)]["table2"]
                
                eps_values = []
                metric_values = []
                
                for eps in all_eps:
                    eps_str = str(eps)
                    if eps_str in table2:
                        eps_values.append(eps)
                        metric_values.append(table2[eps_str][metric])
                
                if eps_values:
                    ax.plot(eps_values, metric_values, 'o-', 
                            label=method,
                            color=METHOD_COLORS.get(method, None),
                            linewidth=2, markersize=8)
        
        ax.set_xlabel(f"{EPSILON} ({norm_type} norm)", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        
        if norm_type == "Linf" and metric != "ssim":
            ax.set_xscale('log')
            ax.set_xticks(all_eps)
            ax.xaxis.set_major_formatter(ScalarFormatter())
        
        if metric == "ssim":
            ax.set_ylim(0.85, 1.005)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), 
               ncol=len(methods), fontsize=12, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(f"{output_dir}/perturbation_metrics_{norm_type}.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_dir}/perturbation_metrics_{norm_type}.png", bbox_inches='tight', dpi=300)
    plt.close()


def plot_computational_metrics(json_data, methods, norm_type, output_dir="plots"):
    """
    Plot computational metrics (runtime, iterations, gradient calls) vs epsilon.
    
    Args:
        json_data: Dictionary of JSON data indexed by (method, norm)
        methods: List of methods to plot
        norm_type: Norm type to plot ("Linf" or "L2")
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ["runtime", "iterations", "gradient_calls"]
    titles = ["Runtime (s)", "Iterations", "Gradient Calls"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Determine all epsilon values across methods
    all_eps = set()
    
    for method in methods:
        if (method, norm_type) in json_data:
            table3 = json_data[(method, norm_type)]["table3"]
            all_eps.update(table3.keys())
    
    all_eps = sorted([float(eps) for eps in all_eps])
    
    # Plot metrics for each method
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        for method in methods:
            if (method, norm_type) in json_data:
                table3 = json_data[(method, norm_type)]["table3"]
                
                eps_values = []
                metric_values = []
                
                for eps in all_eps:
                    eps_str = str(eps)
                    if eps_str in table3:
                        eps_values.append(eps)
                        metric_values.append(table3[eps_str][metric])
                
                if eps_values:
                    ax.plot(eps_values, metric_values, 'o-', 
                            label=method,
                            color=METHOD_COLORS.get(method, None),
                            linewidth=2, markersize=8)
        
        ax.set_xlabel(f"{EPSILON} ({norm_type} norm)", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        
        if norm_type == "Linf":
            ax.set_xscale('log')
            ax.set_xticks(all_eps)
            ax.xaxis.set_major_formatter(ScalarFormatter())
        
        if metric in ["iterations", "gradient_calls"]:
            ax.set_yscale('log')
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), 
               ncol=len(methods), fontsize=12, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(f"{output_dir}/computational_metrics_{norm_type}.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_dir}/computational_metrics_{norm_type}.png", bbox_inches='tight', dpi=300)
    plt.close()


def plot_method_comparison(json_data, methods, norm_type, output_dir="plots"):
    """
    Create a comprehensive comparison plot showing key metrics across methods.
    
    Args:
        json_data: Dictionary of JSON data indexed by (method, norm)
        methods: List of methods to plot
        norm_type: Norm type to plot ("Linf" or "L2")
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a specific epsilon value for comparison
    # For Linf, use 0.03, for L2 use 1.0
    target_eps = "0.03" if norm_type == "Linf" else "1.0"
    
    # Create a figure with a complex layout
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Success rate by model
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Prepare data for plotting
    models = []
    method_success_rates = {method: [] for method in methods}
    
    for method in methods:
        if (method, norm_type) in json_data:
            table1 = json_data[(method, norm_type)]["table1"]
            
            for model, eps_dict in sorted(table1.items()):
                if model not in models:
                    models.append(model)
                if target_eps in eps_dict:
                    method_success_rates[method].append(eps_dict[target_eps])
                else:
                    method_success_rates[method].append(0)
    
    # Plot success rates by model
    x = np.arange(len(models))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        if method_success_rates[method]:
            ax1.bar(x + (i - len(methods)/2 + 0.5) * width, 
                    method_success_rates[method], 
                    width, label=method, 
                    color=METHOD_COLORS.get(method, None))
    
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title(f'Success Rate by Model ({EPSILON}={target_eps})', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylim(0, 105)
    
    # Perturbation metrics
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Prepare data for perturbation metrics
    metrics = ["l2_norm", "linf_norm", "ssim"]
    method_metrics = {method: {metric: 0 for metric in metrics} for method in methods}
    
    for method in methods:
        if (method, norm_type) in json_data:
            table2 = json_data[(method, norm_type)]["table2"]
            
            if target_eps in table2:
                for metric in metrics:
                    method_metrics[method][metric] = table2[target_eps][metric]
    
    # Normalize metrics for better visualization
    max_l2 = max([method_metrics[method]["l2_norm"] for method in methods if method_metrics[method]["l2_norm"] > 0], default=1)
    max_linf = max([method_metrics[method]["linf_norm"] for method in methods if method_metrics[method]["linf_norm"] > 0], default=1)
    
    for method in methods:
        if method_metrics[method]["l2_norm"] > 0:
            method_metrics[method]["l2_norm_norm"] = method_metrics[method]["l2_norm"] / max_l2
        else:
            method_metrics[method]["l2_norm_norm"] = 0
            
        if method_metrics[method]["linf_norm"] > 0:
            method_metrics[method]["linf_norm_norm"] = method_metrics[method]["linf_norm"] / max_linf
        else:
            method_metrics[method]["linf_norm_norm"] = 0
            
        # SSIM is already normalized but invert it (1-SSIM) so smaller is better
        method_metrics[method]["ssim_norm"] = 1 - method_metrics[method]["ssim"]
    
    # Create a radar plot for perturbation metrics
    metrics_for_radar = ["l2_norm_norm", "linf_norm_norm", "ssim_norm"]
    metrics_labels = ["L2 Norm", "L∞ Norm", "1-SSIM"]
    
    angles = np.linspace(0, 2*np.pi, len(metrics_for_radar), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    for method in methods:
        values = [method_metrics[method][m] for m in metrics_for_radar]
        values += values[:1]  # Close the polygon
        
        ax2.plot(angles, values, 'o-', 
                 label=method, 
                 color=METHOD_COLORS.get(method, None),
                 linewidth=2, markersize=8)
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics_labels)
    ax2.set_ylim(0, 1.1)
    ax2.set_title(f'Perturbation Metrics ({EPSILON}={target_eps})', fontsize=14)
    
    # Computational metrics
    ax3 = fig.add_subplot(gs[1, :])
    
    # Prepare data for computational metrics
    comp_metrics = ["runtime", "iterations", "gradient_calls"]
    labels = ["Runtime (s)", "Iterations", "Gradient Calls"]
    
    x = np.arange(len(labels))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        if (method, norm_type) in json_data:
            table3 = json_data[(method, norm_type)]["table3"]
            
            if target_eps in table3:
                values = [table3[target_eps][m] for m in comp_metrics]
                ax3.bar(x + (i - len(methods)/2 + 0.5) * width, 
                       values, width, label=method,
                       color=METHOD_COLORS.get(method, None))
    
    ax3.set_ylabel('Value (log scale)', fontsize=12)
    ax3.set_title(f'Computational Metrics ({EPSILON}={target_eps})', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_yscale('log')
    
    # Add a single legend for the entire figure
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), 
               ncol=len(methods), fontsize=12, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(f"{output_dir}/method_comparison_{norm_type}.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_dir}/method_comparison_{norm_type}.png", bbox_inches='tight', dpi=300)
    plt.close()


def plot_perturbation_visualizations(csv_data, methods, norm_type, output_dir="plots"):
    """
    Create violin plots showing the distribution of perturbation metrics.
    
    Args:
        csv_data: DataFrame with raw CSV data
        methods: List of methods to plot
        norm_type: Norm type to plot ("Linf" or "L2")
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if csv_data.empty:
        print("No CSV data available for perturbation visualizations")
        return
    
    # Print available methods and norms for debugging
    available_methods = csv_data['method'].unique()
    available_norms = csv_data['norm_type'].unique()
    print(f"Available methods in CSV data: {available_methods}")
    print(f"Available norms in CSV data: {available_norms}")
    
    # Filter data for specified methods and norm type
    print(f"Filtering for methods {methods} with norm {norm_type}")
    
    # Create separate filters to debug which one might be failing
    method_filter = csv_data['method'].isin(methods)
    norm_filter = csv_data['norm_type'] == norm_type
    
    print(f"Method filter matches: {method_filter.sum()} rows")
    print(f"Norm filter matches: {norm_filter.sum()} rows")
    
    # Apply both filters
    data = csv_data[method_filter & norm_filter]
    
    print(f"Combined filter matches: {len(data)} rows")
    
    if data.empty:
        print(f"No data available for methods {methods} with norm {norm_type}")
        return
    
    # Create violin plots for L2 norm, Linf norm, and SSIM
    metrics = ["l2_norm", "linf_norm", "ssim"]
    titles = ["L2 Norm Distribution", "L∞ Norm Distribution", "SSIM Distribution"]
    
    for metric, title in zip(metrics, titles):
        plt.figure(figsize=(10, 6))
        
        # Create a list of unique epsilon values
        eps_values = sorted(data['epsilon'].unique())
        print(f"Found epsilon values: {eps_values}")
        
        # Group data by method and epsilon
        grouped_data = []
        labels = []
        
        for method in methods:
            if method not in data['method'].unique():
                print(f"Method {method} not found in filtered data")
                continue
                
            method_data = data[data['method'] == method]
            print(f"Found {len(method_data)} rows for method {method}")
            
            for eps in eps_values:
                eps_data = method_data[method_data['epsilon'] == eps][metric]
                if not eps_data.empty:
                    grouped_data.append(eps_data)
                    labels.append(f"{method}\n{EPSILON}={eps}")
                    print(f"Added {len(eps_data)} samples for {method}, ε={eps}")
        
        if grouped_data:
            # Create violin plot
            violins = plt.violinplot(grouped_data, showmeans=True, showextrema=True)
            
            # Set colors for each method
            for i, violin in enumerate(violins['bodies']):
                method = labels[i].split('\n')[0]
                color = METHOD_COLORS.get(method, None)
                if color:
                    violin.set_facecolor(color)
                    violin.set_alpha(0.7)
            
            # Set x-axis labels
            plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
            
            plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
            plt.title(title, fontsize=14)
            plt.grid(True, alpha=0.3)
            
            if metric == "ssim":
                plt.ylim(0.85, 1.005)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{metric}_distribution_{norm_type}.pdf", bbox_inches='tight', dpi=300)
            plt.savefig(f"{output_dir}/{metric}_distribution_{norm_type}.png", bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved {metric} plot for {norm_type} norm")


def main():
    parser = argparse.ArgumentParser(description="Plot adversarial attack evaluation results")
    parser.add_argument("--type", type=str, required=True, 
                        choices=["success_rates", "perturbation", "computational", 
                                 "comparison", "distributions", "all"],
                        help="Type of plot to generate")
    parser.add_argument("--methods", nargs="+", default=["FGSM", "FFGSM", "DeepFool", "CW"],
                        help="Methods to include in the plots")
    parser.add_argument("--norm", type=str, default="Linf", choices=["Linf", "L2"],
                        help="Norm type to plot")
    parser.add_argument("--targeted", action="store_true", 
                        help="Plot targeted attack results (default: untargeted)")
    parser.add_argument("--output", type=str, default="plots",
                        help="Output directory for plots")
    
    args = parser.parse_args()
    
    print(f"Loading results for methods: {args.methods}, norm: {args.norm}, targeted: {args.targeted}")
    csv_data, json_data = load_results(methods=args.methods, 
                                       norm_types=[args.norm], 
                                       targeted=args.targeted)
    
    if not json_data:
        print("No JSON data found for the specified methods and norm")
        return
    
    print(f"Generating {args.type} plots")
    if args.type in ["success_rates", "all"]:
        plot_success_rates(json_data, args.methods, args.norm, args.output)
    
    if args.type in ["perturbation", "all"]:
        plot_perturbation_metrics(json_data, args.methods, args.norm, args.output)
    
    if args.type in ["computational", "all"]:
        plot_computational_metrics(json_data, args.methods, args.norm, args.output)
    
    if args.type in ["comparison", "all"]:
        plot_method_comparison(json_data, args.methods, args.norm, args.output)
    
    if args.type in ["distributions", "all"] and not csv_data.empty:
        plot_perturbation_visualizations(csv_data, args.methods, args.norm, args.output)
    
    print(f"Plots saved to {args.output}/")


if __name__ == "__main__":
    main() 