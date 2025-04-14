#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Projected Gradient Descent (PGD) projection property demonstration

USAGE::
    >>> python pgd_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Set Matplotlib params for paper-quality figures
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

np.random.seed(42)


def demonstrate_contraction_property():
    """Demonstrate the contraction property of projection operators"""
    print("Demonstrating contraction property of projection operators...")

    # Generate random points
    n_points = 20
    points = np.random.randn(n_points, 2) * 2  # 2D points

    # Define projection onto unit ball
    def project_to_ball(points, radius=1.0):
        norms = np.sqrt(np.sum(points**2, axis=1))
        mask = norms > radius
        points[mask] = points[mask] / norms[mask, None] * radius
        return points

    # Select pairs of points
    pairs = []
    for i in range(0, n_points, 2):
        if i + 1 < n_points:
            pairs.append((i, i + 1))

    # Calculate original and projected distances
    orig_points = points.copy()
    proj_points = project_to_ball(points.copy())

    # Define colormap for consistent colors across plots
    # Use vibrant colors that are distinguishable
    colors = [
        "#FF1493",  # deep pink
        "#00BFFF",  # deep sky blue
        "#FF7F00",  # orange
        "#ADFF2F",  # green yellow
        "#9400D3",  # dark violet
        "#FF4500",  # orange red
        "#1E90FF",  # dodger blue
        "#32CD32",  # lime green
        "#FF69B4",  # hot pink
        "#00FA9A",  # medium spring green
    ]

    # Create figure with stacked subplots - using same total height as PGD plot (5 inches)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 5))

    # Unit circle for both plots
    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)

    # Add unit circle to both plots
    ax1.plot(x, y, "--", color="black", linewidth=1.5)
    ax2.plot(x, y, "--", color="black", linewidth=1.5)

    # Add subtle titles to identify each subplot clearly
    ax1.set_title("Original Points", fontsize=10, pad=5)
    ax2.set_title("After Projection to Unit Ball", fontsize=10, pad=5)

    # Plot pairs of points
    for idx, (i, j) in enumerate(pairs):
        color = colors[idx % len(colors)]

        # Original points and lines
        ax1.plot(
            [orig_points[i, 0], orig_points[j, 0]],
            [orig_points[i, 1], orig_points[j, 1]],
            "-o",
            color=color,
            linewidth=1.2,
            markersize=5,
        )

        # Projected points and lines
        ax2.plot(
            [proj_points[i, 0], proj_points[j, 0]],
            [proj_points[i, 1], proj_points[j, 1]],
            "-o",
            color=color,
            linewidth=1.2,
            markersize=5,
        )

    # Set axis limits and labels
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$y$")
    ax2.grid(True, alpha=0.3)

    # Add a text annotation to explain contraction property with LaTeX
    ax2.text(
        0,
        -1.3,
        r"$\text{Note: } d(\Pi_C(\mathbf{a}), \Pi_C(\mathbf{b})) \leq d(\mathbf{a}, \mathbf{b})$",
        ha="center",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="lightgray", alpha=0.3, ec="gray"),
    )

    # Add equal aspect ratio to both plots
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4)

    # Save figure
    plt.savefig("images/projection_property.png", dpi=300, bbox_inches="tight")
    plt.savefig("images/projection_property.pdf", bbox_inches="tight")
    print("Saved images/projection_property.png and images/projection_property.pdf")

    plt.close()


def visualize_projection_effect():
    """Visualize the effect of projection in PGD"""
    print("Visualizing projection effect in PGD...")

    # Create a simple quadratic problem with the minimum outside the constraint region
    # Minimize f(x,y) = (x+0.7)^2 + (y-0.7)^2 subject to ||[x,y]||_2 ≤ radius
    # The optimum is at (-0.7, 0.7) but the constraint is a circle centered at origin
    def f(x, y):
        return (x + 0.7) ** 2 + (y - 0.7) ** 2

    # Gradient of f
    def grad_f(x, y):
        return np.array([2 * (x + 0.7), 2 * (y - 0.7)])

    # Project onto L2 ball (circle)
    def project_l2(point, radius=0.5):
        norm = np.linalg.norm(point)
        if norm <= radius:
            return point
        else:
            return point * (radius / norm)

    # Initial point and parameters
    radius = 0.5  # Constraint region radius
    x0, y0 = 0.4, -0.3  # Starting point inside constraint region
    alpha = 0.2  # Learning rate
    num_iter = 15  # Number of iterations

    print(f"Starting optimization from point ({x0}, {y0})")
    print(f"Objective function: f(x,y) = (x+0.7)^2 + (y-0.7)^2")
    print(f"Global optimum at: (-0.7, 0.7) - outside constraint region")
    print(f"Constraint region: L2 ball (circle) with radius {radius}")
    print(f"Learning rate: {alpha}, Iterations: {num_iter}")

    # Arrays to store trajectories
    x_traj_no_proj = [x0]
    y_traj_no_proj = [y0]

    x_traj_with_proj = [x0]
    y_traj_with_proj = [y0]

    # Run two separate optimizations to ensure both trajectories are complete
    # 1. Standard Gradient Descent (without projection)
    x_gd, y_gd = x0, y0
    print("\nRunning standard gradient descent:")
    for i in range(num_iter):
        gx, gy = grad_f(x_gd, y_gd)
        x_new = x_gd - alpha * gx
        y_new = y_gd - alpha * gy

        # Debug print every 5 iterations
        if i % 5 == 0:
            print(
                f"  GD Iteration {i}: ({x_gd:.4f}, {y_gd:.4f}) → ({x_new:.4f}, {y_new:.4f}), gradient: ({gx:.4f}, {gy:.4f})"
            )

        # Update
        x_gd, y_gd = x_new, y_new

        # Store points
        x_traj_no_proj.append(x_gd)
        y_traj_no_proj.append(y_gd)

    # 2. Projected Gradient Descent
    x_pgd, y_pgd = x0, y0
    print("\nRunning projected gradient descent:")
    projected_points = []  # Store pre-projection and post-projection points

    for i in range(num_iter):
        gx, gy = grad_f(x_pgd, y_pgd)

        # Update with gradient step
        x_pre_proj = x_pgd - alpha * gx
        y_pre_proj = y_pgd - alpha * gy

        # Project to constraint set
        x_post_proj, y_post_proj = project_l2(
            np.array([x_pre_proj, y_pre_proj]), radius
        )

        # Store pre and post projection information
        projected_points.append(
            {
                "pre_x": x_pre_proj,
                "pre_y": y_pre_proj,
                "post_x": x_post_proj,
                "post_y": y_post_proj,
                "iteration": i,
            }
        )

        # Debug print every 5 iterations
        if i % 5 == 0:
            print(
                f"  PGD Iteration {i}: ({x_pgd:.4f}, {y_pgd:.4f}) → pre-proj: ({x_pre_proj:.4f}, {y_pre_proj:.4f}) → post-proj: ({x_post_proj:.4f}, {y_post_proj:.4f})"
            )

        # Update variables
        x_pgd, y_pgd = x_post_proj, y_post_proj

        # Store points
        x_traj_with_proj.append(x_pgd)
        y_traj_with_proj.append(y_pgd)

    # Debug info about final trajectories
    print("\nTrajectory information:")
    print(f"GD  trajectory length: {len(x_traj_no_proj)}")
    print(f"PGD trajectory length: {len(x_traj_with_proj)}")
    print(f"GD  final point: ({x_traj_no_proj[-1]:.6f}, {y_traj_no_proj[-1]:.6f})")
    print(f"PGD final point: ({x_traj_with_proj[-1]:.6f}, {y_traj_with_proj[-1]:.6f})")

    # Create figure and axes - make a perfect square
    fig, ax = plt.subplots(figsize=(5, 5))

    # Create mesh grid for contour plot - ensure equal ranges
    x_min, x_max = -1.2, 0.8
    y_min, y_max = -0.8, 1.2

    # Adjust limits to make the plot square
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)

    # Set equal ranges for both axes
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_min = x_center - max_range / 2 - 0.1  # Add a small margin
    x_max = x_center + max_range / 2 + 0.1
    y_min = y_center - max_range / 2 - 0.1
    y_max = y_center + max_range / 2 + 0.1

    # Create mesh grid with square dimensions
    x_range = np.linspace(x_min, x_max, 100)
    y_range = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)

    # Add contour plot with contour lines labeled
    CS = ax.contour(X, Y, Z, levels=np.linspace(0.5, 3.5, 7), cmap="viridis")
    ax.clabel(CS, inline=True, fontsize=10, fmt="%.1f")

    # Add constraint boundary (L2 ball - circle)
    circle = Circle(
        (0, 0), radius, fill=False, linestyle="--", edgecolor="red", linewidth=2
    )
    ax.add_patch(circle)

    # Add global minimum marker - slightly smaller size
    ax.scatter(
        -0.7,
        0.7,
        c="yellow",
        marker="*",
        s=120,
        zorder=5,
        edgecolors="k",
        linewidth=0.5,
    )

    # Add trajectories - standard version without projection markers
    ax.plot(
        x_traj_no_proj,
        y_traj_no_proj,
        "b-o",
        linewidth=1.8,
        markersize=5,
        label=r"$\text{GD}$",
    )
    ax.plot(
        x_traj_with_proj,
        y_traj_with_proj,
        "r-o",
        linewidth=1.8,
        markersize=4,
        label=r"$\text{PGD}$",
    )

    # Draw projection operations more clearly
    for point in projected_points:
        # Only add visualization if actual projection occurred
        if (
            abs(point["pre_x"] - point["post_x"]) > 1e-6
            or abs(point["pre_y"] - point["post_y"]) > 1e-6
        ):

            # Draw the projection line with arrow
            ax.annotate(
                "",
                xy=(point["post_x"], point["post_y"]),  # end point
                xytext=(point["pre_x"], point["pre_y"]),  # start point
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle="arc3",
                    color="black",
                    alpha=0.7,
                    linewidth=1.2,
                ),
            )

            # Highlight pre-projection point only when it's outside constraint
            if np.linalg.norm([point["pre_x"], point["pre_y"]]) > 0.5:
                ax.plot(point["pre_x"], point["pre_y"], "kx", markersize=4, alpha=0.7)

    # Make constraint region indicator clearer
    # Add annotation for constraint region - LaTeX style with clearer positioning
    ax.annotate(
        r"$\|\mathbf{x}\|_2 \leq 0.5$",
        xy=(0.25, 0.4),
        xytext=(0.1, 0.55),
        color="white",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="red", alpha=0.4, ec="none"),
    )

    # Add a text annotation to explain the key concept
    ax.text(
        x_min + 0.1,
        y_min + 0.1,
        r"PGD projects points back to feasible region",
        fontsize=8,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="gray"),
    )

    # Set axis limits to the square bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    # Set aspect ratio to be equal
    ax.set_aspect("equal")

    # Create a single legend with all elements - much more compact
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="yellow", markersize=8),
        Line2D([0], [0], color="red", linestyle="--", lw=1.5),
    ]
    custom_labels = [r"$\text{Optimum}$", r"$\text{Constraint}$"]
    all_handles = handles + custom_lines
    all_labels = [r"$\text{GD}$", r"$\text{PGD}$"] + custom_labels

    # Optimize legend placement
    ax.legend(
        all_handles,
        all_labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=1,
        frameon=False,
        fontsize=8,
        handlelength=1,
        handletextpad=0.3,
        labelspacing=0.4,
    )

    # Save figure
    plt.tight_layout()
    plt.savefig("images/pgd_demo.png", dpi=300, bbox_inches="tight")
    plt.savefig("images/pgd_demo.pdf", bbox_inches="tight")
    print("\nSaved images/pgd_demo.png and images/pgd_demo.pdf")

    print("Projection effect visualization complete.")


if __name__ == "__main__":
    print("Running PGD demonstrations with Matplotlib visualizations...")

    # Generate high-quality figures for paper
    demonstrate_contraction_property()
    visualize_projection_effect()
