#!/usr/bin/env python3
"""
Map visualization script for polygon display with objects
"""

import csv
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path


def load_map_from_csv(csv_file):
    """Load polygon coordinates from CSV file."""
    coordinates = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row["x"].strip())
            y = float(row["y"].strip())
            coordinates.append([x, y])
    return np.array(coordinates)


def load_objects_from_csv(csv_file):
    """Load objects from CSV file (Type, x, y, angle)."""
    objects = []
    with open(csv_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Normalize fieldnames by stripping whitespace
        reader.fieldnames = [field.strip() for field in reader.fieldnames]
        for row in reader:
            obj_type = row["Type"].strip()
            x = float(row["x"].strip())
            y = float(row["y"].strip())
            angle = float(row["angle"].strip())
            objects.append({"type": obj_type, "x": x, "y": y, "angle": angle})
    return objects


def load_meta_json(json_file):
    """Load object dimensions from JSON file."""
    with open(json_file, "r") as f:
        return json.load(f)


def visualize_polygon(coordinates, objects, meta_data, title="Map Polygon"):
    """Visualize polygon with objects on top."""
    # Close the polygon by adding the first point at the end
    polygon = np.vstack([coordinates, coordinates[0]])

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot the polygon
    ax.plot(polygon[:, 0], polygon[:, 1], "b-", linewidth=2, label="Workspace")
    ax.fill(polygon[:, 0], polygon[:, 1], alpha=0.2, color="blue")

    # Plot vertices
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        color="cyan",
        s=50,
        zorder=5,
        label="Vertices",
    )

    # Add labels to vertices
    for i, (x, y) in enumerate(coordinates):
        ax.annotate(
            f"P{i}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9
        )

    # Track if we've already added legend entries
    box_added = False
    obj_added = False

    # Plot objects
    for obj in objects:
        obj_type = obj["type"]
        x = obj["x"]
        y = obj["y"]
        angle = obj["angle"]

        if obj_type == "S":
            # Start position - star marker
            ax.plot(x, y, marker="*", markersize=20, color="green", zorder=10)
            ax.text(x, y + 10, "START", ha="center", fontsize=10, fontweight="bold")

        elif obj_type == "B":
            # Box - rectangle centered at (x, y) with rotation
            width = meta_data["B"]["x"]
            length = meta_data["B"]["y"]
            rect = patches.Rectangle(
                (-width / 2, -length / 2),
                width,
                length,
                linewidth=2,
                edgecolor="orange",
                facecolor="orange",
                alpha=0.5,
                label="Box" if not box_added else "",
            )
            box_added = True
            # Apply rotation and translation
            t = (
                patches.Affine2D().rotate_deg(angle)
                + patches.Affine2D().translate(x, y)
                + ax.transData
            )
            rect.set_transform(t)
            ax.add_patch(rect)
            # Plot center
            ax.plot(x, y, "ko", markersize=5, zorder=9)
            ax.text(x, y - 15, "B", ha="center", fontsize=10, fontweight="bold")

        elif obj_type == "O":
            # Object - small rectangle
            width = meta_data["O"]["x"]
            length = meta_data["O"]["y"]
            rect = patches.Rectangle(
                (-width / 2, -length / 2),
                width,
                length,
                linewidth=2,
                edgecolor="red",
                facecolor="red",
                alpha=1,
                label="Object" if not obj_added else "",
            )
            obj_added = True
            # Apply rotation and translation
            t = (
                patches.Affine2D().rotate_deg(angle)
                + patches.Affine2D().translate(x, y)
                + ax.transData
            )
            rect.set_transform(t)
            ax.add_patch(rect)
            # Plot center
            # ax.plot(x, y, "ko", markersize=5, zorder=9)
            # ax.text(x - 10, y - 10, "O", ha="center", fontsize=9, fontweight="bold")

    # Set labels and title
    ax.set_xlabel("X (cm)", fontsize=12)
    ax.set_ylabel("Y (cm)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Set equal aspect ratio
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # Get the path to the files
    script_dir = Path(__file__).parent
    workspace_csv = script_dir.parent / "maps" / "workspace_1.csv"
    objects_csv = script_dir.parent / "maps" / "map_1_1.csv"
    meta_json = script_dir.parent / "maps" / "meta.json"

    # Load data
    print(f"Loading workspace from {workspace_csv}")
    coordinates = load_map_from_csv(workspace_csv)
    print(f"Loaded {len(coordinates)} vertices:")
    for i, (x, y) in enumerate(coordinates):
        print(f"  P{i}: ({x}, {y})")

    print(f"\nLoading objects from {objects_csv}")
    objects = load_objects_from_csv(objects_csv)
    print(f"Loaded {len(objects)} objects:")
    for obj in objects:
        print(f"  {obj['type']} at ({obj['x']}, {obj['y']}) - angle: {obj['angle']}°")

    print(f"\nLoading metadata from {meta_json}")
    meta_data = load_meta_json(meta_json)
    print(f"Box dimensions: {meta_data['B']['x']} x {meta_data['B']['y']}")
    print(f"Object dimensions: {meta_data['O']['x']} x {meta_data['O']['y']}")

    # Visualize
    fig, ax = visualize_polygon(
        coordinates, objects, meta_data, title="Workspace Map with Objects"
    )
    plt.show()
    # save as pdf
    fig.savefig(script_dir.parent / "maps" / "workspace_1_visualization.pdf")
