# Data collection core

This directory contains the foundational classes and utilities for handling hand joint data collected from MediaPipe or other pose estimation sources.

## Core Components

### `joint_data.py`
Provides the `JointData` class, which is used to load, query, and visualize hand pose datasets. It supports:
- Loading .npy files containing world or normalized joint positions.
- Accessing specific joints by name (using the `Joint` enum).
- Plotting joint trajectories and deltas over time.
- Plotting model traces (comparing ground truth vs predicted positions).

### `joints.py`
Defines the `Joint` Enum, which maps the 21 standard MediaPipe hand landmarks to human-readable names (e.g., `W` for Wrist, `IT` for Index Tip).

### `utils.py`
Common mathematical utilities for data normalization and delta processing.
