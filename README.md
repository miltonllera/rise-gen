# Generating Freeform Endoskeletal Robots

**Paper URL**: [Generating Freeform Endoskeletal Robots](https://openreview.net/pdf?id=awvJBtB2op)

**Status**: Under review at ICLR

## Methodology

1. **Simulation Platform**: We built a voxel-based simulation platform that enables two-way coupling between flexible (soft) and rigid components of the robots.

2. **Encoding Morphology**: A variational autoencoder (VAE) was used to generate and optimize robot structures, encoding valid designs into a 512-dimensional latent space.

3. **Control Mechanism**: We built a universal controller using a graph transformer, and aligns the robot’s proprioceptive sensory input from voxels
    with its skeletal graph sensor inputs.

4. **Optimization Process**: Designs were optimized using Proximal Policy Optimization (PPO) within a reinforcement learning framework, while evolutionary strategies (CMA-ES) iteratively improved designs based on fitness scores.

5. **Task Environments and Adaptation**: Robots were tested in diverse task environments (e.g., flat ground, potholes, mountain ranges) to evolve designs that could overcome complex obstacles.

## Repository Structure

The repository is organized into the following directories:

- **`model`**: Models for VAE and controllers.
- **`rl`**: Reinforcement learning implementation.
- **`scripts`**: Scripts for running experiments.
- **`sim`**: Simulation platform implementation.
- **`synthetic`**: Synthetic training data generation.
- **`utils`**: Utility functions and helpers.

## Usage

1. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements_cu121.txt
   # or
   pip install -r requirements_cu118.txt
   ```
3. Train Morphology VAE:
   ```bash
    python scripts/train_vae.py
   ```
4. Run one of the experiments:
   ```bash
   python scripts/evolve_rl.py
   ```