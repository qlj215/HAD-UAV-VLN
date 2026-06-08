# HAD-UAV-VLN

HAD-UAV-VLN is a research codebase for UAV vision-language navigation with a height-aware dual-view design. The project skeleton includes dataset conversion and validation tools, dual-view model components, training/evaluation entry points, and visualization scripts.

## Project Structure

```text
configs/       YAML configuration files for data, model, training, and evaluation.
data_tools/    Dataset conversion, split, and validation utilities.
datasets/      PyTorch dataset and transform definitions.
engine/        Training, evaluation, and metric entry points.
models/        Encoders, fusion modules, policy head, and HAD-VLN model.
scripts/       Shell wrappers for data conversion, baseline training, HAD training, and evaluation.
visualize/     Visualization utilities for samples, trajectories, and gate weights.
```

## Data and Outputs

Large or generated artifacts are intentionally excluded from Git:

- `data/`
- `outputs/`
- Python cache folders such as `__pycache__/`
- local virtual environments such as `.venv/`
- model weights and checkpoints

Keep raw datasets, processed images, checkpoints, and analysis outputs outside Git history. Recreate them with the scripts in `data_tools/`, `engine/`, and `scripts/` as the implementation becomes complete.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Status

This repository is an initial research framework. Some entry-point files are placeholders and should be filled as experiments are implemented.
