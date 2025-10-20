# Repository Guidelines

## Project Structure & Module Organization
Code lives under `projects/mmdet3d_plugin` (custom heads, ops, trackers) and `projects/configs` (train/test recipes named after variant, e.g. `sparse4dv3_temporal_r50_1x8_bs6_256x704.py`).  Utility scripts reside in `tools` for data prep, anchor search, and evaluation.  Documentation and notebooks are in `docs/` and `tutorial/`, while figures and reference assets stay in `resources/`.  Keep large datasets outside the repo and symlink them into `data/` as noted in `docs/quick_start.md`.

## Build, Test, and Development Commands
Use Python 3.8+ with the virtualenv from `docs/quick_start.md`.  Install dependencies via `pip3 install -r requirement.txt`.  Rebuild custom CUDA layers after touching `projects/mmdet3d_plugin/ops` with `python3 setup.py develop`.  Launch training with `bash local_train.sh sparse4dv3_temporal_r50_1x8_bs6_256x704`, which feeds the chosen config to MMDetection3D.  Evaluate checkpoints using `bash local_test.sh sparse4dv3_temporal_r50_1x8_bs6_256x704 path/to/ckpt`, and pass `--eval` flags through the script if additional metrics are required.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and snake_case for functions, PascalCase for classes, and UPPER_CASE for constants.  Keep config filenames descriptive (`model_backbone_schedule_resolution.py`) and align keys with MMDetection3D conventions (`train_cfg`, `test_cfg`, `data`).  Run `yapf -ir projects` before submitting Python changes; keep imports grouped by standard, third-party, and local modules.  Document new heads, losses, or ops with short docstrings describing tensor shapes and expectations.

## Testing Guidelines
Before training, generate metadata using `python3 tools/nuscenes_converter.py --version v1.0-mini --info_prefix data/nuscenes_anno_pkls/nuscenes-mini`.  Validate new anchors or datasets via `python3 tools/anchor_generator.py --ann_file <train.pkl>` and capture summary stats in your PR.  Smoke-test CUDA changes with a tiny nuScenes clip (`--version v1.0-mini`) and ensure `local_test.sh` reports stable NDS/mAP against the chosen baseline.  Share command outputs or TensorBoard screenshots when introducing model tweaks.

## Commit & Pull Request Guidelines
Recent history uses short, descriptive summaries (often Chinese verbs).  Aim for imperative messages such as `ops: optimize deformable aggregation kernel` or `tracking: fix propagated instance filter`.  Squash incidental experiments locally.  PRs should: reference the config or issue touched, describe dataset splits and GPUs used, attach validation metrics/logs, and note any required artifacts (ckpts, anchors).  Include screenshots when updating docs or tutorials.

## Data & Configuration Tips
Store environment-specific paths in `.env` files or shell exports rather than hard-coding them in configs.  Check large assets into external storage and version download URLs in `resources/` or `docs/`.  When creating new configs, start from the closest template in `projects/configs` and document deviations at the top of the file for faster reviews.
