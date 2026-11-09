# CartPole Adaptive World Models

Research code, trained checkpoints and numerical results for few-shot dynamics adaptation and closed-loop model-predictive control in full-angle CartPole.

## Methods

- Frozen learned dynamics backbone.
- Ridge residual correction fitted from deployment observations.
- Original Meta residual adapter.
- Rollout Meta residual adapter.
- Oracle dynamics and condition-specific dense diagnostic references.
- Archived V2 factorial experiments also include residual/LoRA comparisons.

The main experiments use five pretrained seeds: 11, 22, 33, 44 and 55. Configuration and result metadata identify the data splits, adaptation settings and physical conditions. Oracle and dense references have privileged information and are not few-shot deployment claims.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/adaptive_world_model/` | Dynamics, adaptation, environments and control implementation |
| `scripts/cartpole/` | Experiment entry points and analysis utilities |
| `configs/cartpole/` | Recorded experiment configurations |
| `results/cartpole/` | Checkpoint files, browsable numerical tables and scientific figures |
| `artifacts/index.json` | Complete raw-result archive URLs, sizes and SHA-256 digests |
| `artifacts/file-manifests/` | Per-file checksums for the filtered public artifacts |

## Trained models and raw results

Checkpoint files retain their original paths under `results/cartpole/`. They include frozen bases, trained adapter checkpoints, resume states and diagnostic copies; the number of files is not a count of independent models. Per-task fitted parameters, trajectories, support observations, prediction/ranking outputs, fitting traces and additional large tables are in the release archives.

The complete filtered results are attached to the [artifact release](https://github.com/ShravanVenky/cartpole-adaptive-world-models/releases/tag/v2-artifacts-2026-11-08). Large raw data stays in release assets so cloning the code and checkpoints does not require downloading the whole dataset. No Git LFS download is required.

| Run | Included files | Download size | Raw results |
| --- | ---: | ---: | --- |
| `20261005_232356_821640_cartpole_v1` | 7,761 | 78.4 MiB | [Download](https://github.com/ShravanVenky/cartpole-adaptive-world-models/releases/download/v2-artifacts-2026-11-08/cartpole-v1-results.zip) |
| `20261006_140915_614361_cartpole_v2_factorial` | 7,908 | 206.3 MiB | [Download](https://github.com/ShravanVenky/cartpole-adaptive-world-models/releases/download/v2-artifacts-2026-11-08/cartpole-v2-factorial-results.zip) |
| `20261023_v2_control` | 1,724 | 740.3 MiB | [Download](https://github.com/ShravanVenky/cartpole-adaptive-world-models/releases/download/v2-artifacts-2026-11-08/v2-control-results.zip) |
| `20261026_000707_v2_control_diagnostics` | 1,832 | 1,185.9 MiB | [Download](https://github.com/ShravanVenky/cartpole-adaptive-world-models/releases/download/v2-artifacts-2026-11-08/v2-control-diagnostics-results.zip) |
| `20261026_223145_v2_final_validation_full` | 1,465 | 220.9 MiB | [Download](https://github.com/ShravanVenky/cartpole-adaptive-world-models/releases/download/v2-artifacts-2026-11-08/v2-final-validation-results.zip) |
| `20261101_013609_468917_v2_model_planner_mismatch_full` | 1,016 | 1,754.1 MiB | [Download](https://github.com/ShravanVenky/cartpole-adaptive-world-models/releases/download/v2-artifacts-2026-11-08/v2-model-planner-mismatch-results.zip) |
| `20261108_021754_284433_v2_final_hardening_full` | 68,805 | 739.8 MiB | [Download](https://github.com/ShravanVenky/cartpole-adaptive-world-models/releases/download/v2-artifacts-2026-11-08/v2-final-hardening-results.zip) |

Download all results into the checkout, verifying archive hashes:

```powershell
python scripts/download_results.py --all
```

Or download just the final hardening suite:

```powershell
python scripts/download_results.py --run 20261108_021754_284433_v2_final_hardening_full
```

Extraction preserves the original `results/cartpole/<run>/` structure and checks existing files before replacing anything. Each archive includes an `ARTIFACT_MANIFEST.jsonl`; the repository also contains compressed per-file manifests and `SHA256SUMS`.

## Environment and entry points

Python 3.11 or newer; dependencies are declared in `pyproject.toml` and `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Experiment entry points include `scripts/cartpole/run_v2.py`, `run_diagnostics.py`, `run_final_v2.py`, `run_mismatch_v2.py` and `run_hardening_v2.py`. Read their help and configuration before starting compute-intensive work. Published checkpoints and saved numerical results can be inspected without retraining.

## Publication scope

This repository includes the CartPole research project and all seven CartPole result runs. Tests, smoke-test artifacts, generated run reports, personal handoffs, environments, external reference checkouts and the retired Pendulum archive are omitted. The README documents the dataset; it is not an experiment report.

Raw numerical artifacts preserve their original bytes. Historical provenance manifests can refer to deliberately excluded reports/tests, so this filtered publication is not an exact replacement for a private run's preservation archive. The release manifests describe exactly what is included. Raw result folders are ignored by default to avoid accidental large commits; the curated checkpoint/table files already tracked in Git remain tracked.
