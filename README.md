# OX4C

Reference implementation and experimental artifact for the OX4C cybersecurity architecture. The software instantiates Optimization, eXplainability, Continuous Learning, Collaboration, Computational Performance, and Compliance in a common detection lifecycle.

[Architecture](figures/output/fig1_ox4c_architecture.pdf)

## Architecture

| OX4C dimension | Implementation |
|---|---|
| Optimization | A-GRASPQ-FS feature selection with calibration-based deployment |
| eXplainability | Post-decision full-vector TreeSHAP evidence |
| Continuous Learning | Delayed supervised adaptation with latest-batch and rolling-replay policies |
| Collaboration | Heterogeneous classifier pool with conflict-triggered, calibration-selected voting |
| Computational Performance | Latency, throughput, resource, serialized-size, and compact-model measurements |
| Compliance | Machine-readable technical-policy checks for provenance, model/version, audit, explanation, and feature evidence |

The validation control plane authorizes feature deployment, counselor intervention, and model promotion. Audit, provenance, performance, model-version, and feedback records support the lifecycle.

## Repository structure

```text
src/ox4c/                  architecture and services
experiments/               QUT-DV25 and LYCOS-IDS2017 runners
analysis/                  result consolidation and integrity audits
config/feature_selection/  frozen A-GRASPQ-FS search artifacts
results/                   released experiment outputs
figures/scripts/           reproducible plotting programs
figures/output/            publication figures
```

## Environment

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Pinned direct dependency versions for the released results are listed in `requirements-lock.txt`.

## Data

Dataset files are not redistributed.

QUT-DV25 is available from [Harvard Dataverse](https://doi.org/10.7910/DVN/LBMXJY). Place the consolidated CSV at:

```text
data/qut_dv25/QUT-DV25.csv
```

LYCOS-IDS2017 is described in [From CIC-IDS2017 to LYCOS-IDS2017](https://doi.org/10.1145/3486622.3493973). Place the five daily archives under `data/lycos_ids2017/` or pass their extracted parent directory to `--data-root`.

## Experiments

### QUT-DV25

```powershell
python experiments/run_qut_confirmatory.py `
  --qut-data data/qut_dv25/QUT-DV25.csv `
  --output artifacts/qut_dv25 `
  --progress-every 1000

python analysis/analyze_qut.py --input artifacts/qut_dv25
python analysis/audit_qut.py --input artifacts/qut_dv25
```

The runner evaluates baseline, optimization-only, collaboration-only, optimization-plus-collaboration, and complete profiles under grouped configurations 17, 42, and 73. The frozen feature-selection artifacts preserve the reported A-GRASPQ-FS searches and their search costs.

### LYCOS-IDS2017

```powershell
python experiments/run_lycos_continual.py `
  --data-root data/lycos_ids2017 `
  --output artifacts/lycos_ids2017 `
  --label-delays 0 1 2 5

python analysis/audit_lycos.py --input artifacts/lycos_ids2017
```

The runner uses Monday and Tuesday for initial fitting and processes Wednesday through Friday in chronological, predict-before-update batches. Frozen, latest-batch, and rolling-replay policies are evaluated with immediate post-prediction feedback and trusted-label delays of one, two, and five batches.

## Released results

`results/qut_dv25/` contains per-event decisions, full-vector explanation records, feature-selection metadata, classifier competence, technical-policy outcomes, computational measurements, consolidated tables, and the integrity report. Across the three grouped configurations, the baseline, collaboration-only, and optimization-plus-collaboration profiles obtain mean MCC values of 0.9651, 0.9723, and 0.9760, respectively. In the combined profile, 28 of 32 changed conflict decisions are corrections.

`results/lycos_ids2017/` contains batch, update, retention, attack-family, protocol, and integrity records. Rolling replay obtains mean online F1 of 0.9652 with immediate post-prediction feedback and 0.7829 with one complete batch of delay.

## Figures

```powershell
python figures/scripts/fig_qut_component_effects.py
python figures/scripts/fig_qut_evidence_cost.py
python figures/scripts/fig_qut_local_explanations.py
python figures/scripts/fig_lycos_continual_learning.py
```

Each script generates independent PDF and PNG panels under `figures/output/`.
