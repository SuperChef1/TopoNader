# TopoNader

TopoNader is a topology-task-specific adaptation of NADER for surrogate-assisted
optimization. It combines the original four-bar topology/COMSOL workflow with a
Reader-free, event-triggered architecture review pipeline.

The runtime has one optimization channel only:

```text
current FE dataset
  -> train the inherited TopoNader surrogate
  -> PSO over predicted mean and UCB
  -> exploit/explore/diversity batch
  -> MATLAB/COMSOL truth evaluation
  -> ranking and calibration feedback
  -> keep the architecture or schedule a TopoNader audit
```

WAE is not imported by the main package. Its original implementation is retained
outside this project only as an experimental baseline.

## Components

- `topo_nader.architecture`: bounded JSON architecture specifications, model
  factory, parameter/shape/forward/backward validation.
- `topo_nader.agents`: `TopoNaderReflector -> TopoNaderProposer ->
  TopoNaderModifier`. There is deliberately no Reader.
- `topo_nader.training`: chronological validation, early stopping, physical-scale
  metrics, MC-dropout uncertainty, and switchable training-free Jacobian screening.
- `topo_nader.monitoring`: ranking metrics and deterministic audit trigger.
- `topo_nader.optimization`: four-bar geometry, PSO, UCB acquisition, diversity,
  and deduplication.
- `topo_nader.pipeline`: resumable prepare/feedback state machine and parent-child
  inheritance/rollback.
- `topo_nader.reporting`: a standalone report for every completed iteration plus
  cumulative JSON, JSONL, and CSV reports that remain usable after a manual stop.
- `matlab`: adapters plus the runtime copy of `func_outputs_nowrite.m`, the COMSOL
  model, and the geometry mask used by the self-contained FE workflow.

## Upstream reuse boundary

From the original NADER project, TopoNader adapts the multi-agent separation,
structured proposal/modification/reflection loop, retry/logging approach, and
architecture validation concept. NADER's image-classification training, CNN
search space, Reader, and CIFAR/ImageNet metrics are not used.

From `test37_4Beam_Best`, TopoNader keeps the 4 x 4 physical representation,
grid convention, width/angle bounds, coverage constraint, EPSO update ideas,
MATLAB finite-element oracle, COMSOL model, and the original
`1024-512-256 -> latent-12 -> Transformer` surrogate as the initial parent. The WAE branch and train-loss-only
checkpoint selection are not part of the new runtime.

The original source directories are treated as read-only upstreams. Runtime FE
files are kept under `Topo_Nader/matlab`, and TopoNader writes checkpoints and
logs only into its own run directories.

## Quick start

Create a project-local Python environment and install TopoNader. The project does
not import source code from the original NADER repository:

```powershell
cd D:\Nader\Topo_Nader
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\run_TopoNader.ps1 --config configs\default.json validate-config
.\run_TopoNader.ps1 --config configs\default.json init --run-dir runs\trial1
.\run_TopoNader.ps1 --config configs\default.json prepare `
  --run-dir runs\trial1 `
  --dataset ..\test37_4Beam_Best\test37_4Beam_Best\data_g\300ntrain_train_step1.mat `
  --iteration 1
```

`prepare` creates `runs/trial1/iteration_0001/fe_request.mat`. MATLAB evaluates
that request and writes `fe_result.mat`; then run:

```powershell
.\run_TopoNader.ps1 --config configs\default.json feedback `
  --run-dir runs\trial1 `
  --dataset ..\test37_4Beam_Best\test37_4Beam_Best\data_g\300ntrain_train_step1.mat `
  --iteration 1 `
  --fe-result runs\trial1\iteration_0001\fe_result.mat `
  --next-dataset runs\trial1\datasets\dataset_0002.mat
```

For the MATLAB-driven loop, set `TOPONADER_PYTHON` if `python` is not the desired
interpreter and call `TopoNader_main` as documented in the file.

For an architecture audit, set `TOPONADER_LLM_API_KEY` and optionally
`TOPONADER_LLM_MODEL_NAME` / `TOPONADER_LLM_BASE_URL`. Ordinary iterations never
call an LLM. An audit is scheduled only by deterministic feedback rules.

Every architecture is retrained from scratch on the current dataset. TopoNader
does not inherit surrogate weights between iterations or architectures.

## Architecture-audit order

An audit iteration now executes in this order:

```text
train and validate the current parent
  -> give current parent metrics + previous FE feedback to TopoNaderReflector
  -> TopoNaderProposer proposes changes
  -> TopoNaderModifier generates structured children
  -> invalid/duplicate output is returned to Modifier for correction
  -> Proposer supplies additional proposals if the requested count is not reached
  -> legality checks
  -> optional training-free Jacobian screening
  -> quick training screen
  -> equal full training of parent and retained children
  -> provisional model drives the candidate search
  -> discretize and repair the selected designs
  -> re-predict those exact final designs
  -> FE feedback selects child inheritance or parent rollback
```

The zero-cost score is deliberately treated as a heuristic filter rather than a
scientific performance metric. Set `architecture.zero_cost_enabled` to `false`
to disable it during later ablations.

The FE selector targets `optimization.fe_batch_size` designs (10 by default),
but may return fewer after geometry repair and deduplication. It never writes an
empty FE request: no valid candidate raises an error before COMSOL is called.
Feedback with one valid FE result is accepted, while ranking correlations are
recorded conservatively because a single sample cannot define an ordering.

## Method switches and baselines

The `method` section separates the main method components:

- `architecture_adaptation`: enable or disable TopoNader architecture changes.
- `architecture_review_mode`: `event_triggered` or `every_round`.
- `uncertainty_sampling`: add the UCB PSO/exploration channel or use mean PSO only.
- `diversity_sampling`: use farthest-candidate diversity or pure score ordering.

Ready-to-use inherited configurations are provided:

- `configs/baseline_fixed_mean.json`: fixed original surrogate + mean PSO.
- `configs/baseline_fixed_uncertainty.json`: fixed surrogate + identical
  uncertainty/diversity sampling, isolating the effect of architecture adaptation.
- `configs/default.json`: event-triggered TopoNader proposal.
- `configs/baseline_every_round_review.json`: architecture review every round.

These files inherit `default.json`, so common FE budgets and training settings
remain aligned unless explicitly overridden.

## Per-iteration reporting

After every successful `feedback` call, TopoNader writes:

- `iteration_NNNN/iteration_summary.json`: that iteration's standalone result;
- `iterations.jsonl`: machine-readable records for every completed iteration;
- `iterations.csv`: plotting/table input;
- `summary.json`: cumulative FE calls, valid/failed FE results, best performance,
  audit count, architecture changes, rollbacks, and target attainment.

Set `reporting.target_performance` to calculate an iteration-level upper bound on
the FE calls needed to reach a target. The cumulative FE count includes the
initial dataset and counts failed new FE calls as spent budget.

## Important experimental controls

- The objective is maximized; exploration uses `mean + kappa * uncertainty`.
- FE `Inf`, non-finite, and non-positive values are rejected before selection and
  dataset update.
- Parents and children use the same chronological split and training settings.
- Cumulative legacy MAT files are read from unsorted `inputsX/outputsX`; first-round
  files without those keys use a fixed random split. The performance-sorted
  `inputs/outputs` fields are never mistaken for temporal order.
- Training R2 never selects an architecture; ranking accuracy, Kendall tau,
  top-k regret, and then RMSE determine the provisional winner.
- On an audit round, parent and provisional child predictions are stored for the
  same FE batch; post-FE feedback either inherits the child or rolls back.
- Predictions are recomputed after discretization and coverage repair, so logged
  predictions and FE truth always refer to the exact same final design.
- MC-dropout prediction uses the configured experiment seed so repeated objective
  calls and parent-child comparisons use reproducible stochastic masks.
- Every run stores a config snapshot, dataset hash, iteration manifest, model
  metrics, agent transcript, trigger state, and experience log.

Thresholds in `configs/default.json` are initial engineering defaults, not
validated scientific constants. They should be fixed by pilot experiments before
the formal comparison study.
