# TopoNader

TopoNader is a topology-task-specific adaptation of NADER for surrogate-assisted
optimization. It generalizes the original four-bar topology/COMSOL workflow to a
configuration-driven number of bars and material-coverage constraints, together with a
Reader-free, event-triggered architecture review pipeline.

The runtime has one optimization channel only:

```text
current FE dataset
  -> train the inherited TopoNader surrogate
  -> elite-local mean EPSO + global UCB EPSO
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
- `topo_nader.optimization`: variable-bar geometry, PSO, UCB acquisition, diversity,
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

From `test37_4Beam_Best`, TopoNader keeps four parameters per bar
(`[xc, yc, theta_deg, width]`) while allowing the number of bars to vary through configuration. It also keeps the
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
.\run_TopoNader.ps1 --config configs\default.json init
.\run_TopoNader.ps1 --config configs\default.json prepare `
  --dataset ..\test37_4Beam_Best\test37_4Beam_Best\data_g\300ntrain_train_step1.mat `
  --iteration 1
```

The default run directory is `paths.runs_dir/experiment_name`, so the default
configuration creates `runs/toponader-four-bar-beam-elite-local/iteration_0001/fe_request.mat`.
`--run-dir` remains available as an explicit override. MATLAB evaluates
that request and writes `fe_result.mat`; then run:

```powershell
.\run_TopoNader.ps1 --config configs\default.json feedback `
  --dataset ..\test37_4Beam_Best\test37_4Beam_Best\data_g\300ntrain_train_step1.mat `
  --iteration 1 `
  --fe-result runs\toponader-four-bar-beam-elite-local\iteration_0001\fe_result.mat
```

When `--next-dataset` is omitted, feedback writes the next dataset under the same
experiment directory. Reusing an experiment directory with a different resolved
configuration is rejected instead of mixing results.

The main experiment controls in `default.json` are:

```json
{
  "experiment_name": "toponader-four-bar-beam-elite-local",
  "random_seed": 42,
  "data": {"num_bars": 4, "parameters_per_bar": 4},
  "optimization": {
    "coverage": {
      "upper_bound_enabled": true,
      "upper_bound": 0.30,
      "upper_tolerance": 0.02,
      "lower_bound_enabled": false,
      "lower_bound": null
    },
    "elite_local_search": {
      "enabled": true,
      "elite_count": 5,
      "particle_weights": [0.40, 0.20, 0.16, 0.12, 0.12],
      "selected_weights": [0.40, 0.20, 0.20, 0.10, 0.10]
    }
  }
}
```

`upper_bound` is an upper material-use constraint, not an equality target. The
lower-bound path is implemented but disabled by default. The Python pipeline can
use any positive number of bars; COMSOL execution additionally requires the `.mph`
model to contain `r1` through `rN`. Missing features fail explicitly, and prebuilt
features above `N` are disabled for that run.

With `elite_local_search.enabled`, the 10 exploitation designs are searched by
five independent local EPSO swarms centred on the five best true FE samples. The
default 300 particles are allocated `120/60/48/36/36`, and the final exploitation
quota is `4/2/2/1/1`. This changes initialization and bounds only; the EPSO velocity,
personal-best, global-best, mutation, and stall-reset equations are unchanged.
Exploration remains a separate global UCB-EPSO search.

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

The FE selector targets `optimization.fe_batch_size` designs (20 by default:
10 exploitation and 10 exploration). With elite-local search enabled, it either
returns the exact configured counts after geometry repair and deduplication or
raises an error before COMSOL is called; it does not silently shrink the batch.
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
  achieved-coverage statistics, audit count, architecture changes, rollbacks, and
  target attainment.

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
- `random_seed` deterministically derives independent seeds for the split, training,
  mean/UCB PSO, MC-dropout, zero-cost screening, and MATLAB input generation.
- MC-dropout prediction uses its derived experiment seed so repeated objective
  calls and parent-child comparisons use reproducible stochastic masks.
- Every run stores a config snapshot, dataset hash, iteration manifest, model
  metrics, agent transcript, trigger state, and experience log.

For elite-local runs, `iteration_NNNN/manifest.json` also records candidate-pool
finalization diagnostics under `search`: the fraction requiring coverage repair,
absolute and relative width changes, normalized projection distance, historical
and post-repair duplicate counts, unique-candidate yield, and raw-versus-final
ranking stability (Spearman correlation, pairwise order agreement, top-10 overlap,
and priority-value changes). These statistics are computed only for shortlisted
candidates, not for every PSO particle at every generation.

Thresholds in `configs/default.json` are initial engineering defaults, not
validated scientific constants. They should be fixed by pilot experiments before
the formal comparison study.

## Proposal-scoped architecture changes

Each Proposer proposal declares one `design_intent` and a nonempty `changes`
object mapping architecture fields to exact target values. Modifier preserves
unrelated parent fields. Multiple related fields may change together; there is
no fixed limit on the number of edited fields. Necessary additional edits must
be declared in `metadata.supporting_changes` as a field-to-reason object.
Input dimensions remain immutable, including for supporting edits.

The orchestrator rejects unimplemented target values, undeclared extra edits,
empty reasons, and stale supporting declarations and returns the actual
parent-child diff for correction. Candidate attempts in `audit/trace.json` and
accepted candidate metadata store the `proposal_scope_v2` audit. Cosmetic name
and metadata changes are excluded from the architectural diff. If declared as
supporting changes, they are ignored and retained separately under
`ignored_cosmetic_declarations`, without relaxing checks on actual architecture
fields. Renaming alone still fails duplicate-architecture detection. Historical
`proposal_scope_v1` logs are left unchanged. Reasons are
recorded for review; their scientific validity is not automatically verified.
This policy does not change architecture inheritance metrics or FE budgets.
Existing experiments continued with this version use the new proposal policy
at subsequent audits; record that code change when comparing experimental runs.

## Conservative event-triggered inheritance

`default.json` and the named 4-/6-bar TopoNader configurations explicitly enable
the pilot policy below. Existing Baseline configurations remain unchanged.

```json
{
  "trigger": {"min_audit_interval": 5, "max_architecture_age": 10},
  "inheritance": {
    "enabled": true,
    "min_pairwise_improvement": 0.03,
    "max_topk_regret_increase": 0.02,
    "min_feedback_samples": 10,
    "min_pairwise_accuracy": null
  }
}
```

These are pilot values, not statistically established thresholds. For
`event_triggered` review, feedback at round t schedules review at t+1 only if the
minimum interval since the last completed audit is met AND persistent ranking
degradation or the maximum audit age has been reached. Before the first audit,
the interval is counted from round zero. Requests do not update the completed
audit timestamp. After an audit's FE feedback, the timestamp is updated and the
age/degradation counters reset whether the child is inherited or rejected.
`max_architecture_age` retains its existing meaning: the counter of ordinary
feedback rounds since the last completed audit, not since the last architecture
replacement. The explicit `every_round` ablation bypasses the event-triggered
cooldown; it still uses the inheritance guard if enabled.

Training/quick-screen ranking and candidate search remain unchanged. On the
same FE batch, a distinct child must win the original lexicographic comparison,
gain at least 0.03 pairwise accuracy, and increase top-k regret by at most 0.02
absolute units. Fewer than 10 valid FE samples or nonfinite comparison metrics
prevents inheritance. There is no fixed holdout or cross-round confirmation.
The optional absolute quality floor is disabled (`null`). All valid FE samples
are appended even on rejection; rollback does not undo the consumed FE budget.

`feedback.json`, iteration JSON/JSONL reports, experience records, and CSV
columns record decisions, reasons and metric differences. `state/last_metrics.json`
describes the ultimately inherited model, including after rollback.

Legacy configurations that omit these fields resolve to interval zero and a
disabled inheritance guard. Legacy snapshots are compared with those legacy
defaults, never silently upgraded to the new enabled policy. Do not use `init
--force` to bypass a policy mismatch. Use a new experiment name or explicit new
run directory for new-policy experiments, or perform a separately documented
migration. This change does not rewrite existing run snapshots or records.

## Seed-42 C/D ablations

`configs/Ablation-C-4bars-30coverage-seed42.json` inherits the current 4-bar
TopoNader configuration and disables only `inheritance.enabled`. It retains the
five-round minimum audit interval, original training/selection and scoped LLM
generation. FE inheritance returns to the original lexicographic comparison.

`configs/Ablation-D-4bars-30coverage-seed42.json` changes only
`agents.proposal_source` to `random`. Keep `agents.enabled=true` to enable
architecture search. This route replaces the LLM candidate-generation team,
including diagnosis/modification/experience retrieval; it also skips post-FE
LLM reflection. It does not call an LLM or require an API key. Training,
zero-cost screening, inheritance protection, PSO and FE budgets remain aligned.

The random baseline (`local_mutation_v1`) samples one of eight component groups
uniformly per attempt: encoder widths, encoder depth, encoder type, latent
dimension, encoder activation, normalization, dropout, or predictor design.
Widths use factors 0.5/0.75/1.25/1.5 rounded to multiples of 16 and clipped to
configured limits; depth adds/removes a terminal hidden layer; latent dimension
uses factors 0.5/0.75/1.25/1.5/2 clipped to its limits. Activations and
normalizations use the model factory's supported choices; dropout is sampled
from 0/0.05/0.1/0.2/0.3/0.5. Predictor design samples MLP/Transformer and widths
16/32/64/128/256; Transformers use depths 1/2/3/4 and a compatible head count
from 1/2/4/8. These are documented baseline choices, not tuned optimal values.

Random proposals use an iteration-specific seed derived from `random_seed`,
preserve input dimensions, and pass the same scope/legality/parameter/duplicate
checks. Attempts are capped at `candidate_count * candidate_retry_limit**2`;
shortfalls, rejected attempts, seed, policy version and candidate counts are
recorded in `audit/trace.json`. This is local random mutation, not a uniform
sample of the entire valid architecture space. It cannot perfectly match the
distribution or generation cost of LLM proposals. Identical trigger rules do
not imply identical numbers of audits along different optimization trajectories.

Start both ablations independently from the original-load 300-sample dataset,
not from an A/B cumulative dataset, and use separate run directories. For the
current comparison use 250 outer iterations (5000 new FE attempts each). Current
C/D runs use `proposal_scope_v2`; the previous B seed-42 run used v1, so its
comparison with C/D is exploratory until a version-matched B is available.

## Reflector and Proposer role ablations

`agents.reflector_enabled` and `agents.proposer_enabled` default to `true`.
With both enabled the existing prompts, call sequence and candidate checks are
unchanged. Historical snapshots missing these fields mean both enabled; they
are normalized in memory without rewriting the snapshot. Changing either
switch in an existing run is rejected. Use a new experiment name/run directory.
These switches apply to the `llm` route, not the `random` route, and do not
override `agents.enabled` or `method.architecture_adaptation`.

- With `reflector_enabled=false`, skip both pre-audit LLM diagnosis and post-FE
  LLM reflection. Pass measured metrics and recent experience records directly
  to the remaining planning stage. Experience logging/retrieval remains enabled;
  this is a whole-Reflector ablation, not a separate memory ablation.
- With `proposer_enabled=false`, do not call the Proposer. Instead the Modifier
  returns `{proposal, architecture}` in one call, using the diagnosis/context.
  The proposal must still declare one intent and exact targets. Scope checks,
  supporting-change checks, input dimensions, parameter limits, duplicate
  rejection, training selection and FE inheritance remain active. This compares
  separate proposal/implementation roles against a merged Modifier; it does not
  remove the necessity of architectural planning or substitute random search.

Audit traces record `agent_switches` and `candidate_generation_mode`. The direct
Modifier uses the existing candidate/retry limits. LLM calls/tokens and retry
patterns can differ between routes; equal FE/training settings do not imply
equal inference cost. A declared intent and structural scope checks cannot
prove that the LLM's scientific rationale is sound.

Two seed-42 configs inherit the frozen Bv2 seed-42 configuration:
`configs/Ablation-NoReflector-4bars-30coverage-seed42.json` and
`configs/Ablation-NoProposer-4bars-30coverage-seed42.json`. Each starts from
`data/300ntrain_train_step1.mat`, iteration 1 through 250, in its own same-named
`runs` directory. For example, in MATLAB with the existing Python/API environment:

```matlab
cd('D:\Topo_Nader\web_plate');
addpath(fullfile(pwd, 'matlab'));
setenv('TOPONADER_PYTHON', 'D:\Topo_Nader\.venv\Scripts\python.exe');
name = 'Ablation-NoReflector-4bars-30coverage-seed42';
TopoNader_main(fullfile(pwd, 'data', '300ntrain_train_step1.mat'), ...
    1, 250, fullfile(pwd, 'runs', name), ...
    fullfile(pwd, 'configs', [name '.json']));
```

Change `name` to `Ablation-NoProposer-4bars-30coverage-seed42` for the other
experiment; run sequentially if GPU memory is constrained. Keep the existing
Bv2 run as the full-framework control. A single seed is preliminary evidence;
the experiment can support or contradict the usefulness of role separation,
not guarantee that a removed role will reduce performance.

## Human-suggestion implementation baseline

The earlier role ablations remain unchanged: missing
`agents.proposer_disabled_mode` means `modifier_direct`, including in existing
snapshots. They continue to ask Modifier to plan and implement. Never relabel
those results as the human-suggestion baseline or switch their mode mid-run.

The new `proposer_disabled_mode="human_suggestions"` requires an enabled LLM
architecture-search route and `proposer_enabled=false`. Program code samples
from `topo_nader/agents/human_suggestions_v1.json`, resolves exact targets,
checks applicability/legality/parameter limits, and supplies the suggestion to
Modifier. The LLM only implements the specified fields: even explained extra
structural changes are rejected. When Reflector remains enabled its diagnosis
is implementation context, not authority to choose or rewrite suggestions.
Experience storage/retrieval and ordinary validity retries remain available.

The 22-rule library is literature-informed and adapted to this factory; it is
not a published FE-surrogate expert-system benchmark or a literal copy of the
CNN rules in NADER. Each rule records source links, intent and adaptation.
Width/latent multipliers, depth steps and dropout values are project choices.
CNN-only operations unsupported by the schema are explicitly excluded.

Applicable unique rules are permuted without replacement using the independent
`human_suggestions:<iteration>` seed. Neither metrics nor an LLM rank the
suggestions. No-op/invalid/duplicate rules are skipped; exhausted pools and
candidate shortfalls are logged. All existing candidate counts, retry limits,
training, search, FE budget and inheritance protection stay configured as before.
This is a finite expert-rule menu, unlike D's sampled random mutation operators;
it still spends LLM calls to implement suggestions.

New configs pin `human_suggestions_sha256`; initialization checks the library
bytes before training. A changed library requires an explicitly new experiment.
The fingerprint covers the JSON library, not all executable source code.
Trace files record policy version, hash, seed, ordered IDs, skipped rules and
the exact proposal/implementation checks.

Two independent seed-42 experiments are prepared:

- `Ablation-Human-NoProposer-4bars-30coverage-seed42.json`: retain Reflector,
  use program-selected human suggestions and Modifier implementation.
- `Ablation-Human-NoReflector-NoProposer-4bars-30coverage-seed42.json`: skip
  Reflector as well; Modifier implements the program-selected suggestions.

Start each from the original 300 samples and iteration 1, never an earlier
ablation's cumulative dataset. In the MATLAB launch example above replace
`name` with the chosen new name and retain the 1-to-250 budget. Existing
full Bv2 and NoReflector-only groups remain controls; the older direct-Modifier
groups are additional, separately labelled planning baselines.

The library was constructed after observing project pilot results. It includes
balanced alternatives, but is not a pre-observation preregistration and does
not establish that its advice improves FE optimization. Freeze this version
before new runs; report equal-budget outcomes and costs even if the baseline
outperforms the full framework. Sources and mapping decisions are documented in
`analysis/human_suggestions_20261028/design.md`.

## Modifier selection guided by Reflector

`agents.proposer_disabled_mode="human_suggestions_select"` is a separate
opt-in strategy. It requires the same pinned human library and disabled
Proposer as the random human-suggestion mode. It does not change existing
`human_suggestions` or `modifier_direct` runs or their default meaning.

The new seed-42 config is
`configs/Ablation-HumanSelect-NoProposer-4bars-30coverage-seed42.json`.
Reflector first diagnoses the parent using measured feedback/history. Modifier
then receives the diagnosis and the entire applicable library pool, selecting
distinct suggestion IDs in priority order. Program code validates those IDs
and resolves their authoritative exact targets. Modifier implements each
selected suggestion as a separate child of the current parent. All strict
human-suggestion scope, parameter, training, FE and inheritance checks remain.
Modifier selects among existing suggestions; it cannot invent target values,
combine rules, or use a separate Proposer call.

Selection and implementation are separate calls under the Modifier role so
that the implementation targets are fixed before architecture generation.
Invalid/duplicate/unknown IDs, incorrect counts, missing rationales or extra
selection fields are rejected with bounded retries. Failed implementations
consume the normal retry budget; their suggestion IDs are excluded from later
selection rounds in the same audit. There is no silent random fallback. If
candidate generation ultimately fails, the existing parent-only fallback is
retained. Calls/tokens may increase even with the same FE budget.

Traces distinguish `human_suggestions_modifier_select`, record selection
attempts, available/selected IDs, rationales, errors and library provenance.
The deterministic pool permutation only determines presentation order; the
Modifier decides which IDs are used. Existing random-mode traces continue to
mean program selection without LLM ranking.

Start the new config from original data at iteration 1, ending at 250, in its
own same-named runs directory (replace `name` in the MATLAB launch example).
Do not resume an old random-mode or direct-Modifier run using this config.
Configuration mismatch checks enforce that distinction.

Comparison against both roles disabled with random selection measures the
combined effect of diagnosis and LLM selection. It does not isolate diagnosis
alone. The new mode also supports `reflector_enabled=false`, retaining raw
metrics/history for a possible matched selection control; no extra experiment
is started automatically.
