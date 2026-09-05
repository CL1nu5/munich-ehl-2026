# Running the project

The routing and analysis scripts use the Python standard library. Python 3.10+
is a conservative baseline. Run commands from the repository root.

## Without the challenge data

```bash
python3 scripts/route_request.py examples/request.json
# Equivalent shortcut:
make demo
```

The example is handwritten and synthetic. The command loads
`models/two_stage_router.json`, extracts opening-context features, and applies
the policy. It prints JSON and does not contact a model provider or execute
tools. A lack of historical support produces the configured fallback.

You can browse the three notebooks' saved outputs directly on GitHub. Executing
their cells requires generated result tables and a Jupyter environment with
IPython; Jupyter is not needed by the pipeline or notebook generator.

## With the challenge data

Keep the authorized export local:

```text
export/
├── trajectories_v1_01.jsonl
└── trajectories_v1_02.jsonl   # if included in your run
```

The loader reads every `.jsonl` file directly inside the selected export
directory. Archives can stay inside `export/` as well; the loader ignores them.
The export and derived row-level files must not be committed or uploaded.

```bash
make pipeline
```

This runs reconstruction, grouping audit, feature extraction, outcome labeling,
the fixed review calibration, training, routing, and evaluation in order. It
writes local tables and reports to `results/` and **replaces the saved router**
in `models/two_stage_router.json`. It stops if a step fails.
Training and evaluation can take several minutes, with no console output during
some fitting steps.

To select another local export directory or Python interpreter:

```bash
make pipeline EXPORT=export/snapshot PYTHON=python3
```

The fixed calibration requires the original chunk 01 and expects 100 review
rows. A different export needs its own review process; renaming a new chunk to
match the old one does not make those judgments valid. Ambiguous trajectory
groups also need review before proceeding.

### Individual steps

```bash
python3 scripts/load_trajectories.py export/
python3 scripts/audit_trajectory_groups.py export/
python3 scripts/build_feature_table.py export/
python3 scripts/build_evaluation_table.py export/
python3 scripts/build_manual_review_sample.py export/
python3 scripts/train_two_stage_router.py --export export/
python3 scripts/two_stage_router.py
python3 scripts/evaluate_two_stage_router.py --export export/
```

Most scripts expose paths and settings through `--help`. Review these outputs
together:

| Local output | Purpose |
| :--- | :--- |
| `results/two_stage_summary.md` | Training, validation selection, and held-out results |
| `results/evaluator_summary.md` | Baseline comparison and sensitivity checks |
| `results/cost_quality_frontier.svg` | Frontier for the new run |
| `results/two_stage_routes.jsonl` | Per-trajectory routing decisions |

## Saved snapshot versus a new run

The committed notebooks and README chart record a 1,000-request run. Adding
chunks changes the task mix, grouping, split, and fitted models. A run over two
1,000-request chunks should not be expected to reproduce the archived numbers.
The snapshot count alone does not establish an exact data manifest.

After running the pipeline, regenerate the notebooks explicitly:

```bash
make notebooks
```

This replaces the three committed notebooks and writes figures under
`results/figures/`. It does not replace the archived README chart. Review both
the outputs and their written interpretation before publishing a new snapshot.

The original presentation replay reads two fixed rows from the local test
results:

```bash
make replay
# Pause between cases:
python3 scripts/demo_router.py --pause
```

Those rows may no longer be held out when the input data changes. The replay
reports missing rows; use the synthetic request example for a data-free demo.
