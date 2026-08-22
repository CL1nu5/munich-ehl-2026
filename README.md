# Viktor Challenge Starter — Build the Router

Starter kit for the **Viktor Challenge** at the TUM.ai hackathon (Munich, 22–23 Aug 2026).
From real LLM-request logs, build a router that picks the right model for every call —
then prove it works, even though the log shows only the model that ran, and no outputs or token counts.

## Quick start (5 minutes)

```bash
# 1. No dataset yet? Generate a synthetic sample with the same shape:
python scripts/make_synthetic_sample.py            # writes ./export/

# 2. Got the real dataset links (shipped at kickoff)? Then instead: the export ships
#    as trajectories_v1_<index>.jsonl.tar.gz archives — download, verify the posted
#    SHA-256, then:  mkdir -p export && tar xzf trajectories_v1_01.jsonl.tar.gz -C export/

# 3. Sanity-check the export, reconstruct trajectories, print stats:
python scripts/load_trajectories.py export/

# 4. Run the baseline heuristic router + cache-aware cost report:
python scripts/baseline_router.py export/

# 5. Turn results into a cost–quality frontier CSV (+ PNG if matplotlib is installed):
python scripts/plot_frontier.py results/routes.jsonl
```

Python 3.10+, standard library only (matplotlib optional for the PNG).

## Complexity training dataset

Build leakage-safe train, validation, and test files with transparent prompt and
observed-execution complexity scores:

```bash
python scripts/build_complexity_dataset.py export/
```

Outputs are written to the gitignored `results/complexity_dataset/` directory.
Each split has aligned `*_inputs.jsonl`, `*_targets.jsonl`, and
`*_metadata.jsonl` files. Related exact-prefix requests and semantic
near-duplicate prompts are always assigned to the same split. All scaling and
normalization is fitted on training rows only. The complete walkthrough and an
optional TF–IDF baseline are in `notebooks/complexity_pipeline.ipynb`.

## Model catalog and quality cutoffs

Derive a per-model complexity cutoff from published benchmark results, so routing
thresholds trace to citations instead of hand-picked constants:

```bash
python scripts/model_catalog.py            # writes results/model_catalog.json
```

Public benchmark coverage is ragged and the benchmarks disagree about which family is
stronger, so the fit does two things. It compares every pair of models *only on the
benchmarks both were measured on*, reconciling those pairwise differences by weighted
least squares; then it projects the result onto the partial order of head-to-head
dominance, so a model that beat another on every shared benchmark can never be ranked
below it. Fitted ability becomes a quantile of the **training-split** complexity
distribution, with a safety margin that widens where evidence is thin.
`route_score()` / `route_trajectory()` then pick the **weakest** model whose cutoff covers a
request, resolved at tier granularity so a model the benchmarks cannot distinguish from a
better-evidenced one is never preferred. Selection is capability-only: prices are recorded
on each model card but never consulted, so this supplies the quality axis of a cost-quality
frontier rather than the frontier itself. Section 5 of
`notebooks/complexity_pipeline.ipynb` walks the chain, including a `risk_aversion` sweep.

Two findings worth knowing before you use it:

- **The model ids are not anonymised.** Every id is a real public model and
  `scripts/pricing.json` matches each one's public list price to the cent -- including
  `gpt-5.6-sol`/`terra`/`luna`, which are OpenAI's real tier names.
- **The logged policy is not complexity-aware** (correlation between a request's
  complexity and its serving model's fitted capability is +0.03). That is the headroom a
  router is competing for -- and the reason the log says little about which model a hard
  request actually needs.
- **The top five models are not separable** by these benchmarks: they fall inside one
  2.7-point detectability band. Within a band the ordering is noise, so the members are
  treated as interchangeable (`snap_to_band=True`) rather than silently ranked.

Limitations are recorded in `scripts/model_catalog.py` and printed by the notebook.

## Using a coding agent

Point Claude Code / Codex / Cursor / opencode at this repo — `AGENTS.md` briefs your agent.
In Claude Code you also get slash commands:

- `/setup` — set up everything needed to participate
- `/make-presentation` — build a Viktor-branded presentation of your solution
- `/prepare-submission` — package your solution into a formal submission

## What's here

| Path | What |
|---|---|
| `AGENTS.md` | Agent briefing: dataset shape, the cache trap, judging, starter ideas |
| `skills/` | The three guided workflows above (plain Markdown, readable by humans too) |
| `scripts/` | Loader + trajectory reconstruction, baseline router, cache-aware cost model (estimated tokens), frontier plot, synthetic sample |
| `templates/presentation.html` | Self-contained branded slide template |

## Rules that matter

- **License:** challenge use only — no redistribution of the dataset. Full terms ship with the download.
- No GPU or API keys needed. Judge-model rescoring is allowed (credits announced at kickoff).
- Questions → the challenge Discord; the Viktor team answers there all weekend.
