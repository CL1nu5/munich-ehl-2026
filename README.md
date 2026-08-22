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

# 6. Build the complexity dataset, train a model on it, and score the held-out split:
uv sync && python model/train.py
```

Everything under `scripts/` is Python 3.10+ standard library only (matplotlib optional for
the PNG). The learned model in `model/` additionally needs the pinned project dependencies —
`uv sync` installs them. Still no GPU and no API keys: it runs on a laptop CPU.

## Complexity model — dataset, training, evaluation

The whole workflow runs in one go, from the raw export to a scored held-out split:

```bash
python model/train.py                  # static encoder: ~13s from cold, seconds when cached
python model/train.py --encoder minilm # MiniLM instead: ~2 min on a cold cache
python -m model.benchmark              # compare encoders/features on validation
```

`notebooks/complexity_pipeline.ipynb` runs the same pipeline top to bottom with the
diagnostics, plots and the routing bridge; every cell executes in sequence with no manual
steps in between.

| stage | module | what it does |
|---|---|---|
| dataset | `model/data.py` | builds the leakage-safe splits via `scripts/build_complexity_dataset.py`, then re-derives the no-leakage guarantee instead of trusting the manifest |
| embedding | `model/embed.py` | segments, chunks, deduplicates, encodes and caches the request text |
| heads | `model/heads.py` | closed-form multi-output ridge and a small MLP, both multi-task |
| evaluation | `model/evaluate.py` | regression, rank and band metrics, plus the four-panel summary chart |
| orchestration | `model/pipeline.py` | runs the stages and writes `results/complexity_model/` |

**Discipline.** Feature scaling, the ridge penalty, the MLP weights and the choice between
heads are fitted or selected on train/validation only; the test split is scored once, for the
one selected configuration. The logged model id stays audit metadata and never becomes a
feature.

**Where the time goes.** The corpus is ~49M characters, but `user_messages` is byte-for-byte
the join that already formed `user_prompt`, and the system prompt largely repeats across
requests. Dropping the former and deduplicating chunks cuts the work to ~19.7k unique chunks;
the pooled matrix is then cached under a hash of the settings and the chunk text, so re-runs
are free. Measured on this export:

| encoder | cold encode | throughput | validation MAE |
|---|---|---|---|
| `static` (token-embedding lookup) | 2.5s | ~7,800 chunks/s | 4.56 |
| `minilm` (6-layer transformer) | 127s | ~155 chunks/s | 4.47 |

`static` is the default: it gives up 0.08 MAE and buys back roughly 50x the encoding time.
Eleven integer text statistics alone already reach 4.97, so the embeddings are earning their
keep on the half of the target that character counts cannot see.

Dataset outputs are written to the gitignored `results/complexity_dataset/`; each split has
aligned `*_inputs.jsonl`, `*_targets.jsonl` and `*_metadata.jsonl` files. To rebuild the
dataset alone:

```bash
python scripts/build_complexity_dataset.py export/
```

## What routing saves — the cost–quality frontier

```bash
python -m model.frontier                    # held-out frontier + chart
python -m model.frontier --min-support 0    # sensitivity: allow thinly-observed models
```

Measured on the 150 held-out requests, against the policy that actually ran:

| policy | cost | vs logged | est. quality | Δ quality | 95% CI on Δ | survives |
|---|---|---|---|---|---|---|
| logged policy | $13.14 | — | 0.9632 | — | — | — |
| same quality, cheaper | $7.38 | **−43.8%** | 0.9637 | +0.0005 | [−0.0080, +0.0067] | no |
| same cost, better quality | $12.28 | −6.5% | 0.9718 | **+0.0085** | [+0.0021, +0.0139] | **yes** |
| cheapest routable | $5.98 | −54.5% | 0.9606 | −0.0026 | — | — |
| repo baseline heuristic | $11.12 | −15.4% | 0.9622 | −0.0010 | — | — |

The router picks, per request, the model maximising `estimated quality − λ · cost`; sweeping λ
traces the curve. It **chooses** on *predicted* complexity but is **scored** at the request's
*true* complexity stratum, so prediction error costs quality rather than being forgiven.

**Read the CI column before the savings column.** Cost is arithmetic and solid. Of the quality
claims, only the equal-cost gain survives refitting the estimator on resampled training rows.
The 43.8% saving is honestly stated as *"cost down, quality change indistinguishable from
zero"* — not as a free lunch.

### Three findings that constrain how far this goes

**Averaging tool calls, not per-request ratios, changes the ranking.** 21% of requests make ≤2
tool calls, so under a per-request average a single failure on a 1-call request scores 0.0 and
outweighs a 40-call request scoring 0.975. That made `claude-opus-4-8` appear to beat
`claude-opus-5` — a previous-generation flagship out-scoring its successor. Pooling the counts
reverses it and agrees with the head-to-head benchmark evidence. The same bug had made a
"95% saving at no quality cost" look real; priced correctly, that policy is simply the cheapest
*and* clearly worse.

**The §8 benchmark catalog does not transfer to this workload.** Correlation between fitted
benchmark ability and observed tool-call success is **+0.07, rank correlation 0.00** across the
seven models with enough traffic. The model the panel ranks top (`gpt-5.6-terra`) has the worst
observed success here. The catalog's cutoffs therefore carry no validated authority — the router
above does not use them, which is why it is unaffected.

**"Each model degrades above its cutoff" is not evidence.** It holds for all seven models, but
tool-call success falls with complexity for *everyone* (0.982 → 0.940 across quartiles), so any
threshold placed mid-distribution shows the same thing. The like-for-like test — model-specific
cutoff versus one common threshold — lands on different model subsets and is inconclusive.

**Structural caveats.** Prices are assumed (`scripts/pricing.json`), only input tokens are
counted, and token counts are `chars/4`. Every trajectory in `trajectories_v1_01` is a *single
call*, so the cache-reset penalty never fires — cache-aware and uncached cost are identical on
this slice. And the permutation test (does the gain survive shuffled complexity predictions?) is
**not yet run**, so "complexity-aware routing beats a constant policy" remains open. Full
failure-mode list in notebook §11.2.

## Experimental model catalog

`scripts/model_catalog.py` contains useful stdlib-only fitting and routing machinery,
but its bundled model identities, benchmark cards, tier order, and sources are
**unverified assumptions**. The challenge briefing says the ids are anonymized, and
`scripts/pricing.json` is likewise an assumed price sheet unless the organizers replace
it. The catalog CLI is therefore disabled by default. For local experimentation only:

```bash
python scripts/model_catalog.py --allow-unverified-assumptions
```

Do not present its generated rankings or cutoffs as evidence until the cards are replaced
with organizer-approved inputs. The unit tests validate fitting behavior under their
inputs, not the truth of the bundled observations.

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
| `model/` | End-to-end complexity model: dataset setup, cached text embedding, ridge/MLP heads, held-out evaluation |
| `notebooks/` | `complexity_pipeline.ipynb` — the same workflow start to finish, with diagnostics and plots |
| `tests/` | `python -m unittest tests.test_complexity_model` (and one module per script) |
| `templates/presentation.html` | Self-contained branded slide template |

## Rules that matter

- **License:** challenge use only — no redistribution of the dataset. Full terms ship with the download.
- No GPU or API keys needed. Judge-model rescoring is allowed (credits announced at kickoff).
- Questions → the challenge Discord; the Viktor team answers there all weekend.
