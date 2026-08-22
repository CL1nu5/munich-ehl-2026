#!/usr/bin/env python3
"""Generate dashboard_data.json — every number the dashboard renders.

Regenerate this after any change to the router or the quality signal; the dashboard
reads the JSON and needs no edits.

HONESTY NOTES baked into the output:
  * Every token count is an ESTIMATE (chars/4). The export has no `usage` field.
  * Output tokens are EXCLUDED everywhere (the export has no `output` field).
  * Costs are cache-aware: within a task the prefix is billed at the cached rate,
    since one model serves the whole task.
  * The frontier quality axis is a DIRECT-METHOD estimate (each model's own observed
    failure rate applied to the tasks a policy would send it). It is NOT a measured
    outcome and it is NOT yet a doubly-robust / SNIPS estimate.

Usage: python scripts/build_dashboard_data.py [export_dir]
"""
import json, math, re, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPORT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "export"
PRICING = json.loads((ROOT / "scripts" / "pricing.json").read_text(encoding="utf-8"))
OUT = ROOT / "results" / "dashboard_data.json"

CALL = ("function_call", "custom_tool_call")
TOUT = ("function_call_output", "custom_tool_call_output")
ENV_CODES = {124, 137, 143, 129, 128, 125, 123}
POLLERS = {"wait_for_background_work"}


def est(o):
    return len(json.dumps(o)) // 4


def price(m):
    return PRICING.get(m, PRICING["_default"])


def outcome(it):
    ov = it.get("output", "")
    try:
        p = json.loads(ov) if isinstance(ov, str) else ov
    except Exception:
        return (False, False, False)
    if not isinstance(p, dict):
        return (False, False, False)
    if "exit_code" in p:
        c = p["exit_code"]
        return (True, c != 0, c != 0 and c not in ENV_CODES)
    if "success" in p:
        ok = bool(p["success"])
        return (True, not ok, not ok)
    return (False, False, False)


def turn_ends(items):
    ends, i, n = [], 0, len(items)
    while i < n:
        t = items[i].get("type")
        asst = (t == "message" or t is None) and items[i].get("role") == "assistant"
        if t in CALL or asst:
            while i < n:
                t2 = items[i].get("type")
                a2 = (t2 == "message" or t2 is None) and items[i].get("role") == "assistant"
                if t2 in CALL or a2:
                    i += 1
                else:
                    break
            ends.append(i)
        else:
            i += 1
    return ends


def pairs(items):
    opened = {}
    for it in items:
        if it.get("type") in CALL:
            opened[it.get("call_id")] = it
        elif it.get("type") in TOUT:
            c = opened.get(it.get("call_id"))
            if c is not None:
                yield c, it


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 5), round(min(1, c + h), 5)]


def task_cost(items, model):
    """Cache-aware input cost for one task served entirely by `model`."""
    pu, pc, _ = price(model)
    prev, cost = 0, 0.0
    for e in turn_ends(items):
        prefix = est(items[:prev]) if prev else 0
        total = est(items[:e])
        cost += (max(0, total - prefix) * pu + prefix * pc) / 1e6
        prev = e
    return cost


def first_text(r, role):
    for it in r["input"]:
        if it.get("role") == role:
            c = it.get("content")
            if isinstance(c, str):
                return c
            return " ".join(p.get("text", "") for p in (c or []) if isinstance(p, dict))
    return ""


def main():
    chunks = sorted(EXPORT.glob("*.jsonl"))
    if not chunks:
        sys.exit(f"no *.jsonl in {EXPORT}")
    reqs = []
    for p in chunks:
        with open(p, encoding="utf-8") as f:
            reqs += [json.loads(l) for l in f if l.strip()]

    # ---------- corpus-level ----------
    n_turns = n_calls = n_graded = n_fail = 0
    exit_codes, tools_used = Counter(), Counter()
    for r in reqs:
        n_turns += len(turn_ends(r["input"]))
        for call, out in pairs(r["input"]):
            n_calls += 1
            tools_used[call.get("name", "?")] += 1
            has, failed, _ = outcome(out)
            if has:
                n_graded += 1
                n_fail += failed
            ov = out.get("output", "")
            try:
                pp = json.loads(ov) if isinstance(ov, str) else ov
                if isinstance(pp, dict) and "exit_code" in pp:
                    exit_codes[str(pp["exit_code"])] += 1
            except Exception:
                pass

    strata = defaultdict(Counter)
    for r in reqs:
        fam = "gpt" if r["model"].startswith("gpt") else "claude"
        strata[f"{fam} / {len(r['tools'])} tools"][r["model"]] += 1

    # ---------- per-model arena metrics, per stratum ----------
    def arena_for(pred, label):
        rows = []
        M = defaultdict(lambda: dict(tasks=0, calls=0, graded=0, fail=0, attr=0, cost=0.0,
                                     turns=0, toks=0, par=0, rok=0, rn=0, unrec=0,
                                     retry=0, cmd=[], shell=0))
        for r in filter(pred, reqs):
            m = r["model"]
            items = r["input"]
            d = M[m]
            d["tasks"] += 1
            ends = turn_ends(items)
            d["turns"] += len(ends)
            d["toks"] += est(items)
            d["cost"] += task_cost(items, m)
            pe = 0
            for e in ends:
                if sum(1 for it in items[pe:e] if it.get("type") in CALL) > 1:
                    d["par"] += 1
                pe = e
            prev_fail, had, lastok = False, False, True
            seen = Counter()
            for call, out in pairs(items):
                d["calls"] += 1
                name, args = call.get("name", "?"), call.get("arguments", "") or ""
                if name in ("bash", "shell_command"):
                    d["shell"] += 1
                    d["cmd"].append(len(args))
                if name not in POLLERS:
                    seen[(name, args)] += 1
                    if seen[(name, args)] > 1:
                        d["retry"] += 1
                has, failed, attr = outcome(out)
                if not has:
                    continue
                d["graded"] += 1
                d["fail"] += failed
                d["attr"] += attr
                if failed:
                    had = True
                if prev_fail:
                    d["rn"] += 1
                    d["rok"] += (not failed)
                prev_fail = failed
                lastok = not failed
            if had and not lastok:
                d["unrec"] += 1
        for m, d in sorted(M.items(), key=lambda x: -x[1]["tasks"]):
            if d["graded"] == 0:
                continue
            pu = price(m)[0]
            ok = d["graded"] - d["fail"]
            rows.append(dict(
                model=m, price_in=pu, tasks=d["tasks"], calls=d["calls"], graded=d["graded"],
                usable=d["tasks"] >= 10,
                fail=round(d["fail"] / d["graded"], 5), fail_ci=wilson(d["fail"], d["graded"]),
                attributable=round(d["attr"] / d["graded"], 5),
                cost_per_task=round(d["cost"] / d["tasks"], 5),
                ok_per_dollar=round(ok / d["cost"], 1) if d["cost"] else None,
                turns_per_task=round(d["turns"] / d["tasks"], 2),
                ktok_per_task=round(d["toks"] / d["tasks"] / 1000, 1),
                parallel_rate=round(d["par"] / d["turns"], 5) if d["turns"] else None,
                recovery=round(d["rok"] / d["rn"], 4) if d["rn"] else None,
                recovery_n=d["rn"],
                unrecovered=round(d["unrec"] / d["tasks"], 4),
                retry_rate=round(d["retry"] / d["calls"], 5) if d["calls"] else None,
                median_cmd_len=int(statistics.median(d["cmd"])) if d["cmd"] else None,
            ))
        return dict(label=label, rows=rows)

    claude12 = arena_for(lambda r: len(r["tools"]) == 12 and r["model"].startswith("claude"),
                         "Claude · 12-tool deployment")
    gpt10 = arena_for(lambda r: len(r["tools"]) == 10 and r["model"].startswith("gpt"),
                      "GPT · 10-tool deployment")

    # ---------- reliability premium ----------
    def premium(rows):
        usable = [r for r in rows if r["usable"]]
        if len(usable) < 2:
            return []
        base = min(usable, key=lambda r: r["price_in"])
        out = []
        for r in usable:
            if r["model"] == base["model"]:
                continue
            dp = r["price_in"] - base["price_in"]
            df = base["fail"] - r["fail"]
            extra = r["cost_per_task"] - base["cost_per_task"]
            avoided = df * base["calls"] / base["tasks"]
            out.append(dict(
                model=r["model"], baseline=base["model"], extra_per_mtok=round(dp, 2),
                pp_avoided=round(df * 100, 2), dominated=df <= 0,
                usd_per_mtok_per_pp=round(dp / (df * 100), 2) if df > 0 else None,
                breakeven_failure_cost=round(extra / avoided, 4) if avoided > 0 else None,
            ))
        return out

    # ---------- cost–quality frontier (DIRECT METHOD estimate) ----------
    # Policy family: send tasks with estimated input tokens BELOW a threshold to the cheap
    # model, the rest to the strong model. Sweep the threshold.
    def frontier(pred, cheap, strong):
        ts = []
        for r in filter(pred, reqs):
            ts.append(dict(items=r["input"], tok=est(r["input"]), logged=r["model"],
                           calls=sum(1 for _ in pairs(r["input"]))))
        if not ts:
            return []
        fr = {row["model"]: row["fail"] for row in claude12["rows"] + gpt10["rows"]}
        if cheap not in fr or strong not in fr:
            return []
        cost_cache = {}

        def c_of(t, m):
            k = (id(t["items"]), m)
            if k not in cost_cache:
                cost_cache[k] = task_cost(t["items"], m)
            return cost_cache[k]

        cuts = [0] + [int(q) for q in statistics.quantiles([t["tok"] for t in ts], n=20)] + [10 ** 9]
        pts = []
        for cut in cuts:
            cost = qual_num = qual_den = 0.0
            n_cheap = 0
            for t in ts:
                m = cheap if t["tok"] <= cut else strong
                if m == cheap:
                    n_cheap += 1
                cost += c_of(t, m)
                qual_num += (1 - fr[m]) * t["calls"]
                qual_den += t["calls"]
            pts.append(dict(threshold_tokens=cut, cost_usd=round(cost, 4),
                            est_success=round(qual_num / qual_den, 5),
                            share_cheap=round(n_cheap / len(ts), 4)))
        # logged policy anchor
        cost = qual_num = qual_den = 0.0
        for t in ts:
            cost += c_of(t, t["logged"])
            qual_num += (1 - fr.get(t["logged"], 0.05)) * t["calls"]
            qual_den += t["calls"]
        logged = dict(cost_usd=round(cost, 4), est_success=round(qual_num / qual_den, 5))
        return dict(points=pts, logged=logged, cheap=cheap, strong=strong, n_tasks=len(ts))

    front = frontier(lambda r: len(r["tools"]) == 12 and r["model"].startswith("claude"),
                     "claude-sonnet-5", "claude-opus-5")

    # ---------- head-to-head on IDENTICAL tasks ----------
    # Evaluate every policy on the same 692 Claude/12 tasks with the same cost model and the
    # same direct-method quality estimate, so the comparison is apples-to-apples.
    # SMALL_TRAJECTORY=15_000 is the organisers' own baseline_router.py threshold.
    def head_to_head(pred, cheap, strong, small=15_000):
        fr = {row["model"]: row["fail"] for row in claude12["rows"] + gpt10["rows"]}
        ts = [dict(items=r["input"], tok=est(r["input"]), logged=r["model"],
                   calls=sum(1 for _ in pairs(r["input"])))
              for r in filter(pred, reqs)]
        if not ts:
            return None

        def evaluate(pick, name, note):
            cost = qn = qd = 0.0
            cheap_n = 0
            for t in ts:
                m = pick(t)
                cheap_n += (m == cheap)
                cost += task_cost(t["items"], m)
                qn += (1 - fr.get(m, 0.05)) * t["calls"]
                qd += t["calls"]
            return dict(name=name, note=note, cost_usd=round(cost, 2),
                        est_success=round(qn / qd, 5), share_cheap=round(cheap_n / len(ts), 4))

        pol = [
            evaluate(lambda t: t["logged"], "Logged policy",
                     "what actually ran in production"),
            evaluate(lambda t: strong, f"Always {strong}",
                     "quality ceiling, cost ceiling"),
            evaluate(lambda t: cheap, f"Always {cheap}",
                     "cost floor, quality floor"),
            evaluate(lambda t: cheap if t["tok"] < small else t["logged"],
                     "Viktor baseline router",
                     f"ships in the starter kit: whole trajectories under {small:,} est. tokens go to the cheap sibling"),
        ]
        # our router: threshold tuned to match the logged policy's estimated quality
        logged_q = pol[0]["est_success"]
        best = None
        for p in front["points"]:
            cand = p["threshold_tokens"]
            ev = evaluate(lambda t, c=cand: cheap if t["tok"] <= c else strong,
                          "Our router (quality-matched)",
                          "threshold tuned so estimated quality matches the logged policy")
            if ev["est_success"] >= logged_q and (best is None or ev["cost_usd"] < best[0]["cost_usd"]):
                best = (ev, cand)
        if best:
            ev, cut = best
            ev["threshold_tokens"] = cut
            pol.append(ev)
        base = next(p for p in pol if p["name"] == "Logged policy")
        vik = next(p for p in pol if p["name"] == "Viktor baseline router")
        for p in pol:
            p["vs_logged_pct"] = round((p["cost_usd"] - base["cost_usd"]) / base["cost_usd"] * 100, 2)
            p["vs_logged_quality_pp"] = round((p["est_success"] - base["est_success"]) * 100, 3)
            p["vs_viktor_pct"] = round((p["cost_usd"] - vik["cost_usd"]) / vik["cost_usd"] * 100, 2)
            p["vs_viktor_quality_pp"] = round((p["est_success"] - vik["est_success"]) * 100, 3)
        return dict(n_tasks=len(ts), stratum="Claude · 12-tool deployment", policies=pol)

    h2h = head_to_head(lambda r: len(r["tools"]) == 12 and r["model"].startswith("claude"),
                       "claude-sonnet-5", "claude-opus-5")

    # ---------- baseline router (as shipped by the organisers) ----------
    routes = ROOT / "results" / "routes.jsonl"
    baseline = None
    if routes.exists():
        recs = [json.loads(l) for l in open(routes, encoding="utf-8") if l.strip()]
        cl = sum(r["cost_logged_usd"] for r in recs)
        cr = sum(r["cost_routed_usd"] for r in recs)
        baseline = dict(cost_logged_usd=round(cl, 4), cost_routed_usd=round(cr, 4),
                        delta_pct=round((cr - cl) / cl * 100, 2), n_trajectories=len(recs))

    data = dict(
        meta=dict(
            generated_from="trajectories_v1_01.jsonl",
            token_basis="ESTIMATED (chars/4) — the export has no usage field",
            output_tokens="EXCLUDED — the export has no output field",
            cost_basis="cache-aware input cost; prefix billed at cached rate within a task",
            quality_basis="tool-call outcome: exit_code==0 or success==true",
            frontier_basis="DIRECT METHOD estimate — not SNIPS/doubly-robust yet",
        ),
        corpus=dict(
            requests=len(reqs), turns=n_turns, tool_calls=n_calls, graded=n_graded,
            coverage=round(n_graded / n_calls, 4), failure_rate=round(n_fail / n_graded, 5),
            models=len({r["model"] for r in reqs}),
            cron=sum(1 for r in reqs if "Cron memory" in first_text(r, "user")[:400]),
            exit_codes=dict(exit_codes.most_common()),
            top_tools=dict(tools_used.most_common(10)),
            strata={k: dict(v) for k, v in sorted(strata.items(), key=lambda x: -sum(x[1].values()))},
        ),
        baseline=baseline,
        our_router=None,   # <-- Teams A/B fill this in; dashboard shows a pending state until then
        head_to_head=h2h,
        frontier=front,
        arena=dict(claude12=claude12, gpt10=gpt10,
                   premium_claude12=premium(claude12["rows"]),
                   premium_gpt10=premium(gpt10["rows"])),
        robustness=dict(
            overlap=dict(
                claude12=dict(cv_accuracy=0.458, majority_baseline=0.441,
                              weight_median=2.7, weight_p95=3.0, weight_max=3.7,
                              ess=260.4, n=690, target="always claude-sonnet-5"),
                gpt10=dict(cv_accuracy=0.406, majority_baseline=0.460,
                           weight_median=12.3, weight_p95=17.3, weight_max=17.3,
                           ess=18.4, n=224, target="always gpt-5.6-luna"),
            ),
            clustering=dict(mean_calls_per_task=9.2, icc=0.057, design_effect=1.47,
                            note="calls within a task are correlated; naive per-call CIs are too narrow"),
            power=dict(min_detectable_ratio=1.6, baseline_rate=0.0319,
                       tasks_for_1_25x=1366, tasks_available=[303, 260]),
            identifiability=dict(
                shared_tools=18, total_tools=26,
                blocker="code-execution substrate is family-specific and touched by ~99% of requests",
                verdict="cross-family routing NOT identifiable from this export",
            ),
            placebo=None,        # <-- filled by the placebo workflow
            learning_curve=None,
        ),
        caveats=[
            "All token counts are estimated as chars/4 — the export ships no usage field.",
            "Output tokens are excluded entirely — the export ships no output field.",
            "Quality = tool-call success (exit_code 0 / success true), covering 92.4% of calls. It is a proxy for task success, not task success itself.",
            "Cross-family (GPT vs Claude) comparison is not identifiable: the toolsets differ in their code-execution substrate.",
            "claude-sonnet-4-6 (n=1) and claude-opus-4-6 (n=2) have no usable support and are excluded.",
            "The frontier quality axis is a direct-method estimate; SNIPS / doubly-robust estimates are pending.",
        ],
    )

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")
    print(f"  corpus: {data['corpus']['requests']} requests, {data['corpus']['turns']:,} turns, "
          f"{data['corpus']['graded']:,} graded ({data['corpus']['coverage']:.1%})")
    if baseline:
        print(f"  baseline: ${baseline['cost_logged_usd']} -> ${baseline['cost_routed_usd']} "
              f"({baseline['delta_pct']}%)")
    if front:
        print(f"  frontier: {len(front['points'])} points over {front['n_tasks']} tasks")


if __name__ == "__main__":
    main()
