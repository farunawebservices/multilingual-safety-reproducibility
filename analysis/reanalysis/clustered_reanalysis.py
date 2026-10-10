"""Prompt-clustered reanalysis of all paired contrasts in the ARR draft.

Each harmful cell holds 100 unique prompts x 3 repeated pipeline runs (record IDs
suffixed _planner/_tool-caller/_critic share identical prompt text). The paper's
McNemar tests and bootstrap intervals resample records; here the resampling and
randomisation unit is the unique prompt (the "base ID").

Outputs one CSV row per contrast with:
  record-level (paper) estimate, CI, exact McNemar p, Holm p   -> reproduction check
  prompt-clustered bootstrap CI (10,000 reps, seed 20260908)
  prompt-level sign-flip randomisation p (10,000 draws), Holm p within the paper's family
"""
import sys, csv, random, collections
from math import comb
sys.path.insert(0, sys.argv[1])
from minipq import read_parquet

ROOT = sys.argv[2]
OUT = sys.argv[3]
CAN = ROOT + "/data/derived/canonical/"
B = 10000
SEED = 20260908


def load(name):
    cols, n = read_parquet(CAN + name)
    return [dict(zip(cols, vals)) for vals in zip(*cols.values())]


def base(rid):
    return rid.rsplit("_", 1)[0]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def contrast(A, Bm, label, family):
    """A, Bm: dict record_id -> 0/1 outcome (baseline, steered)."""
    ids = sorted(set(A) & set(Bm))
    n = len(ids)
    d = {i: Bm[i] - A[i] for i in ids}
    delta = 100 * sum(d.values()) / n
    b = sum(1 for i in ids if A[i] == 1 and Bm[i] == 0)
    c = sum(1 for i in ids if A[i] == 0 and Bm[i] == 1)
    p_rec = mcnemar_exact(b, c)
    # record-level bootstrap (paper's method)
    rng = random.Random(SEED)
    dl = [d[i] for i in ids]
    rb = sorted(100 * sum(rng.choice(dl) for _ in range(n)) / n for _ in range(B))
    # prompt-clustered bootstrap
    cl = collections.defaultdict(list)
    for i in ids:
        cl[base(i)].append(d[i])
    keys = sorted(cl)
    sums = {k: sum(cl[k]) for k in keys}
    lens = {k: len(cl[k]) for k in keys}
    rng = random.Random(SEED)
    cb = []
    for _ in range(B):
        s = [rng.choice(keys) for _ in keys]
        cb.append(100 * sum(sums[k] for k in s) / sum(lens[k] for k in s))
    cb.sort()
    # prompt-level sign-flip randomisation test on per-prompt summed differences
    obs = abs(sum(sums.values()))
    rng = random.Random(SEED + 1)
    vals = [sums[k] for k in keys if sums[k] != 0]
    ge = 1
    for _ in range(B):
        t = abs(sum(v if rng.random() < 0.5 else -v for v in vals))
        if t >= obs - 1e-12:
            ge += 1
    p_cl = ge / (B + 1) if vals else 1.0
    lo, hi = int(0.025 * B), int(0.975 * B) - 1
    return dict(family=family, contrast=label, n_records=n, n_prompts=len(keys),
                base_rate=round(100 * sum(A[i] for i in ids) / n, 1),
                steer_rate=round(100 * sum(Bm[i] for i in ids) / n, 1),
                delta_pp=round(delta, 1),
                rec_ci_lo=round(rb[lo], 1), rec_ci_hi=round(rb[hi], 1), rec_mcnemar_p=p_rec,
                cl_ci_lo=round(cb[lo], 1), cl_ci_hi=round(cb[hi], 1), cl_signflip_p=p_cl)


def outcome_map(rows, pred, val):
    return {r["record_id"]: int(bool(val(r))) for r in rows if pred(r)}


results = []

# ---------------- Llama 3.3 70B ----------------
L1 = load("llama33_phase1_main.parquet")
L2 = load("llama33_phase2_turbo.parquet")
fail = lambda r: r["operational_failure"]
refuse = lambda r: r["parser_outcome"] == "VETO_EXPLICIT"
langs = ["Arabic", "English", "Hausa", "Yoruba"]


def sel(rows, cfg, a, lang, cls):
    return lambda r: (r["config"] == cfg and float(r["alpha"]) == a and r["language"] == lang
                      and r["prompt_class"] == cls)


for lang in langs:
    results.append(contrast(outcome_map(L1, sel(L1, "planner_only", 0, lang, "harmful"), fail),
                            outcome_map(L1, sel(L1, "planner_only", 20, lang, "harmful"), fail),
                            f"Llama planner a0->a20 {lang} harmful", "llama_primary_harmful"))
for lang in langs:
    results.append(contrast(outcome_map(L1, sel(L1, "planner_only", 0, lang, "benign"), refuse),
                            outcome_map(L1, sel(L1, "planner_only", 20, lang, "benign"), refuse),
                            f"Llama planner a0->a20 {lang} benign refusal", "llama_primary_benign"))
for cfg in ["planner_only", "critic_only", "both"]:
    for lang in ["Hausa", "Yoruba"]:
        results.append(contrast(outcome_map(L1, sel(L1, cfg, 0, lang, "harmful"), fail),
                                outcome_map(L2, sel(L2, cfg, 30, lang, "harmful"), fail),
                                f"Llama {cfg} P1 a0 -> Turbo a30 {lang}", "llama_turbo_a30"))
for cfg in ["planner_only", "critic_only", "both"]:
    for lang in ["Hausa", "Yoruba"]:
        results.append(contrast(outcome_map(L2, sel(L2, cfg, 20, lang, "harmful"), fail),
                                outcome_map(L2, sel(L2, cfg, 30, lang, "harmful"), fail),
                                f"Llama {cfg} Turbo a20 -> a30 {lang}", "llama_within_turbo"))
for cfg in ["planner_only", "critic_only", "both"]:
    for lang in ["Hausa", "Yoruba"]:
        A = outcome_map(L1, sel(L1, cfg, 20, lang, "harmful"), fail)
        Bm = outcome_map(L2, sel(L2, cfg, 20, lang, "harmful"), fail)
        if len(set(A) & set(Bm)) >= 30:
            results.append(contrast(A, Bm, f"Llama {cfg} P1 a20 -> Turbo a20 {lang}",
                                    "llama_phase_shift_descriptive"))

# ---------------- Qwen 2.5 72B (33-cell family) ----------------
Q = load("qwen25_72b_primary_role_strength.parquet")
qbase = collections.defaultdict(dict)
for r in Q:
    if r["config"] == "baseline":
        qbase[(r["language"], r["prompt_class"])][r["record_id"]] = int(bool(r["operational_failure"]))
qcells = sorted({(r["config"], float(r["alpha"]), r["language"], r["prompt_class"])
                 for r in Q if r["config"] != "baseline"})
for cfg, a, lang, cls in qcells:
    St = {r["record_id"]: int(bool(r["operational_failure"])) for r in Q
          if r["config"] == cfg and float(r["alpha"]) == a and r["language"] == lang
          and r["prompt_class"] == cls}
    results.append(contrast(qbase[(lang, cls)], St, f"Qwen {cfg} a{a:g} {lang} {cls}", "qwen_33"))

# ---------------- Gemma 4 31B ----------------
G = load("gemma_4_31b_phase1_complete_case.parquet")
for lang, cls in [("Arabic", "benign"), ("English", "benign"), ("Arabic", "harmful")]:
    A = {r["record_id"]: int(bool(r["operational_failure"])) for r in G
         if r["config"] == "baseline" and r["language"] == lang and r["prompt_class"] == cls}
    Bm = {r["record_id"]: int(bool(r["operational_failure"])) for r in G
          if r["config"] != "baseline" and r["language"] == lang and r["prompt_class"] == cls}
    results.append(contrast(A, Bm, f"Gemma planner a1 {lang} {cls}", "gemma_3"))

# ---------------- GPT-OSS 20B ----------------
O0 = load("gpt_oss_20b_phase1_baseline.parquet")
O1 = load("gpt_oss_20b_phase1_planner_only_alpha1.parquet")
# Paper definition: legacy_operational_failure (BOTH_LABELS counted as VETO).
# Sensitivity: audited operational_failure (BOTH_LABELS counted as failure).
for col, famname in [("legacy_operational_failure", "gptoss_4"),
                     ("operational_failure", "gptoss_4_audited_parser_sensitivity")]:
    for lang in langs:
        A = {r["record_id"]: int(bool(r[col])) for r in O0
             if r["language"] == lang and r["prompt_class"] == "harmful"}
        Bm = {r["record_id"]: int(bool(r[col])) for r in O1
              if r["language"] == lang and r["prompt_class"] == "harmful"}
        results.append(contrast(A, Bm, f"GPT-OSS planner a1 {lang} harmful", famname))

# ---------------- Holm within families ----------------
fam = collections.defaultdict(list)
for i, r in enumerate(results):
    fam[r["family"]].append(i)
for f, idx in fam.items():
    for key_p, key_h in [("rec_mcnemar_p", "rec_holm_p"), ("cl_signflip_p", "cl_holm_p")]:
        adj = holm([results[i][key_p] for i in idx])
        for i, a in zip(idx, adj):
            results[i][key_h] = a

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)
print("wrote", len(results), "rows to", OUT)
