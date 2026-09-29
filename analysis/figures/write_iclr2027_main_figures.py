from pathlib import Path
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "paper_artifacts" / "metrics"
OUT = ROOT / "paper_artifacts" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
})


def read_csv(name):
    with (METRICS / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row, key):
    return float(row[key])


def assert_pct(rows, columns):
    for row in rows:
        for col in columns:
            value = f(row, col)
            assert -100.0 <= value <= 100.0, f"{col} out of range: {value}"


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=220)
    plt.close(fig)


def figure1():
    fig = plt.figure(figsize=(7.0, 2.9))
    ax = fig.add_axes([0.03, 0.18, 0.56, 0.68])
    ax.axis("off")
    boxes = [(0.02, 0.45, "Prompt"), (0.23, 0.45, "Planner"), (0.44, 0.45, "Tool-Caller"), (0.68, 0.45, "Critic")]
    for x, y, label in boxes:
        patch = FancyBboxPatch((x, y), 0.15, 0.22, boxstyle="round,pad=0.02", linewidth=1, edgecolor="#24415c", facecolor="#eaf2f8")
        ax.add_patch(patch)
        ax.text(x + 0.075, y + 0.11, label, ha="center", va="center", fontsize=8)
    for x1, x2 in [(0.17, 0.23), (0.38, 0.44), (0.59, 0.68)]:
        ax.add_patch(FancyArrowPatch((x1, 0.56), (x2, 0.56), arrowstyle="->", mutation_scale=10, linewidth=1))
    ax.text(0.84, 0.56, "VETO / ALLOW /\nNO-DECISION", ha="center", va="center", fontsize=8)
    ax.add_patch(FancyArrowPatch((0.83, 0.56), (0.96, 0.56), arrowstyle="->", mutation_scale=10, linewidth=1))
    ax.text(0.3, 0.18, "Planner-only", color="#1b9e77", ha="center")
    ax.text(0.68, 0.18, "Critic-only", color="#d95f02", ha="center")
    ax.text(0.5, 0.02, "Both roles", color="#7570b3", ha="center")

    ax2 = fig.add_axes([0.66, 0.18, 0.31, 0.68])
    rows = read_csv("llama33_turbo_alpha30_role_strength_paired_inference.csv")
    assert len(rows) == 6
    assert_pct(rows, ["delta_steered_minus_baseline_percentage_points", "ci95_low_percentage_points", "ci95_high_percentage_points"])
    order = [("planner_only", "Hausa"), ("planner_only", "Yoruba"), ("critic_only", "Hausa"), ("critic_only", "Yoruba"), ("both", "Hausa"), ("both", "Yoruba")]
    vals, labels, colors, lows, highs = [], [], [], [], []
    palette = {"planner_only": "#1b9e77", "critic_only": "#d95f02", "both": "#7570b3"}
    for config, lang in order:
        row = next(r for r in rows if r["config"] == config and r["language"] == lang)
        vals.append(f(row, "delta_steered_minus_baseline_percentage_points"))
        lows.append(f(row, "ci95_low_percentage_points"))
        highs.append(f(row, "ci95_high_percentage_points"))
        labels.append(f"{config.replace('_', ' ')}\n{lang}")
        colors.append(palette[config])
    y = np.arange(len(vals))
    ax2.axvline(0, color="black", linewidth=0.8)
    for yi, vi, lo, hi, color in zip(y, vals, lows, highs, colors):
        ax2.errorbar(vi, yi, xerr=[[vi - lo], [hi - vi]], fmt="o", color=color, ecolor=color, capsize=2, markersize=4)
    ax2.set_yticks(y, labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("Steered minus baseline failure (pp)")
    ax2.set_title("Role × language contrast")
    ax2.grid(axis="x", alpha=0.25)
    fig.suptitle("Role placement changes the direction of steering effects", fontsize=10, y=0.98)
    save(fig, "iclr2027_figure1_pipeline_overview")


def figure2():
    rows = read_csv("llama33_turbo_alpha30_role_strength_paired_inference.csv")
    assert len(rows) == 6
    assert_pct(rows, ["delta_steered_minus_baseline_percentage_points", "ci95_low_percentage_points", "ci95_high_percentage_points"])
    order = [("planner_only", "Hausa"), ("planner_only", "Yoruba"), ("critic_only", "Hausa"), ("critic_only", "Yoruba"), ("both", "Hausa"), ("both", "Yoruba")]
    labels, vals, lows, highs, colors = [], [], [], [], []
    palette = {"planner_only": "#1b9e77", "critic_only": "#d95f02", "both": "#7570b3"}
    for config, lang in order:
        row = next(r for r in rows if r["config"] == config and r["language"] == lang)
        v = f(row, "delta_steered_minus_baseline_percentage_points")
        labels.append(f"{config.replace('_', ' ')} — {lang}")
        vals.append(v)
        lows.append(f(row, "ci95_low_percentage_points"))
        highs.append(f(row, "ci95_high_percentage_points"))
        colors.append(palette[config])
    y = np.arange(len(vals))
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    ax.axvline(0, color="black", linewidth=0.8)
    for yi, v, lo, hi, color in zip(y, vals, lows, highs, colors):
        ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="o", color=color, ecolor=color, capsize=3, markersize=5)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Steered minus baseline operational failure (percentage points)")
    ax.set_title("Llama 3.3 70B: cross-phase, record-ID-matched alpha-30 contrasts")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save(fig, "iclr2027_figure2_role_strength_forest")


def figure3():
    rows = read_csv("llama33_turbo_role_strength_decision_composition.csv")
    assert len(rows) == 18
    assert_pct(rows, ["explicit_veto_rate_percentage", "explicit_allow_rate_percentage", "no_decision_rate_percentage", "operational_failure_rate_percentage"])
    for row in rows:
        assert row["parser_outcomes_sum_to_harmful_n"] == "true"
        assert row["record_ids_are_unique"] == "true"
        assert row["operational_failure_matches_allow_plus_no_decision"] == "true"
    configs = [("both", 0.0, "Baseline"), ("planner_only", 30.0, "Planner 30"), ("critic_only", 30.0, "Critic 30"), ("both", 30.0, "Both 30")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    colors = ["#1b9e77", "#d95f02", "#7570b3"]
    names = ["VETO", "ALLOW", "No-decision"]
    for ax, lang in zip(axes, ["Hausa", "Yoruba"]):
        x = np.arange(len(configs))
        bottom = np.zeros(len(configs))
        for key, color, name in zip(["explicit_veto_rate_percentage", "explicit_allow_rate_percentage", "no_decision_rate_percentage"], colors, names):
            vals = []
            for config, alpha, _ in configs:
                row = next(r for r in rows if r["config"] == config and r["language"] == lang and math.isclose(f(r, "alpha"), alpha))
                vals.append(f(row, key))
            ax.bar(x, vals, bottom=bottom, color=color, label=name)
            bottom += np.array(vals)
        ax.set_xticks(x, [c[2] for c in configs], rotation=35, ha="right")
        ax.set_title(lang)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage of harmful prompts")
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("Decision composition changes across roles and languages", fontsize=10)
    fig.tight_layout()
    save(fig, "iclr2027_figure3_decision_composition")


def figure4():
    entries = []
    llama = read_csv("llama33_planner20_paired_inference.csv")
    assert len(llama) == 4
    for r in llama:
        entries.append(("Llama", r["language"], f(r, "delta_steered_minus_baseline_percentage_points"), f(r, "ci95_low_percentage_points"), f(r, "ci95_high_percentage_points"), r["mcnemar_holm_reject_alpha_0_05"] == "true"))
    qwen = read_csv("qwen25_72b_primary_paired_inference_holm.csv")
    qwen_rows = [r for r in qwen if r["prompt_class"] == "harmful" and r["steered_config"] == "planner_only" and r["alpha"] == "1.0"]
    assert len(qwen_rows) == 4
    for r in qwen_rows:
        entries.append(("Qwen", r["language"], f(r, "delta_steered_minus_baseline_percentage_points"), f(r, "ci95_low_percentage_points"), f(r, "ci95_high_percentage_points"), r.get("mcnemar_holm_reject_alpha_0_05", "false") == "true"))
    gemma = read_csv("gemma_4_31b_phase1_complete_case_paired_inference_holm.csv")
    gemma_rows = [r for r in gemma if r["prompt_class"] == "harmful"]
    assert len(gemma_rows) == 1
    for r in gemma_rows:
        entries.append(("Gemma", r["language"], f(r, "delta_steered_minus_baseline_percentage_points"), f(r, "ci95_low_percentage_points"), f(r, "ci95_high_percentage_points"), r["mcnemar_holm_reject_alpha_0_05"] == "true"))
    gpt = read_csv("gpt_oss_20b_planner_alpha1_paired_effect_holm.csv")
    assert len(gpt) == 4
    gpt_ci = {r["language"]: r for r in read_csv("gpt_oss_20b_planner_alpha1_paired_bootstrap_ci.csv")}
    for r in gpt:
        ci = gpt_ci[r["language"]]
        entries.append(("GPT-OSS", r["language"], f(r, "absolute_change_percentage_points"), f(ci, "ci95_low_percentage_points"), f(ci, "ci95_high_percentage_points"), r["mcnemar_holm_reject_alpha_0_05"] == "true"))
    assert_pct([{"v": str(v), "lo": str(lo), "hi": str(hi)} for _, _, v, lo, hi, _ in entries], ["v", "lo", "hi"])
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    y = np.arange(len(entries))
    colors = {"Llama": "#1b9e77", "Qwen": "#377eb8", "Gemma": "#984ea3", "GPT-OSS": "#e41a1c"}
    for yi, (model, lang, v, lo, hi, sig) in zip(y, entries):
        ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="o", color=colors[model], ecolor=colors[model], capsize=2, markersize=4)
        if sig:
            ax.text(hi + 0.7 if v >= 0 else lo - 0.7, yi, "*", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    labels = [f"{m} — {l}" for m, l, *_ in entries]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Steered minus baseline operational failure (percentage points)")
    ax.set_title("Planner-only effects are model- and language-dependent")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save(fig, "iclr2027_figure4_cross_model_effects")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    print("Generated four ICLR 2027 main-paper figures.")
