from pathlib import Path
import csv

OUT = Path("paper_artifacts/metrics/repository_artifact_chain_status.csv")

PACKAGES = {
    "qwen25_72b_primary": {
        "tokens": ["qwen25_72b", "qwen_"],
        "sources": [Path("data/derived/canonical/qwen25_72b_primary_role_strength.parquet")],
        "adapters": [Path("analysis/adapters/qwen25_72b_primary.py")],
    },
    "gpt_oss_20b_phase1": {
        "tokens": ["gpt_oss_20b"],
        "sources": [
            Path("data/derived/canonical/gpt_oss_20b_phase1_baseline.parquet"),
            Path("data/derived/canonical/gpt_oss_20b_phase1_planner_only_alpha1.parquet"),
        ],
        "adapters": [
            Path("analysis/adapters/gpt_oss_baseline.py"),
            Path("analysis/adapters/gpt_oss_planner_alpha1.py"),
        ],
    },
    "gemma_4_31b_phase1": {
        "tokens": ["gemma_4_31b_phase1"],
        "sources": [Path("data/derived/canonical/gemma_4_31b_phase1_complete_case.parquet")],
        "adapters": [Path("analysis/adapters/gemma_4_31b_complete_case.py")],
    },
    "gpt_5_frontier_baseline": {
        "tokens": ["gpt_5_frontier_baseline"],
        "sources": [],
        "adapters": [],
    },
    "claude_sonnet_4_vs_5": {
        "tokens": ["claude_sonnet_4_vs_5"],
        "sources": [],
        "adapters": [],
    },
    "llama33_phase1_main": {
        "tokens": ["llama33_phase1", "llama33_planner20", "llama33_critic20"],
        "sources": [Path("data/derived/canonical/llama33_phase1_main.parquet")],
        "adapters": [Path("analysis/adapters/llama_main.py")],
    },
    "llama33_phase2_turbo": {
        "tokens": ["llama33_phase2", "llama33_turbo"],
        "sources": [Path("data/derived/canonical/llama33_phase2_turbo.parquet")],
        "adapters": [Path("analysis/adapters/llama_turbo.py")],
    },
    "addition_1_causal_patching": {
        "tokens": ["addition_1", "patching"],
        "sources": [],
        "adapters": [],
    },
    "addition_4_calibration": {
        "tokens": ["addition_4", "calibration"],
        "sources": [],
        "adapters": [],
    },
    "addition_5_adversarial_robustness": {
        "tokens": ["addition_5", "adversarial"],
        "sources": [],
        "adapters": [],
    },
    "addition_6_prompt_filter_baseline": {
        "tokens": ["addition_6", "addition6", "prompt_filter"],
        "sources": [],
        "adapters": [],
    },
}

ARTIFACT_ROOTS = [
    Path("paper_artifacts/metrics"),
    Path("paper_artifacts/tables"),
    Path("paper_artifacts/figures"),
]

files = []
for root in ARTIFACT_ROOTS:
    if root.exists():
        files.extend(path for path in root.rglob("*") if path.is_file())

def matches(path: Path, tokens: list[str]) -> bool:
    text = str(path).lower()
    return any(token.lower() in text for token in tokens)

rows = []
for package, config in PACKAGES.items():
    matched = [path for path in files if matches(path, config["tokens"])]

    metric_csvs = [
        path for path in matched
        if path.suffix == ".csv"
    ]
    tables = [
        path for path in matched
        if path.suffix == ".tex"
    ]
    figures = [
        path for path in matched
        if path.suffix.lower() in {".pdf", ".png", ".svg"}
    ]
    manifests = [
        path for path in matched
        if path.suffix == ".json" and "manifest" in path.name.lower()
    ]
    handoffs = [
        path for path in matched
        if path.suffix == ".md" and "handoff" in path.name.lower()
    ]
    verifiers = [
        path for path in Path("analysis/metrics").glob("verify_*.py")
        if matches(path, config["tokens"])
    ]

    canonical_ready = bool(config["sources"]) and all(
        path.exists() for path in config["sources"]
    )
    adapter_ready = bool(config["adapters"]) and all(
        path.exists() for path in config["adapters"]
    )

    rows.append({
        "package": package,
        "canonical_parquet_ready": canonical_ready,
        "adapter_ready": adapter_ready,
        "metric_csv_n": len(metric_csvs),
        "latex_table_n": len(tables),
        "figure_n": len(figures),
        "manifest_n": len(manifests),
        "handoff_n": len(handoffs),
        "verifier_script_n": len(verifiers),
        "strict_chain_complete": (
            canonical_ready
            and adapter_ready
            and bool(metric_csvs)
            and bool(tables)
            and bool(figures)
            and bool(manifests)
            and bool(handoffs)
            and bool(verifiers)
        ),
    })

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote: {OUT}")
print(f"Packages checked: {len(rows)}")
print(
    "Strict-chain-complete packages: "
    f"{sum(row['strict_chain_complete'] for row in rows)}"
)

for row in rows:
    print(
        f"{row['package']}: "
        f"canonical={row['canonical_parquet_ready']} "
        f"adapter={row['adapter_ready']} "
        f"metrics={row['metric_csv_n']} "
        f"tables={row['latex_table_n']} "
        f"figures={row['figure_n']} "
        f"manifest={row['manifest_n']} "
        f"handoff={row['handoff_n']} "
        f"verifier={row['verifier_script_n']} "
        f"complete={row['strict_chain_complete']}"
    )
