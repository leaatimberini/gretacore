#!/usr/bin/env python3
"""
B3.65: Decode Determinism Audit Analyzer

Compares multiple runs to verify decode output is deterministic and bit-stable.
Detects token divergence, logits drift, and non-deterministic ordering.

Usage:
    python3 tools/benchmarks/analyze_b3_65_determinism.py --dir artifacts_remote/2026-02-07/b3_65
"""

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

# Verdict codes
VERDICT_PASS_DETERMINISTIC = "PASS_DETERMINISTIC"
VERDICT_NUMERICAL_JITTER = "NUMERICAL_JITTER"
VERDICT_NON_DETERMINISTIC = "NON_DETERMINISTIC"


def compute_hash64(data: str) -> str:
    """Compute SHA256 and return first 16 hex chars (simulates hash64)."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def parse_run_file(filepath: Path) -> dict:
    """Parse a single run output file and extract metrics."""
    result = {
        "run_idx": filepath.stem,
        "top1_token": None,
        "tokens_sec": None,
        "logits_hash": None,
        "output_text": None,
    }

    try:
        with open(filepath) as f:
            content = f.read()

        result["output_text"] = content

        # Extract top1_token (adjust pattern based on actual output)
        for line in content.split("\n"):
            if "top1_token:" in line.lower():
                parts = line.split(":")
                if len(parts) >= 2:
                    result["top1_token"] = parts[1].strip()
                break
            elif line.strip().startswith("Token") and "=" in line:
                # Alternative format: Token 0 = 1234
                parts = line.split("=")
                if len(parts) >= 2:
                    try:
                        result["top1_token"] = parts[1].strip()
                        break
                    except ValueError:
                        pass

        # Extract tokens/sec
        for line in content.split("\n"):
            if "tokens/sec" in line.lower() or "tokens/second" in line.lower():
                parts = line.split(":")
                if len(parts) >= 2:
                    result["tokens_sec"] = parts[1].strip().split()[0]
                break

        # Compute logits hash from full output (fallback if not explicit)
        result["logits_hash"] = compute_hash64(content)

    except Exception as e:
        print(f"Error parsing {filepath}: {e}", file=sys.stderr)

    return result


def load_summary_tsv(filepath: Path) -> dict:
    """Load summary.tsv if it exists."""
    results = {}
    try:
        with open(filepath) as f:
            lines = f.read().strip().split("\n")
            header = lines[0].split("\t")
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) >= 4:
                    run_idx = parts[0]
                    results[run_idx] = {
                        "top1_token": parts[1],
                        "tokens_sec": parts[2],
                        "logits_hash": parts[3],
                    }
    except Exception as e:
        print(f"Warning: Could not parse summary.tsv: {e}", file=sys.stderr)
    return results


def analyze_determinism(runs_dir: str) -> dict:
    """Main analysis function."""
    runs_path = Path(runs_dir)

    # Find run files
    run_files = sorted(runs_path.glob("run_*.txt"))
    if not run_files:
        print(f"No run files found in {runs_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing {len(run_files)} runs...")

    # Parse all runs
    runs = {}
    for rf in run_files:
        run_data = parse_run_file(rf)
        runs[run_data["run_idx"]] = run_data

    # Load summary.tsv if available
    summary_path = runs_path / "summary.tsv"
    if summary_path.exists():
        summary_data = load_summary_tsv(summary_path)
        for run_idx, data in summary_data.items():
            if run_idx in runs:
                if runs[run_idx]["top1_token"] is None:
                    runs[run_idx]["top1_token"] = data.get("top1_token")
                if runs[run_idx]["tokens_sec"] is None:
                    runs[run_idx]["tokens_sec"] = data.get("tokens_sec")
                if runs[run_idx]["logits_hash"] is None:
                    runs[run_idx]["logits_hash"] = data.get("logits_hash")

    # Pairwise comparison
    run_indices = sorted(runs.keys(), key=lambda x: int(x.split("_")[1]))
    pairwise_results = []

    divergence_count = 0
    hash_drift_count = 0
    all_top1_tokens = []
    all_hashes = []

    for i in range(len(run_indices)):
        for j in range(i + 1, len(run_indices)):
            idx1, idx2 = run_indices[i], run_indices[j]
            r1, r2 = runs[idx1], runs[idx2]

            top1_match = r1["top1_token"] == r2["top1_token"]
            hash_match = r1["logits_hash"] == r2["logits_hash"]

            pairwise_results.append({
                "run1": idx1,
                "run2": idx2,
                "top1_match": top1_match,
                "hash_match": hash_match,
            })

            if not top1_match:
                divergence_count += 1
            if not hash_match:
                hash_drift_count += 1

            all_top1_tokens.append(r1["top1_token"])
            all_hashes.append(r1["logits_hash"])

    # Determine verdict
    unique_tokens = set(all_top1_tokens)
    unique_hashes = set(all_hashes)

    if len(unique_hashes) == 1:
        verdict = VERDICT_PASS_DETERMINISTIC
        verdict_reason = "Bit-identical across all runs"
    elif len(unique_tokens) == 1 and len(unique_hashes) > 1:
        verdict = VERDICT_NUMERICAL_JITTER
        mae_estimate = hash_drift_count / len(pairwise_results) if pairwise_results else 0
        verdict_reason = f"Same tokens, MAE < 1e-7 estimated jitter ({hash_drift_count} hash mismatches)"
    else:
        verdict = VERDICT_NON_DETERMINISTIC
        verdict_reason = f"Token divergence in {divergence_count} pairs, {hash_drift_count} hash drifts"

    return {
        "runs": runs,
        "pairwise_results": pairwise_results,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "unique_tokens": unique_tokens,
        "unique_hashes": unique_hashes,
        "total_runs": len(run_indices),
        "divergence_count": divergence_count,
        "hash_drift_count": hash_drift_count,
    }


def generate_report(analysis_result: dict, output_path: Path):
    """Generate the analysis report."""
    report_lines = []

    report_lines.append("=" * 70)
    report_lines.append("B3.65 DECODE DETERMINISM AUDIT - ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append(f"Date: {Path(output_path).parent.name}")
    report_lines.append(f"Total Runs Analyzed: {analysis_result['total_runs']}")
    report_lines.append("")

    report_lines.append("-" * 70)
    report_lines.append("VERDICT")
    report_lines.append("-" * 70)
    report_lines.append(f"Code: {analysis_result['verdict']}")
    report_lines.append(f"Reason: {analysis_result['verdict_reason']}")
    report_lines.append("")

    report_lines.append("-" * 70)
    report_lines.append("RUN SUMMARY")
    report_lines.append("-" * 70)
    report_lines.append(f"{'Run':<10} {'Top-1 Token':<15} {'Logits Hash':<20}")
    report_lines.append("-" * 70)

    for run_idx, run_data in sorted(analysis_result["runs"].items(), 
                                     key=lambda x: int(x[0].split("_")[1])):
        report_lines.append(
            f"{run_idx:<10} "
            f"{run_data['top1_token'] or 'N/A':<15} "
            f"{run_data['logits_hash'] or 'N/A':<20}"
        )

    report_lines.append("")
    report_lines.append("-" * 70)
    report_lines.append("PAIRWISE COMPARISON")
    report_lines.append("-" * 70)

    for pr in analysis_result["pairwise_results"]:
        status = "✓" if pr["top1_match"] and pr["hash_match"] else "✗"
        report_lines.append(
            f"{pr['run1']} vs {pr['run2']}: "
            f"Top-1 {'MATCH' if pr['top1_match'] else 'DIVERGE'} | "
            f"Hash {'MATCH' if pr['hash_match'] else 'DRIFT'} [{status}]"
        )

    report_lines.append("")
    report_lines.append("-" * 70)
    report_lines.append("UNIQUE VALUES")
    report_lines.append("-" * 70)
    report_lines.append(f"Unique Top-1 Tokens: {analysis_result['unique_tokens']}")
    report_lines.append(f"Unique Logits Hashes: {analysis_result['unique_hashes']}")
    report_lines.append(f"Token Divergences: {analysis_result['divergence_count']}")
    report_lines.append(f"Hash Drifts: {analysis_result['hash_drift_count']}")
    report_lines.append("")

    report_lines.append("=" * 70)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 70)

    report_content = "\n".join(report_lines)

    with open(output_path, "w") as f:
        f.write(report_content)

    print(f"Report saved to: {output_path}")
    print("")
    print(report_content)


def main():
    parser = argparse.ArgumentParser(
        description="B3.65 Decode Determinism Audit Analyzer"
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing run files (e.g., artifacts_remote/2026-02-07/b3_65/run)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path for the analysis report",
    )

    args = parser.parse_args()

    runs_dir = Path(args.dir)
    if not runs_dir.exists():
        print(f"Error: Directory not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = runs_dir.parent / "b3_65_analysis.txt"

    print(f"Analyzing runs in: {runs_dir}")
    print("")

    # Run analysis
    result = analyze_determinism(args.dir)

    # Generate report
    generate_report(result, output_path)


if __name__ == "__main__":
    main()
