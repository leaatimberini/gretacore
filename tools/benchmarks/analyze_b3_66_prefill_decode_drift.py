#!/usr/bin/env python3
"""
B3.66 Prefill vs Decode Drift Analyzer v2

Enhanced with kv_aligned mode for probing KV alignment and attention semantics.
Supports two modes:
  - as_designed: Original drift detection (prefill_last vs decode0 pairing)
  - kv_aligned: Deep probe of Q/K/V projections and attention scores
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
import statistics


def load_traces(traces_dir: str) -> dict:
    """Load all trace JSONL files into dict keyed by (prompt, phase)."""
    traces = defaultdict(list)
    traces_dir = Path(traces_dir)
    
    for f in traces_dir.glob("*_trace.jsonl"):
        prompt = f.stem.replace("_trace", "")
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    traces[(prompt, "prefill_last")].append(json.loads(line))
    
    # Also look for decode0 traces
    for f in traces_dir.glob("*_decode_trace.jsonl"):
        prompt = f.stem.replace("_decode_trace", "")
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    traces[(prompt, "decode0")].append(json.loads(line))
    
    return traces


def load_kv_aligned_traces(traces_dir: str) -> dict:
    """Load traces specifically for kv_aligned mode analysis."""
    traces = defaultdict(list)
    traces_dir = Path(traces_dir)
    
    for f in traces_dir.glob("*.jsonl"):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    data = json.loads(line)
                    # Categorize by tensor type for kv_aligned analysis
                    tensor = data.get("tensor", "")
                    if "q_proj" in tensor.lower():
                        traces["q_proj"].append(data)
                    elif "k_proj" in tensor.lower():
                        traces["k_proj"].append(data)
                    elif "v_proj" in tensor.lower():
                        traces["v_proj"].append(data)
                    elif "attention" in tensor.lower() or "attn" in tensor.lower():
                        traces["attention"].append(data)
    
    return traces


def build_pairs(traces: dict) -> list:
    """Build strict pairs based on matching keys."""
    pairs = []
    
    # Get all prompts and their traces
    prompt_phases = defaultdict(lambda: {"prefill_last": [], "decode0": []})
    for (prompt, phase), items in traces.items():
        prompt_phases[prompt][phase] = items
    
    for prompt, phases in prompt_phases.items():
        pf_traces = phases["prefill_last"]
        dc_traces = phases["decode0"]
        
        # Index decode traces by matching keys
        dc_by_key = {}
        for t in dc_trases:
            key = (t.get("token_id"), t.get("logical_tok_idx"), 
                   t.get("pos_id"), t.get("layer"), t.get("tensor"))
            dc_by_key[key] = t
        
        # Pair each prefill trace with matching decode trace
        for pf in pf_traces:
            key = (pf.get("token_id"), pf.get("logical_tok_idx"),
                   pf.get("pos_id"), pf.get("layer"), pf.get("tensor"))
            if key in dc_by_key:
                pairs.append({
                    "prompt": prompt,
                    "prefill": pf,
                    "decode": dc_by_key[key],
                    "pair_key": key
                })
            else:
                pairs.append({
                    "prompt": prompt,
                    "prefill": pf,
                    "decode": None,
                    "pair_key": key,
                    "status": "MISSING_PAIR"
                })
    
    return pairs


def check_tensor_drift(prefill: dict, decode: dict) -> dict:
    """Check for drift between paired tensors."""
    result = {
        "tensor": prefill.get("tensor"),
        "layer": prefill.get("layer"),
        "status": "PASS",
        "root_cause": None,
        "details": {}
    }
    
    if decode is None:
        result["status"] = "MISSING_PAIR"
        result["root_cause"] = "TRACE_OFFSET"
        return result
    
    # Check hash match (primary indicator)
    pf_hash = prefill.get("hash")
    dc_hash = decode.get("hash")
    
    if pf_hash != dc_hash:
        result["status"] = "FAIL"
        result["root_cause"] = f"{prefill.get('tensor').upper()}_DRIFT"
        result["details"]["hash_mismatch"] = True
        result["details"]["prefill_hash"] = pf_hash
        result["details"]["decode_hash"] = dc_hash
    else:
        result["details"]["hash_match"] = True
    
    # Check nz_count delta
    pf_nz = prefill.get("nz_count", 0)
    dc_nz = decode.get("nz_count", 0)
    if pf_nz != dc_nz:
        result["details"]["nz_delta"] = pf_nz - dc_nz
    
    # Check abs_sum delta
    pf_sum = prefill.get("abs_sum", 0.0)
    dc_sum = decode.get("abs_sum", 0.0)
    if abs(pf_sum - dc_sum) > 1e-6:
        result["details"]["abs_sum_delta"] = pf_sum - dc_sum
    
    return result


def analyze_pairs(pairs: list) -> dict:
    """Analyze all pairs and determine first failure."""
    results = []
    fail_by_cause = defaultdict(int)
    
    for pair in pairs:
        if pair.get("status") == "MISSING_PAIR":
            results.append({
                "prompt": pair["prompt"],
                "token_id": pair["prefill"].get("token_id"),
                "pos_id": pair["prefill"].get("pos_id"),
                "layer": pair["prefill"].get("layer"),
                "tensor": pair["prefill"].get("tensor"),
                "first_fail": "TRACE_OFFSET",
                "root_cause": "TRACE_OFFSET",
                "details": "No matching decode trace found"
            })
            fail_by_cause["TRACE_OFFSET"] += 1
        else:
            drift = check_tensor_drift(pair["prefill"], pair["decode"])
            results.append({
                "prompt": pair["prompt"],
                "token_id": pair["prefill"].get("token_id"),
                "pos_id": pair["prefill"].get("pos_id"),
                "layer": pair["prefill"].get("layer"),
                "tensor": pair["prefill"].get("tensor"),
                "first_fail": drift["root_cause"] if drift["status"] == "FAIL" else None,
                "root_cause": drift["root_cause"],
                "status": drift["status"],
                "details": drift["details"]
            })
            if drift["status"] == "FAIL":
                fail_by_cause[drift["root_cause"]] += 1
    
    # Find first failure in layer order
    fail_order = [
        "EMBED_DRIFT", "RESIDUAL_PRE_ATTN_DRIFT", "ATTN_NORM_DRIFT",
        "Q_PRE_ROPE_DRIFT", "ROPE_DRIFT", "ATTN_OUT_DRIFT",
        "RESIDUAL_POST_ATTN_DRIFT", "FFN_NORM_DRIFT", "MLP_OUT_DRIFT",
        "RESIDUAL_POST_MLP_DRIFT", "LOGITS_DRIFT"
    ]
    
    first_fail = None
    for cause in fail_order:
        if cause in fail_by_cause:
            first_fail = cause
            break
    
    return {
        "results": results,
        "fail_by_cause": dict(fail_by_cause),
        "first_fail": first_fail,
        "total_pairs": len(pairs),
        "missing_pairs": sum(1 for r in results if r.get("status") == "MISSING_PAIR"),
        "pass_count": sum(1 for r in results if r.get("status") == "PASS"),
        "fail_count": sum(1 for r in results if r.get("status") == "FAIL")
    }


# ============================================================================
# KV_ALIGNED MODE ANALYSIS
# ============================================================================

def analyze_kv_aligned(traces: dict, mode: str) -> dict:
    """
    Deep probe analysis for kv_aligned mode.
    
    Registers:
    - q_proj, k_proj, v_proj hashes
    - Attention scores (pre-softmax, post-softmax)
    - KV alignment indicators
    """
    result = {
        "mode": mode,
        "projection_hashes": {
            "q_proj": [],
            "k_proj": [],
            "v_proj": []
        },
        "attention_stats": {
            "pre_softmax": {"min": None, "max": None, "mean": None},
            "post_softmax": {"min": None, "max": None, "mean": None}
        },
        "kv_alignment": {
            "k_hash_consistency": None,
            "v_hash_consistency": None,
            "k_v_aligned": None
        },
        "status": "PASS"
    }
    
    # Collect projection hashes
    for q_item in traces.get("q_proj", []):
        result["projection_hashes"]["q_proj"].append({
            "hash": q_item.get("hash"),
            "layer": q_item.get("layer"),
            "prompt": q_item.get("prompt", "unknown")
        })
    
    for k_item in traces.get("k_proj", []):
        result["projection_hashes"]["k_proj"].append({
            "hash": k_item.get("hash"),
            "layer": k_item.get("layer"),
            "prompt": k_item.get("prompt", "unknown")
        })
    
    for v_item in traces.get("v_proj", []):
        result["projection_hashes"]["v_proj"].append({
            "hash": v_item.get("hash"),
            "layer": v_item.get("layer"),
            "prompt": v_item.get("prompt", "unknown")
        })
    
    # Collect attention scores
    pre_softmax_vals = []
    post_softmax_vals = []
    
    for attn_item in traces.get("attention", []):
        scores = attn_item.get("attention_scores", [])
        if scores:
            pre_softmax_vals.extend(scores.get("pre_softmax", []))
            post_softmax_vals.extend(scores.get("post_softmax", []))
        
        # Also check for raw score fields
        raw_scores = attn_item.get("raw_scores", [])
        if raw_scores:
            pre_softmax_vals.extend(raw_scores)
        
        softmax_scores = attn_item.get("softmax_scores", [])
        if softmax_scores:
            post_softmax_vals.extend(softmax_scores)
    
    # Compute attention stats
    if pre_softmax_vals:
        result["attention_stats"]["pre_softmax"]["min"] = min(pre_softmax_vals)
        result["attention_stats"]["pre_softmax"]["max"] = max(pre_softmax_vals)
        result["attention_stats"]["pre_softmax"]["mean"] = statistics.mean(pre_softmax_vals)
    
    if post_softmax_vals:
        result["attention_stats"]["post_softmax"]["min"] = min(post_softmax_vals)
        result["attention_stats"]["post_softmax"]["max"] = max(post_softmax_vals)
        result["attention_stats"]["post_softmax"]["mean"] = statistics.mean(post_softmax_vals)
    
    # Check KV alignment
    k_hashes = [h["hash"] for h in result["projection_hashes"]["k_proj"]]
    v_hashes = [h["hash"] for h in result["projection_hashes"]["v_proj"]]
    
    if len(set(k_hashes)) == 1:
        result["kv_alignment"]["k_hash_consistency"] = True
    elif k_hashes:
        result["kv_alignment"]["k_hash_consistency"] = False
    
    if len(set(v_hashes)) == 1:
        result["kv_alignment"]["v_hash_consistency"] = True
    elif v_hashes:
        result["kv_alignment"]["v_hash_consistency"] = False
    
    # Determine if KV is aligned (same across prompts for same layer)
    if k_hashes and v_hashes:
        # Check if K and V hashes match for same layer/prompt combos
        k_by_layer = defaultdict(list)
        v_by_layer = defaultdict(list)
        for h in result["projection_hashes"]["k_proj"]:
            k_by_layer[(h["layer"], h["prompt"])].append(h["hash"])
        for h in result["projection_hashes"]["v_proj"]:
            v_by_layer[(h["layer"], h["prompt"])].append(h["hash"])
        
        aligned_count = 0
        total_count = 0
        for key in k_by_layer:
            if key in v_by_layer:
                total_count += 1
                if set(k_by_layer[key]) == set(v_by_layer[key]):
                    aligned_count += 1
        
        if total_count > 0:
            result["kv_alignment"]["k_v_aligned"] = (aligned_count == total_count)
            if not result["kv_alignment"]["k_v_aligned"]:
                result["status"] = "KV_MISMATCH"
    
    return result


# ============================================================================
# REPORTING
# ============================================================================

def write_report(analysis: dict, output: str, mode: str = "as_designed"):
    """Write markdown and TSV reports."""
    
    with open(output, "w") as f:
        f.write(f"# B3.66 Prefill vs Decode Drift Analysis (v2)\n\n")
        f.write(f"**Date**: 2026-02-07\n")
        f.write(f"**Mode**: {mode}\n\n")
        
        if mode == "as_designed":
            write_as_designed_report(f, analysis)
        else:
            write_kv_aligned_report(f, analysis)
        
        # TSV output
        tsv_path = output.replace(".md", ".tsv")
        write_tsv(analysis, tsv_path, mode)


def write_as_designed_report(f, analysis: dict):
    """Write report for as_designed mode."""
    f.write(f"**Total Pairs**: {analysis['total_pairs']}\n")
    f.write(f"**Pass**: {analysis['pass_count']}\n")
    f.write(f"**Fail**: {analysis['fail_count']}\n")
    f.write(f"**Missing Pairs**: {analysis['missing_pairs']}\n\n")
    
    if analysis['first_fail']:
        f.write(f"**FIRST FAILURE**: {analysis['first_fail']}\n\n")
    else:
        f.write("**RESULT**: PASS - All tensors match\n\n")
    
    f.write("## Failure Summary\n\n")
    for cause, count in analysis['fail_by_cause'].items():
        f.write(f"- **{cause}**: {count}\n\n")
    
    f.write("## Detailed Results\n\n")
    f.write("| Prompt | Token | Pos | Layer | Tensor | Status | Root Cause |\n")
    f.write("|--------|-------|-----|-------|--------|--------|------------|\n")
    for r in analysis['results']:
        f.write(f"| {r.get('prompt', '-')} | {r.get('token_id', '-')} | {r.get('pos_id', '-')} | "
                f"{r.get('layer', '-')} | {r.get('tensor', '-')} | {r.get('status', '-')} | "
                f"{r.get('root_cause', '-')} |\n")


def write_kv_aligned_report(f, analysis: dict):
    """Write report for kv_aligned mode."""
    f.write(f"**Status**: {analysis['status']}\n\n")
    
    f.write("## Projection Hashes\n\n")
    for proj_type, items in analysis['projection_hashes'].items():
        f.write(f"### {proj_type.upper()}\n\n")
        f.write("| Layer | Prompt | Hash |\n")
        f.write("|-------|--------|------|\n")
        for item in items:
            f.write(f"| {item['layer']} | {item['prompt']} | `{item['hash']}` |\n")
        f.write("\n")
    
    f.write("## Attention Statistics\n\n")
    f.write("### Pre-Softmax Scores\n\n")
    pre = analysis['attention_stats']['pre_softmax']
    f.write(f"- **Min**: {pre['min']}\n")
    f.write(f"- **Max**: {pre['max']}\n")
    f.write(f"- **Mean**: {pre['mean']}\n\n")
    
    f.write("### Post-Softmax Scores\n\n")
    post = analysis['attention_stats']['post_softmax']
    f.write(f"- **Min**: {post['min']}\n")
    f.write(f"- **Max**: {post['max']}\n")
    f.write(f"- **Mean**: {post['mean']}\n\n")
    
    f.write("## KV Alignment Indicators\n\n")
    kv = analysis['kv_alignment']
    f.write(f"- **K Hash Consistency**: {kv['k_hash_consistency']}\n")
    f.write(f"- **V Hash Consistency**: {kv['v_hash_consistency']}\n")
    f.write(f"- **K-V Aligned**: {kv['k_v_aligned']}\n\n")
    
    if analysis['status'] == "KV_MISMATCH":
        f.write("**WARNING**: KV misalignment detected. This indicates a structural\n")
        f.write("drift between K and V projections that may cause attention score anomalies.\n\n")


def write_tsv(analysis: dict, output: str, mode: str):
    """Write TSV summary for spreadsheet analysis."""
    with open(output, "w") as f:
        f.write("mode\tprompt\tlayer\ttensor\tstatus\troot_cause\thash_match\tfirst_fail\n")
        for r in analysis.get('results', []):
            f.write(f"{mode}\t{r.get('prompt', '-')}\t{r.get('layer', '-')}\t"
                    f"{r.get('tensor', '-')}\t{r.get('status', '-')}\t"
                    f"{r.get('root_cause', '-')}\t{r.get('details', {}).get('hash_mismatch', 'N/A')}\t"
                    f"{r.get('first_fail', '-')}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="B3.66 Prefill vs Decode Drift Analyzer v2"
    )
    parser.add_argument(
        "--traces-dir", "-i",
        required=True,
        help="Directory containing trace JSONL files"
    )
    parser.add_argument(
        "--output", "-o",
        default="B3_66_V2_ANALYSIS.md",
        help="Output markdown file path"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["as_designed", "kv_aligned"],
        default="as_designed",
        help="Analysis mode (default: as_designed)"
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.traces_dir):
        print(f"ERROR: Traces directory not found: {args.traces_dir}")
        sys.exit(1)
    
    if args.mode == "kv_aligned":
        traces = load_kv_aligned_traces(args.traces_dir)
        analysis = analyze_kv_aligned(traces, args.mode)
    else:
        traces = load_traces(args.traces_dir)
        pairs = build_pairs(traces)
        analysis = analyze_pairs(pairs)
    
    write_report(analysis, args.output, args.mode)
    
    print(f"Analysis complete. Results written to: {args.output}")
    print(f"TSV summary written to: {args.output.replace('.md', '.tsv')}")
    
    if args.mode == "kv_aligned":
        print(f"\nKV Alignment Status: {analysis['status']}")
        print(f"K-V Aligned: {analysis['kv_alignment']['k_v_aligned']}")
    else:
        print(f"\nFirst Failure: {analysis['first_fail']}")
        print(f"Pass/Fail: {analysis['pass_count']}/{analysis['fail_count']}")


if __name__ == "__main__":
    main()
