#!/usr/bin/env python3
"""
B3.66 Prefill vs Decode Drift Analyzer

Strict pairing of prefill_last and decode0 tensors to identify first divergence.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


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
        for t in dc_traces:
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


def write_report(analysis: dict, output: str):
    """Write markdown and TSV reports."""
    # Markdown table
    with open(output, "w") as f:
        f.write("# B3.66 Prefill vs Decode Drift Analysis\n\n")
        f.write(f"**Date**: 2026-02-07\n\n")
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
            f.write(f"- {cause}: {count}\n\n")
        
        f.write("## Detailed Results\n\n")
        f.write("| PROMPT | TOKEN_ID | POS_ID | LAYER | TENSOR | FIRST_FAIL | ROOT_CAUSE | DETAILS |\n")
        f.write("|--------|----------|--------|-------|--------|------------|------------|---------|\n")
        
        for r in analysis['results']:
            details = str(r.get('details', {}))
            f.write(f"| {r.get('prompt', '-')} | {r.get('token_id', '-')} | "
                    f"{r.get('pos_id', '-')} | {r.get('layer', '-')} | "
                    f"{r.get('tensor', '-')} | {r.get('first_fail', '-')} | "
                    f"{r.get('root_cause', '-')} | {details[:50]} |\n")
    
    # TSV summary
    tsv_path = output.replace(".txt", ".tsv")
    with open(tsv_path, "w") as f:
        f.write("prompt\ttoken_id\tpos_id\tlayer\ttensor\tfirst_fail\troot_cause\tstatus\n")
        for r in analysis['results']:
            f.write(f"{r.get('prompt', '-')}\t{r.get('token_id', '-')}\t"
                    f"{r.get('pos_id', '-')}\t{r.get('layer', '-')}\t"
                    f"{r.get('tensor', '-')}\t{r.get('first_fail', '-')}\t"
                    f"{r.get('root_cause', '-')}\t{r.get('status', '-')}\n")
    
    print(f"Report written to: {output}")
    print(f"TSV written to: {tsv_path}")


def main():
    parser = argparse.ArgumentParser(description="B3.66 Drift Analyzer")
    parser.add_argument("--dir", required=True, help="Artifacts directory with traces/")
    parser.add_argument("--out", required=True, help="Output markdown file")
    args = parser.parse_args()
    
    traces = load_traces(os.path.join(args.dir, "traces"))
    pairs = build_pairs(traces)
    analysis = analyze_pairs(pairs)
    write_report(analysis, args.out)


if __name__ == "__main__":
    main()
