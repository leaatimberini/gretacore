#!/usr/bin/env python3
"""
B3.64: Numerical Drift Audit Analyzer

Compares prefill_last vs decode0 for the same logical token.
Produces MAE table and root cause verdict.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def hash_to_float(h64: str) -> float:
    """Convert hex hash64 to float for comparison."""
    try:
        return float.fromhex(h64.replace("0x", ""))
    except ValueError:
        return float('nan')


def load_traces(traces_dir: str) -> dict:
    """Load all JSONL traces into structured dict."""
    traces = defaultdict(list)
    for f in Path(traces_dir).glob("*.jsonl"):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    entry = json.loads(line)
                    traces[entry.get("prompt_id", "unknown")].append(entry)
    return dict(traces)


def pair_traces(prefill_traces: list, decode_traces: list) -> list:
    """
    Pair prefill_last with decode0 by (pos_id, logical_tok_idx, layer, tensor).
    Returns list of (prefill_entry, decode_entry) tuples.
    """
    # Index decode traces by key
    decode_index = {}
    for entry in decode_traces:
        key = (
            entry.get("pos_id"),
            entry.get("logical_tok_idx"),
            entry.get("layer"),
            entry.get("tensor")
        )
        if None not in key:
            decode_index[key] = entry
    
    pairs = []
    for pre in prefill_traces:
        key = (
            pre.get("pos_id"),
            pre.get("logical_tok_idx"),
            pre.get("layer"),
            pre.get("tensor")
        )
        if None not in key and key in decode_index:
            pairs.append((pre, decode_index[key]))
    
    return pairs


def calculate_mae(pre: dict, dec: dict) -> float:
    """Calculate MAE from abs_sum values if available."""
    pre_sum = pre.get("abs_sum", 0)
    dec_sum = dec.get("abs_sum", 0)
    if isinstance(pre_sum, (int, float)) and isinstance(dec_sum, (int, float)):
        return abs(pre_sum - dec_sum)
    return float('nan')


def extract_topk(logits_entry: dict) -> tuple:
    """Extract top1 id/logit and top5 from logits entry."""
    topk = logits_entry.get("topk", {})
    top1_id = topk.get("top1_id", -1)
    top1_logit = topk.get("top1_logit", float('nan'))
    top5_ids = topk.get("top5_ids", [])
    top5_logits = topk.get("top5_logits", [])
    return top1_id, top1_logit, top5_ids, top5_logits


def main():
    parser = argparse.ArgumentParser(description="B3.64 Numerical Drift Analyzer")
    parser.add_argument("--dir", required=True, help="Artifacts dir (e.g., artifacts_remote/2026-02-06/b3_64)")
    parser.add_argument("--out", required=True, help="Output file path")
    args = parser.parse_args()
    
    traces_dir = os.path.join(args.dir, "traces")
    if not os.path.exists(traces_dir):
        print(f"ERROR: Traces dir not found: {traces_dir}")
        sys.exit(1)
    
    # Load traces
    all_traces = load_traces(traces_dir)
    print(f"Loaded {len(all_traces)} prompts from traces")
    
    # Process each prompt
    results = []
    for prompt_id, entries in all_traces.items():
        # Separate by phase
        prefill = [e for e in entries if e.get("phase") == "prefill_last"]
        decode0 = [e for e in entries if e.get("phase") == "decode0"]
        
        # Pair traces
        pairs = pair_traces(prefill, decode0)
        print(f"Prompt {prompt_id}: {len(pairs)} pairs found")
        
        for pre, dec in pairs:
            mae = calculate_mae(pre, dec)
            top1_pre = extract_topk(pre)
            top1_dec = extract_topk(dec)
            top1_match = (top1_pre[0] == top1_dec[0])
            
            results.append({
                "prompt_id": prompt_id,
                "pos_id": pre.get("pos_id"),
                "logical_tok_idx": pre.get("logical_tok_idx"),
                "layer": pre.get("layer"),
                "tensor": pre.get("tensor"),
                "mae": mae,
                "top1_pre": top1_pre[0],
                "top1_dec": top1_dec[0],
                "top1_match": top1_match
            })
    
    # Generate output table
    output_lines = []
    output_lines.append("=" * 120)
    output_lines.append("B3.64 NUMERICAL DRIFT AUDIT ANALYSIS")
    output_lines.append("=" * 120)
    output_lines.append("")
    
    # Table header
    header = ["PROMPT", "POS", "TOK_IDX", "LAYER", "TENSOR", "MAE", "TOP1_PRE", "TOP1_DEC", "MATCH"]
    output_lines.append("| " + " | ".join(header) + " |")
    output_lines.append("|" + "|".join(["-" * len(h) for h in header]) + "|")
    
    # Table rows
    first_fail = None
    for r in results:
        mae_str = f"{r['mae']:.6e}" if r['mae'] != float('nan') else "N/A"
        row = [
            r["prompt_id"],
            str(r.get("pos_id", "N/A")),
            str(r.get("logical_tok_idx", "N/A")),
            str(r.get("layer", "N/A")),
            r.get("tensor", "N/A")[:20],
            mae_str,
            str(r["top1_pre"]),
            str(r["top1_dec"]),
            "✓" if r["top1_match"] else "✗"
        ]
        output_lines.append("| " + " | ".join(row) + " |")
        if first_fail is None and (not r["top1_match"] or (r['mae'] != float('nan') and r['mae'] > 1e-6)):
            first_fail = r
    
    output_lines.append("")
    
    # Determine root cause
    root_cause = "PASS"
    if first_fail:
        tensor = first_fail.get("tensor", "")
        if "norm" in tensor.lower():
            root_cause = "NORM_NUMERICS"
        elif "q_pre" in tensor or "q_post" in tensor:
            root_cause = "ROPE_NUMERICS"
        elif "attn_out" in tensor:
            root_cause = "ATTN_NUMERICS"
        elif "residual" in tensor.lower():
            root_cause = "RESIDUAL_NUMERICS"
        elif "ffn" in tensor.lower() or "ffn_norm" in tensor.lower():
            root_cause = "FFN_NORM_NUMERICS"
        elif "logits" in tensor.lower():
            root_cause = "LOGITS_NUMERICS"
        else:
            root_cause = "UNKNOWN_NUMERICS"
    elif len(pairs) == 0:
        root_cause = "TRACE_OFFSET"
    
    output_lines.append("-" * 120)
    output_lines.append(f"FIRST_FAIL: {first_fail}")
    output_lines.append(f"ROOT_CAUSE: {root_cause}")
    output_lines.append(f"TOTAL_PAIRS: {len(results)}")
    output_lines.append(f"MATCH_RATE: {sum(1 for r in results if r['top1_match']) / max(len(results), 1) * 100:.1f}%")
    output_lines.append("=" * 120)
    
    # Write output
    with open(args.out, "w") as f:
        f.write("\n".join(output_lines))
    
    print(f"Analysis written to: {args.out}")
    print(f"ROOT_CAUSE: {root_cause}")


if __name__ == "__main__":
    main()
