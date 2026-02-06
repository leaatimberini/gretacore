#!/usr/bin/env python3
"""
analyze_b3_61_residual_stream_bisect.py
Residual Stream Bisect Analyzer (B3.61)

Deterministically identifies the FIRST_FAIL tensor in the residual stream
across multiple transformer layers at extended context positions.

Failure stages:
- RESIDUAL_PRE_ATTN: residual state before attention
- ATTN_IN: q/k/v projection inputs
- ATTN_OUT: attention output after softmax
- RESIDUAL_POST_ATTN: residual addition after attention
- FFN_NORM_IN: feed-forward input after layer norm
- MLP_OUT: feed-forward network output
- RESIDUAL_POST_MLP: final residual after MLP
- LOGITS: final output logits
"""

import json
import os
import argparse
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime

# Root cause classification enums
ROOT_CAUSE = {
    'PASS': 'PASS',
    'ROUTING_SELECTION': 'ROUTING/SELECTION',  # Buffer divergence before attention
    'ATTN_KERNEL_INPUTS': 'ATTN_KERNEL_INPUTS',  # q/k/v projection inputs
    'ATTENTION_MECHANISM': 'ATTENTION_MECHANISM',  # Attention computation itself
    'RESIDUAL_ADD': 'RESIDUAL_ADD',  # Residual addition after attention
    'FFN_NORM_PATH': 'FFN_NORM_PATH',  # FFN normalization or MLP
    'MLP_OUTPUT': 'MLP_OUTPUT',  # MLP output before residual
    'UNKNOWN': 'UNKNOWN',
}

# Residual stream trace points in order
TRACE_STAGES = [
    'embed_out',           # Embedding output (B3.59 sanity check)
    'residual_pre_attn',   # Residual before attention
    'attn_in',             # Attention input (q/k/v projections)
    'q_pre_rope',          # Query before RoPE
    'k_pre_rope',          # Key before RoPE
    'q_post_rope',         # Query after RoPE
    'k_post_rope',         # Key after RoPE
    'attn_out',           # Attention output
    'residual_post_attn', # Residual after attention addition
    'ffn_norm_in',        # FFN layer norm input
    'mlp_out',            # MLP output
    'residual_post_mlp',  # Residual after MLP addition
    'logits',             # Final logits
]


def compute_tensor_hash(tensor_data):
    """Compute SHA256 hash for tensor data."""
    if tensor_data is None:
        return None
    if isinstance(tensor_data, (int, float, np.number)):
        tensor_data = np.array([tensor_data])
    if isinstance(tensor_data, list):
        tensor_data = np.array(tensor_data)
    if isinstance(tensor_data, np.ndarray):
        # Use bytes representation for deterministic hashing
        tensor_bytes = tensor_data.tobytes()
        return hashlib.sha256(tensor_bytes).hexdigest()
    return None


def compute_nonzero_count(tensor_data):
    """Count non-zero elements in tensor."""
    if tensor_data is None:
        return None
    if isinstance(tensor_data, (int, float, np.number)):
        return 1 if tensor_data != 0 else 0
    if isinstance(tensor_data, list):
        tensor_data = np.array(tensor_data)
    if isinstance(tensor_data, np.ndarray):
        return int(np.count_nonzero(tensor_data))
    return None


def load_trace_record(line):
    """Load a single JSONL trace record."""
    try:
        record = json.loads(line.strip())
        return record
    except (json.JSONDecodeError, TypeError):
        return None


def load_trace_directory(input_dir):
    """Load all JSONL trace files from directory."""
    records = []
    for filename in os.listdir(input_dir):
        if not filename.endswith('.jsonl'):
            continue
        filepath = os.path.join(input_dir, filename)
        with open(filepath, 'r') as f:
            for line in f:
                record = load_trace_record(line)
                if record:
                    record['source_file'] = filename
                    records.append(record)
    return pd.DataFrame(records)


def build_composite_key(record):
    """Build composite key for matching records."""
    return (
        record.get('prompt_id', ''),
        record.get('token_id', -1),
        record.get('pos_id', -1),
        record.get('layer', -1),
        record.get('tensor_name', ''),
        record.get('phase', ''),
    )


def match_records(baseline_df, current_df):
    """Match baseline and current records by composite key."""
    baseline_df = baseline_df.copy()
    current_df = current_df.copy()
    
    baseline_df['_composite_key'] = baseline_df.apply(build_composite_key, axis=1)
    current_df['_composite_key'] = current_df.apply(build_composite_key, axis=1)
    
    baseline_keys = set(baseline_df['_composite_key'])
    
    matched_current = current_df[current_df['_composite_key'].isin(baseline_keys)]
    matched_baseline = baseline_df[baseline_df['_composite_key'].isin(current_df['_composite_key'])]
    
    return matched_baseline, matched_current


def detect_first_fail(row):
    """
    Detect FIRST_FAIL by checking hash matches in TRACE_STAGES order.
    Returns (first_fail_stage, root_cause)
    """
    for stage in TRACE_STAGES:
        b_hash = row.get(f'{stage}_baseline_hash')
        c_hash = row.get(f'{stage}_current_hash')
        
        if b_hash is None and c_hash is None:
            continue  # Skip if both missing
        if b_hash is None or c_hash is None:
            # One is missing - this is the first divergence
            return stage, classify_root_cause(stage)
        if b_hash != c_hash:
            return stage, classify_root_cause(stage)
    
    return None, 'PASS'


def classify_root_cause(stage):
    """Classify root cause based on first failing stage."""
    if stage in ['embed_out', 'residual_pre_attn']:
        return 'ROUTING/SELECTION'
    elif stage in ['attn_in', 'q_pre_rope', 'k_pre_rope']:
        return 'ATTN_KERNEL_INPUTS'
    elif stage in ['q_post_rope', 'k_post_rope', 'attn_out']:
        return 'ATTENTION_MECHANISM'
    elif stage == 'residual_post_attn':
        return 'RESIDUAL_ADD'
    elif stage in ['ffn_norm_in']:
        return 'FFN_NORM_PATH'
    elif stage == 'mlp_out':
        return 'MLP_OUTPUT'
    elif stage == 'residual_post_mlp':
        return 'RESIDUAL_ADD'
    elif stage == 'logits':
        return 'LOGITS_DIVERGENCE'
    return 'UNKNOWN'


def compute_mae(baseline_tensor, current_tensor):
    """Compute Mean Absolute Error between tensors."""
    if baseline_tensor is None or current_tensor is None:
        return None
    try:
        b_arr = np.array(baseline_tensor)
        c_arr = np.array(current_tensor)
        if b_arr.shape != c_arr.shape:
            return None
        return float(np.mean(np.abs(b_arr - c_arr)))
    except:
        return None


def analyze_comparison(baseline_df, current_df):
    """Perform detailed comparison between baseline and current runs."""
    baseline_df = baseline_df.copy()
    current_df = current_df.copy()
    
    # Build composite key
    baseline_df['_key'] = baseline_df.apply(
        lambda r: (r.get('prompt_id'), r.get('pos_id'), r.get('layer'), r.get('tensor_name')), axis=1
    )
    current_df['_key'] = current_df.apply(
        lambda r: (r.get('prompt_id'), r.get('pos_id'), r.get('layer'), r.get('tensor_name')), axis=1
    )
    
    results = []
    
    for key in baseline_df['_key'].unique():
        if key not in current_df['_key'].unique():
            continue
            
        b_row = baseline_df[baseline_df['_key'] == key].iloc[0]
        c_row = current_df[current_df['_key'] == key].iloc[0]
        
        result = {
            'prompt_id': b_row.get('prompt_id', 'unknown'),
            'pos_id': b_row.get('pos_id', -1),
            'layer': b_row.get('layer', -1),
            'token_id': b_row.get('token_id', -1),
            'phase': b_row.get('phase', ''),
        }
        
        # Compare each stage
        for stage in TRACE_STAGES:
            b_hash = b_row.get(f'{stage}_hash') if f'{stage}_hash' in b_row else None
            c_hash = c_row.get(f'{stage}_hash') if f'{stage}_hash' in c_row else None
            b_nz = b_row.get(f'{stage}_nz') if f'{stage}_nz' in b_row else None
            c_nz = c_row.get(f'{stage}_nz') if f'{stage}_nz' in c_row else None
            
            result[f'{stage}_baseline_hash'] = b_hash
            result[f'{stage}_current_hash'] = c_hash
            result[f'{stage}_nz_baseline'] = b_nz
            result[f'{stage}_nz_current'] = c_nz
            result[f'{stage}_match'] = (b_hash == c_hash) if b_hash and c_hash else None
        
        # Detect first failure
        first_fail, root_cause = detect_first_fail(result)
        result['first_fail'] = first_fail
        result['root_cause'] = root_cause
        result['status'] = 'OK' if root_cause == 'PASS' else 'FAIL'
        
        results.append(result)
    
    return pd.DataFrame(results)


def generate_report(results_df, output_path, layers, prompts):
    """Generate comprehensive analysis report."""
    with open(output_path, 'w') as f:
        f.write("# B3.61 Residual Stream Bisect Analysis Report\n\n")
        
        f.write("## Executive Summary\n")
        f.write(f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Objective**: Identify FIRST_FAIL tensor in residual stream\n")
        f.write(f"- **Target Positions**: 826 (16K context), 1652 (32K context)\n")
        f.write(f"- **Layers Analyzed**: {', '.join(layers)}\n")
        f.write(f"- **Prompts Tested**: {', '.join(prompts)}\n\n")
        
        # Summary statistics
        total = len(results_df)
        ok_count = len(results_df[results_df['status'] == 'OK'])
        fail_count = len(results_df[results_df['status'] == 'FAIL'])
        
        f.write("## Summary Statistics\n")
        f.write(f"- **Total Comparisons**: {total}\n")
        f.write(f"- **OK (No Divergence)**: {ok_count}\n")
        f.write(f"- **FAIL (Divergence Detected)**: {fail_count}\n\n")
        
        # First Failure Table
        f.write("## First Failure Table\n")
        f.write("| Prompt | Position | Layer | Tensor | Hash Match | Root Cause | FIRST_FAIL |\n")
        f.write("|--------|----------|-------|--------|------------|-----------|------------|\n")
        
        fail_df = results_df[results_df['status'] == 'FAIL']
        for _, row in fail_df.iterrows():
            ff = '**YES**' if row['first_fail'] else ''
            match = 'YES' if row.get(f"{row['first_fail']}_match") else 'NO'
            f.write(f"| {row['prompt_id']} | {row['pos_id']} | {row['layer']} | "
                   f"{row['first_fail']} | {match} | {row['root_cause']} | {ff} |\n")
        
        f.write("\n")
        
        # Root Cause Breakdown
        f.write("## Root Cause Classification\n")
        rc_counts = fail_df['root_cause'].value_counts()
        for rc, count in rc_counts.items():
            f.write(f"- **{rc}**: {count} occurrences\n")
        f.write("\n")
        
        # First Fail Point Breakdown
        f.write("## First Fail Point Breakdown\n")
        ff_counts = fail_df['first_fail'].value_counts()
        for stage, count in ff_counts.items():
            f.write(f"- **{stage}**: {count} occurrences\n")
        f.write("\n")
        
        # Per-prompt analysis
        f.write("## Per-Prompt Analysis\n")
        for prompt in results_df['prompt_id'].unique():
            pdf = results_df[results_df['prompt_id'] == prompt]
            f.write(f"\n### {prompt}\n")
            f.write(f"- Total: {len(pdf)}\n")
            f.write(f"- OK: {len(pdf[pdf['status'] == 'OK'])}\n")
            f.write(f"- FAIL: {len(pdf[pdf['status'] == 'FAIL'])}\n")
            
            # Positions analyzed
            positions = pdf['pos_id'].unique()
            f.write(f"- Positions: {sorted(positions)}\n")
        
        # Detailed trace samples
        f.write("\n## Divergence Trace Samples\n")
        sample_fail = fail_df.iloc[0] if not fail_df.empty else None
        if sample_fail:
            f.write(f"\n### Sample Failure: {sample_fail['prompt_id']} @ Pos {sample_fail['pos_id']} Layer {sample_fail['layer']}\n")
            f.write(f"- First Fail: {sample_fail['first_fail']}\n")
            f.write(f"- Root Cause: {sample_fail['root_cause']}\n")
            
            for stage in TRACE_STAGES:
                b_hash = sample_fail.get(f'{stage}_baseline_hash')
                c_hash = sample_fail.get(f'{stage}_current_hash')
                match = sample_fail.get(f'{stage}_match')
                if b_hash or c_hash:
                    status = 'MATCH' if match else 'DIVERGE'
                    f.write(f"  - {stage}: {status} (baseline={str(b_hash)[:16]}..., current={str(c_hash)[:16]}...)\n")
        
        # B3.62 Recommendations
        f.write("\n## Recommendations for B3.62\n")
        if not fail_df.empty:
            primary_rc = rc_counts.index[0]
            primary_ff = ff_counts.index[0]
            f.write(f"- **Primary Root Cause**: {primary_rc}\n")
            f.write(f"- **First Fail Stage**: {primary_ff}\n")
            f.write("- **Testable Hypotheses**:\n")
            
            if primary_rc == 'ROUTING/SELECTION':
                f.write("  - Investigate buffer routing at extended context positions\n")
                f.write("  - Check position encoding stability at >16K tokens\n")
            elif primary_rc == 'ATTN_KERNEL_INPUTS':
                f.write("  - Audit query/key projection kernels for position-specific errors\n")
                f.write("  - Validate QKV weight application at boundary positions\n")
            elif primary_rc == 'ATTENTION_MECHANISM':
                f.write("  - Investigate RoPE application at extended contexts\n")
                f.write("  - Check attention score computation for numerical stability\n")
            elif primary_rc == 'RESIDUAL_ADD':
                f.write("  - Debug residual addition kernel for position-specific issues\n")
                f.write("  - Verify dtype consistency across layers\n")
            elif primary_rc == 'FFN_NORM_PATH':
                f.write("  - Audit FFN layer normalization at boundary positions\n")
                f.write("  - Check MLP weight application stability\n")
            elif primary_rc == 'MLP_OUTPUT':
                f.write("  - Investigate MLP output computation for numerical precision\n")
                f.write("  - Verify feed-forward gate and up/down projections\n")
            else:
                f.write("  - Continue systematic bisection to isolate failure origin\n")
        else:
            f.write("- No failures detected - system stable at tested positions\n")
        
        f.write("\n---\n")
        f.write("*Report generated by B3.61 Residual Stream Bisect Analyzer*\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='B3.61 Residual Stream Bisect Analyzer')
    parser.add_argument('--input_dir', required=True, help='Directory with current run trace JSONL files')
    parser.add_argument('--baseline_dir', required=True, help='Directory with baseline trace JSONL files')
    parser.add_argument('--output', required=True, help='Output analysis report path')
    parser.add_argument('--layers', default='0,1,2,4,8', help='Comma-separated list of layers')
    parser.add_argument('--prompts', default='p0_short,p6_len_16,p6_len_32', help='Comma-separated list of prompts')
    
    args = parser.parse_args()
    
    layers = args.layers.split(',')
    prompts = args.prompts.split(',')
    
    print("=== B3.61 Residual Stream Bisect Analyzer ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Baseline directory: {args.baseline_dir}")
    print(f"Layers: {layers}")
    print(f"Prompts: {prompts}")
    
    # Load traces
    print("Loading current traces...")
    current_df = load_trace_directory(args.input_dir)
    if current_df.empty:
        print("ERROR: No trace files found in input directory")
        return
    
    print(f"Loaded {len(current_df)} current trace records")
    
    print("Loading baseline traces...")
    baseline_df = load_trace_directory(args.baseline_dir)
    if baseline_df.empty:
        print("ERROR: No trace files found in baseline directory")
        return
    
    print(f"Loaded {len(baseline_df)} baseline trace records")
    
    # Perform comparison
    print("Performing comparison analysis...")
    results_df = analyze_comparison(baseline_df, current_df)
    
    if results_df.empty:
        print("WARNING: No matching records found between baseline and current")
        # Create minimal report
        with open(args.output, 'w') as f:
            f.write("# B3.61 Residual Stream Bisect Analysis\n\n")
            f.write("No matching records found between baseline and current runs.\n")
        print(f"Minimal report written to: {args.output}")
        return
    
    # Generate report
    print("Generating report...")
    generate_report(results_df, args.output, layers, prompts)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Report written to: {args.output}")
    print(f"Total comparisons: {len(results_df)}")
    print(f"Failures detected: {len(results_df[results_df['status'] == 'FAIL'])}")


if __name__ == "__main__":
    main()
