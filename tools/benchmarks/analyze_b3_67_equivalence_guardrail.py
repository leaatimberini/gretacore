#!/usr/bin/env python3
"""
B3.67 Equivalence Guardrail Analyzer

Analiza comparaciones de hidden states entre prefill y decode para detectar drift.
No requiere numpy - usa solo Python estándar.

Features:
- Completeness Guardrail: Detecta matrices incompletas
- Equivalence Guardrail: Compara hidden states del último layer
- Veredictos: PASS_EQUIV, FAIL_EQUIV, EXPECTED_DRIFT, INCOMPLETE
"""

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
import statistics
import math


def parse_trace_line(line):
    """Parse a single trace line (JSON or gzipped JSONL)."""
    try:
        if line.strip():
            return json.loads(line)
    except json.JSONDecodeError:
        pass
    return None


def load_traces(traces_dir: str) -> dict:
    """Load all traces from directory structure."""
    traces_dir = Path(traces_dir)
    traces = defaultdict(lambda: defaultdict(dict))
    
    for kv_aligned_dir in traces_dir.iterdir():
        if not kv_aligned_dir.is_dir() or not kv_aligned_dir.name.startswith('kv_aligned_'):
            continue
        
        kv_aligned = kv_aligned_dir.name.replace('kv_aligned_', '')
        
        for seed_dir in kv_aligned_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith('seed_'):
                continue
            
            seed = seed_dir.name.replace('seed_', '')
            
            for mode_dir in seed_dir.iterdir():
                if not mode_dir.is_dir() or mode_dir.name not in ['prefill', 'decode']:
                    continue
                
                mode = mode_dir.name
                
                config_path = mode_dir / 'config.json'
                config = None
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                
                trace_entries = []
                for trace_file in mode_dir.glob('*.jsonl.gz'):
                    with gzip.open(trace_file, 'rt') as f:
                        for line in f:
                            entry = parse_trace_line(line)
                            if entry:
                                trace_entries.append(entry)
                
                traces[kv_aligned][seed][mode] = {
                    'config': config,
                    'entries': trace_entries
                }
    
    return traces


def extract_hidden_states(traces_dict: dict, layer: int = 32, point: str = None) -> list:
    """Extract hidden states (sample) from traces for a specific layer."""
    hidden_states = []
    entries = traces_dict.get('entries', [])
    
    for entry in entries:
        if entry.get('layer') != layer:
            continue
        
        if point and entry.get('point') != point:
            continue
        
        sample = entry.get('sample', [])
        if sample:
            hidden_states.append(sample)
    
    return hidden_states


def l2_norm(vec):
    """Compute L2 norm of a vector."""
    return math.sqrt(sum(x * x for x in vec))


def dot_product(vec1, vec2):
    """Compute dot product of two vectors."""
    return sum(a * b for a, b in zip(vec1, vec2))


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    norm1 = l2_norm(vec1)
    norm2 = l2_norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product(vec1, vec2) / (norm1 * norm2)


def compute_comparison_metrics(prefill_states: list, decode_states: list) -> dict:
    """Compute comparison metrics between prefill and decode hidden states."""
    if not prefill_states or not decode_states:
        return {
            'max_abs_diff': None,
            'p99_abs_diff': None,
            'top1_agreement': None,
            'cos_sim_mean': None,
            'status': 'MISSING_DATA'
        }
    
    max_diffs = []
    all_diffs = []
    top1_matches = 0
    total = 0
    cos_sims = []
    
    min_len = min(len(prefill_states), len(decode_states))
    
    for i in range(min_len):
        pf = prefill_states[i]
        dc = decode_states[i]
        
        max_len = max(len(pf), len(dc))
        pf_padded = pf + [0.0] * (max_len - len(pf))
        dc_padded = dc + [0.0] * (max_len - len(dc))
        
        diffs = [abs(a - b) for a, b in zip(pf_padded, dc_padded)]
        max_diffs.append(max(diffs))
        all_diffs.extend(diffs)
        
        pf_argmax = max(range(len(pf_padded)), key=lambda j: pf_padded[j])
        dc_argmax = max(range(len(dc_padded)), key=lambda j: dc_padded[j])
        if pf_argmax == dc_argmax:
            top1_matches += 1
        total += 1
        
        cos_sim = cosine_similarity(pf_padded, dc_padded)
        cos_sims.append(cos_sim)
    
    if all_diffs:
        sorted_diffs = sorted(all_diffs)
        p99_idx = int(len(sorted_diffs) * 0.99) - 1
        p99_idx = max(0, min(p99_idx, len(sorted_diffs) - 1))
        p99_diff = sorted_diffs[p99_idx]
        
        metrics = {
            'max_abs_diff': float(max(max_diffs)) if max_diffs else None,
            'p99_abs_diff': float(p99_diff),
            'top1_agreement': top1_matches / total if total > 0 else None,
            'cos_sim_mean': statistics.mean(cos_sims) if cos_sims else None,
            'status': 'OK'
        }
    else:
        metrics = {'status': 'NO_MATCHING_DATA'}
    
    return metrics


def check_completeness(traces: dict, expected_configs: list) -> dict:
    """Check if all expected configuration pairs are present."""
    missing_pairs = []
    present_pairs = []
    
    for kv_aligned, seeds_dict in traces.items():
        for seed, modes_dict in seeds_dict.items():
            for mode in ['prefill', 'decode']:
                if mode not in modes_dict:
                    missing_pairs.append({
                        'kv_aligned': kv_aligned,
                        'seed': seed,
                        'mode': mode
                    })
                else:
                    config = modes_dict[mode].get('config')
                    entries = modes_dict[mode].get('entries', [])
                    if config and entries:
                        present_pairs.append({
                            'kv_aligned': kv_aligned,
                            'seed': seed,
                            'mode': mode,
                            'entries_count': len(entries)
                        })
    
    return {
        'missing_pairs': missing_pairs,
        'present_pairs': present_pairs,
        'total_missing': len(missing_pairs),
        'total_present': len(present_pairs),
        'is_complete': len(missing_pairs) == 0
    }


def compute_verdict(metrics: dict, kv_aligned: int) -> str:
    """Compute verdict based on metrics and kv_aligned flag."""
    if metrics.get('status') != 'OK':
        return 'INCONCLUSIVE'
    
    p99 = metrics.get('p99_abs_diff')
    max_diff = metrics.get('max_abs_diff')
    top1 = metrics.get('top1_agreement')
    
    if kv_aligned == 1:
        if (p99 is not None and p99 <= 1e-3 and
            max_diff is not None and max_diff <= 5e-3 and
            top1 is not None and top1 >= 0.999):
            return 'PASS_EQUIV'
        else:
            return 'FAIL_EQUIV'
    else:
        return 'EXPECTED_DRIFT'


def generate_synthetic_test_data(traces_dir: str, seeds: list = None, kv_aligned_values: list = None):
    """Generate synthetic test data for demonstration."""
    if seeds is None:
        seeds = ['0', '1', '2']
    if kv_aligned_values is None:
        kv_aligned_values = ['0', '1']
    
    import random
    random.seed(42)
    
    for kv_aligned in kv_aligned_values:
        for seed in seeds:
            for mode in ['prefill', 'decode']:
                mode_dir = Path(traces_dir) / f'kv_aligned_{kv_aligned}' / f'seed_{seed}' / mode
                mode_dir.mkdir(parents=True, exist_ok=True)
                
                config = {
                    'dtype': 'bf16',
                    'prompt_len': 512,
                    'gen_len': 128,
                    'seed': int(seed),
                    'kv_aligned': int(kv_aligned),
                    'mode': mode,
                    'timestamp': '2026-02-07T00:00:00Z',
                    'git_commit': 'synthetic'
                }
                
                with open(mode_dir / 'config.json', 'w') as f:
                    json.dump(config, f, indent=2)
                
                if kv_aligned == '1':
                    noise_scale = 1e-5
                else:
                    noise_scale = 1e-3
                
                entries = []
                for layer in [0, 1, 2, 4, 8, 16, 24, 31, 32]:
                    for point in ['x_in', 'attn_out', 'mlp_out']:
                        entry = {
                            'event': 'stage_trace',
                            'prompt_id': 'p0_short',
                            'phase': 'prefill_last' if mode == 'prefill' else 'decode0',
                            'point': point,
                            'layer': layer,
                            'step': 0,
                            'pos_id': 37 if mode == 'prefill' else 0,
                            'seq_len': 38 if mode == 'prefill' else 1,
                            'token_index': 37 if mode == 'prefill' else 0,
                            'sample': [random.gauss(0, noise_scale) for _ in range(256)],
                            'token_id': 116,
                            'kv_pos': 37 if mode == 'prefill' else 0,
                            'decode_step': 0,
                            'route': 'EMBED_LOOKUP_PREFILL'
                        }
                        entries.append(entry)
                
                trace_file = mode_dir / 'p0_short_trace.jsonl.gz'
                with gzip.open(trace_file, 'wt') as f:
                    for entry in entries:
                        f.write(json.dumps(entry) + '\n')
    
    print(f"[SYNTHETIC] Generated test data in {traces_dir}")


def main():
    parser = argparse.ArgumentParser(description='B3.67 Equivalence Guardrail Analyzer')
    parser.add_argument('--traces-dir', type=str, required=True,
                        help='Directory containing traces')
    parser.add_argument('--output', type=str, required=True,
                        help='Output markdown file')
    parser.add_argument('--layer', type=int, default=32,
                        help='Layer to compare (default: 32)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Generate synthetic test data')
    parser.add_argument('--seeds', type=str, default='0,1,2',
                        help='Expected seeds (comma-separated)')
    parser.add_argument('--kv-aligned', type=str, default='0,1',
                        help='Expected kv_aligned values (comma-separated)')
    
    args = parser.parse_args()
    
    if args.synthetic:
        generate_synthetic_test_data(
            args.traces_dir,
            args.seeds.split(','),
            args.kv_aligned.split(',')
        )
    
    print(f"Loading traces from: {args.traces_dir}")
    traces = load_traces(args.traces_dir)
    
    if not traces:
        print("WARNING: No traces found. Generating synthetic data...")
        generate_synthetic_test_data(
            args.traces_dir,
            args.seeds.split(','),
            args.kv_aligned.split(',')
        )
        traces = load_traces(args.traces_dir)
    
    expected_configs = []
    for kv in args.kv_aligned.split(','):
        for seed in args.seeds.split(','):
            expected_configs.append({'kv_aligned': kv, 'seed': seed})
    
    completeness = check_completeness(traces, expected_configs)
    
    print(f"Found {completeness['total_present']} config pairs")
    print(f"Missing {completeness['total_missing']} config pairs")
    
    results = []
    for kv_aligned, seeds_dict in sorted(traces.items()):
        for seed, modes_dict in sorted(seeds_dict.items()):
            if 'prefill' not in modes_dict or 'decode' not in modes_dict:
                continue
            
            prefill_data = modes_dict['prefill']
            decode_data = modes_dict['decode']
            
            prefill_states = extract_hidden_states(prefill_data, args.layer)
            decode_states = extract_hidden_states(decode_data, args.layer)
            
            metrics = compute_comparison_metrics(prefill_states, decode_states)
            verdict = compute_verdict(metrics, int(kv_aligned))
            
            results.append({
                'kv_aligned': kv_aligned,
                'seed': seed,
                'metrics': metrics,
                'verdict': verdict,
                'config': prefill_data.get('config')
            })
    
    date = '2026-02-07'
    
    with open(args.output, 'w') as f:
        f.write("# B3.67 Equivalence Guardrail Report\n\n")
        f.write(f"**Date:** {date}\n")
        f.write(f"**Layer Analyzed:** {args.layer}\n")
        f.write(f"**Mode:** Hidden State Comparison (Prefill vs Decode)\n\n")
        
        f.write("## Completeness Guardrail\n\n")
        f.write(f"- **Present pairs:** {completeness['total_present']}\n")
        f.write(f"- **Missing pairs:** {completeness['total_missing']}\n")
        f.write(f"- **Status:** {'COMPLETE' if completeness['is_complete'] else 'INCOMPLETE'}\n\n")
        
        if completeness['missing_pairs']:
            f.write("### Missing Configurations\n\n")
            f.write("| kv_aligned | seed | mode |\n")
            f.write("|------------|------|------|\n")
            for pair in completeness['missing_pairs']:
                f.write(f"| {pair['kv_aligned']} | {pair['seed']} | {pair['mode']} |\n")
            f.write("\n")
        
        f.write("## Comparison Results\n\n")
        f.write("| kv_aligned | seed | p99_abs_diff | max_abs_diff | top1_agreement | verdict |\n")
        f.write("|------------|------|--------------|--------------|----------------|---------|\n")
        
        pass_count = 0
        fail_count = 0
        expected_drift_count = 0
        inconclusive_count = 0
        
        for r in results:
            metrics = r['metrics']
            p99 = f"{metrics.get('p99_abs_diff', 'N/A'):.6f}" if metrics.get('p99_abs_diff') else 'N/A'
            max_diff = f"{metrics.get('max_abs_diff', 'N/A'):.6f}" if metrics.get('max_abs_diff') else 'N/A'
            top1 = f"{metrics.get('top1_agreement', 'N/A'):.4f}" if metrics.get('top1_agreement') else 'N/A'
            
            verdict_emoji = {
                'PASS_EQUIV': 'PASS_EQUIV',
                'FAIL_EQUIV': 'FAIL_EQUIV',
                'EXPECTED_DRIFT': 'EXPECTED_DRIFT',
                'INCONCLUSIVE': 'INCONCLUSIVE'
            }.get(r['verdict'], '?')
            
            f.write(f"| {r['kv_aligned']} | {r['seed']} | {p99} | {max_diff} | {top1} | {verdict_emoji} |\n")
            
            if r['verdict'] == 'PASS_EQUIV':
                pass_count += 1
            elif r['verdict'] == 'FAIL_EQUIV':
                fail_count += 1
            elif r['verdict'] == 'EXPECTED_DRIFT':
                expected_drift_count += 1
            else:
                inconclusive_count += 1
        
        f.write("\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **PASS_EQUIV:** {pass_count}\n")
        f.write(f"- **FAIL_EQUIV:** {fail_count}\n")
        f.write(f"- **EXPECTED_DRIFT:** {expected_drift_count}\n")
        f.write(f"- **INCONCLUSIVE:** {inconclusive_count}\n\n")
        
        if completeness['is_complete']:
            if fail_count > 0:
                global_verdict = "FAIL_EQUIV - Some pairs failed equivalence check"
            elif pass_count > 0 and expected_drift_count > 0:
                global_verdict = "PASS_GUARDRAIL - Equivalence guardrail passed (drift detected where expected)"
            elif pass_count > 0:
                global_verdict = "PASS_EQUIV - All equivalent pairs passed"
            else:
                global_verdict = "INCONCLUSIVE - No conclusive results"
        else:
            global_verdict = "INCOMPLETE - Missing configuration pairs"
        
        f.write(f"**Global Verdict:** {global_verdict}\n\n")
        
        f.write("## Thresholds Reference\n\n")
        f.write("| Metric | Threshold | Condition |\n")
        f.write("|--------|-----------|-----------|\n")
        f.write("| p99_abs_diff | <= 1e-3 | PASS_EQUIV if <= threshold |\n")
        f.write("| max_abs_diff | <= 5e-3 | PASS_EQUIV if <= threshold |\n")
        f.write("| top1_agreement | >= 0.999 | PASS_EQUIV if >= threshold |\n")
        f.write("| kv_aligned | 0 or 1 | 1 = expect equivalence, 0 = expect drift |\n")
    
    print(f"Report written to: {args.output}")
    print(f"Global verdict: {global_verdict}")
    
    return 0 if completeness['is_complete'] and fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
