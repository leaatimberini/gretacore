#!/usr/bin/env python3
"""
B3.67 Equivalence Guardrail Analyzer

Analiza comparaciones de logits entre prefill y decode para verificar equivalencia
con guardrails según kv_aligned.

Veredictos:
- kv_aligned=1: PASS_EQUIV / FAIL_EQUIV (debe cumplir umbrales)
- kv_aligned=0: EXPECTED_DRIFT (drift permitido, solo registrar métricas)

Strict pairing: tokens deben coincidir por token_id y token_idx.

Completeness Guardrail:
- Con config.json: verifica que la matriz completa este ejecutada
- Sin config.json: modo best effort con PASS_GUARDRAIL_LOCAL
"""

import argparse
import gzip
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# Constantes de umbrales
# -----------------------------------------------------------------------------
THRESHOLDS = {
    "p99_abs_diff_max": 1e-3,        # 0.001
    "max_abs_diff_max": 5e-3,        # 0.005
    "top1_agreement_min": 0.999      # 99.9%
}

# Salidas esperadas
EXPECTED_FILES = {
    "prefill": ["logits.jsonl.gz", "metadata.json"],
    "decode": ["logits.jsonl.gz", "metadata.json"]
}


# -----------------------------------------------------------------------------
# Funciones de métricas
# -----------------------------------------------------------------------------

def max_abs_diff(logits_a: list, logits_b: list) -> float:
    """Computa la diferencia absoluta máxima entre dos logits."""
    diffs = [abs(a - b) for a, b in zip(logits_a, logits_b)]
    return max(diffs) if diffs else 0.0


def p99_abs_diff(logits_a: list, logits_b: list) -> float:
    """Computa el percentil 99 de diferencias absolutas."""
    diffs = [abs(a - b) for a, b in zip(logits_a, logits_b)]
    if not diffs:
        return 0.0
    sorted_diffs = sorted(diffs)
    idx = int(len(sorted_diffs) * 0.99)
    return sorted_diffs[min(idx, len(sorted_diffs) - 1)]


def top1_agreement(logits_a: list, logits_b: list) -> float:
    """Computa si el argmax coincide."""
    if not logits_a or not logits_b:
        return 0.0
    argmax_a = max(range(len(logits_a)), key=lambda i: logits_a[i])
    argmax_b = max(range(len(logits_b)), key=lambda i: logits_b[i])
    return 1.0 if argmax_a == argmax_b else 0.0


def cos_sim_mean(logits_a: list, logits_b: list) -> float:
    """Computa similaridad coseno."""
    if not logits_a or not logits_b:
        return 0.0
    dot = sum(a * b for a, b in zip(logits_a, logits_b))
    norm_a = math.sqrt(sum(a * a for a in logits_a))
    norm_b = math.sqrt(sum(b * b for b in logits_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_metrics(prefill_logits: list, decode_logits: list) -> dict:
    """Computa todas las métricas para un par de logits."""
    return {
        "max_abs_diff": max_abs_diff(prefill_logits, decode_logits),
        "p99_abs_diff": p99_abs_diff(prefill_logits, decode_logits),
        "top1_agreement": top1_agreement(prefill_logits, decode_logits),
        "cos_sim_mean": cos_sim_mean(prefill_logits, decode_logits)
    }


# -----------------------------------------------------------------------------
# Validación de artifacts
# -----------------------------------------------------------------------------

def validate_artifacts(prefill_dir: str, decode_dir: str, kv_aligned: int, seed: int) -> tuple[bool, str]:
    """
    Valida que existan los artifacts necesarios.
    Returns (is_valid, error_message).
    """
    for mode, expected in EXPECTED_FILES.items():
        dir_path = prefill_dir if mode == "prefill" else decode_dir
        for fname in expected:
            fpath = os.path.join(dir_path, fname)
            if not os.path.exists(fpath):
                return False, f"ERROR: Missing {mode}/{fname} in seed={seed}, kv_aligned={kv_aligned}"
    return True, ""


# -----------------------------------------------------------------------------
# Carga de datos
# -----------------------------------------------------------------------------

def load_logits(filepath: str) -> dict:
    """Carga logits desde JSONL o JSONL.gz."""
    data = {}
    open_func = gzip.open if filepath.endswith('.gz') else open

    with open_func(filepath, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            token_idx = item.get("token_idx", 0)
            data[token_idx] = {
                "token_id": item.get("token_id"),
                "logits": item.get("logits", [])
            }
    return data


def load_metadata(filepath: str) -> dict:
    """Carga metadata JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Evaluación de veredictos
# -----------------------------------------------------------------------------

def evaluate_verdict(metrics: dict, kv_aligned: int) -> str:
    """
    Evalúa el veredicto según kv_aligned y umbrales.

    Returns:
        - "PASS_EQUIV": kv_aligned=1 y cumple todos los umbrales
        - "FAIL_EQUIV": kv_aligned=1 y no cumple umbrales
        - "EXPECTED_DRIFT": kv_aligned=0 (drift permitido)
    """
    if kv_aligned == 0:
        return "EXPECTED_DRIFT"

    # kv_aligned == 1: verificar umbrales
    if (metrics["p99_abs_diff"] <= THRESHOLDS["p99_abs_diff_max"] and
        metrics["max_abs_diff"] <= THRESHOLDS["max_abs_diff_max"] and
        metrics["top1_agreement"] >= THRESHOLDS["top1_agreement_min"]):
        return "PASS_EQUIV"
    else:
        return "FAIL_EQUIV"


# -----------------------------------------------------------------------------
# Análisis
# -----------------------------------------------------------------------------

def analyze_pair(prefill_dir: str, decode_dir: str, kv_aligned: int, seed: int) -> dict:
    """
    Analiza un par prefill vs decode.

    Returns:
        dict con métricas, veredicto y first_fail info
    """
    # Validar artifacts
    is_valid, error_msg = validate_artifacts(prefill_dir, decode_dir, kv_aligned, seed)
    if not is_valid:
        return {
            "error": error_msg,
            "verdict": "ERROR",
            "seed": seed,
            "kv_aligned": kv_aligned
        }

    # Cargar logits
    prefill_logits_file = os.path.join(prefill_dir, "logits.jsonl.gz")
    decode_logits_file = os.path.join(decode_dir, "logits.jsonl.gz")
    metadata_file = os.path.join(prefill_dir, "metadata.json")

    prefill_data = load_logits(prefill_logits_file)
    decode_data = load_logits(decode_logits_file)

    # Cargar metadata
    metadata = {}
    if os.path.exists(metadata_file):
        metadata = load_metadata(metadata_file)

    # Strict pairing: construir mapa por (token_id, token_idx)
    paired_count = 0
    missing_pairs = 0

    # Index decode por token_id
    decode_by_token_id = {}
    for token_idx, item in decode_data.items():
        token_id = item.get("token_id")
        if token_id is not None:
            decode_by_token_id[token_id] = item

    # Computar métricas
    all_max_diffs = []
    all_p99_diffs = []
    all_agreements = []
    all_cos_sims = []

    first_fail = None

    for token_idx, prefill_item in prefill_data.items():
        token_id = prefill_item.get("token_id")

        # Strict pairing: buscar por token_id
        if token_id in decode_by_token_id:
            decode_item = decode_by_token_id[token_id]
        else:
            missing_pairs += 1
            continue

        if not prefill_item["logits"] or not decode_item["logits"]:
            continue

        paired_count += 1

        metrics = compute_metrics(prefill_item["logits"], decode_item["logits"])

        all_max_diffs.append(metrics["max_abs_diff"])
        all_p99_diffs.append(metrics["p99_abs_diff"])
        all_agreements.append(metrics["top1_agreement"])
        all_cos_sims.append(metrics["cos_sim_mean"])

        # First fail detection (token-level)
        if first_fail is None and metrics["max_abs_diff"] > THRESHOLDS["max_abs_diff_max"]:
            first_fail = {
                "token_idx": token_idx,
                "token_id": token_id,
                "max_abs_diff": metrics["max_abs_diff"],
                "p99_abs_diff": metrics["p99_abs_diff"],
                "top1_agreement": metrics["top1_agreement"]
            }

    # Aggregated metrics
    result = {
        "seed": seed,
        "kv_aligned": kv_aligned,
        "pair_count": paired_count,
        "missing_pairs": missing_pairs,
        "metrics": {
            "max_abs_diff": max(all_max_diffs) if all_max_diffs else 0.0,
            "p99_abs_diff": max(all_p99_diffs) if all_p99_diffs else 0.0,
            "top1_agreement": sum(all_agreements) / len(all_agreements) if all_agreements else 0.0,
            "cos_sim_mean": sum(all_cos_sims) / len(all_cos_sims) if all_cos_sims else 0.0
        },
        "first_fail": first_fail,
        "metadata": metadata
    }

    # Evaluar veredicto
    result["verdict"] = evaluate_verdict(result["metrics"], kv_aligned)

    return result


def analyze_runs(runs_dir: str) -> tuple[dict, list]:
    """
    Analiza todos los runs en el directorio.
    Returns (results_dict, errors_list).

    Estructura esperada:
    runs/kv_aligned_{0,1}/seed_{0,1,2}/{prefill,decode}/logits.jsonl.gz
    """
    results = {}
    errors = []

    runs_path = Path(runs_dir)

    # Buscar directorios kv_aligned_*
    for kv_dir in sorted(runs_path.iterdir()):
        if not kv_dir.is_dir() or not kv_dir.name.startswith("kv_aligned_"):
            continue

        try:
            kv_aligned = int(kv_dir.name.split("_")[-1])
        except ValueError:
            continue

        results[kv_aligned] = {}

        # Buscar directorios seed_*
        for seed_dir in sorted(kv_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue

            try:
                seed = int(seed_dir.name.split("_")[-1])
            except ValueError:
                continue

            prefill_dir = str(seed_dir / "prefill")
            decode_dir = str(seed_dir / "decode")

            # Verificar que existen ambos directorios
            if not os.path.exists(prefill_dir) or not os.path.exists(decode_dir):
                errors.append(f"WARNING: Missing prefill/decode dirs for seed={seed}, kv_aligned={kv_aligned}")
                continue

            result = analyze_pair(prefill_dir, decode_dir, kv_aligned, seed)
            results[kv_aligned][seed] = result

            if "error" in result:
                errors.append(result["error"])

    return results, errors


# -----------------------------------------------------------------------------
# Aggregación
# -----------------------------------------------------------------------------

def aggregate_results(results: dict) -> dict:
    """Agrega resultados de todos los runs."""
    summary = {
        "total_runs": 0,
        "pass_equiv": 0,
        "fail_equiv": 0,
        "expected_drift": 0,
        "errors": 0,
        "first_fail": None
    }

    for kv_aligned, seeds in results.items():
        for seed, result in seeds.items():
            summary["total_runs"] += 1
            verdict = result.get("verdict", "UNKNOWN")

            if verdict == "PASS_EQUIV":
                summary["pass_equiv"] += 1
            elif verdict == "FAIL_EQUIV":
                summary["fail_equiv"] += 1
            elif verdict == "EXPECTED_DRIFT":
                summary["expected_drift"] += 1
            elif verdict == "ERROR":
                summary["errors"] += 1

            # Track first failure for FAIL_EQUIV
            if verdict == "FAIL_EQUIV" and summary["first_fail"] is None:
                summary["first_fail"] = {
                    "kv_aligned": kv_aligned,
                    "seed": seed,
                    "max_abs_diff": result.get("metrics", {}).get("max_abs_diff", 0),
                    "p99_abs_diff": result.get("metrics", {}).get("p99_abs_diff", 0),
                    "top1_agreement": result.get("metrics", {}).get("top1_agreement", 0)
                }

    # Veredicto global
    if summary["total_runs"] > 0:
        if summary["fail_equiv"] > 0:
            summary["global_verdict"] = "FAIL_GUARDRAIL"
        elif summary["pass_equiv"] > 0 and summary["expected_drift"] >= 0:
            summary["global_verdict"] = "PASS_GUARDRAIL"
        elif summary["expected_drift"] > 0 and summary["pass_equiv"] == 0:
            summary["global_verdict"] = "EXPECTED_DRIFT_ONLY"
        else:
            summary["global_verdict"] = "INCONCLUSIVE"
    else:
        summary["global_verdict"] = "NO_DATA"

    return summary


# -----------------------------------------------------------------------------
# Reportes
# -----------------------------------------------------------------------------

def write_metrics_json(results: dict, output_dir: str):
    """Escribe metrics.json por cada par prefill/decode."""
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    for kv_aligned, seeds in results.items():
        kv_metrics_dir = os.path.join(metrics_dir, f"kv_aligned_{kv_aligned}")
        os.makedirs(kv_metrics_dir, exist_ok=True)

        for seed, result in seeds.items():
            metrics_file = os.path.join(kv_metrics_dir, f"seed_{seed}_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(result, f, indent=2)


def write_summary_json(summary: dict, output_path: str):
    """Escribe summary.json global."""
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)


def write_markdown_report(results: dict, summary: dict, errors: list, output_path: str, date: str, missing_list: list = None):
    """Escribe el reporte markdown."""
    with open(output_path, 'w') as f:
        f.write("# B3.67 Equivalence Guardrail Analysis\n\n")
        f.write(f"**Date**: {date}\n")
        f.write(f"**Benchmark**: B3.67\n\n")

        # Errors section
        if errors:
            f.write("## Errors\n\n")
            for err in errors:
                f.write(f"- {err}\n")
            f.write("\n")

        # Completeness section
        f.write("## Completeness Status\n\n")
        config_present = summary.get('config_present', False)
        f.write(f"- **Config Present**: {config_present}\n")
        
        if config_present:
            f.write(f"- **Expected Pairs**: {summary.get('expected_pairs', 'N/A')}\n")
            f.write(f"- **Found Pairs**: {summary.get('found_pairs', 'N/A')}\n")
            f.write(f"- **Missing Pairs**: {summary.get('missing_pairs_count', 'N/A')}\n")
            f.write(f"- **Completeness Verdict**: `{summary.get('completeness_verdict', 'N/A')}`\n\n")
            
            if missing_list:
                f.write("### Missing Pairs\n\n")
                for missing in missing_list:
                    f.write(f"- {missing}\n")
                f.write("\n")
        else:
            f.write("- **Note**: No config.json found. Running in best-effort mode.\n")
            f.write("- **Global Verdict**: `PASS_GUARDRAIL_LOCAL` (completeness unknown)\n\n")

        # Threshold Configuration
        f.write("## Threshold Configuration\n\n")
        f.write("| Metric | Threshold |\n")
        f.write("|--------|-----------|\n")
        for metric, value in THRESHOLDS.items():
            f.write(f"| {metric} | {value} |\n")
        f.write("\n")

        # Summary
        f.write("## Summary\n\n")
        f.write(f"- **Total Runs**: {summary['total_runs']}\n")
        f.write(f"- **PASS_EQUIV**: {summary['pass_equiv']}\n")
        f.write(f"- **FAIL_EQUIV**: {summary['fail_equiv']}\n")
        f.write(f"- **EXPECTED_DRIFT**: {summary['expected_drift']}\n")
        f.write(f"- **Errors**: {summary['errors']}\n")
        f.write(f"- **Global Verdict**: `{summary['global_verdict']}`\n\n")

        # Detailed Results
        f.write("## Detailed Results\n\n")
        f.write("| KV Aligned | Seed | Pairs | Missing | Max Abs Diff | P99 Abs Diff | Top1 Agreement | Verdict |\n")
        f.write("|------------|------|-------|---------|--------------|--------------|----------------|---------|\n")

        for kv_aligned in sorted(results.keys()):
            seeds = results[kv_aligned]
            for seed in sorted(seeds.keys()):
                r = seeds[seed]
                metrics = r.get("metrics", {})
                f.write(f"| {kv_aligned} | {seed} | {r.get('pair_count', 'N/A')} | {r.get('missing_pairs', 0)} | "
                        f"{metrics.get('max_abs_diff', 'N/A'):.6f} | "
                        f"{metrics.get('p99_abs_diff', 'N/A'):.6f} | "
                        f"{metrics.get('top1_agreement', 'N/A'):.4f} | "
                        f"`{r.get('verdict', 'N/A')}` |\n")

        f.write("\n")

        # Metrics by kv_aligned
        f.write("## Metrics Summary by KV Aligned\n\n")
        for kv_aligned in sorted(results.keys()):
            f.write(f"### KV Aligned = {kv_aligned}\n\n")
            seeds = results[kv_aligned]
            max_diff = 0.0
            p99_diff = 0.0
            min_agree = 1.0

            for seed, r in seeds.items():
                m = r.get("metrics", {})
                max_diff = max(max_diff, m.get("max_abs_diff", 0))
                p99_diff = max(p99_diff, m.get("p99_abs_diff", 0))
                min_agree = min(min_agree, m.get("top1_agreement", 1.0))

            f.write(f"- **Max Max Abs Diff**: {max_diff:.6f}\n")
            f.write(f"- **Max P99 Abs Diff**: {p99_diff:.6f}\n")
            f.write(f"- **Min Top1 Agreement**: {min_agree:.6f}\n\n")


# -----------------------------------------------------------------------------
# Completeness Guardrail
# -----------------------------------------------------------------------------

def load_config(config_path: str) -> dict | None:
    """Carga config.json si existe."""
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def get_expected_pairs(config: dict) -> set:
    """
    Construye el set esperado de pares (kv_aligned, seed, dtype, prompt_len, gen_len).
    """
    expected = set()
    seeds = config.get('seeds', [0, 1, 2])
    kv_aligned_values = config.get('kv_aligned', [0, 1])
    dtype = config.get('dtype', 'bf16')
    prompt_len = config.get('prompt_len', 512)
    gen_len = config.get('gen_len', 128)
    
    for kv in kv_aligned_values:
        for seed in seeds:
            expected.add((kv, seed, dtype, prompt_len, gen_len))
    return expected


def get_found_pairs(results: dict) -> set:
    """
    Construye el set de pares encontrados en results.
    """
    found = set()
    for kv_aligned, seeds in results.items():
        for seed, result in seeds.items():
            if 'error' not in result:
                meta = result.get('metadata', {})
                dtype = meta.get('dtype', 'bf16')
                prompt_len = meta.get('prompt_len', 512)
                gen_len = meta.get('gen_len', 128)
                found.add((kv_aligned, seed, dtype, prompt_len, gen_len))
    return found


def check_completeness(config: dict | None, results: dict) -> tuple[dict, list]:
    """
    Verifica completitud de la matriz.
    
    Returns: (summary_extra, missing_pairs)
    """
    if config is None:
        # Modo best effort: no hay config, no podemos verificar completitud
        return {
            'config_present': False,
            'expected_pairs': 0,
            'found_pairs': sum(len(seeds) for seeds in results.values()),
            'missing_pairs_count': -1,
            'missing_pairs': [],
            'completeness_verdict': 'UNKNOWN_COMPLETENESS'
        }, []
    
    expected = get_expected_pairs(config)
    found = get_found_pairs(results)
    
    missing = expected - found
    
    return {
        'config_present': True,
        'expected_pairs': len(expected),
        'found_pairs': len(found),
        'missing_pairs_count': len(missing),
        'missing_pairs': sorted([str(p) for p in missing]),
        'completeness_verdict': 'COMPLETE' if not missing else 'INCOMPLETE'
    }, sorted([str(p) for p in missing])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="B3.67 Equivalence Guardrail Analyzer")
    parser.add_argument("--traces-dir", required=True, help="Directory with runs/")
    parser.add_argument("--output", required=True, help="Output markdown file path")
    parser.add_argument("--date", default=None, help="Date string (default: today)")
    
    args = parser.parse_args()
    
    date = args.date or __import__('datetime').datetime.now().strftime("%Y-%m-%d")
    
    # Cargar config.json
    parent_dir = os.path.dirname(args.traces_dir.rstrip('/'))
    config_path = os.path.join(parent_dir, 'config.json')
    config = load_config(config_path)
    
    # Analizar runs
    print(f"Analyzing runs in: {args.traces_dir}")
    results, errors = analyze_runs(args.traces_dir)
    
    # Verificar completitud
    completeness_info, missing_list = check_completeness(config, results)
    
    if not results and not errors:
        print("ERROR: No runs found")
        sys.exit(1)
    
    # Imprimir errores
    if errors:
        for err in errors:
            print(err)
    
    # Escribir metrics.json por par
    output_dir = os.path.dirname(args.output)
    write_metrics_json(results, output_dir)
    
    # Agregar resultados
    summary = aggregate_results(results)
    
    # Agregar info de completitud
    summary.update(completeness_info)
    
    # Determinar veredicto global considerando completitud
    if summary['missing_pairs_count'] == -1:
        # Sin config: veredicto LOCAL
        summary['global_verdict'] = 'PASS_GUARDRAIL_LOCAL'
    elif summary['missing_pairs_count'] > 0:
        # Matriz incompleta
        summary['global_verdict'] = 'INCOMPLETE'
    else:
        # Matriz completa: evaluar normalmente
        if summary.get('fail_equiv', 0) > 0:
            summary['global_verdict'] = 'FAIL_GUARDRAIL'
        elif summary.get('pass_equiv', 0) > 0 and summary.get('expected_drift', 0) >= 0:
            summary['global_verdict'] = 'PASS_GUARDRAIL'
        elif summary.get('expected_drift', 0) > 0 and summary.get('pass_equiv', 0) == 0:
            summary['global_verdict'] = 'EXPECTED_DRIFT_ONLY'
        else:
            summary['global_verdict'] = 'INCONCLUSIVE'
    
    # Escribir summary.json
    base_name = os.path.splitext(os.path.basename(args.output))[0]
    summary_json_path = os.path.join(output_dir, f"{base_name}_summary.json")
    write_summary_json(summary, summary_json_path)
    
    # Escribir reporte markdown
    write_markdown_report(results, summary, errors, args.output, date, missing_list)
    
    print(f"Analysis complete.")
    print(f"Metrics JSON: {output_dir}/metrics/")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Report: {args.output}")
    print(f"Global Verdict: {summary['global_verdict']}")
    print(f"Config Present: {summary['config_present']}")
    if summary.get('missing_pairs_count', 0) > 0:
        print(f"Missing Pairs: {summary['missing_pairs_count']}")
    
    if summary.get('first_fail'):
        print(f"First Failure: KV={summary['first_fail']['kv_aligned']}, Seed={summary['first_fail']['seed']}")
    
    # Exit code based on verdict
    if summary['global_verdict'] in ['FAIL_GUARDRAIL', 'INCOMPLETE']:
        print(f"ERROR: Benchmark {summary['global_verdict']} - check report for details")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
