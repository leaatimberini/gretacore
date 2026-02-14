#!/usr/bin/env python3
"""
parse_b3_89_events.py — Compact live summary table for B3.89 benchmark logs.

Reads JSON events emitted by remote_b3_89_executor.sh and prints a table.

Usage:
    # One-shot summary of a completed log:
    python3 tools/benchmarks/parse_b3_89_events.py /tmp/b3_89_remote.log

    # Live tail (re-reads every 2s):
    python3 tools/benchmarks/parse_b3_89_events.py /tmp/b3_89_remote.log --follow

No external dependencies required (stdlib only).
"""

import json
import sys
import time
import os

def parse_events(filepath):
    """Parse all JSON event lines from the log file."""
    events = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
                if "event" in ev:
                    events.append(ev)
            except json.JSONDecodeError:
                continue
    return events


def build_summary(events):
    """Build a summary dict from parsed events."""
    suite = {"total": 0, "completed": 0, "mode": "?", "wall_s": 0}
    tests = []  # list of dicts: variant, ctx, run_idx, status, prefill_s, wall_s, attn_impl
    last_heartbeat = None

    for ev in events:
        etype = ev.get("event", "")
        if etype == "SUITE_START":
            suite["total"] = ev.get("total_tests", 0)
            suite["mode"] = ev.get("mode", "?")
            suite["variants"] = ev.get("variants", "?")
            suite["contexts"] = ev.get("contexts", "?")
        elif etype == "TEST_END":
            tests.append({
                "variant": ev.get("variant", "?"),
                "ctx": ev.get("ctx", "?"),
                "run_idx": ev.get("run_idx", "?"),
                "status": ev.get("exit_status", "?"),
                "exit_code": ev.get("exit_code", "?"),
                "prefill_s": ev.get("prefill_s", "NA"),
                "wall_s": ev.get("wall_s", "NA"),
                "attn_impl": ev.get("attn_impl", "NA"),
                "model_load_s": ev.get("model_load_s", "NA"),
                "decode_s": ev.get("decode_s", "NA"),
            })
            suite["completed"] = ev.get("test_index", suite["completed"])
        elif etype == "HEARTBEAT":
            last_heartbeat = ev
        elif etype == "SUITE_END":
            suite["wall_s"] = ev.get("suite_wall_s", 0)
            suite["completed"] = ev.get("completed", suite["completed"])

    return suite, tests, last_heartbeat


def fmt_seconds(s):
    """Format seconds as Xm Ys or Xs."""
    try:
        s = float(s)
    except (ValueError, TypeError):
        return str(s)
    if s >= 60:
        return f"{int(s)//60}m{int(s)%60}s"
    return f"{s:.1f}s"


def print_table(suite, tests, last_hb):
    """Print a compact summary table."""
    # Header
    mode = suite.get("mode", "?")
    total = suite.get("total", "?")
    done = suite.get("completed", 0)
    wall = suite.get("wall_s", 0)

    print(f"\n{'='*78}")
    print(f" B3.89 Summary | mode={mode} | {done}/{total} tests | suite_wall={fmt_seconds(wall)}")
    print(f"{'='*78}")

    if not tests:
        if last_hb:
            variant = last_hb.get("variant", "?")
            ctx = last_hb.get("ctx", "?")
            run_idx = last_hb.get("run_idx", "?")
            elapsed = last_hb.get("elapsed_s", "?")
            est_pct = last_hb.get("est_pct", "?")
            gpu = last_hb.get("gpu_use", "NA")
            vram = last_hb.get("vram_used", "NA")
            eta = last_hb.get("eta_remaining_s", -1)
            eta_str = fmt_seconds(eta) if eta >= 0 else "?"
            print(f"  Running: variant={variant} ctx={ctx} run={run_idx}")
            print(f"  Elapsed: {fmt_seconds(elapsed)} ({est_pct}%) | GPU: {gpu} | VRAM: {vram} | ETA: {eta_str}")
        else:
            print("  (no test data yet)")
        print()
        return

    # Table header
    hdr = f"{'variant':>10} {'ctx':>6} {'run':>4} {'status':>6} {'prefill_s':>10} {'wall_s':>10} {'attn_impl':>18}"
    print(hdr)
    print("-" * len(hdr))

    for t in tests:
        prefill = t.get("prefill_s", "NA")
        if prefill != "NA":
            try:
                prefill = f"{float(prefill):.3f}"
            except (ValueError, TypeError):
                pass
        wall_s = t.get("wall_s", "NA")
        if wall_s != "NA":
            try:
                wall_s = f"{float(wall_s):.1f}"
            except (ValueError, TypeError):
                pass
        print(f"{t['variant']:>10} {str(t['ctx']):>6} {str(t['run_idx']):>4} {t['status']:>6} {str(prefill):>10} {str(wall_s):>10} {t['attn_impl']:>18}")

    print()

    # Current run if still going
    if last_hb and done < total:
        variant = last_hb.get("variant", "?")
        ctx = last_hb.get("ctx", "?")
        run_idx = last_hb.get("run_idx", "?")
        elapsed = last_hb.get("elapsed_s", "?")
        est_pct = last_hb.get("est_pct", "?")
        eta = last_hb.get("suite_eta_s", -1)
        eta_str = fmt_seconds(eta) if eta >= 0 else "?"
        print(f"  ▸ Running: variant={variant} ctx={ctx} run={run_idx} elapsed={fmt_seconds(elapsed)} ({est_pct}%) suite_eta={eta_str}")
        print()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <logfile> [--follow]")
        sys.exit(1)

    filepath = sys.argv[1]
    follow = "--follow" in sys.argv or "-f" in sys.argv

    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found")
        sys.exit(1)

    if follow:
        last_size = 0
        try:
            while True:
                cur_size = os.path.getsize(filepath)
                if cur_size != last_size:
                    last_size = cur_size
                    events = parse_events(filepath)
                    suite, tests, last_hb = build_summary(events)
                    # Clear screen
                    print("\033[2J\033[H", end="")
                    print_table(suite, tests, last_hb)
                time.sleep(2)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        events = parse_events(filepath)
        suite, tests, last_hb = build_summary(events)
        print_table(suite, tests, last_hb)


if __name__ == "__main__":
    main()
