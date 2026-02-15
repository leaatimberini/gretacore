#!/usr/bin/env python3
import json
import glob
import os
import sys

def load_perf_data(root_dir):
    data = {}
    pattern = os.path.join(root_dir, "*", "ctx_*_run*", "perf.json")
    files = glob.glob(pattern)
    
    if not files:
        files = glob.glob(os.path.join(root_dir, "*", "*", "perf.json")) # Try deeper for baseline?
        
    for p in files:
        try:
            with open(p) as f:
                d = json.load(f)
            
            if d.get("exit_status") != "OK":
                continue
                
            variant = d.get("variant", "unknown")
            ctx = d.get("context_len", 0)
            
            if variant not in data:
                data[variant] = {}
            if ctx not in data[variant]:
                data[variant][ctx] = []
                
            val = d["timings"].get("prefill_s", 0)
            wall = d.get("wall_time_sec", 0)
            attn = d["timings"].get("attn_impl", "?")
            
            data[variant][ctx].append({
                "prefill_s": val,
                "wall_s": wall,
                "attn_impl": attn
            })
        except:
            pass
    return data

def main():
    root = "artifacts_remote/2026-02-15/b3_90"
    if not os.path.exists(root):
        root = "artifacts_remote/2026-02-14/b3_89" # Fallback if run failed
        
    data = load_perf_data(root)
    print(f"Analyzing {root}")
    
    print(f"{'Ctx':<8} {'Variant':<10} {'Prefill':<10} {'Speedup':<10} {'V4/V3':<10}")
    
    for ctx in [8192, 16384]:
        base = 0
        v3 = 0
        v4 = 0
        
        # Get baseline first
        if "baseline" in data and ctx in data["baseline"]:
            base = data["baseline"][ctx][0]["prefill_s"]
            
        for var in ["baseline", "v3", "v4"]:
            val = 0
            if var in data and ctx in data[var]:
                val = data[var][ctx][0]["prefill_s"]
                
            if var == "v3": v3 = val
            if var == "v4": v4 = val
            
            speedup = base / val if val > 0 and base > 0 else 0
            v4v3 = v3 / val if var == "v4" and val > 0 and v3 > 0 else 0
            
            s_str = f"{speedup:.2f}x" if speedup else "-"
            v_str = f"{v4v3:.2f}x" if v4v3 else "-"
            
            print(f"{ctx:<8} {var:<10} {val:<10.2f} {s_str:<10} {v_str:<10}")

if __name__ == "__main__":
    main()
