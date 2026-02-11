#!/bin/bash
# =============================================================================
# B3.89 Kernel Resource Dump (MI300X) - V2
# =============================================================================
set -euo pipefail

KERNEL_FILE="src/rt/backend/hip/kernels/attention_kernels.hip"
ASM_FILE="kernels_dump_gpu.s"

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
HIPCC="$ROCM_PATH/bin/hipcc"
ARCH="gfx942"
INCLUDES="-Isrc/rt/backend/hip/include -Isrc/rt/include -Isrc/compute/include"
DEFS="-D__HIP_PLATFORM_AMD__=1"

VARIANT_FLAG=${1:-""}

echo "=== Dumping Kernel Resources for $ARCH ==="
echo "Command: $HIPCC --offload-arch=$ARCH --offload-device-only -S ... $VARIANT_FLAG"

# Compile to GPU assembly
$HIPCC --offload-arch=$ARCH --offload-device-only -S $KERNEL_FILE -o $ASM_FILE $INCLUDES $DEFS $VARIANT_FLAG

python3 -c "
import sys
import re

asm_file = '$ASM_FILE'
try:
    with open(asm_file, 'r') as f:
        content = f.read()
except Exception as e:
    print(f'Error reading {asm_file}: {e}')
    sys.exit(1)

if '.amdgpu_metadata' not in content:
    print('Error: .amdgpu_metadata not found in assembly.')
    # Print first 20 lines of assembly for debug
    print('--- Assembly Preview (20 lines) ---')
    print('\n'.join(content.split('\n')[:20]))
    sys.exit(1)

metadata_block = content.split('.amdgpu_metadata')[1].split('.end_amdgpu_metadata')[0]

# More flexible kernel splitting
# Kernels are entries in a list under amdhsa.kernels
kernels_parts = re.split(r'\.name:\s+', metadata_block)

found = False
for part in kernels_parts[1:]:
    name = part.split('\n')[0].strip()
    # Print all found kernels for debugging if not found yet
    # print(f'Debug: Found kernel {name}')
    
    if 'flash_attention_prefill_kernel' in name:
        found = True
        # Extract fields
        vgpr = re.search(r'\.vgpr_count:\s+(\d+)', part)
        sgpr = re.search(r'\.sgpr_count:\s+(\d+)', part)
        scratch = re.search(r'\.private_segment_fixed_size:\s+(\d+)', part) or re.search(r'\.scratch_memory_size:\s+(\d+)', part)
        lds = re.search(r'\.group_segment_fixed_size:\s+(\d+)', part)
        
        vgpr_val = vgpr.group(1) if vgpr else '?'
        sgpr_val = sgpr.group(1) if sgpr else '?'
        scratch_val = scratch.group(1) if scratch else '0'
        lds_val = lds.group(1) if lds else '0'
        
        print(f'\nKernel: {name}')
        print(f'  VGPRs:   {vgpr_val}')
        print(f'  SGPRs:   {sgpr_val}')
        print(f'  LDS:     {lds_val} bytes')
        print(f'  Scratch: {scratch_val} bytes')
        
        if int(scratch_val) > 0:
            print('  WARNING: SCRATCH SPILLING DETECTED!')
        else:
            print('  GATE PASSED: No scratch spilling.')

if not found:
    print('Error: flash_attention_prefill_kernel not found in metadata.')
    # Print a few kernel names for debugging
    names = [p.split('\n')[0].strip() for p in kernels_parts[1:5]]
    print(f'First few kernels found: {names}')
"

rm -f $ASM_FILE
