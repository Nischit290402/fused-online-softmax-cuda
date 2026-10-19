"""Nsight Compute runs for the version 2 kernels. Reports go to results/v2/.

usage: python run_ncu_v2.py              all runs
       python run_ncu_v2.py B4 8192      one run (optional block as a third argument)
"""
import os
import subprocess
import sys

from analyze import ncu_path
from run_ncu import METRICS, SECTIONS, find_ncu

OUT_DIR = os.path.join('results', 'v2')
RUNS = [(v, S, 256) for v in ('B3v2', 'B4', 'B5') for S in (2048, 8192, 16384)]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    ncu = find_ncu()
    print('using', ncu)
    if len(sys.argv) >= 3:
        runs = [(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 256)]
    else:
        runs = RUNS
    for variant, S, block in runs:
        out_csv = ncu_path(OUT_DIR, variant, S, block)
        cmd = [ncu, '--profile-from-start', 'off', '--page', 'raw', '--csv', '--print-units', 'base',
               '--log-file', out_csv]
        for sec in SECTIONS:
            cmd += ['--section', sec]
        cmd += ['--metrics', ','.join(METRICS), sys.executable, 'profile_one_v2.py', variant, str(S), str(block)]
        print(f"[{variant} S={S} block={block}] {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        print(f"  exit code {rc}, report: {out_csv}")
        if rc != 0:
            print("  ncu failed; see run_ncu.py for the counter-permission fix.")
            break


if __name__ == '__main__':
    main()
