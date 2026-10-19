import glob
import os
import shutil
import subprocess
import sys

from analyze import ncu_path

# SpeedOfLight: DRAM and SM % of peak. MemoryWorkloadAnalysis: L2 hit rate, DRAM rate.
# Occupancy and LaunchStats: occupancy limits, registers, shared memory. WarpStateStats: why warps stall.
SECTIONS = ['SpeedOfLight', 'MemoryWorkloadAnalysis', 'Occupancy', 'LaunchStats', 'WarpStateStats']
# exact DRAM bytes, split into reads and writes (the sections above only give a rate)
METRICS = ['dram__bytes_read.sum', 'dram__bytes_write.sum']

# (variant, S, block): the doc's 10 runs, B3 at the same sizes, and B1/B2 at block 512,
# the fastest block size in the sweep, to test the L2 explanation for it
RUNS = ([(v, S, 256) for v in ('torch.softmax', 'B1', 'B2', 'B3') for S in (2048, 8192, 16384)]
        + [('composed', 8192, 256), ('B1', 8192, 512), ('B2', 8192, 512)])


def find_ncu() -> str:
    # 1 the NCU environment variable  2 PATH  3 the usual Windows install folders (newest version)
    env = os.environ.get('NCU')
    if env:
        return env
    found = shutil.which('ncu')
    if found:
        return found
    patterns = [r'C:\Program Files\NVIDIA Corporation\Nsight Compute*\ncu.exe',
                r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\nsight-compute*\ncu.exe']
    if os.environ.get('CUDA_PATH'):
        patterns.append(os.path.join(os.environ['CUDA_PATH'], 'nsight-compute*', 'ncu.exe'))
    hits = sorted(h for pat in patterns for h in glob.glob(pat))
    if hits:
        return hits[-1]
    raise SystemExit("ncu not found. Nsight Compute is not on PATH and not in the usual folders.\n"
                     "Find it with:  where /r \"C:\\Program Files\" ncu.exe\n"
                     "then run:      set NCU=<full path to ncu.exe>   and start this script again.\n"
                     "If it is not installed, install Nsight Compute from NVIDIA.")


def ncu_cmd(variant: str, S: int, block: int, out_csv: str) -> list:
    cmd = [find_ncu(), '--profile-from-start', 'off', '--page', 'raw', '--csv',
           '--print-units', 'base', '--log-file', out_csv]
    for sec in SECTIONS:
        cmd += ['--section', sec]
    cmd += ['--metrics', ','.join(METRICS)]
    return cmd + [sys.executable, 'profile_one.py', variant, str(S), str(block)]


def main() -> None:
    # python run_ncu.py                    all runs in RUNS
    # python run_ncu.py B1 8192 [block]    one run
    os.makedirs('results', exist_ok=True)
    print('using', find_ncu())
    if len(sys.argv) >= 3:
        runs = [(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 256)]
    else:
        runs = RUNS
    for variant, S, block in runs:
        out_csv = ncu_path('results', variant, S, block)
        cmd = ncu_cmd(variant, S, block, out_csv)
        print(f"[{variant} S={S} block={block}] {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        print(f"  exit code {rc}, report: {out_csv}")
        if rc != 0:
            print("  ncu failed. If the message mentions ERR_NVGPUCTRPERM, enable GPU performance "
                  "counters for all users in the NVIDIA Control Panel (Developer menu) or use an "
                  "administrator shell.")
            break


if __name__ == '__main__':
    main()
