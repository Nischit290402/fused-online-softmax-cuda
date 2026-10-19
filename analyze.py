import csv
import os
import sys

PEAK_GBPS = 352.06              # theoretical peak, see results/env.txt
USABLE_MIB = 7.5 * 1024         # usable GPU memory line in graph 2
SIZES = [1024, 2048, 4096, 8192, 16384, 32768]
# v1 kernels, then the version 2 kernels (csrc/softmax_kernels_v2.cu); the 8 colours are the validated palette in order
ORDER = ['composed', 'torch.softmax', 'B1', 'B2', 'B3', 'B3v2', 'B4', 'B5']
COLORS = {'composed': '#2a78d6', 'torch.softmax': '#eb6834', 'B1': '#1baf7a', 'B2': '#eda100',
          'B3': '#e87ba4', 'B3v2': '#008300', 'B4': '#6250d6', 'B5': '#e34948'}
MARKERS = {'composed': 'o', 'torch.softmax': 's', 'B1': '^', 'B2': 'D', 'B3': 'P', 'B3v2': 'X', 'B4': 'v', 'B5': '<'}
V2_ONLY = ('B3v2', 'B4', 'B5')     # taken from results/v2 sweeps; torch.softmax and B2 come from the v1 sweeps
MODEL_BYTES = {'composed': 16, 'B1': 8, 'B2': 6, 'B3': 4}   # idealized model, bytes per element
INK, INK2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e1', '#fcfcfb'

TIME_SCALE = {'second': 1.0, 'msecond': 1e-3, 'usecond': 1e-6, 'nsecond': 1e-9}
BYTE_SCALE = {'byte': 1.0, 'Kbyte': 1e3, 'Mbyte': 1e6, 'Gbyte': 1e9}


def ncu_path(ncu_dir: str, variant: str, S: int, block: int = 256) -> str:
    suffix = '' if block == 256 else f'_b{block}'
    return os.path.join(ncu_dir, f"ncu_{variant.replace('.', '_')}_{S}{suffix}.csv")


def _num(text):
    try:
        return float(text.strip().replace(',', ''))
    except ValueError:
        return None


def parse_ncu_csv(path: str) -> dict:
    """ncu --page raw --csv file -> time and DRAM bytes summed over all kernel rows;
    percentages are averaged with the kernel time as weight."""
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()
    # some copies of the file come back with every line wrapped in an extra layer of quotes
    lines = [ln[1:-1].replace('""', '"') if ln.startswith('"""') else ln for ln in lines]
    start = next((i for i, ln in enumerate(lines) if ln.lstrip().startswith('"ID"')), None)
    if start is None:
        raise ValueError(f"{path}: no ncu CSV header found (no counter access, or the run failed)")
    table = [r for r in csv.reader(lines[start:]) if r]
    header, units = table[0], table[1]
    rows = [r for r in table[2:] if len(r) == len(header)]
    if not rows:
        raise ValueError(f"{path}: no kernel rows")
    col = {name: i for i, name in enumerate(header)}

    def column(names, scales=None, default=1.0):
        for n in names:
            if n in col:
                i = col[n]
                unit = units[i].strip() if i < len(units) else ''
                k = scales.get(unit, default) if scales else 1.0
                vals = [_num(r[i]) for r in rows]
                return [None if v is None else v * k for v in vals]
        return None

    secs = column(['gpu__time_duration.sum'], TIME_SCALE, 1e-9)
    if secs is None or None in secs:
        raise ValueError(f"{path}: gpu__time_duration.sum missing")
    total_s = sum(secs)
    read = column(['dram__bytes_read.sum'], BYTE_SCALE, 1.0)
    write = column(['dram__bytes_write.sum'], BYTE_SCALE, 1.0)
    # older command line had no dram__bytes_read/write; total bytes = dram__bytes.sum.per_second x kernel time
    bps = column(['dram__bytes.sum.per_second'])
    total_bytes = None if bps is None or None in bps else sum(b * t for b, t in zip(bps, secs))
    dram = column(['gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
                   'dram__throughput.avg.pct_of_peak_sustained_elapsed'])
    sm = column(['sm__throughput.avg.pct_of_peak_sustained_elapsed'])
    occ = column(['sm__warps_active.avg.pct_of_peak_sustained_active'])
    l2 = column(['lts__t_sector_hit_rate.pct'])
    regs = column(['launch__registers_per_thread'])
    # WarpStateStats: average warps stalled per issued instruction, by reason (only in files profiled with it)
    stalls = {}
    for name in header:
        if name.startswith('smsp__average_warps_issue_stalled_') and name.endswith('_per_issue_active.ratio'):
            v = column([name])
            if v is not None and None not in v:
                reason = name[len('smsp__average_warps_issue_stalled_'):-len('_per_issue_active.ratio')]
                stalls[reason] = sum(a * w for a, w in zip(v, secs)) / total_s

    def total(v):
        return None if v is None or None in v else sum(v)

    def wmean(v):
        if v is None or None in v:
            return None
        return sum(a * w for a, w in zip(v, secs)) / total_s

    return {'n_kernels': len(rows), 'time_us': total_s * 1e6,
            'dram_read_bytes': total(read), 'dram_write_bytes': total(write),
            'dram_total_bytes': (total(read) + total(write)) if total(read) is not None and total(write) is not None else total_bytes,
            'dram_pct_peak': wmean(dram), 'sm_pct_peak': wmean(sm),
            'occupancy_pct': wmean(occ), 'l2_hit_pct': wmean(l2),
            'regs_per_thread': max(regs) if regs and None not in regs else None,
            'stalls': stalls}


def top_stalls(parsed: dict, k: int = 3) -> str:
    s = sorted(parsed.get('stalls', {}).items(), key=lambda kv: -kv[1])[:k]
    return '; '.join(f'{r} {v:.2f}' for r, v in s)


def dram_bytes_per_elem(parsed: dict, S: int) -> float:
    return parsed['dram_total_bytes'] / (S * S)


def _fmt(v, digits=2):
    return '' if v is None else round(v, digits)


def fill_sweep_with_ncu(sweep_csv: str, ncu_dir: str = 'results') -> int:
    """Write the Nsight columns into the sweep rows that have an ncu file (expf rows, any block)."""
    with open(sweep_csv, newline='') as f:
        rd = csv.DictReader(f)
        fields, rows = rd.fieldnames, list(rd)
    n = 0
    for r in rows:
        p = ncu_path(ncu_dir, r['variant'], int(r['S']), int(r['block']))
        if r['status'] != 'ok' or r['exp_impl'] != 'expf' or not os.path.exists(p):
            continue
        d = parse_ncu_csv(p)
        r['dram_read_bytes'] = _fmt(d['dram_read_bytes'], 0)
        r['dram_write_bytes'] = _fmt(d['dram_write_bytes'], 0)
        r['dram_pct_peak'] = _fmt(d['dram_pct_peak'])
        r['sm_pct_peak'] = _fmt(d['sm_pct_peak'])
        r['occupancy_pct'] = _fmt(d['occupancy_pct'])
        r['regs_per_thread'] = _fmt(d['regs_per_thread'], 0)
        n += 1
    with open(sweep_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return n


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _series(rows):
    """variant -> S -> {'t': [...], 'mem': [...], 'status': set} for the base cells."""
    out = {}
    for r in rows:
        if int(r['block']) != 256 or r['exp_impl'] != 'expf':
            continue
        cell = out.setdefault(r['variant'], {}).setdefault(int(r['S']), {'t': [], 'mem': [], 'status': set()})
        cell['status'].add(r['status'])
        if r['status'] == 'ok':
            cell['t'].append(float(r['t_median_us']))
            cell['mem'].append(float(r['peak_mem_MiB']))
    return out


def make_charts(sweep_csv: str, ncu_dir: str = 'results', sweep_csv2: str = None,
                peak_gbps: float = PEAK_GBPS, out_dir: str = 'results', v2_csvs=()) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = []
    for p in (sweep_csv, sweep_csv2):
        if p:
            with open(p, newline='') as f:
                rows += list(csv.DictReader(f))
    for p in v2_csvs:
        if p and os.path.exists(p):
            with open(p, newline='') as f:
                rows += [r for r in csv.DictReader(f) if r['variant'] in V2_ONLY]
    data = _series(rows)
    os.makedirs(out_dir, exist_ok=True)
    ticks = [str(s // 1024) + 'K' for s in SIZES]

    for fname, key, ylabel, title in (
            ('graph1_bandwidth.png', 'bw', 'Effective bandwidth (GB/s)',
             'Softmax effective bandwidth, RTX 2070 Max-Q (fp16, block 256; B3v2, B4, B5 from v2)'),
            ('graph2_memory.png', 'mem', 'Peak GPU memory above the input (MiB, log scale)',
             'Softmax peak memory above the input')):
        fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150, facecolor=SURFACE)
        _style(ax)
        ends = {}
        for idx, v in enumerate(ORDER):
            if v not in data:
                continue
            xs, ys, last_ok = [], [], None
            for S in SIZES:
                cell = data[v].get(S)
                if not cell:
                    continue
                if cell['t']:
                    t = sum(cell['t']) / len(cell['t'])
                    y = 4 * S * S / (t * 1e-6) / 1e9 if key == 'bw' else sum(cell['mem']) / len(cell['mem'])
                    xs.append(S)
                    ys.append(y)
                    last_ok = (S, y)
                elif 'oom' in cell['status'] and last_ok:
                    ax.plot([S], [last_ok[1]], marker='x', color=COLORS[v], ms=10, mew=2.4, ls='none')
            if not xs:
                continue
            # in the memory graph the fused kernels overlap, so draw larger markers first
            # and smaller ones on top; every series stays visible
            ms = 6.5 if key == 'bw' else max(3.5, 13 - 1.4 * idx)
            ax.plot(xs, ys, color=COLORS[v], lw=2, marker=MARKERS[v], ms=ms,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=v)
            # series that end within 3% of each other share one label ("B4 = B5") so labels never overlap
            for (ex, ey), names in ends.items():
                if ex == xs[-1] and abs(ey - ys[-1]) <= 0.03 * max(ey, ys[-1]):
                    names.append(v)
                    break
            else:
                ends[(xs[-1], ys[-1])] = [v]
        for (x_end, y_end), names in ends.items():
            ax.annotate(' = '.join(names), (x_end, y_end), xytext=(8, 0), textcoords='offset points',
                        va='center', fontsize=9, color=INK2)
        if key == 'bw':
            ax.axhline(peak_gbps, color=MUTED, ls='--', lw=1.2)
            ax.text(SIZES[-1] * 2.5, peak_gbps, f'theoretical peak {peak_gbps:.0f} GB/s ', ha='right',
                    va='bottom', fontsize=9, color=INK2)
            ax.set_ylim(0, peak_gbps * 1.1)
        else:
            ax.set_yscale('log')
            ax.axhline(USABLE_MIB, color=MUTED, ls='--', lw=1.2)
            ax.text(SIZES[-1] * 2.5, USABLE_MIB, '7.5 GB usable ', ha='right', va='bottom',
                    fontsize=9, color=INK2)
        ax.set_xscale('log', base=2)
        ax.set_xticks(SIZES)
        ax.set_xticklabels(ticks)
        ax.set_xlim(SIZES[0] * 0.85, SIZES[-1] * 2.6)
        ax.set_xlabel('Sequence length S')
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc='upper center',
                  bbox_to_anchor=(0.5, -0.15), ncol=4)
        fig.savefig(os.path.join(out_dir, fname), facecolor=SURFACE, bbox_inches='tight')
        plt.close(fig)

    # Table 1 (S = 8192) and a summary of every ncu file
    summary, table1 = [], []
    for v in ORDER:
        for S, block in ((2048, 256), (8192, 256), (8192, 512), (16384, 256)):
            p = ncu_path(ncu_dir, v, S, block)
            if not os.path.exists(p):
                continue
            d = parse_ncu_csv(p)
            line = {'variant': v, 'S': S, 'block': block, 'kernels_per_call': d['n_kernels'],
                    'dram_bytes_per_elem': _fmt(dram_bytes_per_elem(d, S)),
                    'model_bytes_per_elem': MODEL_BYTES.get(v, ''),
                    'dram_pct_peak': _fmt(d['dram_pct_peak']), 'sm_pct_peak': _fmt(d['sm_pct_peak']),
                    'l2_hit_pct': _fmt(d['l2_hit_pct']), 'occupancy_pct': _fmt(d['occupancy_pct']),
                    'regs_per_thread': _fmt(d['regs_per_thread'], 0), 'time_us_ncu': _fmt(d['time_us'], 1),
                    'dram_bytes_source': 'read+write' if d['dram_read_bytes'] is not None else 'rate x time',
                    'top_stalls': top_stalls(d)}
            summary.append(line)
            if S == 8192 and block == 256:
                table1.append(line)
    for name, lines in (('ncu_summary.csv', summary), ('table1.csv', table1)):
        if lines:
            with open(os.path.join(out_dir, name), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(lines[0].keys()))
                w.writeheader()
                w.writerows(lines)
    if not summary:
        print("no ncu_*.csv files found, so table1.csv was not written")


if __name__ == '__main__':
    # python analyze.py results/sweep_run1.csv results/sweep_run2.csv
    run1, run2 = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
    for p in (run1, run2):
        if p:
            print(f"{p}: filled ncu columns in {fill_sweep_with_ncu(p)} rows")
    make_charts(run1, 'results', run2,
                v2_csvs=[os.path.join('results', 'v2', f'sweep_v2_run{i}.csv') for i in (1, 2)])
