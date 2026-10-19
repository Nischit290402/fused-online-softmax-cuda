"""Chart and tables for the version 2 kernels.

usage: python analyze_v2.py results/v2/sweep_v2_run1.csv results/v2/sweep_v2_run2.csv
writes results/v2/graph_v2_bandwidth.png and results/v2/tables_v2.md. B3 with the v1 launch is read
from the v1 sweeps (results/sweep_run1.csv, results/sweep_run2.csv) and drawn dashed.
"""
import csv
import os
import sys

from analyze import (INK, INK2, MUTED, PEAK_GBPS, SIZES, SURFACE, _style, dram_bytes_per_elem,
                     ncu_path, parse_ncu_csv, top_stalls)

OUT = os.path.join('results', 'v2')
# order and colours checked with the palette validator: no two neighbours too close in hue
ORDER = ['torch.softmax', 'B4', 'B2', 'B5', 'B3v2']   # adjacent pairs validated with these colours
COLORS = {'torch.softmax': '#eb6834', 'B4': '#6250d6', 'B2': '#eda100', 'B5': '#e34948', 'B3v2': '#008300'}  # same as analyze.py
MARKERS = {'torch.softmax': 's', 'B4': 'v', 'B2': 'D', 'B5': '<', 'B3v2': 'X'}   # same as analyze.py
LABELS = {'torch.softmax': 'torch.softmax', 'B4': 'B4: 16-B loads, chunked (m, d)', 'B2': 'B2 (v1 kernel)',
          'B5': 'B5: one warp per row', 'B3v2': 'B3, widened block'}


def load(paths):
    cells = {}
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                k = (r['variant'], int(r['S']), int(r['block']), r['exp_impl'])
                c = cells.setdefault(k, {'t': [], 'status': r['status'], 'launched': r.get('block_launched', '')})
                if r['status'] == 'ok':
                    c['t'].append(float(r['t_median_us']))
    return cells


def mean(v):
    return sum(v) / len(v) if v else None


def bw(S, t):
    return 4 * S * S / (t * 1e-6) / 1e9


def chart(v2, v1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150, facecolor=SURFACE)
    _style(ax)
    ends = {}
    for v in ORDER:
        pts = [(S, bw(S, mean(v2[(v, S, 256, 'expf')]['t']))) for S in SIZES
               if (v, S, 256, 'expf') in v2 and v2[(v, S, 256, 'expf')]['t']]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=COLORS[v], lw=2, marker=MARKERS[v], ms=6.5, markeredgecolor=SURFACE,
                markeredgewidth=1.2, label=LABELS[v])
        for (ex, ey), names in ends.items():
            if ex == xs[-1] and abs(ey - ys[-1]) <= 0.03 * max(ey, ys[-1]):
                names.append(v)
                break
        else:
            ends[(xs[-1], ys[-1])] = [v]
    for (ex, ey), names in ends.items():
        ax.annotate(' = '.join(names), (ex, ey), xytext=(8, 0), textcoords='offset points', va='center',
                    fontsize=9, color=INK2)
    pts = [(S, bw(S, mean(v1[('B3', S, 256, 'expf')]['t']))) for S in SIZES
           if ('B3', S, 256, 'expf') in v1 and v1[('B3', S, 256, 'expf')]['t']]
    if pts:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color='#e87ba4', lw=1.6, ls='--', marker=MARKERS['B3v2'], ms=5,
                markeredgecolor=SURFACE, label='B3, v1 launch (256 threads)')
    ax.axhline(PEAK_GBPS, color=MUTED, ls='--', lw=1.2)
    ax.text(SIZES[-1] * 2.5, PEAK_GBPS, f'theoretical peak {PEAK_GBPS:.0f} GB/s ', ha='right', va='bottom',
            fontsize=9, color=INK2)
    ax.set_ylim(0, PEAK_GBPS * 1.1)
    ax.set_xscale('log', base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([f'{S // 1024}K' for S in SIZES])
    ax.set_xlim(SIZES[0] * 0.85, SIZES[-1] * 2.6)
    ax.set_xlabel('Sequence length S')
    ax.set_ylabel('Effective bandwidth (GB/s)')
    ax.set_title('Version 2 kernels against B2 and torch.softmax (fp16, block 256)', loc='left',
                 fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    fig.savefig(os.path.join(OUT, 'graph_v2_bandwidth.png'), facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def tables(v2, v1):
    L = ['### Version 2: median time per call (µs, mean of two runs, block 256, accurate exp)\n',
         '| S | torch.softmax | B2 | B3 (v1 launch) | B3v2 | B4 | B5 | best v2 / B2 | best v2 / torch |',
         '|---|---|---|---|---|---|---|---|---|']
    for S in SIZES:
        t = {v: mean(v2.get((v, S, 256, 'expf'), {'t': []})['t']) for v in ('torch.softmax', 'B2', 'B3v2', 'B4', 'B5')}
        t['B3v1'] = mean(v1.get(('B3', S, 256, 'expf'), {'t': []})['t'])
        best = min(x for x in (t['B3v2'], t['B4'], t['B5']) if x is not None)
        f = lambda x: f'{x:,.1f}' if x is not None else 'skip'
        L.append(f"| {S:,} | {f(t['torch.softmax'])} | {f(t['B2'])} | {f(t['B3v1'])} | {f(t['B3v2'])} | "
                 f"{f(t['B4'])} | {f(t['B5'])} | {t['B2'] / best:.2f}x faster | {best / t['torch.softmax']:.2f}x the time |")
    L.append('\n### B3 threads per block launched by v2\n')
    L.append('| S | requested | launched |')
    L.append('|---|---|---|')
    for S in SIZES:
        c = v2.get(('B3v2', S, 256, 'expf'))
        if c:
            L.append(f"| {S:,} | 256 | {c['launched'] or 'skip'} |")
    L.append('\n### B4 block size and fast exp2 (S = 8192 unless stated, median µs)\n')
    L.append('| block 128 | block 256 | block 512 | block 1024 | B4 ex2.approx 8192 | B4 ex2.approx 16384 | B5 ex2.approx 8192 | B5 ex2.approx 16384 |')
    L.append('|---|---|---|---|---|---|---|---|')
    g = lambda k: f"{mean(v2[k]['t']):,.0f}" if k in v2 and v2[k]['t'] else ''
    L.append('| ' + ' | '.join([g(('B4', 8192, b, 'expf')) for b in (128, 256, 512, 1024)] +
                              [g((v, S, 256, 'ex2.approx')) for v in ('B4', 'B5') for S in (8192, 16384)]) + ' |')
    L.append('\n### Nsight Compute, version 2 (B2 rows are the v1 reports, same kernel)\n')
    L.append('| variant | S | DRAM B/elem | DRAM % peak | SM % peak | occupancy % | regs | time (µs) | top stall reasons |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for v, d_ in (('B2', 'results'), ('B3v2', OUT), ('B4', OUT), ('B5', OUT)):
        for S in (2048, 8192, 16384):
            p = ncu_path(d_, v, S)
            if not os.path.exists(p):
                continue
            d = parse_ncu_csv(p)
            L.append(f"| {v} | {S:,} | {dram_bytes_per_elem(d, S):.2f} | {d['dram_pct_peak']:.0f} | {d['sm_pct_peak']:.0f} | "
                     f"{d['occupancy_pct']:.0f} | {d['regs_per_thread']:.0f} | {d['time_us']:,.0f} | {top_stalls(d)} |")
    with open(os.path.join(OUT, 'tables_v2.md'), 'w', newline='\n') as f:
        f.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    v2 = load(sys.argv[1:3] or [os.path.join(OUT, 'sweep_v2_run1.csv'), os.path.join(OUT, 'sweep_v2_run2.csv')])
    v1 = load([os.path.join('results', 'sweep_run1.csv'), os.path.join('results', 'sweep_run2.csv')])
    chart(v2, v1)
    tables(v2, v1)
    print('wrote results/v2/graph_v2_bandwidth.png and results/v2/tables_v2.md')
