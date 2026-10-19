"""Extra README figures and markdown tables, built from the sweep CSVs and the ncu CSVs.

usage: python make_readme_figures.py results/sweep_run1.csv results/sweep_run2.csv
writes results/graph3_bytes_per_elem.png, results/graph4_block_size.png,
       results/graph5_limiters.png, results/readme_tables.md
"""
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze import (COLORS, INK, INK2, MARKERS, MODEL_BYTES, MUTED, PEAK_GBPS, SIZES, SURFACE,
                     _style, dram_bytes_per_elem, ncu_path, parse_ncu_csv, top_stalls)

FUSED = ['torch.softmax', 'B1', 'B2', 'B3']
ALL = ['composed'] + FUSED
NCU_S = [2048, 8192, 16384]
NCU_CELLS = [(2048, 256), (8192, 256), (8192, 512), (16384, 256)]


def load_sweeps(paths):
    """(variant, S, block, exp) -> list of t_median_us over the runs, plus peak memory."""
    cells = {}
    for p in paths:
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                k = (r['variant'], int(r['S']), int(r['block']), r['exp_impl'])
                c = cells.setdefault(k, {'t': [], 'mem': None, 'status': r['status']})
                if r['status'] == 'ok':
                    c['t'].append(float(r['t_median_us']))
                    c['mem'] = float(r['peak_mem_MiB'])
    return cells


def mean(v):
    return sum(v) / len(v) if v else None


def spread_pct(v):
    return (max(v) - min(v)) / min(v) * 100 if len(v) > 1 else 0.0


def ncu_table():
    out = {}
    for v in ALL:
        for S, block in NCU_CELLS:
            p = ncu_path('results', v, S, block)
            if os.path.exists(p):
                d = parse_ncu_csv(p)
                d['bpe'] = dram_bytes_per_elem(d, S)
                out[(v, S) if block == 256 else (v, S, block)] = d
    return out


def fig_bytes(ncu):
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=150, facecolor=SURFACE)
    _style(ax)
    w = 0.19
    for i, v in enumerate(FUSED):
        xs, ys = [], []
        for j, S in enumerate(NCU_S):
            if (v, S) in ncu:
                xs.append(j + (i - 1.5) * w)
                ys.append(ncu[(v, S)]['bpe'])
        ax.bar(xs, ys, width=w - 0.02, color=COLORS[v], label=v, edgecolor=SURFACE, linewidth=2)
        for x, y in zip(xs, ys):
            ax.text(x, y - 0.15, f'{y:.1f}', ha='center', va='top', fontsize=7.5, color=SURFACE, fontweight='bold')
        if v in MODEL_BYTES:
            for x in xs:
                ax.plot([x - w / 2 + 0.01, x + w / 2 - 0.01], [MODEL_BYTES[v]] * 2, color=INK, lw=1.4, ls=(0, (2, 1.5)))
    ax.plot([], [], color=INK, lw=1.4, ls=(0, (2, 1.5)), label='idealized model (no L2 hits)')
    ax.set_xticks(range(len(NCU_S)))
    ax.set_xticklabels([f'S = {S}\nrow = {2 * S // 1024} KiB' for S in NCU_S])
    ax.set_ylabel('DRAM bytes per element (ncu)')
    ax.set_ylim(0, 9.5)
    ax.set_title('Measured DRAM traffic per element vs the idealized pass model', loc='left', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=5)
    fig.savefig('results/graph3_bytes_per_elem.png', facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def fig_block(cells):
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150, facecolor=SURFACE)
    _style(ax)
    blocks = [128, 256, 512, 1024]
    w = 0.38
    for i, v in enumerate(['B1', 'B2']):
        xs = [j + (i - 0.5) * w for j in range(len(blocks))]
        ys = [mean(cells[(v, 8192, b, 'expf')]['t']) for b in blocks]
        ax.bar(xs, ys, width=w - 0.03, color=COLORS[v], label=v, edgecolor=SURFACE, linewidth=2)
        for x, y in zip(xs, ys):
            ax.text(x, y + 30, f'{y:,.0f}', ha='center', va='bottom', fontsize=8, color=INK2)
    t_torch = mean(cells[('torch.softmax', 8192, 256, 'expf')]['t'])
    ax.axhline(t_torch, color=COLORS['torch.softmax'], lw=1.6, ls='--')
    ax.text(-0.45, t_torch - 40, f'torch.softmax, block 256: {t_torch:,.0f} µs', ha='left', va='top', fontsize=8.5, color=INK2,
            bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1.5))
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels([f'block {b}' for b in blocks])
    ax.set_ylabel('median time at S = 8192 (µs, lower is better)')
    ax.set_title('Threads per block changes B1 and B2 time by up to 40%', loc='left', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.savefig('results/graph4_block_size.png', facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def fig_limiters(ncu):
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150, facecolor=SURFACE)
    _style(ax)
    for v in ALL:
        for S, ms in ((8192, 9), (16384, 13)):
            if (v, S) not in ncu:
                continue
            d = ncu[(v, S)]
            ax.plot([d['dram_pct_peak']], [d['sm_pct_peak']], marker=MARKERS[v], ms=ms, color=COLORS[v],
                    markeredgecolor=SURFACE, markeredgewidth=1.5, ls='none',
                    label=v if S == 8192 else None)
            off = {('B3', 8192): (-8, 6, 'right'), ('torch.softmax', 8192): (8, 4, 'left')}.get((v, S), (9, -3, 'left'))
            ax.annotate(f"{v} {S // 1024}K, occ {d['occupancy_pct']:.0f}%", (d['dram_pct_peak'], d['sm_pct_peak']),
                        xytext=off[:2], textcoords='offset points', fontsize=7.5, color=INK2, ha=off[2])
    ax.plot([0, 100], [0, 100], color=MUTED, lw=0.8, ls=':')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('DRAM throughput, % of peak (ncu)')
    ax.set_ylabel('SM throughput, % of peak (ncu)')
    ax.set_title('No kernel is close to either roof: latency and occupancy set the time', loc='left', fontsize=11, color=INK)
    ax.text(97, 4, 'near x = 100: DRAM-bound\nnear y = 100: SM-bound', ha='right', va='bottom', fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=5)
    fig.savefig('results/graph5_limiters.png', facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def tables(cells, ncu):
    L = []
    L.append('### Table A. Median time per call (µs, CUDA events, mean of the two sweep runs; lower is better)\n')
    L.append('| S | composed | torch.softmax | B1 (3-pass) | B2 (online, 2-pass) | B3 (row in smem) |')
    L.append('|---|---|---|---|---|---|')
    for S in SIZES:
        row = [f'{S:,}']
        for v in ALL:
            c = cells.get((v, S, 256, 'expf'))
            row.append('skip (row > smem)' if c and c['status'] == 'skip_smem' else f"{mean(c['t']):,.1f}")
        L.append('| ' + ' | '.join(row) + ' |')
    L.append('\n### Table B. Effective bandwidth (GB/s, 4·S² bytes / time; theoretical peak 352 GB/s)\n')
    L.append('| S | composed | torch.softmax | B1 | B2 | B3 | best custom / torch |')
    L.append('|---|---|---|---|---|---|---|')
    for S in SIZES:
        row, bws = [f'{S:,}'], {}
        for v in ALL:
            c = cells.get((v, S, 256, 'expf'))
            if c and c['t']:
                bws[v] = 4 * S * S / (mean(c['t']) * 1e-6) / 1e9
                row.append(f'{bws[v]:.0f}')
            else:
                row.append('skip')
        best = max(bws.get(v, 0) for v in ('B1', 'B2', 'B3'))
        row.append(f"{best / bws['torch.softmax'] * 100:.0f}%")
        L.append('| ' + ' | '.join(row) + ' |')
    L.append('\n### Table C. Nsight Compute (ncu time is at locked base clocks, so it is slower than Table A)\n')
    stalls = any(d.get('stalls') for d in ncu.values())
    L.append('| variant | S | block | kernels/call | DRAM B/elem | model B/elem | L2 hit % | DRAM % peak | SM % peak | occupancy % | regs/thread | time (µs) |'
             + (' top stall reasons (warps per issue) |' if stalls else ''))
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|' + ('---|' if stalls else ''))
    for v in ALL:
        for S, block in NCU_CELLS:
            d = ncu.get((v, S) if block == 256 else (v, S, block))
            if not d:
                continue
            L.append(f"| {v} | {S:,} | {block} | {d['n_kernels']} | {d['bpe']:.2f} | {MODEL_BYTES.get(v, 4 if v == 'torch.softmax' else '')} | "
                     f"{d['l2_hit_pct']:.0f} | {d['dram_pct_peak']:.0f} | {d['sm_pct_peak']:.0f} | {d['occupancy_pct']:.0f} | "
                     f"{d['regs_per_thread']:.0f} | {d['time_us']:,.0f} |" + (f" {top_stalls(d)} |" if stalls else ''))
    L.append('\n### Table D. Peak GPU memory above the input (MiB)\n')
    L.append('| S | input | composed | torch.softmax, B1, B2, B3 |')
    L.append('|---|---|---|---|')
    for S in SIZES:
        L.append(f"| {S:,} | {2 * S * S / 2**20:,.0f} | {cells[('composed', S, 256, 'expf')]['mem']:,.0f} | "
                 f"{cells[('torch.softmax', S, 256, 'expf')]['mem']:,.0f} |")
    L.append('\n### Table E. Knobs at S = 8192 (median µs, mean of two runs)\n')
    L.append('| variant | block 128 | block 256 | block 512 | block 1024 | `__expf` at 8192 | `__expf` at 16384 |')
    L.append('|---|---|---|---|---|---|---|')
    for v in ('B1', 'B2'):
        t = [mean(cells[(v, 8192, b, 'expf')]['t']) for b in (128, 256, 512, 1024)]
        f8 = mean(cells[(v, 8192, 256, '__expf')]['t'])
        f16 = mean(cells[(v, 16384, 256, '__expf')]['t'])
        e8 = (1 - f8 / t[1]) * 100
        e16 = (1 - f16 / mean(cells[(v, 16384, 256, 'expf')]['t'])) * 100
        L.append(f"| {v} | {t[0]:,.0f} | {t[1]:,.0f} | {t[2]:,.0f} | {t[3]:,.0f} | {f8:,.0f} ({e8:.1f}% faster) | {f16:,.0f} ({e16:.1f}% faster) |")
    worst = max((spread_pct(c['t']), k) for k, c in cells.items() if len(c['t']) > 1)
    L.append(f'\nRun-to-run spread: the largest relative difference between run 1 and run 2 is '
             f'{worst[0]:.1f}% ({worst[1][0]}, S = {worst[1][1]}, block {worst[1][2]}).')
    with open('results/readme_tables.md', 'w', newline='\n') as f:
        f.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    cells = load_sweeps(sys.argv[1:] or ['results/sweep_run1.csv', 'results/sweep_run2.csv'])
    ncu = ncu_table()
    fig_bytes(ncu)
    fig_block(cells)
    fig_limiters(ncu)
    tables(cells, ncu)
    print('wrote graph3_bytes_per_elem.png, graph4_block_size.png, graph5_limiters.png, readme_tables.md')
