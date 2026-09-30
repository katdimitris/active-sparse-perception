"""Plot accuracy against compute: static models as points, recurrent models as
early-exit curves, from the JSON files written by tools/eval_exits.py.  Also prints,
for each static model, what the recurrent curve achieves at the same compute and how
much compute it needs to reach the same accuracy.

    python tools/plot_frontier.py --static results/imagenet200/static_n*.json \
        --recurrent results/imagenet200/recurrent_9-16-25_memory.json \
        --title "ImageNet-200, 192 px" --out assets/imagenet200_frontier.png
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED = '#1d1d1f', '#6b6b70'
COLORS = ['#2066c9', '#d0632a', '#2a9d6e']


def load(paths):
    return [json.load(open(p)) for p in paths]


def best_at_most(curve, gflops):
    return max((acc for c, acc in curve if c <= gflops + 1e-9), default=float('nan'))


def cheapest_reaching(curve, acc):
    return min((c for c, a in curve if a >= acc), default=float('nan'))


def place_labels(fig, ax, static, recurrent, avoid=()):
    """Label each static point at the first position around it that stays inside the axes
    and clears the curves, the markers, the dashed static line and the other labels."""
    fig.canvas.draw()
    pts = [p for d in recurrent for p in d['early_exit']]
    line = [(d['gflops'][0], d['top1'][0]) for d in static]
    for (x0, y0), (x1, y1) in zip(line, line[1:]):
        pts += [(x0 + (x1 - x0) * t / 30, y0 + (y1 - y0) * t / 30) for t in range(31)]
    markers = [p for d in recurrent for p in zip(d['gflops'], d['top1'])] + line
    pts_disp, markers_disp = ax.transData.transform(pts), ax.transData.transform(markers)
    boxes = [t.get_window_extent() for t in avoid]
    axes_box = ax.get_window_extent()
    offsets = [(-8, 6, 'right', 'bottom'), (8, -6, 'left', 'top'), (8, 6, 'left', 'bottom'),
               (-8, -6, 'right', 'top'), (10, 0, 'left', 'center'), (-10, 0, 'right', 'center'),
               (0, 12, 'center', 'bottom'), (0, -12, 'center', 'top'),
               # a second ring, further out, joined to the point by a leader line
               (22, -22, 'left', 'top'), (-22, 22, 'right', 'bottom'), (22, 22, 'left', 'bottom'),
               (-22, -22, 'right', 'top')]
    for xy in line:
        own = ax.transData.transform(xy)
        others = [q for q in pts_disp if ((q - own) ** 2).sum() > 14 ** 2]
        other_markers = [q for q in markers_disp if ((q - own) ** 2).sum() > 1]
        best = None
        for dx, dy, ha, va in offsets:
            leader = dict(arrowstyle='-', lw=.6, color=MUTED, shrinkA=0, shrinkB=4) if abs(dx) > 15 else None
            t = ax.annotate(f"N={dict(zip(line, static))[xy]['tokens'][0]}\n{xy[1]:.2f}", xy,
                            textcoords='offset points', xytext=(dx, dy), ha=ha, va=va, fontsize=8.5,
                            color=INK, zorder=6, arrowprops=leader,
                            bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=.85))
            fig.canvas.draw()
            box = t.get_window_extent()
            # covering a marker or another label is worse than crossing a line
            score = sum(box.padded(6).contains(*q) for q in others) \
                + 10 * sum(box.padded(14).contains(*q) for q in other_markers) \
                + 10 * sum(box.padded(3).overlaps(b) for b in boxes)
            if not (axes_box.contains(box.x0, box.y0) and axes_box.contains(box.x1, box.y1)):
                score += 1000
            if best is None or score < best[0]:
                if best is not None:
                    best[1].remove()
                best = (score, t, box)
            else:
                t.remove()
            if score == 0:
                break
        boxes.append(best[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--static', nargs='+', required=True)
    ap.add_argument('--recurrent', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', help='legend labels for the recurrent models')
    ap.add_argument('--title', default='')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    static = sorted(load(a.static), key=lambda d: d['gflops'][0])
    recurrent = load(a.recurrent)
    labels = a.labels or ['recurrent ' + '→'.join(map(str, d['tokens'])) for d in recurrent]

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    round_labels = []
    ax.plot([d['gflops'][0] for d in static], [d['top1'][0] for d in static], '--o', color=INK, lw=1.4,
            ms=7, mec='white', mew=1., zorder=5, label='SparseFormer, static (N tokens, one pass)')
    for k, (d, label) in enumerate(zip(recurrent, labels)):
        col = COLORS[k % len(COLORS)]
        ax.plot([c for c, _ in d['early_exit']], [v for _, v in d['early_exit']], '-', color=col, lw=2.,
                zorder=4, label=f'{label}, confidence-based early exit')
        ax.plot(d['gflops'], d['top1'], 's', color=col, ms=6.5, mec='white', mew=1., zorder=6,
                label=f'{label}, fixed exit after round 1 / 2 / 3')
        for r, (c, v) in enumerate(zip(d['gflops'], d['top1'])):
            round_labels.append(ax.annotate(f'r{r + 1}', (c, v), textcoords='offset points', xytext=(6, -11),
                                            fontsize=8.5, color=col))
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(x0, x1 + .03 * (x1 - x0))               # room for labels at the edges
    ax.set_ylim(y0, y1 + .07 * (y1 - y0))
    place_labels(fig, ax, static, recurrent, avoid=round_labels)
    ax.set_xlabel('GFLOPs per image')
    ax.set_ylabel('top-1 accuracy (%)')
    if a.title:
        ax.set_title(a.title, fontsize=11)
    ax.grid(alpha=.3, lw=.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(fontsize=8.5, loc='lower right', framealpha=.95)
    fig.tight_layout()
    fig.savefig(a.out, dpi=160)
    print(f'wrote {a.out}')

    for d, label in zip(recurrent, labels):
        curve = d['early_exit']
        print(f'\n{label} (exits: '
              + ', '.join(f'{c:.3f} GF / {v:.2f}%' for c, v in zip(d['gflops'], d['top1'])) + ')')
        for s in static:
            c, v = s['gflops'][0], s['top1'][0]
            need = cheapest_reaching(curve, v)
            print(f"  vs static N={s['tokens'][0]} ({c:.3f} GF, {v:.2f}%):  same compute -> "
                  f'{best_at_most(curve, c):.2f}%   same accuracy -> {need:.3f} GF '
                  f'({100 * (need / c - 1):+.0f}% compute)')


if __name__ == '__main__':
    main()
