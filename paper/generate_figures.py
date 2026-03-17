"""Generate publication-quality figures for DW-Bench paper.

Best practices applied:
- Colorblind-safe palette (Tol's muted scheme)
- Consistent font: Computer Modern (LaTeX default)
- No chartjunk: minimal gridlines, no 3D, no unnecessary borders
- High DPI for print (300+)
- Tight, consistent spacing
- Clear data-ink ratio maximization
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from collections import defaultdict

# ── Publication style ────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'text.usetex': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colorblind-safe palette (Tol's muted)
COLORS = {
    'FT':     '#332288',  # Indigo
    'VR':     '#88CCEE',  # Cyan
    'GA':     '#44AA99',  # Teal
    'TU':     '#DDCC77',  # Sand
    'RC':     '#CC6677',  # Rose
    'Oracle': '#117733',  # Green
    'FT_v2':  '#332288',
    'GA_v2':  '#44AA99',
    'TU_v2':  '#DDCC77',
}

rd = Path('evaluation/results')
DS = ['adventureworks', 'tpc-ds', 'tpc-di', 'omop_cdm']
DS_ALL = DS + ['syn_logistics']
OUT = Path('paper/figures')
OUT.mkdir(exist_ok=True)


def load_all(bl, cond, tag, datasets=None):
    r = []
    for d in (datasets or DS):
        f = rd / f'{bl}_{cond}_{d}_{tag}.json'
        if f.exists():
            r.extend(json.load(open(f, encoding='utf-8'))['results'])
    return r

def by_difficulty(results):
    dd = defaultdict(list)
    for r in results:
        dd[r['difficulty']].append(1 if r['scores']['exact_match'] else 0)
    return {d: np.mean(v)*100 for d, v in dd.items()}

def by_subtype(results):
    dd = defaultdict(list)
    for r in results:
        dd[r['subtype']].append(1 if r['scores']['exact_match'] else 0)
    return {s: np.mean(v)*100 for s, v in dd.items()}

def micro_em(results):
    if not results: return 0
    return sum(1 for r in results if r['scores']['exact_match']) / len(results) * 100

def macro_em(results):
    st = by_subtype(results)
    return np.mean(list(st.values())) if st else 0


# ════════════════════════════════════════════════════════════════════
# Figure 1: Difficulty by Baseline (side-by-side panels)
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True)

bls = [('flat_text', 'FT'), ('vector_rag', 'VR'), ('graph_aug', 'GA'),
       ('tool_use', 'TU'), ('react_code', 'RC')]

for ax, (tag, title) in zip(axes, [('gemini-2.5-flash', 'Gemini 2.5 Flash'),
                                     ('deepseek-chat', 'DeepSeek-V3')]):
    diffs = ['easy', 'medium', 'hard']
    x = np.arange(len(diffs))
    w = 0.14
    n = len(bls)

    for i, (bl, bl_short) in enumerate(bls):
        data = by_difficulty(load_all(bl, 'original', tag, DS_ALL))
        if not data:
            continue
        vals = [data.get(d, 0) for d in diffs]
        offset = (i - n/2 + 0.5) * w
        ax.bar(x + offset, vals, w, label=bl_short,
               color=COLORS[bl_short], edgecolor='white', linewidth=0.3)

    # Oracle dashed line
    oracle = by_difficulty(load_all('oracle', 'original', tag, DS_ALL))
    oracle_vals = [oracle.get(d, 0) for d in diffs]
    ax.plot(x, oracle_vals, 's--', markersize=4, label='Oracle',
            color=COLORS['Oracle'], linewidth=1.2)

    ax.set_xlabel('Difficulty')
    if ax == axes[0]:
        ax.set_ylabel('Exact Match (%)')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Easy', 'Medium', 'Hard'])
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(axis='y', alpha=0.2, linewidth=0.4)

axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left',
               frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig(OUT / 'difficulty_comparison.pdf')
plt.savefig(OUT / 'difficulty_comparison.png')
plt.close()
print('✓ difficulty_comparison.pdf')


# ════════════════════════════════════════════════════════════════════
# Figure 2: Subtype Heatmap (Gemini, all 5 baselines)
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(4.5, 5.5))

bl_pairs = [('flat_text', 'FT'), ('vector_rag', 'VR'), ('graph_aug', 'GA'),
            ('tool_use', 'TU'), ('react_code', 'RC')]
all_subtypes = set()
data_map = {}
for bl, bl_short in bl_pairs:
    results = load_all(bl, 'original', 'gemini-2.5-flash', DS_ALL)
    if results:
        st = by_subtype(results)
        data_map[bl_short] = st
        all_subtypes.update(st.keys())

subtypes = sorted(all_subtypes, key=lambda s: data_map.get('TU', {}).get(s, 0))
bl_names = [bl for _, bl in bl_pairs if bl in data_map]

matrix = np.array([[data_map.get(bl, {}).get(s, 0) for bl in bl_names]
                    for s in subtypes])

im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

ax.set_xticks(range(len(bl_names)))
ax.set_xticklabels(bl_names, fontsize=9, fontweight='bold')
ax.set_yticks(range(len(subtypes)))
ax.set_yticklabels([s.replace('_', '\\_') for s in subtypes], fontsize=8)

for i in range(len(subtypes)):
    for j in range(len(bl_names)):
        v = matrix[i, j]
        color = 'white' if v < 35 or v > 75 else 'black'
        ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                fontsize=7, fontweight='bold', color=color)

ax.set_title('Tier 1: EM (%) by Subtype (Gemini)', fontweight='bold', pad=10)
cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cbar.ax.set_ylabel('Exact Match (%)', fontsize=9)
cbar.ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig(OUT / 'subtype_heatmap.pdf')
plt.savefig(OUT / 'subtype_heatmap.png')
plt.close()
print('✓ subtype_heatmap.pdf')


# ════════════════════════════════════════════════════════════════════
# Figure 3: Micro vs Macro EM — Triviality Illusion
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 3.2))

all_bls = [('flat_text', 'FT'), ('vector_rag', 'VR'), ('graph_aug', 'GA'),
           ('tool_use', 'TU'), ('react_code', 'RC')]

models_list = []
micro_vals = []
macro_vals = []

for tag, tag_label in [('gemini-2.5-flash', 'Gemini'), ('deepseek-chat', 'DeepSeek')]:
    for bl, bl_short in all_bls:
        results = load_all(bl, 'original', tag, DS_ALL)
        if results:
            models_list.append(f'{tag_label}\n{bl_short}')
            micro_vals.append(micro_em(results))
            macro_vals.append(macro_em(results))

x = np.arange(len(models_list))
w = 0.35

ax.bar(x - w/2, micro_vals, w, label='Micro-EM',
       color='#88CCEE', edgecolor='white', linewidth=0.3)
ax.bar(x + w/2, macro_vals, w, label='Macro-EM',
       color='#CC6677', edgecolor='white', linewidth=0.3)

for i in range(len(models_list)):
    delta = macro_vals[i] - micro_vals[i]
    y = max(micro_vals[i], macro_vals[i]) + 1.5
    ax.text(i, y, f'{delta:+.0f}', ha='center', fontsize=6.5,
            fontweight='bold', color='#882255' if delta < -5 else '#117733')

ax.set_ylabel('Exact Match (%)')
ax.set_title('Micro vs. Macro EM Across Baselines', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_list, fontsize=7.5)
ax.set_ylim(0, 100)
ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
ax.legend(fontsize=9, frameon=False, loc='upper right')
ax.grid(axis='y', alpha=0.2, linewidth=0.4)

gem_count = sum(1 for m in models_list if 'Gemini' in m)
ax.axvline(gem_count - 0.5, color='gray', linestyle=':', alpha=0.5, linewidth=0.7)

plt.tight_layout()
plt.savefig(OUT / 'triviality_illusion.pdf')
plt.savefig(OUT / 'triviality_illusion.png')
plt.close()
print('✓ triviality_illusion.pdf')


# ════════════════════════════════════════════════════════════════════
# Figure 4: Obfuscation Penalty (grouped bars)
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 3.0))

labels = []
penalties = []

for tag, tag_label in [('gemini-2.5-flash', 'Gemini'), ('deepseek-chat', 'DeepSeek')]:
    for bl, bl_short in all_bls:
        orig = load_all(bl, 'original', tag, DS)
        obf = load_all(bl, 'obfuscated', tag, DS)
        if orig and obf:
            delta = micro_em(obf) - micro_em(orig)
            labels.append(f'{tag_label}\n{bl_short}')
            penalties.append(delta)

colors = ['#CC6677' if p < -20 else '#DDCC77' if p < -10 else '#117733'
          for p in penalties]
bars = ax.bar(range(len(labels)), penalties, color=colors,
              edgecolor='white', linewidth=0.3)

for i, (p, bar) in enumerate(zip(penalties, bars)):
    va = 'top' if p < 0 else 'bottom'
    y = p - 1.2 if p < 0 else p + 0.5
    ax.text(bar.get_x() + bar.get_width()/2, y,
            f'{p:.1f}', ha='center', va=va, fontsize=6.5, fontweight='bold')

ax.set_ylabel('EM Change (%)')
ax.set_title('Obfuscation Penalty by Baseline', fontweight='bold')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=7.5)
ax.axhline(0, color='black', linewidth=0.6)
ax.set_ylim(min(penalties) - 5, 5)
ax.grid(axis='y', alpha=0.2, linewidth=0.4)

gem_count_obf = sum(1 for l in labels if 'Gemini' in l)
ax.axvline(gem_count_obf - 0.5, color='gray', linestyle=':', alpha=0.5, linewidth=0.7)

plt.tight_layout()
plt.savefig(OUT / 'obfuscation_penalty.pdf')
plt.savefig(OUT / 'obfuscation_penalty.png')
plt.close()
print('✓ obfuscation_penalty.pdf')


# ════════════════════════════════════════════════════════════════════
# Figure 5: Tier 2 — Value-Level Results by Subtype (BOTH models)
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.8), sharey=True)

t2_bls = [('flat_text_v2', 'FT', COLORS['FT_v2']),
          ('graph_aug_v2', 'GA', COLORS['GA_v2']),
          ('tool_use_v2', 'TU', COLORS['TU_v2'])]

subtype_order = ['cascade_count', 'row_provenance', 'row_impact',
                 'value_origin', 'multi_hop_trace',
                 'value_propagation', 'cross_silo_reachability', 'shared_source']
subtype_labels = ['cascade\ncount', 'row\nprov.', 'row\nimpact',
                  'value\norigin', 'multi-hop\ntrace',
                  'value\nprop.', 'cross-silo\nreach.', 'shared\nsource']

for ax, (tag, title) in zip(axes, [('gemini-2.5-flash', 'Gemini 2.5 Flash'),
                                     ('deepseek-chat', 'DeepSeek-V3')]):
    t2_data = {}
    for bl, label, color in t2_bls:
        f = rd / f'{bl}_original_syn_logistics_{tag}.json'
        if f.exists():
            results = json.load(open(f, encoding='utf-8'))['results']
            st = by_subtype(results)
            t2_data[label] = st

    x = np.arange(len(subtype_order))
    w = 0.25
    n = len(t2_bls)

    for i, (bl, label, color) in enumerate(t2_bls):
        vals = [t2_data.get(label, {}).get(s, 0) for s in subtype_order]
        offset = (i - n/2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=label, color=color,
                      edgecolor='white', linewidth=0.3)
        for j, v in enumerate(vals):
            if v >= 90 or (v == 0 and subtype_order[j] in ['cascade_count', 'row_provenance']):
                ax.text(x[j] + offset, v + 1.5, f'{v:.0f}',
                        ha='center', va='bottom', fontsize=5.5, fontweight='bold')

    ax.set_xlabel('')
    if ax == axes[0]:
        ax.set_ylabel('Exact Match (%)')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subtype_labels, fontsize=6)
    ax.set_ylim(0, 112)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(axis='y', alpha=0.2, linewidth=0.4)

axes[1].legend(fontsize=8, frameon=False, loc='upper right', ncol=3)
fig.suptitle('Tier 2: Value-Level Results by Subtype', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT / 'tier2_subtype.pdf')
plt.savefig(OUT / 'tier2_subtype.png')
plt.close()
print('✓ tier2_subtype.pdf (both models)')


print('\n✅ All 5 figures generated with publication-quality styling!')

