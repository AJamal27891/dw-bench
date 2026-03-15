"""Generate publication-quality figures for DW-Bench paper."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 300,
})

rd = Path('evaluation/results')
DS = ['adventureworks', 'tpc-ds', 'tpc-di', 'omop_cdm']
OUT = Path('paper/figures')
OUT.mkdir(exist_ok=True)

def load_all(bl, cond, tag):
    r = []
    for d in DS:
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


# ── Figure 1: Difficulty Bar Chart (Gemini + DeepSeek + Oracle) ──────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

for ax, (tag, title) in zip(axes, [('gemini-2.5-flash', 'Gemini 2.5 Flash'),
                                     ('deepseek-chat', 'DeepSeek-V3')]):
    diffs = ['easy', 'medium', 'hard']
    bls = [('flat_text', 'Flat Text', '#4ECDC4'),
           ('vector_rag', 'Vector RAG', '#FF6B6B'),
           ('graph_aug', 'Graph-Aug', '#45B7D1')]

    x = np.arange(len(diffs))
    w = 0.22

    for i, (bl, label, color) in enumerate(bls):
        data = by_difficulty(load_all(bl, 'original', tag))
        vals = [data.get(d, 0) for d in diffs]
        ax.bar(x + (i-1)*w, vals, w, label=label, color=color,
               edgecolor='white', linewidth=0.5)

    # Oracle line
    oracle = by_difficulty(load_all('oracle', 'original', tag))
    oracle_vals = [oracle.get(d, 0) for d in diffs]
    ax.plot(x, oracle_vals, 'k--o', markersize=6, label='Oracle', linewidth=1.5)

    ax.set_xlabel('Difficulty')
    ax.set_ylabel('Exact Match (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(['Easy', 'Medium', 'Hard'])
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / 'difficulty_comparison.pdf', bbox_inches='tight')
plt.savefig(OUT / 'difficulty_comparison.png', bbox_inches='tight')
print(f'Saved: {OUT}/difficulty_comparison.pdf')


# ── Figure 2: Subtype Heatmap (Gemini) ────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

bls = [('flat_text', 'FT'), ('vector_rag', 'VR'), ('graph_aug', 'GA')]
all_subtypes = set()
data_map = {}
for bl, bl_short in bls:
    st = by_subtype(load_all(bl, 'original', 'gemini-2.5-flash'))
    data_map[bl_short] = st
    all_subtypes.update(st.keys())

# Sort by GA performance (ascending = hardest at top)
subtypes = sorted(all_subtypes, key=lambda s: data_map['GA'].get(s, 0))

matrix = np.array([[data_map[bl].get(s, 0) for bl in ['FT', 'VR', 'GA']]
                    for s in subtypes])

im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

ax.set_xticks(range(3))
ax.set_xticklabels(['Flat Text', 'Vector RAG', 'Graph-Aug'])
ax.set_yticks(range(len(subtypes)))
ax.set_yticklabels([s.replace('_', ' ') for s in subtypes], fontsize=9)

# Annotate cells
for i in range(len(subtypes)):
    for j in range(3):
        v = matrix[i, j]
        color = 'white' if v < 30 or v > 80 else 'black'
        ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                fontsize=8, fontweight='bold', color=color)

ax.set_title('Gemini 2.5 Flash: EM (%) by Subtype', fontsize=13)
plt.colorbar(im, ax=ax, shrink=0.8, label='Exact Match (%)')
plt.tight_layout()
plt.savefig(OUT / 'subtype_heatmap.pdf', bbox_inches='tight')
plt.savefig(OUT / 'subtype_heatmap.png', bbox_inches='tight')
print(f'Saved: {OUT}/subtype_heatmap.pdf')


# ── Figure 3: Micro vs Macro EM (the "Triviality Illusion") ─────────
fig, ax = plt.subplots(figsize=(8, 4.5))

models = ['Gemini FT', 'Gemini VR', 'Gemini GA',
          'DeepSeek FT', 'DeepSeek VR', 'DeepSeek GA']
micro_vals = [73.9, 68.4, 70.4, 68.4, 74.5, 82.3]
macro_vals = [61.1, 53.0, 68.0, 56.1, 54.2, 73.9]

x = np.arange(len(models))
w = 0.35

bars1 = ax.bar(x - w/2, micro_vals, w, label='Micro-EM',
               color='#45B7D1', edgecolor='white')
bars2 = ax.bar(x + w/2, macro_vals, w, label='Macro-EM',
               color='#FF6B6B', edgecolor='white')

# Annotate deltas
for i in range(len(models)):
    delta = macro_vals[i] - micro_vals[i]
    y = max(micro_vals[i], macro_vals[i]) + 1.5
    ax.text(i, y, f'{delta:+.1f}%', ha='center', fontsize=8,
            fontweight='bold', color='#c0392b' if delta < -5 else '#27ae60')

ax.set_ylabel('Exact Match (%)')
ax.set_title('The Triviality Illusion: Micro vs. Macro EM')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
ax.set_ylim(0, 95)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.axvline(2.5, color='gray', linestyle='--', alpha=0.5)
ax.text(1.0, 88, 'Gemini 2.5 Flash', ha='center', fontsize=9, style='italic')
ax.text(4.0, 88, 'DeepSeek-V3', ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig(OUT / 'triviality_illusion.pdf', bbox_inches='tight')
plt.savefig(OUT / 'triviality_illusion.png', bbox_inches='tight')
print(f'Saved: {OUT}/triviality_illusion.pdf')


# ── Figure 4: Obfuscation Penalty ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))

labels = ['Gemini\nFT', 'Gemini\nVR', 'Gemini\nGA',
          'DeepSeek\nFT', 'DeepSeek\nVR', 'DeepSeek\nGA']
orig =   [73.9, 68.4, 70.4, 68.4, 74.5, 82.3]
obfusc = [64.9, 50.1, 49.1, 40.7, 44.4, 59.2]
penalty = [o - r for o, r in zip(obfusc, orig)]

colors = ['#e74c3c' if p < -20 else '#f39c12' if p < -10 else '#27ae60'
          for p in penalty]
bars = ax.bar(range(len(labels)), penalty, color=colors, edgecolor='white')

for i, (p, bar) in enumerate(zip(penalty, bars)):
    ax.text(bar.get_x() + bar.get_width()/2, p - 1.5,
            f'{p:.1f}%', ha='center', va='top', fontsize=9, fontweight='bold')

ax.set_ylabel('EM Change (%)')
ax.set_title('Obfuscation Penalty: Original → Obfuscated')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylim(-35, 2)
ax.grid(axis='y', alpha=0.3)
ax.axvline(2.5, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(OUT / 'obfuscation_penalty.pdf', bbox_inches='tight')
plt.savefig(OUT / 'obfuscation_penalty.png', bbox_inches='tight')
print(f'Saved: {OUT}/obfuscation_penalty.pdf')

print('\n✅ All 4 figures generated!')
