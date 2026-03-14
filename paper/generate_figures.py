"""Auto-generate paper figures from evaluation result JSON files.

Usage: python paper/generate_figures.py
Output: paper/figures/*.pdf

Run this AFTER evaluations complete to fill in the paper placeholders.
"""
import json
import sys
import os
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Color palette
COLORS = {
    'FT': '#4C72B0',  # steel blue
    'VR': '#DD8452',  # warm orange
    'GA': '#55A868',  # forest green
}
DIFF_COLORS = {'easy': '#92C5DE', 'medium': '#F4A582', 'hard': '#CA0020'}

results_dir = Path(__file__).parent.parent / 'evaluation' / 'results'
figures_dir = Path(__file__).parent / 'figures'
figures_dir.mkdir(exist_ok=True)

BASELINES = ['flat_text', 'vector_rag', 'graph_aug']
BL_SHORT = {'flat_text': 'FT', 'vector_rag': 'VR', 'graph_aug': 'GA'}
BL_LABELS = {'flat_text': 'Flat Text', 'vector_rag': 'Vector RAG',
             'graph_aug': 'Graph-Aug'}
DATASETS = ['adventureworks', 'tpc-ds', 'tpc-di', 'omop_cdm']
MODEL = 'gemini-2.5-flash'


def load_all_results(suffix='original'):
    """Load all result files for a given suffix."""
    all_data = {}
    all_results = []
    for bl in BASELINES:
        for ds in DATASETS:
            f = results_dir / f"{bl}_{suffix}_{ds}_{MODEL}.json"
            if f.exists() and f.stat().st_size > 100:
                d = json.load(open(f, encoding='utf-8'))
                if 'results' in d:
                    all_data[(bl, ds)] = d
                    for r in d['results']:
                        r['_baseline'] = bl
                        r['_dataset'] = ds
                        all_results.append(r)
    return all_data, all_results


def fig3_difficulty_bars(all_results):
    """Figure 3: Grouped bar chart by difficulty."""
    fig, ax = plt.subplots(figsize=(6, 4))
    diffs = ['easy', 'medium', 'hard']
    x = np.arange(len(diffs))
    width = 0.25

    for i, bl in enumerate(BASELINES):
        ems = []
        for diff in diffs:
            items = [r for r in all_results
                     if r['_baseline'] == bl and r['difficulty'] == diff]
            if items:
                em = sum(int(r['scores']['exact_match']) for r in items) / len(items) * 100
            else:
                em = 0
            ems.append(em)
        bars = ax.bar(x + i * width, ems, width,
                      label=BL_LABELS[bl], color=COLORS[BL_SHORT[bl]],
                      edgecolor='white', linewidth=0.5)
        # Add value labels
        for bar, val in zip(bars, ems):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Difficulty', fontsize=12)
    ax.set_ylabel('Exact Match (%)', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Easy', 'Medium', 'Hard'], fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(figures_dir / 'fig3_difficulty_bars.pdf', dpi=300)
    plt.close()
    print("  Created fig3_difficulty_bars.pdf")


def fig4_subtype_heatmap(all_results):
    """Figure 4: Heatmap of EM by subtype × baseline."""
    try:
        import seaborn as sns
    except ImportError:
        print("  SKIP fig4 (seaborn not installed)")
        return

    subtypes = sorted(set(r['subtype'] for r in all_results))
    # Sort by difficulty then name
    diff_order = {'easy': 0, 'medium': 1, 'hard': 2}
    diff_map = {}
    for r in all_results:
        diff_map[r['subtype']] = r['difficulty']
    subtypes = sorted(subtypes, key=lambda s: (diff_order.get(diff_map.get(s, 'medium'), 1), s))

    data = np.zeros((len(subtypes), 3))
    for i, st in enumerate(subtypes):
        for j, bl in enumerate(BASELINES):
            items = [r for r in all_results
                     if r['subtype'] == st and r['_baseline'] == bl]
            if items:
                data[i, j] = sum(int(r['scores']['exact_match']) for r in items) / len(items) * 100

    fig, ax = plt.subplots(figsize=(5, 7))
    sns.heatmap(data, annot=True, fmt='.0f', cmap='RdYlGn',
                xticklabels=['Flat Text', 'Vector RAG', 'Graph-Aug'],
                yticklabels=subtypes, vmin=0, vmax=100,
                cbar_kws={'label': 'Exact Match (%)'}, ax=ax)
    ax.set_title('EM (%) by Subtype × Baseline', fontsize=12)
    plt.tight_layout()
    plt.savefig(figures_dir / 'fig4_subtype_heatmap.pdf', dpi=300)
    plt.close()
    print("  Created fig4_subtype_heatmap.pdf")


def fig5_delta_bars(all_results):
    """Figure 5: GA−FT delta by subtype (horizontal bars)."""
    subtypes = sorted(set(r['subtype'] for r in all_results))

    deltas = {}
    for st in subtypes:
        ft_items = [r for r in all_results
                    if r['subtype'] == st and r['_baseline'] == 'flat_text']
        ga_items = [r for r in all_results
                    if r['subtype'] == st and r['_baseline'] == 'graph_aug']
        if ft_items and ga_items:
            ft_em = sum(int(r['scores']['exact_match']) for r in ft_items) / len(ft_items) * 100
            ga_em = sum(int(r['scores']['exact_match']) for r in ga_items) / len(ga_items) * 100
            deltas[st] = ga_em - ft_em

    # Sort by delta
    sorted_items = sorted(deltas.items(), key=lambda x: x[1])
    names = [x[0] for x in sorted_items]
    vals = [x[1] for x in sorted_items]
    colors = [COLORS['GA'] if v >= 0 else COLORS['FT'] for v in vals]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.barh(names, vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Δ Exact Match: Graph-Aug − Flat Text (%)', fontsize=10)
    ax.set_title('Where Graph-Aug Wins (green) vs Flat Text Wins (blue)', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars, vals):
        x_pos = bar.get_width() + (1 if val >= 0 else -1)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{val:+.1f}%', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(figures_dir / 'fig5_delta_bars.pdf', dpi=300)
    plt.close()
    print("  Created fig5_delta_bars.pdf")


def fig7_obfuscation(all_data_orig, all_data_obf):
    """Figure 7: Original vs Obfuscated paired bars."""
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(3)
    width = 0.35

    orig_ems = []
    obf_ems = []
    for bl in BASELINES:
        o_total, o_correct = 0, 0
        b_total, b_correct = 0, 0
        for ds in DATASETS:
            if (bl, ds) in all_data_orig:
                d = all_data_orig[(bl, ds)]
                o_total += d['metrics']['total']
                o_correct += d['metrics']['exact_match'] * d['metrics']['total']
            if (bl, ds) in all_data_obf:
                d = all_data_obf[(bl, ds)]
                b_total += d['metrics']['total']
                b_correct += d['metrics']['exact_match'] * d['metrics']['total']
        orig_ems.append(o_correct / o_total * 100 if o_total else 0)
        obf_ems.append(b_correct / b_total * 100 if b_total else 0)

    bars1 = ax.bar(x - width / 2, orig_ems, width, label='Original',
                   color=[COLORS[BL_SHORT[bl]] for bl in BASELINES],
                   edgecolor='white')
    bars2 = ax.bar(x + width / 2, obf_ems, width, label='Obfuscated',
                   color=[COLORS[BL_SHORT[bl]] for bl in BASELINES],
                   alpha=0.5, hatch='//', edgecolor='gray')

    ax.set_ylabel('Exact Match (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([BL_LABELS[bl] for bl in BASELINES], fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(figures_dir / 'fig7_obfuscation.pdf', dpi=300)
    plt.close()
    print("  Created fig7_obfuscation.pdf")


def main():
    print("Generating DW-Bench paper figures...")
    print(f"Results dir: {results_dir}")
    print(f"Output dir: {figures_dir}\n")

    all_data_orig, all_results_orig = load_all_results('original')
    print(f"Loaded {len(all_results_orig)} original results")

    if all_results_orig:
        fig3_difficulty_bars(all_results_orig)
        fig4_subtype_heatmap(all_results_orig)
        fig5_delta_bars(all_results_orig)
    else:
        print("  No original results found, skipping figures 3-5")

    all_data_obf, _ = load_all_results('obfuscated')
    if all_data_obf:
        fig7_obfuscation(all_data_orig, all_data_obf)
    else:
        print("  No obfuscated results found, skipping figure 7")

    print("\nDone! Figures saved to paper/figures/")


if __name__ == '__main__':
    main()
