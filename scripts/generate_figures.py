"""
Generate all formal figures for the thesis (English labels for compatibility).
  1. Bland-Altman plots (SBP & DBP)
  2. SHAP feature importance bar charts (SBP & DBP)
  3. Multi-model comparison chart
  4. Ablation study charts
  5. System architecture diagram
  6. Error distribution histograms
  7. BHS error distribution pie charts

Output: thesis-quality PNGs at 300 DPI in output/figures/
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import norm

# ── Style config ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

OUT_DIR = "output/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────
with open("data/processed/experiments_all.json") as f:
    exp = json.load(f)

with open("data/processed/ablation_results.json") as f:
    abl = json.load(f)

with open("data/processed/shap_sbp.json") as f:
    shap_sbp = json.load(f)

with open("data/processed/shap_dbp.json") as f:
    shap_dbp = json.load(f)

data_npz = np.load("data/processed/pulsedb_features.npz", allow_pickle=True)
y_sbp_true = data_npz["y_sbp"]
y_dbp_true = data_npz["y_dbp"]

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 1: Bland-Altman Plots (SBP + DBP)                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_bland_altman():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    colors = {'SBP': '#E74C3C', 'DBP': '#3498DB'}

    for ax, bp_type in zip(axes, ['SBP', 'DBP']):
        ba = exp['bland_altman'][bp_type]
        errors = np.array(ba['errors'])
        mean_err = ba['mean']
        std_err = ba['std']
        loa_lower = ba['loa_lower']
        loa_upper = ba['loa_upper']
        within_5 = ba['within_5']
        within_10 = ba['within_10']
        within_15 = ba['within_15']

        y_true = y_sbp_true if bp_type == 'SBP' else y_dbp_true
        means = y_true + errors / 2.0

        ax.scatter(means, errors, alpha=0.35, s=8, c=colors[bp_type],
                   edgecolors='none', label='_nolegend_')

        ax.axhline(y=mean_err, color='black', linestyle='-', linewidth=1.2)
        ax.axhline(y=loa_lower, color='gray', linestyle='--', linewidth=1.0)
        ax.axhline(y=loa_upper, color='gray', linestyle='--', linewidth=1.0)
        ax.fill_between([means.min(), means.max()], loa_lower, loa_upper,
                        alpha=0.04, color='gray')

        ax.text(0.98, 0.92,
                f'Bias: {mean_err:+.2f} mmHg\n'
                f'1.96 SD: [{loa_lower:.1f}, {loa_upper:.1f}]\n'
                f'Within 5 mmHg: {within_5:.1f}%\n'
                f'Within 10 mmHg: {within_10:.1f}%\n'
                f'Within 15 mmHg: {within_15:.1f}%',
                transform=ax.transAxes, fontsize=9, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='gray', alpha=0.85))

        ax.set_xlabel('Mean of True & Predicted (mmHg)')
        ax.set_ylabel('Predicted - True (mmHg)')
        ax.set_title(f'{bp_type} Bland-Altman Plot', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig1_bland_altman.png')
    plt.close()
    print("  [OK] fig1_bland_altman.png")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 2: SHAP Feature Importance (SBP + DBP Top-15)                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_shap_importance():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#E74C3C', '#3498DB']

    for ax, shap_data, bp_type, color in zip(axes,
                                              [shap_sbp, shap_dbp],
                                              ['SBP', 'DBP'], colors):
        top15 = shap_data[:15]
        names = [n.replace('_', ' ') for n, _ in reversed(top15)]
        values = [v for _, v in reversed(top15)]

        bars = ax.barh(names, values, color=color, alpha=0.85, edgecolor='white',
                       linewidth=0.5)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title(f'{bp_type} SHAP Feature Importance (Top-15)',
                     fontweight='bold')

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', fontsize=8.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=2)
    plt.savefig(f'{OUT_DIR}/fig2_shap_importance.png')
    plt.close()
    print("  [OK] fig2_shap_importance.png")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 3: Multi-Model Comparison (MAE + R-squared + BHS table)        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_model_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    models = exp['multi_model']
    model_names = [m['model'] for m in models]
    sbp_mae = [m['sbp_mae'] for m in models]
    dbp_mae = [m['dbp_mae'] for m in models]
    sbp_r2 = [m['sbp_r2'] for m in models]
    dbp_r2 = [m['dbp_r2'] for m in models]

    x = np.arange(len(model_names))
    width = 0.35

    # --- MAE ---
    ax = axes[0]
    b1 = ax.bar(x - width/2, sbp_mae, width, label='SBP', color='#E74C3C',
                alpha=0.85, edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x + width/2, dbp_mae, width, label='DBP', color='#3498DB',
                alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('MAE (mmHg)')
    ax.set_title('MAE Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
    ax.legend(loc='upper left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for bar, val in zip(b1, sbp_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{val:.2f}', ha='center', fontsize=7)
    for bar, val in zip(b2, dbp_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{val:.2f}', ha='center', fontsize=7)

    # --- R-squared ---
    ax = axes[1]
    b1 = ax.bar(x - width/2, sbp_r2, width, label='SBP', color='#E74C3C',
                alpha=0.85, edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x + width/2, dbp_r2, width, label='DBP', color='#3498DB',
                alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('R-squared')
    ax.set_title('R-squared Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
    ax.legend(loc='upper left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for bar, val in zip(b1, sbp_r2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=7)
    for bar, val in zip(b2, dbp_r2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=7)

    # --- BHS Grade Table ---
    ax = axes[2]
    ax.axis('off')
    ax.set_title('BHS Grade', fontweight='bold', y=1.02)
    bhs_data = [[m['model'], m['sbp_bhs'], m['dbp_bhs']] for m in models]
    table = ax.table(cellText=bhs_data, colLabels=['Model', 'SBP', 'DBP'],
                     cellLoc='center', loc='center',
                     colColours=['#34495E']*3)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    grade_colors = {'A': '#27AE60', 'B': '#2ECC71', 'C': '#F39C12', 'D': '#E74C3C'}
    for i, m in enumerate(models):
        for j, bp in enumerate(['sbp_bhs', 'dbp_bhs']):
            cell = table[i+1, j+1]
            cell.set_facecolor(grade_colors.get(m[bp], 'white'))
            cell.set_text_props(color='white', fontweight='bold')
    for j in range(3):
        table[0, j].set_text_props(color='white', fontweight='bold')

    plt.tight_layout(w_pad=3)
    plt.savefig(f'{OUT_DIR}/fig3_model_comparison.png')
    plt.close()
    print("  [OK] fig3_model_comparison.png")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 4: Ablation Studies (Feature Groups + Data Volume)             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Feature group ablation ---
    ax = axes[0]
    feat_abl = abl['ablation_features']
    groups_en = ['Statistical\n(14)', 'Time-domain\n(16)',
                 'Derivative\nVPG+APG (18)', 'Frequency\n(7)', 'All 55']
    sbp_r2_vals = [f['sbp_r2'] for f in feat_abl]
    dbp_r2_vals = [f['dbp_r2'] for f in feat_abl]

    x = np.arange(len(groups_en))
    width = 0.35
    b1 = ax.bar(x - width/2, [v*100 for v in sbp_r2_vals], width,
                label='SBP', color='#E74C3C', alpha=0.85,
                edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x + width/2, [v*100 for v in dbp_r2_vals], width,
                label='DBP', color='#3498DB', alpha=0.85,
                edgecolor='white', linewidth=0.5)
    ax.set_ylabel('R-squared x 100')
    ax.set_title('Feature Group Ablation', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(groups_en, fontsize=9)
    ax.legend(loc='upper left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 80)

    # Highlight "All 55" bar
    for bar in b1[-1:]:
        bar.set_edgecolor('#E74C3C')
        bar.set_linewidth(2)
    for bar in b2[-1:]:
        bar.set_edgecolor('#3498DB')
        bar.set_linewidth(2)

    # --- Data volume ablation ---
    ax = axes[1]
    vol_abl = abl['ablation_volume']
    ratios = [v['ratio'] for v in vol_abl]
    sbp_r2_vol = [v['sbp_r2'] * 100 for v in vol_abl]
    dbp_r2_vol = [v['dbp_r2'] * 100 for v in vol_abl]
    sbp_mae_vol = [v['sbp_mae'] for v in vol_abl]
    dbp_mae_vol = [v['dbp_mae'] for v in vol_abl]

    ax2 = ax.twinx()
    l1, = ax.plot(ratios, sbp_mae_vol, 'o-', color='#E74C3C', lw=2,
                  markersize=8, label='SBP MAE')
    l2, = ax.plot(ratios, dbp_mae_vol, 's--', color='#3498DB', lw=2,
                  markersize=8, label='DBP MAE')
    b1 = ax2.bar([r-0.04 for r in ratios], sbp_r2_vol, 0.08,
                 color='#E74C3C', alpha=0.2, label='SBP R2')
    b2 = ax2.bar([r+0.04 for r in ratios], dbp_r2_vol, 0.08,
                 color='#3498DB', alpha=0.2, label='DBP R2')

    ax.set_xlabel('Fraction of Training Data')
    ax.set_ylabel('MAE (mmHg)')
    ax2.set_ylabel('R-squared x 100')
    ax.set_title('Data Volume Ablation', fontweight='bold')
    ax.set_xticks(ratios)
    ax.set_xticklabels([f'{r:.0%}' for r in ratios])
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8.5)
    ax.spines['top'].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig4_ablation.png')
    plt.close()
    print("  [OK] fig4_ablation.png")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 5: Error Distribution Histograms                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_error_histograms():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, bp_type in zip(axes, ['SBP', 'DBP']):
        ba = exp['bland_altman'][bp_type]
        errors = np.array(ba['errors'])
        mean_err = ba['mean']; std_err = ba['std']

        ax.hist(errors, bins=50, density=True,
                color='#3498DB' if bp_type == 'DBP' else '#E74C3C',
                alpha=0.75, edgecolor='white', linewidth=0.3)

        x_fit = np.linspace(errors.min(), errors.max(), 200)
        ax.plot(x_fit, norm.pdf(x_fit, mean_err, std_err), 'k-', lw=1.5,
                label=f'N({mean_err:.2f},{std_err:.2f}²)')

        ax.axvline(x=-5, color='green', ls='--', lw=1, alpha=0.7,
                   label='AAMI +/-5 mmHg')
        ax.axvline(x=5, color='green', ls='--', lw=1, alpha=0.7)

        within_5 = np.mean(np.abs(errors) <= 5) * 100
        ax.text(0.98, 0.95, f'Within +/-5 mmHg: {within_5:.1f}%',
                transform=ax.transAxes, fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='gray', alpha=0.8))
        ax.set_xlabel('Prediction Error (mmHg)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{bp_type} Error Distribution', fontweight='bold')
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig5_error_distribution.png')
    plt.close()
    print("  [OK] fig5_error_distribution.png")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 6: System Architecture Diagram                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    ax.axis('off')
    ax.text(8, 8.8, 'System Architecture: PPG-based Cuffless BP Estimation & LLM Health Report',
            ha='center', fontsize=15, fontweight='bold')

    c_in = '#E8F8F5'; c_proc = '#EBF5FB'; c_feat = '#FEF9E7'
    c_ml = '#FDEDEC'; c_clin = '#F4ECF7'; c_nl = '#E8F6F3'; c_out = '#FDEBD0'

    def box(x, y, w, h, text, color, fs=9.5, bold=False):
        fb = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.08",
                            fc=color, ec='#555', lw=1.2, alpha=0.92)
        ax.add_patch(fb)
        wt = 'bold' if bold else 'normal'
        for i, line in enumerate(text.split('\n')):
            ax.text(x, y + (len(text.split('\n'))/2-i-0.5)*(fs/10),
                    line, ha='center', va='center', fontsize=fs, fontweight=wt)

    def arr(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.6))

    # Row 1: Input
    y = 7.5
    box(3, y, 4.5, 1.1, 'PulseDB Dataset\n(.mat files, 70 subjects, 3,591 segments)', c_in, bold=True)

    # Row 2: Signal Processing
    y2 = 6.2
    box(3, y2, 5, 1.1, 'Signal Preprocessing Pipeline\nQuality Gate -> Baseline Correction -> Notch Filter\n-> Motion Artifact Detection -> Spike Removal', c_proc)

    arr(3, y-0.55, 3, y2+0.55)

    # Row 3: Peak Detection
    y3 = 5.0
    box(3, y3, 5, 1.0, 'Peak & Fiducial Point Detection\nSystolic Peak / Onset / Dicrotic Notch\nVPG & APG a-b-c-d-e points', c_proc)

    arr(3, y2-0.55, 3, y3+0.5)

    # Row 4: Features
    y4 = 3.7
    box(3, y4, 5, 1.1, '55-Dim Feature Extraction\nTime-domain (16) + Derivative VPG/APG (18)\n+ Statistical (14) + Frequency (7)', c_feat, bold=True)

    arr(3, y3-0.5, 3, y4+0.55)

    # Row 5: ML + Clinical (split)
    y5 = 2.4
    box(1.2, y5, 3.8, 1.0, 'ML BP Estimation\nRF / XGBoost / SVR / KNN / MLP', c_ml, bold=True)
    box(5.8, y5, 3.8, 1.0, 'Clinical Knowledge Layer\nAHA/ESC/NICE Guidelines\n5-level Risk Stratification', c_clin)

    arr(3, y4-0.55, 1.2, y5+0.5)
    arr(3, y4-0.55, 5.8, y5+0.5)

    # Row 6: Outputs
    y6 = 1.1
    box(1.2, y6, 2.6, 0.9, 'SBP/DBP\nNumeric Output', c_out, bold=True)
    box(5.8, y6, 2.6, 0.9, 'Risk Level\nhealthy/.../danger', c_out)

    arr(1.2, y5-0.5, 1.2, y6+0.45)
    arr(5.8, y5-0.5, 5.8, y6+0.45)

    # Row 7: LLM NL Generation
    y7 = 0.1
    box(9.5, y7+0.5, 6, 1.0, 'LLM Health Report Generation\n(DeepSeek / Grok API + Template Fallback)\n-> Natural Language Health Report', c_nl, bold=True)

    arr(1.2+1.3, y6-0.45, 9.5-3, y7+0.95)
    arr(5.8+1.3, y6-0.45, 9.5-3, y7+0.95)

    plt.tight_layout(pad=0.5)
    plt.savefig(f'{OUT_DIR}/fig6_architecture.png')
    plt.close()
    print("  [OK] fig6_architecture.png")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 7: BHS Error Distribution (Pie Charts)                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_bhs_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, bp_type in zip(axes, ['SBP', 'DBP']):
        ba = exp['bland_altman'][bp_type]
        errors = np.abs(np.array(ba['errors']))
        bins = [0, 5, 10, 15, np.inf]
        labels = ['<=5 mmHg', '5-10 mmHg', '10-15 mmHg', '>15 mmHg']
        counts = [np.sum((errors >= bins[i]) & (errors < bins[i+1]))
                  for i in range(len(bins)-1)]
        pcts = [c / len(errors) * 100 for c in counts]
        colors_pie = ['#27AE60', '#2ECC71', '#F39C12', '#E74C3C']

        wedges, texts, autotexts = ax.pie(pcts, labels=None, autopct='%1.1f%%',
                                          colors=colors_pie, startangle=90,
                                          textprops={'fontsize': 10})

        legend_labels = [f'{l} ({c} seg, {p:.1f}%)'
                         for l, c, p in zip(labels, counts, pcts)]
        ax.legend(wedges, legend_labels, title='Absolute Error',
                  loc='lower center', bbox_to_anchor=(0.5, -0.12),
                  ncol=2, fontsize=9)

        ax.set_title(f'{bp_type} Error Distribution (BHS)', fontweight='bold')

        p5 = pcts[0]; p10 = pcts[0]+pcts[1]; p15 = pcts[0]+pcts[1]+pcts[2]
        grade = ('A' if p5>=60 and p10>=85 and p15>=95 else
                 'B' if p5>=50 and p10>=75 and p15>=90 else
                 'C' if p5>=40 and p10>=65 and p15>=85 else 'D')
        gcolor = {'A':'#27AE60','B':'#2ECC71','C':'#F39C12','D':'#E74C3C'}[grade]
        ax.text(0, 1.28, f'BHS Grade: {grade}', ha='center', fontsize=14,
                fontweight='bold', transform=ax.transAxes, color=gcolor)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig7_bhs_distribution.png')
    plt.close()
    print("  [OK] fig7_bhs_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating thesis figures...\n")

    plot_bland_altman()
    plot_shap_importance()
    plot_model_comparison()
    plot_ablation()
    plot_error_histograms()
    plot_architecture()
    plot_bhs_distribution()

    print(f"\nDone! All 7 figures saved to {OUT_DIR}/")
