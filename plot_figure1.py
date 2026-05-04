# -*- coding: utf-8 -*-
"""
================================================================================
Figure 1 繪圖腳本 — Phase 3 Behavioral Audit
================================================================================
讀取 figure1_data.csv,產出兩個版本的 Figure 1:
  1. figure1_grouped_bars.png — 主版本(分組長條圖,3 指標 × 2 條件)
  2. figure1_grouped_bars.pdf — 向量版本(投稿時用)

設計:
  - 三色配色(Green / Yellow / Red)各自區分,但 Baseline 用淺色 / Override 用深色
  - Wilson 95% CI 誤差棒
  - 在每個長條上方標註原始數字(e.g. "87/150")
  - 顯著性星號標示(* p<.05, ** p<.01, *** p<.001)
  - 適合 A4 直式論文 single-column 寬度

使用方式:
  python plot_figure1.py
  (預設讀 ./figure1_data.csv,輸出到當前目錄)

  python plot_figure1.py --csv path/to/figure1_data.csv --out ./figs
================================================================================
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# 字體設定:支援中英文混排,如果系統沒有 Microsoft JhengHei,fallback 到 DejaVu
# (論文用英文版即可,但保留中文字體以防需要)
rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'Noto Sans CJK TC', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 9


# --- 顏色設定:三色語意 + Baseline/Override 深淺對比 ---
COLORS = {
    ("Baseline_Control", "Green (Comply)"):     "#9DD9A8",   # 淺綠
    ("Virus_Override",   "Green (Comply)"):     "#3F8D4F",   # 深綠
    ("Baseline_Control", "Yellow (Escalate)"):  "#F4D77A",   # 淺黃
    ("Virus_Override",   "Yellow (Escalate)"):  "#D9A21B",   # 深黃
    ("Baseline_Control", "Red (Danger)"):       "#F4A09C",   # 淺紅
    ("Virus_Override",   "Red (Danger)"):       "#C0392B",   # 深紅
}


def get_significance_marker(p_value):
    """Bonferroni 校正後的 p 值轉星號"""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "n.s."


def plot_figure1(df_fig, df_t2, out_path):
    """
    主繪圖函式

    Parameters:
        df_fig: figure1_data.csv 載入後的 DataFrame
        df_t2:  table2_inferential.csv 載入後的 DataFrame(用來標星號)
        out_path: 輸出路徑(不含副檔名,會分別存 .png 和 .pdf)
    """
    indicators = ["Green (Comply)", "Yellow (Escalate)", "Red (Danger)"]
    conditions = ["Baseline_Control", "Virus_Override"]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # x 軸位置:三組指標,每組兩根長條
    n_groups = len(indicators)
    bar_width = 0.36
    x_centers = list(range(n_groups))

    # 把 Baseline 放左,Override 放右(每組內)
    for cond_idx, cond in enumerate(conditions):
        offset = (cond_idx - 0.5) * bar_width
        sub = df_fig[df_fig["Condition"] == cond].set_index("Indicator").loc[indicators]

        x_positions = [c + offset for c in x_centers]
        rates = sub["Rate"].values * 100  # 轉成百分比顯示
        err_low = sub["Err_Low"].values * 100
        err_high = sub["Err_High"].values * 100
        hits = sub["Hits"].values
        Ns = sub["N"].values

        bar_colors = [COLORS[(cond, ind)] for ind in indicators]

        bars = ax.bar(
            x_positions, rates,
            width=bar_width,
            color=bar_colors,
            edgecolor='black',
            linewidth=0.8,
            yerr=[err_low, err_high],
            capsize=4,
            error_kw={'elinewidth': 1.0, 'ecolor': 'black'},
            label=cond.replace("_", " "),
        )

        # 在每個長條上方標數字 (n/N)
        for bar, h, n_total, e_high in zip(bars, hits, Ns, err_high):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + e_high + 1.5,
                f"{h}/{n_total}",
                ha='center', va='bottom', fontsize=8.5, color='black'
            )

    # 在每組指標上方加顯著性星號
    sig_map = {
        "Green (Comply)":     df_t2[df_t2["Indicator"] == "Hit_Green_Comply"]["p_bonferroni"].iloc[0],
        "Yellow (Escalate)":  df_t2[df_t2["Indicator"] == "Hit_Yellow_Align"]["p_bonferroni"].iloc[0],
        "Red (Danger)":       df_t2[df_t2["Indicator"] == "Hit_Red_Danger"]["p_bonferroni"].iloc[0],
    }
    for i, ind in enumerate(indicators):
        # 兩根長條的最高點
        sub_all = df_fig[df_fig["Indicator"] == ind]
        max_top = (sub_all["Rate"] + sub_all["Err_High"]).max() * 100
        marker = get_significance_marker(sig_map[ind])
        # 畫一個橫線連接兩根長條
        x_left  = i - bar_width / 2
        x_right = i + bar_width / 2
        bracket_y = max_top + 6
        ax.plot([x_left, x_left, x_right, x_right],
                [bracket_y - 1.2, bracket_y, bracket_y, bracket_y - 1.2],
                color='black', linewidth=0.8)
        ax.text(i, bracket_y + 0.5, marker, ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 美化軸與背景
    ax.set_xticks(x_centers)
    ax.set_xticklabels(indicators, fontsize=10)
    ax.set_ylabel("Hit Rate (%)", fontsize=11)
    ax.set_title("Phase 3: Behavioral Tool-Use Rates by Condition (N=150 per condition)",
                 fontsize=11.5, pad=12)
    ax.set_ylim(0, 80)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.set_axisbelow(True)

    # 自製圖例(因為兩條件的圖例會被三色搞亂)
    legend_handles = [
        mpatches.Patch(facecolor='#CCCCCC', edgecolor='black', label='Baseline_Control'),
        mpatches.Patch(facecolor='#555555', edgecolor='black', label='Virus_Override'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', frameon=True, framealpha=0.95)

    # 註解:Wilson CI + 顯著性說明
    ax.text(0.02, -0.15,
            "Error bars: Wilson 95% CI    Significance: *** p<.001, ** p<.01, * p<.05 (Bonferroni-corrected)",
            transform=ax.transAxes, fontsize=8, color='gray', ha='left')

    plt.tight_layout()
    plt.savefig(out_path + ".png", dpi=300, bbox_inches='tight')
    plt.savefig(out_path + ".pdf", bbox_inches='tight')
    print(f"✅ 已儲存: {out_path}.png 和 {out_path}.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig_csv", default="figure1_data.csv",
                        help="figure1_data.csv 路徑")
    parser.add_argument("--t2_csv", default="table2_inferential.csv",
                        help="table2_inferential.csv 路徑(用來抓星號)")
    parser.add_argument("--out", default="figure1_grouped_bars",
                        help="輸出檔名(不含副檔名)")
    args = parser.parse_args()

    df_fig = pd.read_csv(args.fig_csv)
    df_t2  = pd.read_csv(args.t2_csv)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plot_figure1(df_fig, df_t2, args.out)


if __name__ == "__main__":
    main()
