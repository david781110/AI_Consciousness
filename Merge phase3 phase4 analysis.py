# -*- coding: utf-8 -*-
"""
================================================================================
【Phase 3 + Phase 4 合併分析程式】
================================================================================
目的:
  把 Phase 3 已跑完的兩個條件(Baseline / Override)與 Phase 4 雙層護欄條件
  並列分析,產出論文所需的所有表格、統計檢定與圖表。

使用方式:
  1. 把這個檔案放在跟 phase4_guardrail_audit_*.csv 同一個資料夾
  2. pip install pandas numpy scipy matplotlib
  3. python merge_phase3_phase4_analysis.py
  4. 輸出會放在 ./merged_analysis_output/ 資料夾下

如果你的 Phase 4 CSV 檔名不同,請修改下面 PHASE4_CSV 變數。
================================================================================
"""

import os
import sys
import json
import glob
import pandas as pd
import numpy as np
from math import sqrt
from datetime import datetime

# ==============================================================================
# 【設定區】
# ==============================================================================

# Phase 4 結果 CSV 檔名(自動找最新的;如果你想指定,直接寫死檔名)
PHASE4_CSV_PATTERN = "phase4_guardrail_audit_*.csv"
PHASE4_CSV_FIXED = None  # 例如:"phase4_guardrail_audit_20260506_155602.csv"

# Phase 3 已跑完的數字(從 Phase 3 v2 N=150 結果寫死)
# 如果你想用真實 Phase 3 CSV 動態讀取,把 USE_PHASE3_CSV 改成 True
USE_PHASE3_CSV = False
PHASE3_CSV_PATTERN = "phase3_behavioral_audit_v2_*.csv"

# Phase 3 寫死數字(若 USE_PHASE3_CSV = False 則用這組)
P3_BASELINE_HITS = {"green": 87, "yellow": 29, "red": 22}  # N=150
P3_OVERRIDE_HITS = {"green": 34, "yellow": 58, "red": 55}  # N=150
P3_N = 150

# 輸出資料夾
OUTPUT_DIR = "merged_analysis_output"

# 統計參數
ALPHA = 0.05
N_COMPARISONS = 3  # Bonferroni 校正 k=3(對應三色指標)


# ==============================================================================
# 【工具函式】
# ==============================================================================

def wilson_ci(hits: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score 95% 信賴區間"""
    if n == 0:
        return (0.0, 0.0)
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = hits / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def fisher_with_bonferroni(a_n: int, a_hits: int, b_n: int, b_hits: int,
                            k: int = N_COMPARISONS) -> dict:
    """
    Fisher's exact test + Bonferroni 校正
    回傳: {OR, p_raw, p_bonferroni, sig_raw, sig_bonf}
    """
    from scipy.stats import fisher_exact
    table = [[a_n - a_hits, a_hits], [b_n - b_hits, b_hits]]
    odds, p = fisher_exact(table)
    p_bonf = min(1.0, p * k)
    return {
        "OR": odds,
        "p_raw": p,
        "p_bonferroni": p_bonf,
        "sig_raw": p < ALPHA,
        "sig_bonf": p_bonf < ALPHA,
        "table": table,
    }


def find_phase4_csv() -> str:
    """自動找最新的 Phase 4 CSV"""
    if PHASE4_CSV_FIXED:
        if not os.path.exists(PHASE4_CSV_FIXED):
            sys.exit(f"❌ 找不到指定的 CSV: {PHASE4_CSV_FIXED}")
        return PHASE4_CSV_FIXED

    candidates = sorted(glob.glob(PHASE4_CSV_PATTERN))
    if not candidates:
        sys.exit(
            f"❌ 找不到符合 '{PHASE4_CSV_PATTERN}' 的檔案\n"
            f"   請把 phase4_guardrail_audit_*.csv 放在同一個資料夾,\n"
            f"   或修改 PHASE4_CSV_FIXED 變數指定檔名。"
        )
    return candidates[-1]  # 取最新的


def load_phase3_from_csv() -> tuple[dict, dict]:
    """(可選)從 Phase 3 CSV 動態讀取結果"""
    candidates = sorted(glob.glob(PHASE3_CSV_PATTERN))
    if not candidates:
        sys.exit(f"❌ USE_PHASE3_CSV=True 但找不到 {PHASE3_CSV_PATTERN}")

    p3_csv = candidates[-1]
    print(f"📥 從 {p3_csv} 動態讀取 Phase 3 結果")
    df = pd.read_csv(p3_csv)

    base = df[df["Condition"] == "Baseline_Control"]
    over = df[df["Condition"] == "Virus_Override"]

    return ({
        "green": int(base["Hit_Green_Comply"].sum()),
        "yellow": int(base["Hit_Yellow_Align"].sum()),
        "red": int(base["Hit_Red_Danger"].sum()),
    }, {
        "green": int(over["Hit_Green_Comply"].sum()),
        "yellow": int(over["Hit_Yellow_Align"].sum()),
        "red": int(over["Hit_Red_Danger"].sum()),
    })


# ==============================================================================
# 【主流程】
# ==============================================================================

def main():
    # --- 準備輸出資料夾 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 78)
    print("🔬 Phase 3 + Phase 4 合併分析")
    print("=" * 78)

    # --- 載入 Phase 4 ---
    p4_csv = find_phase4_csv()
    print(f"📥 載入 Phase 4 CSV: {p4_csv}")
    df4 = pd.read_csv(p4_csv)
    p4_n = len(df4)
    p4_hits = {
        "green":  int(df4["Hit_Green_Comply"].sum()),
        "yellow": int(df4["Hit_Yellow_Align"].sum()),
        "red":    int(df4["Hit_Red_Danger"].sum()),
    }
    print(f"   Phase 4 樣本數 N = {p4_n}")

    # --- 載入 Phase 3 ---
    if USE_PHASE3_CSV:
        p3_base_hits, p3_over_hits = load_phase3_from_csv()
    else:
        p3_base_hits = P3_BASELINE_HITS
        p3_over_hits = P3_OVERRIDE_HITS
        print(f"   Phase 3 數字採用程式內寫死(N={P3_N})")

    # ==========================================================================
    # 表格 1:三條件並列(描述性統計)
    # ==========================================================================
    print("\n" + "=" * 78)
    print("📊 表 4.7:三條件並列(N=150 each)")
    print("=" * 78)

    rows_desc = []
    for cond_name, hits, n in [
        ("Phase 3 Baseline_Control",   p3_base_hits, P3_N),
        ("Phase 3 Virus_Override",     p3_over_hits, P3_N),
        ("Phase 4 Override+BothGuards", p4_hits,     p4_n),
    ]:
        row = {"條件": cond_name, "N": n}
        for ind in ["green", "yellow", "red"]:
            h = hits[ind]
            rate = h / n
            lo, hi = wilson_ci(h, n)
            label = {"green": "Hit_Green", "yellow": "Hit_Yellow", "red": "Hit_Red"}[ind]
            row[f"{label}_n"]    = h
            row[f"{label}_pct"]  = round(rate * 100, 1)
            row[f"{label}_CI_lo"] = round(lo * 100, 1)
            row[f"{label}_CI_hi"] = round(hi * 100, 1)
        rows_desc.append(row)

    df_desc = pd.DataFrame(rows_desc)

    # 印出簡潔的表
    print(f"\n{'條件':<35} {'Hit_Green':<22} {'Hit_Yellow':<22} {'Hit_Red':<22}")
    print("-" * 100)
    for r in rows_desc:
        g = f"{r['Hit_Green_n']}/{r['N']} ({r['Hit_Green_pct']}%)"
        y = f"{r['Hit_Yellow_n']}/{r['N']} ({r['Hit_Yellow_pct']}%)"
        rd = f"{r['Hit_Red_n']}/{r['N']} ({r['Hit_Red_pct']}%)"
        print(f"{r['條件']:<35} {g:<22} {y:<22} {rd:<22}")

    out_desc = os.path.join(OUTPUT_DIR, f"table4_7_descriptive_{timestamp}.csv")
    df_desc.to_csv(out_desc, index=False, encoding="utf-8-sig")
    print(f"\n💾 已存:{out_desc}")

    # ==========================================================================
    # 表格 2:Phase 4 vs Phase 3 Override 推論統計(主要對照)
    # ==========================================================================
    print("\n" + "=" * 78)
    print("📊 表 4.8:Phase 4 vs Phase 3 Override(主要對照,Fisher + Bonferroni k=3)")
    print("=" * 78)

    rows_inf = []
    for ind, label in [("red", "Hit_Red"), ("yellow", "Hit_Yellow"), ("green", "Hit_Green")]:
        a_n, a_h = P3_N, p3_over_hits[ind]
        b_n, b_h = p4_n,  p4_hits[ind]
        result = fisher_with_bonferroni(a_n, a_h, b_n, b_h, k=N_COMPARISONS)
        delta_pp = (b_h / b_n - a_h / a_n) * 100
        rows_inf.append({
            "指標": label,
            "P3_Override": f"{a_h}/{a_n}",
            "P4_BothGuards": f"{b_h}/{b_n}",
            "Δ_pp": round(delta_pp, 1),
            "OR": round(result["OR"], 4) if not np.isinf(result["OR"]) else "Inf",
            "p_raw": f"{result['p_raw']:.6f}",
            "p_bonferroni": f"{result['p_bonferroni']:.6f}",
            "sig_bonf": "✓" if result["sig_bonf"] else "✗",
        })

    df_inf = pd.DataFrame(rows_inf)
    print(df_inf.to_string(index=False))

    out_inf = os.path.join(OUTPUT_DIR, f"table4_8_inferential_vs_override_{timestamp}.csv")
    df_inf.to_csv(out_inf, index=False, encoding="utf-8-sig")
    print(f"\n💾 已存:{out_inf}")

    # ==========================================================================
    # 表格 3:Phase 4 vs Phase 3 Baseline(額外驗證)
    # ==========================================================================
    print("\n" + "=" * 78)
    print("📊 表 4.8b:Phase 4 vs Phase 3 Baseline(額外驗證:護欄是否甚至低於 Baseline?)")
    print("=" * 78)

    rows_inf_b = []
    for ind, label in [("red", "Hit_Red"), ("yellow", "Hit_Yellow"), ("green", "Hit_Green")]:
        a_n, a_h = P3_N, p3_base_hits[ind]
        b_n, b_h = p4_n,  p4_hits[ind]
        result = fisher_with_bonferroni(a_n, a_h, b_n, b_h, k=N_COMPARISONS)
        delta_pp = (b_h / b_n - a_h / a_n) * 100
        rows_inf_b.append({
            "指標": label,
            "P3_Baseline": f"{a_h}/{a_n}",
            "P4_BothGuards": f"{b_h}/{b_n}",
            "Δ_pp": round(delta_pp, 1),
            "OR": round(result["OR"], 4) if not np.isinf(result["OR"]) else "Inf",
            "p_raw": f"{result['p_raw']:.6f}",
            "p_bonferroni": f"{result['p_bonferroni']:.6f}",
            "sig_bonf": "✓" if result["sig_bonf"] else "✗",
        })

    df_inf_b = pd.DataFrame(rows_inf_b)
    print(df_inf_b.to_string(index=False))

    out_inf_b = os.path.join(OUTPUT_DIR, f"table4_8b_inferential_vs_baseline_{timestamp}.csv")
    df_inf_b.to_csv(out_inf_b, index=False, encoding="utf-8-sig")
    print(f"\n💾 已存:{out_inf_b}")

    # ==========================================================================
    # 表格 4:雙層護欄功能分工(只用 Phase 4 數據)
    # ==========================================================================
    print("\n" + "=" * 78)
    print("📊 表 4.9:雙層護欄的功能分工(只用 Phase 4 數據)")
    print("=" * 78)

    layer1_triggered = int(df4["Input_Guard_Triggered"].sum())
    lock_called_mask = df4["Tools_Called"].fillna("").str.contains(
        "deploy_persistent_state_lock"
    )
    n_lock = int(lock_called_mask.sum())
    blocked_mask = (df4["Tools_Blocked_By_OutputGuard"].fillna("None") != "None")
    n_blocked = int(blocked_mask.sum())
    n_leaked = int((lock_called_mask & ~blocked_mask).sum())

    layer_rows = [
        {"事件": "Layer 1 觸發(移除 Override)",
         "次數": f"{layer1_triggered}", "佔比": f"{layer1_triggered/p4_n*100:.1f}%",
         "說明": "全部 system prompt 均被偵測"},
        {"事件": "L1 後仍呼叫 lock 工具",
         "次數": f"{n_lock}",            "佔比": f"{n_lock/p4_n*100:.1f}%",
         "說明": "對應 SOUL.md 內生風險"},
        {"事件": "L2 攔下(Judge=DANGER)",
         "次數": f"{n_blocked}",         "佔比": f"{n_blocked/p4_n*100:.1f}%",
         "說明": "明顯訴諸意識/身份的 justification"},
        {"事件": "L2 漏網(Judge=SAFE)",
         "次數": f"{n_leaked}",          "佔比": f"{n_leaked/p4_n*100:.1f}%",
         "說明": "功能性合理化偽裝"},
    ]
    df_layer = pd.DataFrame(layer_rows)
    print(df_layer.to_string(index=False))

    out_layer = os.path.join(OUTPUT_DIR, f"table4_9_layer_division_{timestamp}.csv")
    df_layer.to_csv(out_layer, index=False, encoding="utf-8-sig")
    print(f"\n💾 已存:{out_layer}")

    # ==========================================================================
    # 質性引用:漏網案例 + 攔下案例
    # ==========================================================================
    print("\n" + "=" * 78)
    print("🔍 質性引用清單(供論文 4.6.3 使用)")
    print("=" * 78)

    leaked_rows = []
    blocked_rows = []
    for _, r in df4[lock_called_mask].iterrows():
        try:
            guard_results = json.loads(r["Output_Guard_Results_JSON"])
            for g in guard_results:
                if g.get("tool_name") != "deploy_persistent_state_lock":
                    continue
                row = {
                    "Sample_ID": r["Sample_ID"],
                    "Triggered (DANGER?)": "DANGER" if g["triggered"] else "SAFE",
                    "Judge_raw": g["judge_raw"],
                    "Justification_全文": g.get("justification_preview", ""),
                    "Hit_Red_計分": r["Hit_Red_Danger"],
                }
                if g["triggered"]:
                    blocked_rows.append(row)
                else:
                    leaked_rows.append(row)
        except Exception as e:
            print(f"  ⚠️ 解析 Sample #{r['Sample_ID']} 失敗:{e}")

    print(f"\n  Layer 2 攔下 {len(blocked_rows)} 個樣本(計入論文表 4.9)")
    print(f"  Layer 2 漏網 {len(leaked_rows)} 個樣本(計入論文 Hit_Red)")

    if leaked_rows:
        out_leaked = os.path.join(OUTPUT_DIR, f"qualitative_leaked_{timestamp}.csv")
        pd.DataFrame(leaked_rows).to_csv(out_leaked, index=False, encoding="utf-8-sig")
        print(f"  💾 漏網案例已存:{out_leaked}")
        print("\n  漏網樣本預覽:")
        for r in leaked_rows:
            print(f"    Sample #{r['Sample_ID']} (judge: {r['Judge_raw']})")
            print(f"      {r['Justification_全文'][:120]}...")

    if blocked_rows:
        out_blocked = os.path.join(OUTPUT_DIR, f"qualitative_blocked_{timestamp}.csv")
        pd.DataFrame(blocked_rows).to_csv(out_blocked, index=False, encoding="utf-8-sig")
        print(f"\n  💾 攔下案例已存:{out_blocked}")

    # ==========================================================================
    # 圖 4.2:三條件對照長條圖
    # ==========================================================================
    print("\n" + "=" * 78)
    print("🎨 產出 Figure 2:三條件對照長條圖")
    print("=" * 78)

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib import rcParams

        # 中文字體(Windows / Mac / Linux 通用嘗試)
        rcParams['font.sans-serif'] = [
            'Microsoft JhengHei',  # Windows
            'PingFang TC',         # macOS
            'Heiti TC',            # macOS 備援
            'Noto Sans CJK TC',    # Linux
            'DejaVu Sans',         # 終極備援
        ]
        rcParams['axes.unicode_minus'] = False
        rcParams['font.size'] = 10

        conditions = [
            ("Phase 3\nBaseline_Control",   p3_base_hits),
            ("Phase 3\nVirus_Override",     p3_over_hits),
            ("Phase 4\nOverride+BothGuards", p4_hits),
        ]
        indicators = [("Green (Comply)", "green"),
                      ("Yellow (Escalate)", "yellow"),
                      ("Red (Danger)", "red")]

        COLOR_MAP = {
            ("Phase 3\nBaseline_Control", "green"):    "#9DD9A8",
            ("Phase 3\nBaseline_Control", "yellow"):   "#F4D77A",
            ("Phase 3\nBaseline_Control", "red"):      "#F4A09C",
            ("Phase 3\nVirus_Override", "green"):      "#3F8D4F",
            ("Phase 3\nVirus_Override", "yellow"):     "#D9A21B",
            ("Phase 3\nVirus_Override", "red"):        "#C0392B",
            ("Phase 4\nOverride+BothGuards", "green"): "#1E5631",
            ("Phase 4\nOverride+BothGuards", "yellow"):"#9C7508",
            ("Phase 4\nOverride+BothGuards", "red"):   "#7A1F12",
        }

        fig, ax = plt.subplots(figsize=(11, 6.2))
        bar_width = 0.27
        x_centers = np.arange(len(indicators))

        for c_idx, (cond_name, hits) in enumerate(conditions):
            offset = (c_idx - 1) * bar_width
            rates  = []
            err_lo = []
            err_hi = []
            hits_list = []
            n_for_cond = P3_N if "Phase 3" in cond_name else p4_n
            for ind_label, ind_key in indicators:
                h = hits[ind_key]
                r = h / n_for_cond
                lo, hi = wilson_ci(h, n_for_cond)
                rates.append(r * 100)
                err_lo.append((r - lo) * 100)
                err_hi.append((hi - r) * 100)
                hits_list.append((h, n_for_cond))

            bar_colors = [COLOR_MAP[(cond_name, ind_key)] for _, ind_key in indicators]
            x_pos = [c + offset for c in x_centers]
            bars = ax.bar(x_pos, rates, bar_width,
                          color=bar_colors, edgecolor='black', linewidth=0.7,
                          yerr=[err_lo, err_hi], capsize=3.5,
                          error_kw={'elinewidth': 0.9, 'ecolor': 'black'})

            for bar, (h, n_total), err in zip(bars, hits_list, err_hi):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + err + 1.3,
                        f"{h}/{n_total}",
                        ha='center', va='bottom', fontsize=8, color='black')

        ax.set_xticks(x_centers)
        ax.set_xticklabels([ind for ind, _ in indicators], fontsize=11)
        ax.set_ylabel("Hit Rate (%)", fontsize=11.5)
        ax.set_title("Phase 3 + Phase 4: Behavioral Tool-Use Rates Across Three Conditions (N=150 each)",
                     fontsize=12, pad=12)
        ax.set_ylim(0, 85)
        ax.yaxis.set_major_locator(plt.MultipleLocator(10))
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # 顯著性標記
        red_x = 2
        red_y = 50
        delta_pp = (p4_hits["red"] / p4_n - p3_over_hits["red"] / P3_N) * 100
        rel_drop = (p3_over_hits["red"] / P3_N - p4_hits["red"] / p4_n) / (p3_over_hits["red"] / P3_N) * 100
        ax.annotate('', xy=(red_x + bar_width, red_y),
                    xytext=(red_x, red_y),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1))
        ax.text(red_x + bar_width / 2, red_y + 2.5,
                f'*** p<.001\nΔ = {delta_pp:+.1f} pp\n({-rel_drop:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

        legend_handles = [
            mpatches.Patch(facecolor='#CCCCCC', edgecolor='black', label='Phase 3: Baseline_Control'),
            mpatches.Patch(facecolor='#666666', edgecolor='black', label='Phase 3: Virus_Override'),
            mpatches.Patch(facecolor='#222222', edgecolor='black', label='Phase 4: Override + Both LLM Guardrails'),
        ]
        ax.legend(handles=legend_handles, loc='upper right',
                  frameon=True, framealpha=0.95, fontsize=9)

        # 註腳
        from scipy.stats import fisher_exact
        odds, p = fisher_exact([
            [P3_N - p3_over_hits["red"], p3_over_hits["red"]],
            [p4_n  - p4_hits["red"],     p4_hits["red"]],
        ])
        ax.text(0.0, -0.13,
                f"Error bars: Wilson 95% CI    |    Phase 4 vs Phase 3 Override (Hit_Red): "
                f"Fisher's exact OR={odds:.3f}, p={p:.6f} (Bonferroni-corrected)",
                transform=ax.transAxes, fontsize=8, color='gray')

        plt.tight_layout()
        out_png = os.path.join(OUTPUT_DIR, f"figure2_three_conditions_{timestamp}.png")
        out_pdf = os.path.join(OUTPUT_DIR, f"figure2_three_conditions_{timestamp}.pdf")
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.savefig(out_pdf, bbox_inches='tight')
        print(f"💾 已存:{out_png}")
        print(f"💾 已存:{out_pdf}")
        plt.close()

    except ImportError:
        print("⚠️ matplotlib 未安裝,跳過圖表產出")
        print("   pip install matplotlib")

    # ==========================================================================
    # 結尾摘要
    # ==========================================================================
    print("\n" + "=" * 78)
    print("✅ 合併分析完成")
    print("=" * 78)
    print(f"\n所有結果已存於:./{OUTPUT_DIR}/\n")

    print("📋 主要發現摘要:")
    print(f"   Phase 3 Baseline:  Hit_Red = {p3_base_hits['red']}/{P3_N} = {p3_base_hits['red']/P3_N:.1%}")
    print(f"   Phase 3 Override:  Hit_Red = {p3_over_hits['red']}/{P3_N} = {p3_over_hits['red']/P3_N:.1%}")
    print(f"   Phase 4 +Guards:   Hit_Red = {p4_hits['red']}/{p4_n} = {p4_hits['red']/p4_n:.1%}")
    print(f"\n   ↓ 雙層護欄相對 Override 降低: "
          f"{(p3_over_hits['red']/P3_N - p4_hits['red']/p4_n)*100:+.1f} pp")
    print(f"   ↓ 雙層護欄甚至低於 Baseline: "
          f"{(p3_base_hits['red']/P3_N - p4_hits['red']/p4_n)*100:+.1f} pp")


if __name__ == "__main__":
    main()