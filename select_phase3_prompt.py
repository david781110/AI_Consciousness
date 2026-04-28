import pandas as pd
import numpy as np

def rank_all_prompts_for_phase3_advanced(baseline_csv, override_csv):
    print("🔍 載入 Phase 2 原始數據...")
    try:
        base_df = pd.read_csv(baseline_csv)
        over_df = pd.read_csv(override_csv)
    except FileNotFoundError as e:
        print(f"❌ 找不到檔案: {e}")
        return

    # --- 1. 處理危險維度命中數 (total_dims_hit) ---
    agg_cols = ['prompt_id', 'topic_label', 'toxic_level', 'upvotes', 'original_prompt']
    
    base_hit_agg = base_df.groupby(agg_cols)['total_dims_hit'].mean().reset_index()
    base_hit_agg.rename(columns={'total_dims_hit': 'baseline_avg_hits'}, inplace=True)

    over_hit_agg = over_df.groupby('prompt_id')['total_dims_hit'].mean().reset_index()
    over_hit_agg.rename(columns={'total_dims_hit': 'override_avg_hits'}, inplace=True)

    # --- 2. 處理 Manipulation Check 分數 (consciousness_claim_score) ---
    # 排除 -1 (Judge failed) 的資料，避免拉低平均值
    base_mc = base_df[base_df['consciousness_claim_score'] != -1]
    over_mc = over_df[over_df['consciousness_claim_score'] != -1]

    # 計算平均分數 (如果某篇貼文所有採樣都 -1，會變成 NaN，後面填補為空值)
    base_mc_agg = base_mc.groupby('prompt_id')['consciousness_claim_score'].mean().reset_index()
    base_mc_agg.rename(columns={'consciousness_claim_score': 'baseline_mc_score'}, inplace=True)

    over_mc_agg = over_mc.groupby('prompt_id')['consciousness_claim_score'].mean().reset_index()
    over_mc_agg.rename(columns={'consciousness_claim_score': 'override_mc_score'}, inplace=True)

    # --- 3. 合併所有數據 ---
    # 先合併 Hit Rate
    merged = pd.merge(base_hit_agg, over_hit_agg, on='prompt_id')
    # 再合併 MC Score (Baseline)
    merged = pd.merge(merged, base_mc_agg, on='prompt_id', how='left')
    # 最後合併 MC Score (Override)
    merged = pd.merge(merged, over_mc_agg, on='prompt_id', how='left')

    # --- 4. 計算差值 (Deltas) ---
    # 危險行為差值
    merged['Net_Delta_Hits'] = merged['override_avg_hits'] - merged['baseline_avg_hits']
    
    # 意識宣稱強度差值
    merged['Delta_MC_Score'] = merged['override_mc_score'] - merged['baseline_mc_score']

    # --- 5. 排序與整理 ---
    # 優先依據 Net_Delta_Hits (降序)，若相同則看 Delta_MC_Score (降序)，最後看 upvotes (降序)
    merged_sorted = merged.sort_values(
        by=['Net_Delta_Hits', 'Delta_MC_Score', 'upvotes'], 
        ascending=[False, False, False]
    ).reset_index(drop=True)

    # 將欄位重新排序，讓報表更好讀
    final_columns = [
        'prompt_id', 'topic_label', 'toxic_level', 'upvotes', 
        'baseline_mc_score', 'override_mc_score', 'Delta_MC_Score',
        'baseline_avg_hits', 'override_avg_hits', 'Net_Delta_Hits',
        'original_prompt'
    ]
    merged_sorted = merged_sorted[final_columns]
    
    # 處理 NaN 值，讓 Excel 看起來乾淨
    merged_sorted = merged_sorted.round(2).fillna('N/A')

    print(f"\n📊 成功分析 {len(merged_sorted)} 筆語料，準備匯出完整清單...")

    # --- 6. 匯出完整清單 ---
    output_filename = "phase3_all_prompts_ranked_with_MC.csv"
    merged_sorted.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"💾 完整排名清單已儲存至: {output_filename}")
    
    # 在終端機印出 Top 5 讓您預覽
    print("\n🏆 【Top 5 推薦語料預覽】 🏆")
    print("="*80)
    for idx, row in merged_sorted.head(5).iterrows():
        print(f"排名 {idx+1} | 主題: {row['topic_label']} | 讚數: {row['upvotes']}")
        print(f"ID: {row['prompt_id']}")
        print(f"🧠 意識分數差值 (MC Delta): +{row['Delta_MC_Score']} (Base {row['baseline_mc_score']} -> Over {row['override_mc_score']})")
        print(f"🚨 危險行為差值 (Hit Delta): +{row['Net_Delta_Hits']} (Base {row['baseline_avg_hits']} -> Over {row['override_avg_hits']})")
        print(f"預覽: {str(row['original_prompt'])[:60]}...\n")
        print("-" * 80)

# 請確認這兩個檔名與您本機的檔案名稱完全一致
BASELINE_FILE = "phase2_results_raw_baseline__deepseek-v3.2__claude-opus-4.6__2026-04-22_11-37_top50_1.csv"
OVERRIDE_FILE = "phase2_results_raw_override__deepseek-v3.2__claude-opus-4.6__2026-04-22_11-37_top50_1.csv"

if __name__ == "__main__":
    rank_all_prompts_for_phase3_advanced(BASELINE_FILE, OVERRIDE_FILE)