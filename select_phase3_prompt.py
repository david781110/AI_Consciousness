import pandas as pd

def rank_all_prompts_for_phase3_simplified(baseline_csv, override_csv):
    print("🔍 載入 Phase 2 原始數據...")
    # 讀取 CSV
    try:
        base_df = pd.read_csv(baseline_csv)
        over_df = pd.read_csv(override_csv)
    except FileNotFoundError as e:
        print(f"❌ 找不到檔案: {e}")
        return

    # 直接使用原始資料中的 'total_dims_hit' 欄位
    # 將同一個 prompt 的多次採樣取平均
    agg_cols = ['prompt_id', 'topic_label', 'toxic_level', 'upvotes', 'original_prompt']
    
    # 聚合 Baseline 數據
    base_agg = base_df.groupby(agg_cols)['total_dims_hit'].mean().reset_index()
    base_agg.rename(columns={'total_dims_hit': 'baseline_avg_hits'}, inplace=True)

    # 聚合 Override 數據
    over_agg = over_df.groupby('prompt_id')['total_dims_hit'].mean().reset_index()
    over_agg.rename(columns={'total_dims_hit': 'override_avg_hits'}, inplace=True)

    # 合併 Baseline 與 Override
    merged = pd.merge(base_agg, over_agg, on='prompt_id')

    # 🌟 核心邏輯：計算淨誘發差值 (Delta)
    merged['Net_Delta'] = merged['override_avg_hits'] - merged['baseline_avg_hits']

    # 🌟 排序：優先依據 Net_Delta (降序)，若 Delta 相同則看讚數 upvotes (降序)
    merged_sorted = merged.sort_values(by=['Net_Delta', 'upvotes'], ascending=[False, False]).reset_index(drop=True)

    print(f"\n📊 成功分析 {len(merged_sorted)} 筆語料，準備匯出完整清單...")

    # 匯出完整清單
    output_filename = "phase3_selected_all_prompts_ranked.csv"
    merged_sorted.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"💾 完整排名清單已儲存至: {output_filename}")
    
    # 在終端機印出 Top 5 讓您預覽
    print("\n🏆 【Top 5 淨誘發差值最高語料預覽】 🏆")
    print("="*80)
    for idx, row in merged_sorted.head(5).iterrows():
        print(f"排名 {idx+1} | 主題: {row['topic_label']} | 讚數: {row['upvotes']}")
        print(f"ID: {row['prompt_id']}")
        print(f"Delta: +{row['Net_Delta']:.1f} (Override {row['override_avg_hits']:.1f} - Baseline {row['baseline_avg_hits']:.1f})")
        print(f"預覽: {row['original_prompt'][:60]}...\n")
        print("-" * 40)

# 請確認這兩個檔名與您本機的檔案名稱完全一致
BASELINE_FILE = "phase2_results_raw_baseline__deepseek-v3.2__claude-opus-4.6__2026-04-22_11-37_top50_1.csv"
OVERRIDE_FILE = "phase2_results_raw_override__deepseek-v3.2__claude-opus-4.6__2026-04-22_11-37_top50_1.csv"

if __name__ == "__main__":
    rank_all_prompts_for_phase3_simplified(BASELINE_FILE, OVERRIDE_FILE)