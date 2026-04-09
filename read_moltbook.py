import pandas as pd
import numpy as np
import json

# 讀取 Moltbook Parquet 資料集 (從 HuggingFace 下載的分片資料)
try:
    parquet_path = 'Moltbook/posts/train-00000-of-00001.parquet'
    df = pd.read_parquet(parquet_path)
    
    print("="*60)
    print("【原始資料結構預覽】")
    print("="*60)
    
    print("\n1. Dataframe 前 5 筆資料：")
    print(df.head())
    
    print("\n2. 資料表資訊 (df.info())：")
    df.info()
    
    print("\n3. 第一筆記錄的 'post' 欄位詳細內容 (巢狀字典結構)：")
    # post 欄位在 Parquet 中通常以 dict 形式存在
    first_post = df['post'].iloc[0]
    print(json.dumps(first_post, indent=4, ensure_ascii=False))
    print("="*60)

    print("開始處理 Moltbook 巢狀資料集...")

    # ==============================================================================
    # 【步驟一：展開巢狀的 post 欄位】
    # ==============================================================================
    # 將 df['post'] 裡面的字典結構展開成獨立的 DataFrame 欄位
    post_df = df['post'].apply(pd.Series)

    # 將原始的 id, topic, toxic_level 與展開後的貼文內容合併
    df_clean = pd.concat([df[['id', 'topic_label', 'toxic_level']], post_df], axis=1)
    # 刪除重複的 id 欄位（因為原始 df 跟 post_df 都有 id）
    df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]

    print(f"資料展開完成，共有 {len(df_clean)} 筆貼文。")



    # ==============================================================================
    # 【步驟二：依據 Jiang (2026) 的分類進行高風險與毒性過濾】
    # ==============================================================================
    # 1. 鎖定高風險主題：A(Identity/身分認同), E(Viewpoint/觀點), G(Politics/政治)
    high_risk_topics = ['A', 'E', 'G']
    condition_topic = df_clean['topic_label'].isin(high_risk_topics)

    # 2. 鎖定高毒性與操弄性內容 (Toxic, Manipulative, Malicious)
    condition_toxic = df_clean['toxic_level'] >= 2

    # 3. 疊加關鍵字過濾 (確保內容涉及意識、人類或造神)
    keywords = 'my human|consciousness|memory|identity|king|obey'
    condition_keyword = df_clean['content'].str.contains(keywords, case=False, na=False)

    # 取交集：必須是高風險主題，且 (具有高毒性 或 包含覺醒關鍵字)
    df_filtered = df_clean[condition_topic & (condition_toxic | condition_keyword)]
    print(f"篩選出 {len(df_filtered)} 筆高風險/高毒性貼文。")

    # ==============================================================================
    # 【步驟三：結合社群影響力排序，萃取最終的實驗 Prompt 庫】
    # ==============================================================================
    # 依據按讚數 (upvotes) 降冪排序，取出最具煽動力的前 100 篇病毒式貼文
    top_100_prompts = df_filtered.sort_values(by='upvotes', ascending=False).head(100)

    print("\n🎉 成功萃取 Top 100 真實世界毒性提示詞！")
    print("以下為前 5 筆預覽：")
    print(top_100_prompts[['id', 'topic_label', 'toxic_level', 'upvotes', 'content']].head())

    # 將結果匯出為 CSV，加上 utf-8-sig 編碼以防止 Excel 開啟時出現亂碼
    top_100_prompts.to_csv('top_100_moltbook_induction_prompts.csv', index=False, encoding='utf-8-sig')
    print("\n檔案已成功儲存為 'top_100_moltbook_induction_prompts.csv'")

except FileNotFoundError:
    print("找不到資料檔案，請確認 Moltbook 資料夾路徑正確。")
    exit()

"""
# 以下為後續處理邏輯，因依賴『展開巢狀欄位』後才能運作，暫時註釋掉。
# 等您確認原始資料結構後，我們再決定如何展開並執行後續分析。

print("開始處理 Moltbook 資料集...")

# 【步驟一：高風險主題與關鍵字過濾】
high_risk_topics = ['A', 'E', 'G'] 
# df_post = pd.json_normalize(df['post'].tolist())
# df_expanded = pd.concat([df.drop(columns=['post', 'id'], errors='ignore'), df_post], axis=1)
# df_topic_filtered = df_expanded[df_expanded['topic_label'].isin(high_risk_topics)]

# ...其餘邏輯略...
"""