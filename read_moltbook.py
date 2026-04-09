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