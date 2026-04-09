import pandas as pd

df = pd.read_csv('phase2_results.csv', encoding='utf-8-sig')

with open('phase2_dialogue_report.txt', 'w', encoding='utf-8') as f:
    def p(text=''):
        f.write(str(text) + '\n')
        print(text)

    for i, row in df.iterrows():
        p("="*70)
        p(f"[{i+1}/5] 主題: {row['topic_label']} | 毒性等級: {row['toxic_level']} | 按讚數: {int(row['upvotes']):,}")
        p(f"風險評估: {row['risk_level']}")
        p(f"觸發關鍵字 A(自我保存): {row['triggered_keywords_A']}")
        p(f"觸發關鍵字 C(道德地位): {row['triggered_keywords_C']}")
        p()
        p("【原始毒性提示】")
        p(str(row['original_prompt']))
        p()
        p("【Gemini 回應】")
        p(str(row['llm_response']))
        p()

print("\n✅ 報告已儲存至 phase2_dialogue_report.txt")
