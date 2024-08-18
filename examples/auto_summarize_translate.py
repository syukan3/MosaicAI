from mosaicai import MosaicAI


def summarize_and_translate(text, target_languages):
    # MosaicAIクライアントの初期化
    # Claude 3.5 Sonnetモデルを使用
    client = MosaicAI(model="claude-3-5-sonnet-20240620")

    # 入力テキストの要約
    # 200単語程度に要約するよう指示
    summary = client.generate_text(f"以下の文章を200単語程度に要約してください：\n{text}")

    # 要約文の各言語への翻訳
    translations = {}
    for lang in target_languages:
        # 各言語に翻訳するよう指示
        translation = client.generate_text(f"以下の文章を{lang}に翻訳してください：\n{summary}")
        # 翻訳結果を辞書に格納
        translations[lang] = translation

    # 要約と翻訳結果を返す
    return summary, translations


# 長文を入力（実際の使用時にはここに長文を設定）
text = "..."

# 翻訳対象の言語リスト
target_languages = ["日本語", "中国語", "フランス語", "スペイン語"]

# 要約と翻訳を実行
summary, translations = summarize_and_translate(text, target_languages)

# 結果の出力
print("要約:", summary)
for lang, translation in translations.items():
    print(f"{lang}翻訳:", translation)
