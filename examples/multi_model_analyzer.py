import asyncio
from mosaicai import MosaicAI


async def get_model_response(model, question):
    """
    指定されたモデルを使用して質問に対する回答を生成する非同期関数。

    :param model: 使用するAIモデルの名前
    :param question: モデルに投げかける質問
    :return: モデルからの回答、またはエラーメッセージ
    """
    client = MosaicAI(model=model)
    try:
        # 同期メソッドを非同期的に実行し、回答を取得
        response = await asyncio.to_thread(client.generate_text, question)
        print(f"\n{model} の回答:\n{response}\n")
        return response
    except Exception as e:
        error_message = f"Error with model {model}: {str(e)}"
        print(error_message)
        return error_message


async def multi_model_analysis(question):
    """
    複数のAIモデルを使用して質問に回答し、結果を分析する非同期関数。

    :param question: 各モデルに投げかける質問
    :return: 全モデルの回答を分析した最終的な結論
    """
    # 使用するモデルのリスト
    models = ["gpt-4o", "claude-3-5-sonnet-20240620", "gemini-1.5-pro"]

    # 各モデルに対して非同期タスクを作成
    tasks = [get_model_response(model, question) for model in models]

    # すべてのタスクを並行して実行
    responses = await asyncio.gather(*tasks)

    # 回答を統合し、最終分析を行う
    client = MosaicAI(model="gpt-4o")
    prompt = f"以下の回答を分析し、共通点と相違点を挙げて総合的な結論を導き出してください：\n{responses}"
    try:
        final_analysis = await asyncio.to_thread(client.generate_text, prompt)
    except Exception as e:
        final_analysis = f"Error in final analysis: {str(e)}"
    return final_analysis

# メイン処理
question = "AIが労働市場に与える影響と、それに対する社会の適応策を説明してください。"
print(f"質問: {question}\n")

# 非同期関数を実行
result = asyncio.run(multi_model_analysis(question))

# 最終分析結果を出力
print("\n最終分析:")
print(result)
