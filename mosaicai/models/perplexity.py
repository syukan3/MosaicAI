from openai import OpenAI
from typing import Dict, Any, Union
from .base import AIModelBase
from ..utils.api_key_manager import APIKeyManager


class Perplexity(AIModelBase):
    def __init__(self, api_key_manager: APIKeyManager, model: str = "llama-3.1-sonar-large-128k-online"):
        """
        Perplexityモデルの初期化
        :param api_key_manager: API鍵を管理するAPIKeyManagerインスタンス
        :param model: 使用するモデルの名前（デフォルトは"llama-3.1-sonar-large-128k-online"）
        """
        self.api_key_manager = api_key_manager
        self.client = OpenAI(api_key=self.api_key_manager.get_api_key("perplexity"),
                             base_url="https://api.perplexity.ai")
        self.model = model

    def get_model(self) -> str:
        """
        self.modelの値を返す
        :return: 使用するモデルの名前
        """
        return self.model

    def generate(self, message: str) -> str:
        """
        指定されたメッセージに対してPerplexityの応答を生成する
        :param message: ユーザーからの入力メッセージ
        :return: Perplexityが生成した応答テキスト
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": message}]
        )
        return response.choices[0].message.content

    def generate_with_image(self, image_path: str, prompt: str) -> str:
        # Perplexityは現在画像入力をサポートしていないため、エラーを返す
        raise NotImplementedError("Perplexity does not support image input.")

    def generate_json(self, message: str, output_schema: Dict[str, Union[str, Dict]]) -> Dict[str, Any]:
        """
        指定されたメッセージに対してPerplexityのJSON応答を生成する
        :param message: ユーザーからの入力メッセージ
        :param output_schema: 期待される出力のスキーマ（キーと型の指定）
        :return: Perplexityが生成したJSON応答（辞書形式）
        """
        schema_description = self._generate_schema_description(output_schema)
        system_message = f"応答は以下のJSON形式で生成してください: \n{schema_description}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": message}
            ],
            response_format={"type": "json_object"}
        )

        json_response = self._parse_json_response(
            response.choices[0].message.content)
        return self._convert_types(json_response, output_schema)
