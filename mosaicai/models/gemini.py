import google.generativeai as genai
from typing import Dict, Any, Union
from PIL import Image
import json
from .base import AIModelBase
from ..utils.api_key_manager import APIKeyManager


class Gemini(AIModelBase):
    def __init__(self, api_key_manager: APIKeyManager, model: str = 'gemini-1.5-pro'):
        """
        Geminiモデルの初期化
        :param api_key_manager: API鍵を管理するAPIKeyManagerインスタンス
        :param model: 使用するモデルの名前（デフォルトは'gemini-1.5-pro'）
        """
        self.api_key_manager = api_key_manager
        api_key = self.api_key_manager.get_api_key("gemini")
        if not api_key:
            raise ValueError("Gemini APIキーが設定されていません。")
        # Google Generative AI APIの設定
        genai.configure(api_key=api_key)
        # Generative AIモデルのインスタンスを作成
        self.model = genai.GenerativeModel(model)

    def get_model(self) -> str:
        """
        self.modelの値を返す
        :return: 使用するモデルの名前
        """
        return self.model

    def generate(self, message: str) -> str:
        """
        指定されたメッセージに対してGeminiの応答を生成する
        :param message: ユーザーからの入力メッセージ
        :return: Geminiが生成した応答テキスト
        """
        # Gemini APIを使用してコンテンツを生成
        response = self.model.generate_content(message)
        # 生成された応答テキストを返す
        return response.text

    def generate_with_image(self, message: str, image_path: str) -> str:
        """
        指定されたメッセージと画像に対してGeminiの応答を生成する
        :param message: ユーザーからの入力メッセージ
        :param image_path: 入力画像のファイルパス
        :return: Geminiが生成した応答テキスト
        """
        # 画像ファイルを開く
        image = Image.open(image_path)
        # メッセージと画像を使用してコンテンツを生成
        response = self.model.generate_content([message, image])
        # 生成された応答テキストを返す
        return response.text

    def generate_json(self, message: str, output_schema: Dict[str, Union[str, Dict]]) -> Dict[str, Any]:
        """
        指定されたメッセージに対してGeminiのJSON応答を生成する
        :param message: ユーザーからの入力メッセージ
        :param output_schema: 期待される出力のスキーマ（キーと型の指定）
        :return: Geminiが生成したJSON応答（辞書形式）
        """
        # スキーマの説明を生成
        schema_description = self._generate_schema_description(output_schema)
        # プロンプトを作成
        prompt = f"応答は以下のJSON形式で生成してください。```json```をつける必要はありません。: \n{schema_description}\n\n{message}"

        # Gemini APIを使用してコンテンツを生成
        response = self.model.generate_content(prompt)
        # 生成されたJSON応答をパースして返す
        json_response = self._parse_json_response(response.text)
        return self._convert_types(json_response, output_schema)

    def generate_with_image_json(self, message: str, image_path: str, output_schema: Dict[str, Union[str, Dict]]) -> Dict[str, Any]:
        """
        画像を含むメッセージに対してGeminiのJSON応答を生成する
        :param message: ユーザーからの入力メッセージ
        :param image_path: 画像ファイルのパス
        :param output_schema: 期待される出力のスキーマ（キーと型の指定）
        :return: Geminiが生成したJSON応答（辞書形式）
        """
        # 画像ファイルを開く
        image = Image.open(image_path)
        # スキーマの説明を生成
        schema_description = self._generate_schema_description(output_schema)
        # プロンプトを作成
        prompt = f"応答は以下のJSON形式で生成してください。JSONのみを出力し、バッククォートや説明テキストは含めないでください: \n<JSONSchema>{schema_description}</JSONSchema>\n\n{message}"

        # メッセージと画像を使用してコンテンツを生成
        response = self.model.generate_content([prompt, image])
        # 生成されたJSON応答をパースして返す
        json_response = self._parse_json_response(response.text)
        return self._convert_types(json_response, output_schema)