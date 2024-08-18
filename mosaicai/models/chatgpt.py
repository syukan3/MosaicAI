import base64
from typing import Dict, Any, Union, Type
import json
import openai
from openai import OpenAI
from .base import AIModelBase
from ..utils.api_key_manager import APIKeyManager
from pydantic import BaseModel


class ChatGPT(AIModelBase):
    def __init__(self, api_key_manager: APIKeyManager, model: str = "gpt-4o"):
        """
        ChatGPTモデルの初期化
        :param api_key_manager: API鍵を管理するAPIKeyManagerインスタンス
        :param model: 使用するモデルの名前（デフォルトは"gpt-4o"）
        """
        self.api_key_manager = api_key_manager
        api_key = self.api_key_manager.get_api_key("openai")
        if not api_key:
            raise ValueError("OpenAI APIキーが設定されていません。")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def get_model(self) -> str:
        """
        self.modelの値を返す
        :return: 使用するモデルの名前
        """
        return self.model

    def generate(self, message: str) -> str:
        """
        指定されたメッセージに対してChatGPTの応答を生成する
        :param message: ユーザーからの入力メッセージ
        :return: ChatGPTが生成した応答テキスト
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": message}]
        )
        return response.choices[0].message.content

    def generate_with_image(self, message: str, image_path: str) -> str:
        """
        画像を含むメッセージに対してChatGPTの応答を生成する
        :param message: ユーザーからの入力メッセージ
        :param image_path: 画像ファイルのパス
        :return: ChatGPTが生成した応答テキスト
        """
        with open(image_path, "rb") as image_file:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": message},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode()}"}}
                        ],
                    }
                ]
            )
        return response.choices[0].message.content

    def generate_json(self, message: str, output_schema: Union[Dict[str, Union[str, Dict]], Type[BaseModel]]) -> Dict[str, Any]:
        """
        指定されたメッセージに対してChatGPTのJSON応答を生成する
        :param message: ユーザーからの入力メッセージ
        :param output_schema: 期待される出力のスキーマ（キーと型の指定）
        :return: ChatGPTが生成したJSON応答（辞書形式）
        """

        # Structured Output（JSON形式での応答）: 正式リリースしたら統合する
        if self.model == 'gpt-4o-2024-08-06':
            system_message = f"応答はJSON形式で生成してください。日本語で回答してください。"
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": message}
                ],
                tools=[
                    openai.pydantic_function_tool(output_schema),
                ]
            )
            return response.choices[0].message.tool_calls[0].function.parsed_arguments.dict()

        else:
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

            json_response = self._parse_json_response(response.choices[0].message.content)
            return self._convert_types(json_response, output_schema)

    def generate_with_image_json(self, message: str, image_path: str, output_schema: Union[Dict[str, Union[str, Dict]], Type[BaseModel]]) -> Dict[str, Any]:
        """
        画像を含むメッセージに対してChatGPTのJSON応答を生成する
        :param message: ユーザーからの入力メッセージ
        :param image_path: 画像ファイルのパス
        :param output_schema: 期待される出力のスキーマ（キーと型の指定）
        :return: ChatGPTが生成したJSON応答（辞書形式）
        """
        schema_description = self._generate_schema_description(output_schema)
        system_message = f"応答は以下のJSON形式で生成してください: \n{schema_description}"

        with open(image_path, "rb") as image_file:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": [
                        {"type": "text", "text": message},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode()}"
                        }}
                    ]}
                ],
                response_format={"type": "json_object"}
            )

        json_response = self._parse_json_response(response.choices[0].message.content)
        return self._convert_types(json_response, output_schema)
