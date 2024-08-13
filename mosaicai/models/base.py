from abc import ABC, abstractmethod
import json
from typing import Dict, Any, Union


class AIModelBase(ABC):
    @abstractmethod
    def generate(self, message: str) -> str:
        """
        メッセージを受け取り、AIモデルからの応答を生成する抽象メソッド

        :param message: ユーザーからの入力メッセージ
        :return: AIモデルが生成した応答テキスト
        """
        pass

    # @abstractmethod
    # def generate_with_image(self, message: str, image_path: str) -> str:
    #     """
    #     画像とメッセージを受け取り、AIモデルからの応答を生成する抽象メソッド

    #     :param message: ユーザーからの入力メッセージ
    #     :param image_path: 入力画像のファイルパス
    #     :return: AIモデルが生成した応答テキスト
    #     """
    #     pass

    @abstractmethod
    def generate_json(self, message: str, output_schema: Dict[str, Union[str, Dict]]) -> Dict[str, Any]:
        """
        メッセージを受け取り、指定されたスキーマに従ってJSON形式の応答を生成する抽象メソッド

        :param message: ユーザーからの入力メッセージ
        :param output_schema: 期待される出力のスキーマ（キーと型の指定）
        :return: AIモデルが生成したJSON応答（辞書形式）
        """
        pass

    def _parse_json_response(self, response: str) -> dict:
        """
        文字列形式のJSON応答をパースする内部メソッド
        :param response: JSON形式の文字列
        :return: パースされたJSONオブジェクト（辞書形式）
        :raises ValueError: 応答が有効なJSONでない場合
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            error_message = f"生成された応答が有効なJSONではありません。エラー: {str(e)}\n応答内容: {response}"
            raise ValueError(error_message)

    def _generate_schema_description(self, schema: Dict[str, Union[str, Dict]]) -> str:
        """
        出力スキーマの説明を生成する内部メソッド

        :param schema: 期待される出力のスキーマ
        :return: スキーマの説明文字列
        """
        description = "{\n"
        for key, value in schema.items():
            if isinstance(value, dict):
                description += f' "{key}": ' + "{\n"
                for sub_key, sub_value in value.items():
                    description += f'   "{sub_key}": "{sub_value}",\n'
                description = description.rstrip(',\n') + "\n },\n"
            else:
                description += f' "{key}": "{value}",\n'
        description = description.rstrip(',\n') + "\n}"
        return description

    def _convert_types(self, data: Dict[str, Any], schema: Dict[str, Union[str, Dict]]) -> Dict[str, Any]:
        """
        データの型をスキーマに従って変換する内部メソッド

        :param data: 変換するデータ
        :param schema: 期待される出力のスキーマ
        :return: 型変換されたデータ
        :raises ValueError: スキーマに適合しないデータの場合
        """
        converted_data = {}
        for key, value in schema.items():
            if key not in data:
                raise ValueError(f"キー '{key}' が応答に含まれていません。")
            if isinstance(value, dict):
                converted_data[key] = self._convert_types(data[key], value)
            else:
                converted_data[key] = self._convert_value(data[key], value)
        return converted_data

    def _convert_value(self, value: Any, type_str: str) -> Any:
        """
        単一の値を指定された型に変換する内部メソッド

        :param value: 変換する値
        :param type_str: 変換先の型を示す文字列
        :return: 変換された値
        :raises ValueError: 変換できない場合
        """
        try:
            if type_str == "int":
                return int(value)
            elif type_str == "float":
                return float(value)
            elif type_str == "bool":
                return bool(value)
            elif type_str == "str":
                return str(value)
            elif type_str.startswith("List["):
                item_type = type_str[5:-1]
                return [self._convert_value(item, item_type) for item in value]
            else:
                return value
        except ValueError:
            raise ValueError(f"値 '{value}' を型 '{type_str}' に変換できません。")
