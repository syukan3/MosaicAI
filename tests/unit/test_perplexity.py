import pytest
from unittest.mock import Mock, patch
from openai import OpenAI
from typing import Dict, Any, Union
from mosaicai.models.perplexity import Perplexity
from mosaicai.utils.api_key_manager import APIKeyManager
from pydantic import BaseModel


class OutputSchema(BaseModel):
    key_str: str
    key_int: int
    key_float: float
    key_bool: bool
    key_list: list


# テストフィクスチャ
@pytest.fixture
def mock_api_key_manager():
    """APIKeyManagerのモックを作成するフィクスチャ"""
    manager = Mock(spec=APIKeyManager)
    manager.get_api_key.return_value = "test_api_key"
    return manager


@pytest.fixture
def perplexity_instance(mock_api_key_manager):
    """Perplexityインスタンスを作成するフィクスチャ"""
    return Perplexity(mock_api_key_manager)


# テストケース
def test_init(perplexity_instance):
    """__init__メソッドのテスト"""
    assert isinstance(perplexity_instance.client, OpenAI)
    assert perplexity_instance.model == "llama-3.1-sonar-large-128k-online"


def test_get_model(perplexity_instance):
    """get_modelメソッドのテスト"""
    assert perplexity_instance.get_model() == "llama-3.1-sonar-large-128k-online"


@patch('openai.OpenAI')
def test_generate(mock_openai, perplexity_instance):
    """generateメソッドのテスト"""
    # モックの設定
    mock_chat = Mock()
    mock_openai.return_value.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="Test response"))]
    )
    perplexity_instance.client = mock_openai.return_value

    # メソッドの呼び出しとアサーション
    result = perplexity_instance.generate("Test message")
    assert result == "Test response"
    mock_openai.return_value.chat.completions.create.assert_called_once_with(
        model="llama-3.1-sonar-large-128k-online",
        messages=[{"role": "user", "content": "Test message"}]
    )


def test_generate_with_image(perplexity_instance):
    """generate_with_imageメソッドのテスト"""
    with pytest.raises(NotImplementedError):
        perplexity_instance.generate_with_image("image.jpg", "Test prompt")


@patch('openai.OpenAI')
def test_generate_json(mock_openai, perplexity_instance):
    """generate_jsonメソッドのテスト"""
    # モックの設定
    mock_openai.return_value.chat.completions.create.return_value = Mock(
        choices=[
            Mock(message=Mock(
                content='{"key_str": "test", "key_int": 42, "key_float": 3.14, "key_bool": true, "key_list": ["a", "b", "c"]}'))
        ]
    )
    perplexity_instance.client = mock_openai.return_value
    output_schema = OutputSchema
    # メソッドの呼び出しとアサーション
    result = perplexity_instance.generate_json("Test message", output_schema)
    assert result == {"key_str": "test", "key_int": 42, "key_float": 3.14, "key_bool": True, "key_list": ["a", "b", "c"]}
    mock_openai.return_value.chat.completions.create.assert_called_once()
    assert mock_openai.return_value.chat.completions.create.call_args[1]['response_format'] == {
        "type": "json_object"}


# 追加のテストケース
def test_custom_model(mock_api_key_manager):
    """カスタムモデルを指定した場合のテスト"""
    custom_model = "custom-model-name"
    perplexity = Perplexity(mock_api_key_manager, model=custom_model)
    assert perplexity.get_model() == custom_model


@patch('openai.OpenAI')
def test_generate_json_type_conversion(mock_openai, perplexity_instance):
    """generate_jsonメソッドの型変換テスト"""
    mock_openai.return_value.chat.completions.create.return_value = Mock(
        choices=[
            Mock(message=Mock(
                content='{"key_str": "test", "key_int": 42, "key_float": 3.14, "key_bool": true, "key_list": ["a", "b", "c"]}'))
        ]
    )
    perplexity_instance.client = mock_openai.return_value

    output_schema = OutputSchema

    result = perplexity_instance.generate_json("Test message", output_schema)
    assert result == {"key_str": "test", "key_int": 42, "key_float": 3.14,
                      "key_bool": True, "key_list": ["a", "b", "c"]}
    assert isinstance(result["key_str"], str)
    assert isinstance(result["key_int"], int)
    assert isinstance(result["key_float"], float)
    assert isinstance(result["key_bool"], bool)
    assert isinstance(result["key_list"], list)

