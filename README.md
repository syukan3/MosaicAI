# MosaicAI

[![PyPI version](https://badge.fury.io/py/mosaicai.svg)](https://badge.fury.io/py/mosaicai)
[![Python Versions](https://img.shields.io/pypi/pyversions/mosaicai.svg)](https://pypi.org/project/mosaicai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MosaicAIは、複数の最先端AI言語モデル（ChatGPT、Claude、Gemini、Perplexity）を統一的なインターフェースで利用できるPythonライブラリです。

## 特徴

- 複数のAI言語モデルを簡単に利用できる統一されたインターフェース
- API Keyの安全な管理と再利用
- 拡張性を持つ設計で、新しいAIモデルの追加が容易

## インストール

pipを使用してMosaicAIをインストールできます：

```bash
$ pip install mosaicai
```

## 要件

- Python 3.7以上
- 各AI言語モデルのAPIキー

## 使用方法

1. まず、プロジェクトのルートディレクトリに`.env`ファイルを作成し、以下のように設定してください：

```
# OpenAI (ChatGPT) API Key
OPENAI_API_KEY=YOUR-OPENAI_API_KEY

# Anthropic (Claude) API Key
ANTHROPIC_API_KEY=YOUR-ANTHROPIC_API_KEY

# Google (Gemini) API Key
GOOGLE_API_KEY=YOUR-GOOGLE_API_KEY

# Perplexity API Key
PERPLEXITY_API_KEY=YOUR-PERPLEXITY_API_KEY

# Default model (optional)
DEFAULT_MODEL=gemini

# Logging configuration (optional)
LOG_LEVEL=INFO
LOG_FILE=mosaicai.log

# Request configuration (optional)
REQUEST_TIMEOUT=30
MAX_RETRIES=3
```

2. 基本的な使用例を以下に示します：

```python
from mosaicai import MosaicAI

# クライアントの初期化
client = MosaicAI(model="gpt-4o")

# テキスト生成
response = client.generate_text("AIの未来について教えてください")
print(response)

# 画像を使用した生成
image_path = "path/to/your/image.jpg"
response = client.generate_with_image("この画像について説明してください", image_path)
print(response)

# JSON生成
schema = {
    "title": "str",
    "main_points": "List[str]",
    "value": "int"
}
json_response = client.generate_json("AIの倫理的課題について3つのポイントを挙げてください", schema)
print(json_response)
```

より詳細な使用例については、[examples](examples)ディレクトリを参照してください。

## ドキュメンテーション

詳細なドキュメントは[こちら](https://mosaicai.readthedocs.io/)で見ることができます。

## ライセンス

MosaicAIはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 貢献

MosaicAIはオープンソースプロジェクトです。バグの報告、機能の提案、コードの貢献など、ご協力いただけます。詳細は[CONTRIBUTING.md](CONTRIBUTING.md)を参照してください。