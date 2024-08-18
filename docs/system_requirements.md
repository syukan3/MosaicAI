# MosaicAI プロジェクト要件定義書

## 1. プロジェクト概要

MosaicAIは、複数の最先端AI言語モデル（ChatGPT、Claude、Gemini、Perplexity）を統一的なインターフェースで利用できるPythonライブラリです。
このプロジェクトは、既存のコードベースを改善し、より効率的で使いやすいPyPIライブラリとして再設計することを目的としています。

## 2. 目的

- 複数のAI言語モデルを簡単に利用できる統一されたインターフェースを提供する
- PyPIライブラリとして公開し、簡単にインストールして使用できるようにする
- API Keyを安全に管理し、再利用可能にする
- 拡張性を持たせ、新しいAIモデルの追加を容易にする

## 3. 機能要件

### 3.1 サポートするAIモデル

- ChatGPT (OpenAI)
- Claude (Anthropic)
- Gemini (Google)
- Perplexity

### 3.2 基本機能

- テキスト生成：各AIモデルを使用してテキストを生成する
- 画像認識：画像を入力として受け取り、説明を生成する（対応モデルのみ）
- JSON生成：指定されたスキーマに従ってJSON形式の出力を生成する
- 画像JSON生成：指定されたスキーマに従って画像からJSON形式の出力を生成する


### 3.3 API Key管理

- API Keyを安全に保存し、再利用可能にする
- 環境変数、設定ファイル、またはプログラム的に API Key を設定できるようにする

### 3.4 エラーハンドリング

- AIモデルからのエラーレスポンスを適切に処理し、ユーザーにわかりやすいエラーメッセージを提供する

### 3.5 ロギング

- 詳細なログ機能を提供し、デバッグや監視を容易にする

### 3.6 拡張性

- 新しいAIモデルを簡単に追加できるプラグイン機構を実装する

## 4. 非機能要件

### 4.1 パフォーマンス

- リクエスト処理の低レイテンシを維持する
- 並行リクエスト処理による高スループットを実現する

### 4.2 セキュリティ

- API Keyを安全に暗号化して保存する
- HTTPS通信を使用してAIサービスとの通信を行う

### 4.3 信頼性

- ネットワークエラーやサービス障害に対して適切に対応し、再試行メカニズムを実装する

### 4.4 拡張性

- 新しいAIモデルやサービスを容易に追加できる設計にする

### 4.5 保守性

- 明確なコード構造と十分なドキュメンテーションを提供する
- ユニットテストとインテグレーションテストを充実させる

### 4.6 使いやすさ

- 直感的なAPIデザインにより、ユーザーが簡単に利用開始できるようにする
- 詳細なドキュメントとサンプルコードを提供する

## 5. システム設計

### 5.1 アーキテクチャ

- モジュール構成：
  - `mosaicai/`: メインパッケージ
    - `__init__.py`: パッケージの初期化
    - `client.py`: メインのクライアントクラス
    - `models/`: 各AIモデルの実装
      - `base.py`: 基底モデルクラス
      - `chatgpt.py`: ChatGPTモデル
      - `claude.py`: Claudeモデル
      - `gemini.py`: Geminiモデル
      - `perplexity.py`: Perplexityモデル
    - `utils/`: ユーティリティ関数
      - `api_key_manager.py`: API Key管理
      - `error_handler.py`: エラーハンドリング
      - `logger.py`: ロギング機能
    - `exceptions.py`: カスタム例外クラス
    - `config.py`: 設定管理

### 5.2 クラス設計
#### 5.2.1 MosaicAI（メインクライアントクラス）

```python
class MosaicAI:
    def __init__(self, model: str, config: Dict[str, Any] = None):
        # 初期化処理

    def initialize_model(self, model: str):
        # モデルを初期化するメソッド

    def generate_text(self, prompt: str) -> str:
        # テキスト生成メソッド

    def generate_with_image(self, image_path: str, prompt: str) -> str:
        # 画像説明生成メソッド

    def generate_json(self, message: str, output_schema: Union[Dict[str, Union[str, Dict]], Type[BaseModel]]) -> Dict[str, Any]:
        # 指定されたメッセージに対してJSON応答を生成する

    def generate_with_image_json(self, message: str, image_path: str, output_schema: Union[Dict[str, Union[str, Dict]], Type[BaseModel]]) -> Dict[str, Any]:
        # 画像を含むメッセージに対してJSON応答を生成する

    def set_api_key(self, model: str, api_key: str):
        # APIキーを設定するメソッド

    @classmethod
    def from_config_file(cls, config_path: str) -> 'MosaicAI':
        # 設定ファイルからインスタンスを生成するクラスメソッド
```

#### 5.2.2 BaseModel（基底モデルクラス）

```python
class AIModelBase(ABC):
    """全てのAIモデルの基底クラス。"""
    @abstractmethod
    def generate(self, message: str) -> str:
        """メッセージを受け取り、AIモデルからの応答を生成する抽象メソッド。"""
        pass

    def generate_json(self, message: str, output_schema: Union[Dict[str, Union[str, Dict]], Type[BaseModel]]) -> Dict[str, Any]:
        """JSON形式の応答を生成する抽象メソッド"""
        pass

    def _parse_json_response(self, response: str) -> dict:
        """JSON応答をパースする内部メソッド"""
        pass

    def _generate_schema_description(self, schema: Union[Dict[str, Union[str, Dict]], Type[BaseModel]]) -> str:
        """スキーマの説明を生成する内部メソッド"""
        pass

    def _convert_types(self, data: Dict[str, Any], schema: Union[Dict[str, Union[str, Dict]], Type[BaseModel]]) -> Dict[str, Any]:
        """データの型をスキーマに従って変換する内部メソッド"""
        pass

    def _convert_value(self, value: Any, type_info: Union[str, Dict[str, Any]]) -> Any:
        """単一の値を指定された型に変換する内部メソッド"""
        pass
```

### 5.3 API Key管理

- `APIKeyManager` クラスを実装し、以下の機能を提供する：
  - API Keyの暗号化と安全な保存
  - 保存されたAPI Keyの読み込みと復号
  - 環境変数からのAPI Keyの読み込み

### 5.4 エラーハンドリング

- カスタム例外クラスを定義し、詳細なエラー情報を提供
- グローバルな例外ハンドラを実装し、一貫したエラーレスポンスを保証

### 5.5 ロギング

- 構造化ロギングを実装し、ログレベルの動的な設定を可能にする
- ログローテーションとログファイルの圧縮機能を提供

## 6. インターフェース設計

### 6.1 公開API

```python
from mosaicai import MosaicAI

# クライアントの初期化
client = MosaicAI()

# API Keyの設定
client.set_api_key("chatgpt", "your-api-key-here")

# テキスト生成
response = client.generate_text("AIの未来について教えてください")

# 画像説明生成
image_description = client.generate_with_image("path/to/image.jpg", "この画像を詳しく説明してください")

# JSON生成
from pydantic import BaseModel
from typing import List
class AIEthicsResponse(BaseModel):
    title: str
    points: List[str]
    summary: str
    integer_value: int
    float_value: float
    boolean_flag: bool
    nested_object: Dict[str, Any]
    array_of_numbers: List[float]

json_response = client.generate_json("AIの倫理的課題について3つのポイントを挙げてください", AIEthicsResponse)

# 画像付きJSON生成
class AIProductFeatures(BaseModel):
    product_name: str
    features: List[str]
    overall_impression: str
    price: float
    is_available: bool
    release_date: str
    specifications: Dict[str, Union[str, int, float, bool]]
    ratings: List[int]

image_json_response = client.generate_with_image_json("path/to/image.jpg", "この画像に基づいて、製品の特徴を3つ挙げてください", AIProductFeatures)
```

### 6.2 設定ファイル形式

MosaicAIは、APIキーとデフォルト設定を管理するために、ルートディレクトリに配置された`.env`ファイルを使用します。

```
# OpenAI (ChatGPT) API Key
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

# Anthropic (Claude) API Key
ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"

# Google (Gemini) API Key
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

# Perplexity API Key
PERPLEXITY_API_KEY="YOUR_PERPLEXITY_API_KEY"

# Default model (optional)
DEFAULT_MODEL=gemini

# Logging configuration (optional)
LOG_LEVEL=INFO
LOG_FILE=mosaicai.log

# Request configuration (optional)
REQUEST_TIMEOUT=30
MAX_RETRIES=3
```

-  `APIKeyManager`クラス ( `mosaicai/utils/api_key_manager.py` にあります) は、`.env`ファイルからAPIキーを読み込みます。
-  各APIキーは、対応するAIモデルの名前で始まります。
-  オプションで、デフォルトのAIモデル、ログ設定、リクエスト設定を指定できます。


## 7. テスト計画

### 7.1 ユニットテスト

- 各モデルクラスのメソッドをテスト
- ユーティリティ関数のテスト
- エラーハンドリングのテスト

### 7.2 インテグレーションテスト

- 実際のAIサービスとの連携テスト
- 非同期処理の動作確認

### 7.3 負荷テスト

- 並行リクエスト処理の性能テスト
- 長時間運用時の安定性テスト

## 8. ドキュメンテーション

- README.md：プロジェクトの概要、インストール方法、基本的な使用例
- API ドキュメント：各クラスとメソッドの詳細な説明
- チュートリアル：一般的なユースケースに基づいた使用例
- コントリビューションガイド：プロジェクトへの貢献方法

## 9. デプロイメント

### 9.1 パッケージング

- `setup.py` または `pyproject.toml` を使用してパッケージを構成
- 依存関係の管理に `poetry` または `pipenv` を使用

### 9.2 CI/CD

- GitHub Actionsを使用して自動テストとデプロイを設定
- PyPIへの自動パブリッシュを実装

### 9.3 バージョニング

- セマンティックバージョニングを採用
- CHANGELOGの自動生成と管理

## 10. 今後の拡張計画

- 新しいAIモデルのサポート追加
- ストリーミングレスポンスのサポート
- マルチモーダル（テキスト、画像、音声）入力の統合
- ファインチューニング機能の追加

この要件定義書に基づいて開発を進めることで、MosaicAIプロジェクトをより効率的で拡張性の高いPyPIライブラリとして再構築できます。各セクションの詳細な実装については、開発チームで議論し、必要に応じて調整を行ってください。
