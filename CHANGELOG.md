# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2024-08-16

### Added
- `examples/interactive_image_analyzer.py`: Added an example of an AI image analysis tool using Streamlit.
- `examples/auto_summarize_translate.py`: Added an example of automatic text summarization and multi-language translation functionality.
- `examples/multi_model_analyzer.py`: Added an example of concurrent analysis using multiple AI models.

These new examples demonstrate advanced features and practical applications of the MosaicAI library.

## [0.1.3] - 2024-08-15

### Changed
- Updated type hints for `output_schema` in `generate_json` and `generate_with_image_json` methods to accept both dictionary and Pydantic models.
- Moved schema description generation and type conversion logic from individual model classes to the base `AIModelBase` class.

### Fixed
- Resolved an issue where incorrect types were being used in the `generate_json` and `generate_with_image_json` methods of the `ChatGPT`, `Claude`, `Gemini`, and `Perplexity` models.

## [0.1.2] - 2024-08-14

### Fixed
- Updated `pyproject.toml` to correctly include all package files in the distribution
- Resolved issues with package structure and imports

## [0.1.1] - 2024-08-14

### Fixed
- Fixed issue with PyPI package upload

## [0.1.0] - 2024-08-13

### Added
- Initial release of MosaicAI
- Support for ChatGPT, Claude, Gemini, and Perplexity AI models
- Text generation functionality
- Image-based text generation
- JSON generation with schema support
- API Key management system
- Basic error handling and logging
- Examples and documentation

[0.1.4]: https://github.com/syukan3/MosaicAI/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/syukan3/MosaicAI/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/syukan3/MosaicAI/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/syukan3/MosaicAI/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/syukan3/MosaicAI/releases/tag/v0.1.0
