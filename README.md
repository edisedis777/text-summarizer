# Text Summarizer

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A sophisticated text summarization tool powered by BART (Bidirectional and Auto-Regressive Transformer) from Facebook, enhanced with advanced features like batch processing, quantization, and rich console output.

## Features

- **Advanced Summarization**: Uses BART-large-CNN model for high-quality summaries
- **Text Preprocessing**: Cleans and normalizes input text
- **Long Text Handling**: Splits and summarizes lengthy documents
- **Batch Processing**: Summarize multiple texts efficiently with async support
- **Quantization**: Optional model quantization for faster inference
- **Rich Output**: Beautiful console formatting using Rich library
- **Configurable**: Flexible parameters via configuration class
- **Error Handling**: Robust validation and logging
- **Memory Management**: Context manager support and cache clearing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/advanced-text-summarizer.git
cd advanced-text-summarizer
```
Install dependencies:
```bash
pip install -r requirements.txt
```

### Requirements
Python 3.8+
torch
transformers
rich

### Usage

Command Line
Summarize a single text:

```bash
python summarizer.py --text "Your long text here that needs summarization"
```

Summarize multiple texts:

```bash
python summarizer.py --batch "Text 1" "Text 2" "Text 3"
```


### Python API
```
from summarizer import AdvancedTextSummarizer, SummarizerConfig

# Basic usage
summarizer = AdvancedTextSummarizer()
summary = summarizer.summarize("Your text here")
summarizer.print_summary(summary)

# With custom configuration
config = SummarizerConfig(
    max_length=200,
    min_length=50,
    quantize=True
)
with AdvancedTextSummarizer(config) as summarizer:
    summary = summarizer.summarize("Your text here")
```


### Example Output

```
╭────────────────────── Text Summary ──────────────────────╮
│                                                          │
│   Original Text    Your input text goes here...          │
│   Summary          [bold green]A concise summary...[/]   │
│                                                          │
╰──────────── Generated with BART ─────────────╯
```

### Configuration Options

| Parameter  |       Description        |	       Default            |
| ---------- | ------------------------ | ------------------------- |
| model_name | Pre-trained model to use	| "facebook/bart-large-cnn" |
|  quantize  | Enable model quantization| False                     |
| max_length | Maximum summary length	  | 150                       |
| min_length | Minimum summary length	  | 50                        |
| length_penalty | Penalty for longer summaries |	1.0               |
|repetition_penalty	| Penalty for repeated tokens	| 1.5             |
| num_beams	 | Number of beams for search |	4                       |
| batch_size | Batch size for processing	| 2                       |


### Development
Running Tests
```
python summarizer.py  # Uncomment run_tests() in main
```

### Contributing
Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -am 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Create a Pull Request

###License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments
Built with Transformers by Hugging Face
Enhanced display with Rich
Inspired by [Text Summarization with DistillBart Model](https://machinelearningmastery.com/text-summarization-with-distillbart-model/)
