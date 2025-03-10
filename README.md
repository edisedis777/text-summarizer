# Text Summarizer

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A sophisticated text summarization tool powered by BART (Bidirectional and Auto-Regressive Transformer) from Facebook, enhanced with advanced features like batch processing, quantization, and rich console output.


<img width="710" alt="Screenshot 2025-03-10 at 22 35 32" src="https://github.com/user-attachments/assets/021120dd-3c75-45cd-aeba-d8e64bb353a3" />



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

## Usage

### Command Line
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

### Sample Text from [Wikipedia](https://en.wikipedia.org/wiki/Reinforcement_learning)

"Reinforcement learning (RL) is an interdisciplinary area of machine learning and optimal control concerned with how an intelligent agent should take actions in a dynamic environment in order to maximize a reward signal. Reinforcement learning is one of the three basic machine learning paradigms, alongside supervised learning and unsupervised learning.

Q-learning at its simplest stores data in tables. This approach becomes infeasible as the number of states/actions increases (e.g., if the state space or action space were continuous), as the probability of the agent visiting a particular state and performing a particular action diminishes.

Reinforcement learning differs from supervised learning in not needing labelled input-output pairs to be presented, and in not needing sub-optimal actions to be explicitly corrected. Instead, the focus is on finding a balance between exploration (of uncharted territory) and exploitation (of current knowledge) with the goal of maximizing the cumulative reward (the feedback of which might be incomplete or delayed).[1] The search for this balance is known as the exploration–exploitation dilemma.

The environment is typically stated in the form of a Markov decision process (MDP), as many reinforcement learning algorithms use dynamic programming techniques.[2] The main difference between classical dynamic programming methods and reinforcement learning algorithms is that the latter do not assume knowledge of an exact mathematical model of the Markov decision process, and they target large MDPs where exact methods become infeasible.[3]"

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
- Fork the repository
- Create a feature branch (git checkout -b feature/amazing-feature)
- Commit your changes (git commit -am 'Add amazing feature')
- Push to the branch (git push origin feature/amazing-feature)
- Create a Pull Request

###License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments
- Built with Transformers by Hugging Face
- Enhanced display with Rich
- Sample text from Wikipedia
- Inspired by [Text Summarization with DistillBart Model](https://machinelearningmastery.com/text-summarization-with-distillbart-model/)
