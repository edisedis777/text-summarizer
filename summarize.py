import functools
import logging
import textwrap
import re
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Optional
from contextlib import contextmanager

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.progress import Progress
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set the quantization backend
torch.backends.quantized.engine = 'qnnpack' if torch.backends.quantized.supported_engines else 'fbgemm'

@dataclass
class SummarizerConfig:
    """Configuration class for the text summarizer."""
    model_name: str = "facebook/bart-large-cnn"
    quantize: bool = False
    max_length: int = 150
    min_length: int = 50
    length_penalty: float = 1.0
    repetition_penalty: float = 1.5
    num_beams: int = 4
    batch_size: int = 2

class AdvancedTextSummarizer:
    def __init__(self, config: SummarizerConfig = SummarizerConfig()):
        """
        Initialize the advanced text summarizer.
        
        Args:
            config (SummarizerConfig): Configuration object with model parameters
            
        Raises:
            RuntimeError: If model initialization fails
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        
        self.console = Console()
        self.logger.info(f"Initializing summarizer with model: {config.model_name}")
        
        try:
            self.device = "cuda" if torch.cuda.is_available() and not config.quantize else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
            
            if config.quantize:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            self.model.to(self.device)
            self.max_input_length = min(1024, self.tokenizer.model_max_length)
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize summarizer: {str(e)}")

    def __enter__(self):
        """Context manager enter method."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method to clean up resources."""
        self.clear_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize input text.
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Cleaned and normalized text
        """
        text = re.sub(r"\s+", " ", text.strip())
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"[^\w\s.,!?;:()\-\"']", "", text)
        return text

    def split_long_text(self, text: str, max_tokens: Optional[int] = None) -> List[str]:
        """
        Split long text into chunks within model's token limit.
        
        Args:
            text (str): Input text
            max_tokens (int, optional): Maximum tokens per chunk
            
        Returns:
            List[str]: List of text chunks
        """
        if max_tokens is None:
            max_tokens = min(512, self.max_input_length - 50)
        
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return [text]
            
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunks.append(self.tokenizer.decode(chunk_tokens, skip_special_tokens=True))
        return chunks

    def _validate_params(self, max_length: int, min_length: int, 
                        length_penalty: float, repetition_penalty: float,
                        num_beams: int) -> None:
        """Validate summarization parameters."""
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if not isinstance(min_length, int) or min_length <= 0 or min_length > max_length:
            raise ValueError("min_length must be a positive integer less than max_length")
        if not isinstance(length_penalty, (int, float)) or length_penalty <= 0:
            raise ValueError("length_penalty must be a positive number")
        if not isinstance(repetition_penalty, (int, float)) or repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be a positive number")
        if not isinstance(num_beams, int) or num_beams < 1:
            raise ValueError("num_beams must be a positive integer")

    @functools.lru_cache(maxsize=200)
    def cached_summarize(self, text: str, max_length: int = 130, min_length: int = 30,
                        length_penalty: float = 1.0, repetition_penalty: float = 1.5,
                        num_beams: int = 4, early_stopping: bool = True) -> str:
        """Cached summarization function."""
        self._validate_params(max_length, min_length, length_penalty, 
                            repetition_penalty, num_beams)
        
        if not text or len(text.strip()) < 50:
            return text
            
        with torch.no_grad():
            encoded = self.tokenizer.encode(
                text, 
                truncation=True,
                max_length=self.max_input_length, 
                return_tensors="pt"
            ).to(self.device)
            
            summary_ids = self.model.generate(
                encoded,
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                num_beams=num_beams,
                early_stopping=early_stopping,
                no_repeat_ngram_size=2,
                top_k=50
            )
            return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def clear_cache(self) -> None:
        """Clear the LRU cache."""
        self.cached_summarize.cache_clear()
        self.logger.info("Cache cleared")

    async def summarize_batch_async(self, texts: List[str], batch_size: int = 2, **kwargs) -> List[str]:
        """
        Asynchronously summarize multiple texts in batches.
        
        Args:
            texts (List[str]): List of input texts
            batch_size (int): Number of texts to process at once
            **kwargs: Additional summarization arguments
            
        Returns:
            List[str]: List of summaries
        """
        async def process_text(text):
            return await asyncio.to_thread(self.summarize, text, **kwargs)
        
        summaries = await asyncio.gather(*[process_text(text) for text in texts])
        return [s['summary'] for s in summaries]

    def summarize(self, text: str, max_length: int = 150, min_length: int = 50,
                 length_penalty: float = 1.0, repetition_penalty: float = 1.5,
                 num_beams: int = 4, early_stopping: bool = True) -> Dict[str, str]:
        """
        Generate a summary with advanced features.
        
        Args:
            text (str): Text to summarize
            max_length (int): Maximum summary length
            min_length (int): Minimum summary length
            length_penalty (float): Penalty for longer summaries
            repetition_penalty (float): Penalty for repeated tokens
            num_beams (int): Number of beams for beam search
            early_stopping (bool): Stop when all beams finish
            
        Returns:
            Dict[str, str]: Dictionary with original, cleaned, and summarized text
        """
        self._validate_params(max_length, min_length, length_penalty, 
                            repetition_penalty, num_beams)
        self.logger.info(f"Starting summarization of text (length: {len(text)})")
        
        cleaned_text = self.preprocess_text(text)
        if len(cleaned_text.strip()) < 50:
            return {"original_text": text, "cleaned_text": cleaned_text, 
                   "summary": "Text too short to summarize."}

        with torch.no_grad():
            chunks = self.split_long_text(cleaned_text)
            if len(chunks) > 1:
                chunk_summaries = []
                for chunk in chunks:
                    summary = self.cached_summarize(
                        chunk, max(75, max_length // len(chunks)),
                        max(30, min_length // len(chunks)),
                        length_penalty, repetition_penalty, num_beams, early_stopping
                    )
                    chunk_summaries.append(summary)
                
                combined_summaries = " ".join(chunk_summaries)
                final_summary = combined_summaries if len(combined_summaries.split()) <= max_length else \
                    self.cached_summarize(combined_summaries, max_length, min_length,
                                        length_penalty, repetition_penalty, num_beams, early_stopping)
            else:
                final_summary = self.cached_summarize(
                    cleaned_text, max_length, min_length,
                    length_penalty, repetition_penalty, num_beams, early_stopping
                )
            
            self.logger.info("Summarization completed successfully")
            return {"original_text": text, "cleaned_text": cleaned_text, "summary": final_summary}

    def print_summary(self, summary_data: Dict[str, str]) -> None:
        """Print a formatted summary using Rich."""
        table = Table(box=box.ROUNDED, show_header=False, padding=(1, 2))
        table.add_column("Category", style="cyan")
        table.add_column("Content", style="white", no_wrap=False)
        
        original_text = textwrap.fill(summary_data["original_text"], width=80)
        table.add_row("Original Text", original_text)
        
        if summary_data["cleaned_text"] != summary_data["original_text"]:
            cleaned_text = textwrap.fill(summary_data["cleaned_text"], width=80)
            table.add_row("Cleaned Text", cleaned_text)
        
        summary_text = textwrap.fill(summary_data["summary"], width=80)
        table.add_row("Summary", f"[bold green]{summary_text}[/bold green]")
        
        panel = Panel(table, title="[bold blue]Text Summary[/bold blue]",
                     subtitle="[italic]Generated with BART[/italic]")
        self.console.print(panel)

def main():
    """Main function with command-line support."""
    import argparse
    parser = argparse.ArgumentParser(description="Advanced Text Summarizer")
    parser.add_argument("--text", help="Text to summarize")
    parser.add_argument("--batch", nargs="+", help="Multiple texts to summarize")
    args = parser.parse_args()
    
    console = Console()
    console.print(Panel("[bold]Advanced Text Summarizer[/bold]\nUsing BART model",
                       title="🤖 Text Summarizer", subtitle="v1.0", style="blue"))
    
    with AdvancedTextSummarizer() as summarizer:
        if args.text:
            summary = summarizer.summarize(args.text)
            summarizer.print_summary(summary)
        
        elif args.batch:
            with Progress() as progress:
                task = progress.add_task("[green]Summarizing batch...", total=len(args.batch))
                summaries = asyncio.run(summarizer.summarize_batch_async(args.batch))
                table = Table(box=box.ROUNDED)
                table.add_column("#", style="cyan")
                table.add_column("Original", style="white")
                table.add_column("Summary", style="green")
                
                for i, (text, summary) in enumerate(zip(args.batch, summaries), 1):
                    table.add_row(str(i), textwrap.fill(text, 40), textwrap.fill(summary, 40))
                    progress.update(task, advance=1)
                
                console.print(Panel(table, title="[bold blue]Batch Summaries[/bold blue]"))

def run_tests():
    """Run basic unit tests."""
    summarizer = AdvancedTextSummarizer()
    assert summarizer.preprocess_text("  test   text  ") == "test text"
    assert len(summarizer.split_long_text("short text")) == 1
    assert summarizer.summarize("short")["summary"] == "Text too short to summarize."
    print("All tests passed!")

if __name__ == "__main__":
    main()
    # Uncomment to run tests
    # run_tests()