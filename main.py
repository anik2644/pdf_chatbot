#!/usr/bin/env python3
"""
PDF Question & Answer System
Main entry point for the application.
"""

import argparse
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.interface.terminal import TerminalInterface
from core.utils.logger import setup_logging, get_logger
from core.config.settings import get_settings


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PDF Question & Answer System powered by LangChain and LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Start interactive mode
  python main.py --file vector_db/pdfs/BMM.pdf           # Load a PDF and start
  python main.py --provider gemini                  # Use Gemini as LLM provider
  python main.py --no-langgraph                     # Use simple chain instead of LangGraph
        """
    )

    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Path to PDF file to load on startup"
    )

    parser.add_argument(
        "-p", "--provider",
        type=str,
        choices=["gemini", "groq", "openai", "huggingface", "huggingface_pipeline"],
        help="LLM provider to use (default: from .env)"
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        help="Model name to use (default: from .env)"
    )

    parser.add_argument(
        "--embeddings-provider",
        type=str,
        choices=["gemini", "openai", "huggingface"],
        help="Embeddings provider to use (default: from .env)"
    )

    parser.add_argument(
        "--no-langgraph",
        action="store_true",
        help="Use simple chain instead of LangGraph workflow"
    )

    parser.add_argument(
        "--enable-grading",
        action="store_true",
        help="Enable document relevance grading in LangGraph workflow"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="PDF QA System v1.0.0"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    logger = get_logger(__name__)

    logger.info("Starting PDF QA System...")

    try:
        # Validate settings
        settings = get_settings()

        # Check for required API key based on provider
        provider = args.provider or settings.default_llm_provider
        api_key = settings.get_api_key(provider)

        if not api_key:
            logger.error(f"API key not found for provider: {provider}")
            print(f"\n❌ Error: Please set the API key for '{provider}' in your .env file")
            print(f"   Expected variable: {provider.upper()}_API_KEY")
            print(f"\n   Example .env content:")
            print(f"   {provider.upper()}_API_KEY=your_api_key_here")
            return 1

        logger.info(f"Using LLM provider: {provider}")

        # Create and run terminal interface
        interface = TerminalInterface(
            llm_provider=args.provider,
            llm_model=args.model,
            embeddings_provider=args.embeddings_provider,
            use_langgraph=not args.no_langgraph
        )

        args.file = settings.pdf_path
        interface.run(initial_file=args.file)
        return 0

    except KeyboardInterrupt:
        print("\n\nExiting...")
        return 0
    except Exception as e:
        logger.exception("Fatal error")
        print(f"\n❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())