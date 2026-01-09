#!/usr/bin/env python3
"""Quick test script for the PDF QA System."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdf_qa_system.agents.qa_agent import QAAgent
from pdf_qa_system.utils.logger import setup_logging


def main():
    setup_logging(level="INFO")

    # Initialize agent
    print("🚀 Initializing QA Agent...")
    agent = QAAgent(
        llm_provider="gemini",
        use_langgraph=True
    )

    # Load PDF
    pdf_path = "data/pdfs/BMM.pdf"
    print(f"📄 Loading PDF: {pdf_path}")
    agent.load_documents(file_path=pdf_path)

    # Ask questions
    print("\n" + "=" * 50)
    print("Ready! Type your questions (type 'quit' to exit)")
    print("=" * 50 + "\n")

    while True:
        question = input("\n❓ Your question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        if not question:
            continue

        print("\n🤔 Thinking...")
        result = agent.query(question)

        print(f"\n✅ Answer:\n{result['answer']}")

        if result.get('sources'):
            print(f"\n📚 Sources: {', '.join(result['sources'])}")


if __name__ == "__main__":
    main()