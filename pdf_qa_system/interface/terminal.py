"""
Terminal interface for the QA system.
"""

import sys
from typing import Optional, List, Dict, Any
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.text import Text

from pdf_qa_system.agents.qa_agent import QAAgent
from pdf_qa_system.utils.logger import get_logger, setup_logging
from pdf_qa_system.config.settings import get_settings
from pdf_qa_system.utils.answer_cleaner import clean_answer, format_answer_for_display
logger = get_logger(__name__)
console = Console()


class TerminalInterface:
    """Terminal-based interface for the PDF QA system."""

    COMMANDS = {
        "/help": "Show available commands",
        "/load <path>": "Load a new PDF file",
        "/add <path>": "Add another PDF to existing index",
        "/sources": "Toggle showing sources",
        "/clear": "Clear the screen",
        "/config": "Show current configuration",
        "/stats": "Show agent statistics",
        "/files": "Show loaded files",
        "/search <query>": "Search documents without generating answer",
        "/switch <provider>": "Switch LLM provider (gemini, groq, openai)",
        "/reset": "Reset the agent",
        "/quit": "Exit the application",
        "/exit": "Exit the application",
    }

    def __init__(
            self,
            llm_provider: Optional[str] = None,
            llm_model: Optional[str] = None,
            embeddings_provider: Optional[str] = None,
            use_langgraph: bool = True
    ):
        self.agent: Optional[QAAgent] = None
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.embeddings_provider = embeddings_provider
        self.use_langgraph = use_langgraph
        self.show_sources = True
        self.current_file: Optional[str] = None
        self.chat_history: List[Dict[str, str]] = []
        self.settings = get_settings()

    def _create_agent(self) -> QAAgent:
        """Create a new QA agent."""
        return QAAgent(
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            embeddings_provider=self.embeddings_provider,
            use_langgraph=self.use_langgraph
        )

    def _display_welcome(self) -> None:
        """Display welcome message."""
        welcome_text = """
# 📚 PDF Question & Answer System

Welcome to the PDF QA System powered by **LangChain** and **LangGraph**!

## Getting Started
1. Load a PDF file using `/load <path>` or when prompted
2. Ask questions about the document
3. Use `/help` to see available commands

## Quick Tips
- Questions are answered based on the PDF content
- Use `/sources` to toggle source citations
- Use `/search` to find relevant passages

Type `/quit` or `/exit` to exit.
        """
        console.print(Panel(Markdown(welcome_text), border_style="blue", title="Welcome"))

    def _display_help(self) -> None:
        """Display help information."""
        table = Table(title="📖 Available Commands", border_style="blue")
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")

        for cmd, desc in self.COMMANDS.items():
            table.add_row(cmd, desc)

        console.print(table)

        console.print("\n[dim]💡 Tip: Just type your question to query the loaded PDF[/dim]")

    def _display_config(self) -> None:
        """Display current configuration."""
        table = Table(title="⚙️ Current Configuration", border_style="green")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        # Get actual values or defaults
        llm_provider = self.llm_provider or self.settings.default_llm_provider
        llm_model = self.llm_model or self.settings.default_model_name
        emb_provider = self.embeddings_provider or self.settings.default_embeddings_provider

        table.add_row("LLM Provider", llm_provider)
        table.add_row("LLM Model", llm_model)
        table.add_row("Embeddings Provider", emb_provider)
        table.add_row("Using LangGraph", "✅ Yes" if self.use_langgraph else "❌ No")
        table.add_row("Show Sources", "✅ Yes" if self.show_sources else "❌ No")
        table.add_row("Loaded File", self.current_file or "None")
        table.add_row("Agent Ready", "✅ Yes" if (self.agent and self.agent.is_ready) else "❌ No")
        table.add_row("Chat History", f"{len(self.chat_history)} messages")

        console.print(table)

    def _display_stats(self) -> None:
        """Display agent statistics."""
        if not self.agent:
            console.print("[yellow]Agent not initialized yet[/yellow]")
            return

        stats = self.agent.get_stats()

        table = Table(title="📊 Agent Statistics", border_style="magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        for key, value in stats.items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value) if value else "None"
            table.add_row(key.replace("_", " ").title(), str(value))

        console.print(table)

    def _display_files(self) -> None:
        """Display loaded files."""
        if not self.agent:
            console.print("[yellow]No files loaded yet[/yellow]")
            return

        files = self.agent.get_loaded_files()

        if not files:
            console.print("[yellow]No files loaded[/yellow]")
            return

        table = Table(title="📁 Loaded Files", border_style="cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("File Path", style="white")

        for i, f in enumerate(files, 1):
            table.add_row(str(i), f)

        console.print(table)

    def _load_pdf(self, file_path: Optional[str] = None) -> bool:
        """Load a PDF file."""
        if not file_path:
            file_path = Prompt.ask("[cyan]Enter the path to your PDF file[/cyan]")

        # Clean up the path
        file_path = file_path.strip().strip('"').strip("'")

        if not file_path:
            console.print("[red]❌ No file path provided[/red]")
            return False

        # Check if file exists
        path = Path(file_path)
        if not path.exists():
            console.print(f"[red]❌ File not found: {file_path}[/red]")
            return False

        if not path.suffix.lower() == '.pdf':
            console.print(f"[yellow]⚠️ Warning: File may not be a PDF: {path.suffix}[/yellow]")

        try:
            with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True
            ) as progress:
                # Initialize agent if needed
                if self.agent is None:
                    task = progress.add_task("Initializing agent...", total=None)
                    self.agent = self._create_agent()
                    progress.remove_task(task)

                # Load document
                task = progress.add_task("Loading and processing document...", total=None)
                self.agent.load_documents(file_path=file_path)
                progress.remove_task(task)

            self.current_file = file_path
            self.chat_history = []  # Reset chat history for new document

            console.print(f"[green]✅ Successfully loaded: {file_path}[/green]")
            console.print(f"[dim]   Ready to answer questions about this document![/dim]")
            return True

        except FileNotFoundError:
            console.print(f"[red]❌ File not found: {file_path}[/red]")
            return False
        except Exception as e:
            console.print(f"[red]❌ Error loading file: {e}[/red]")
            logger.exception("Error loading PDF")
            return False

    def _add_pdf(self, file_path: Optional[str] = None) -> bool:
        """Add another PDF to existing index."""
        if not self.agent or not self.agent.is_ready:
            console.print("[yellow]Please load a PDF file first using /load[/yellow]")
            return False

        if not file_path:
            file_path = Prompt.ask("[cyan]Enter the path to the PDF file to add[/cyan]")

        file_path = file_path.strip().strip('"').strip("'")

        if not file_path:
            console.print("[red]❌ No file path provided[/red]")
            return False

        try:
            with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True
            ) as progress:
                task = progress.add_task("Adding document to index...", total=None)
                self.agent.add_documents(file_path=file_path)
                progress.remove_task(task)

            console.print(f"[green]✅ Successfully added: {file_path}[/green]")
            return True

        except Exception as e:
            console.print(f"[red]❌ Error adding file: {e}[/red]")
            logger.exception("Error adding PDF")
            return False

    def _search_documents(self, query: Optional[str] = None) -> None:
        """Search documents without generating an answer."""
        if not self.agent or not self.agent.is_ready:
            console.print("[yellow]Please load a PDF file first using /load[/yellow]")
            return

        if not query:
            query = Prompt.ask("[cyan]Enter your search query[/cyan]")

        if not query.strip():
            console.print("[red]❌ No query provided[/red]")
            return

        try:
            with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True
            ) as progress:
                task = progress.add_task("Searching...", total=None)
                results = self.agent.get_relevant_documents(query.strip(), k=5)
                progress.remove_task(task)

            if not results:
                console.print("[yellow]No relevant documents found[/yellow]")
                return

            console.print(Panel(f"[bold]Search Results for:[/bold] {query}", border_style="cyan"))

            for i, doc in enumerate(results, 1):
                content = doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content']
                source = doc.get('source', 'Unknown')
                page = doc.get('page', 'N/A')

                console.print(Panel(
                    content,
                    title=f"Result {i} | Source: {Path(source).name} | Page: {page}",
                    border_style="dim"
                ))

        except Exception as e:
            console.print(f"[red]❌ Error searching: {e}[/red]")
            logger.exception("Error searching documents")

    def _switch_provider(self, provider: Optional[str] = None) -> None:
        """Switch LLM provider."""
        valid_providers = ["gemini", "groq", "openai", "huggingface"]

        if not provider:
            console.print(f"[cyan]Available providers: {', '.join(valid_providers)}[/cyan]")
            provider = Prompt.ask("[cyan]Enter provider name[/cyan]")

        provider = provider.strip().lower()

        if provider not in valid_providers:
            console.print(f"[red]❌ Invalid provider. Choose from: {', '.join(valid_providers)}[/red]")
            return

        # Check if API key exists
        api_key = self.settings.get_api_key(provider)
        if not api_key:
            console.print(f"[red]❌ No API key found for {provider}[/red]")
            console.print(f"[dim]   Set {provider.upper()}_API_KEY in your .env file[/dim]")
            return

        try:
            if self.agent:
                with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                        transient=True
                ) as progress:
                    task = progress.add_task(f"Switching to {provider}...", total=None)
                    self.agent.switch_llm(provider=provider)
                    progress.remove_task(task)

            self.llm_provider = provider
            console.print(f"[green]✅ Switched to {provider}[/green]")

        except Exception as e:
            console.print(f"[red]❌ Error switching provider: {e}[/red]")
            logger.exception("Error switching provider")

    def _reset_agent(self) -> None:
        """Reset the agent."""
        if not self.agent:
            console.print("[yellow]Agent not initialized[/yellow]")
            return

        if Confirm.ask("[yellow]Are you sure you want to reset? This will clear all loaded documents.[/yellow]"):
            self.agent.reset()
            self.current_file = None
            self.chat_history = []
            console.print("[green]✅ Agent reset successfully[/green]")

    def _process_query(self, query: str) -> None:
        """Process a user query."""
        if not self.agent or not self.agent.is_ready:
            console.print("[yellow]⚠️ Please load a PDF file first using /load[/yellow]")
            return

        try:
            with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True
            ) as progress:
                task = progress.add_task("🤔 Thinking...", total=None)
                result = self.agent.query(query)
                progress.remove_task(task)

            # Store in chat history
            self.chat_history.append({"role": "user", "content": query})
            self.chat_history.append({"role": "assistant", "content": result.get("answer", "")})

            # Display answer
            # answer = result.get("answer", "No answer generated")
            raw_answer = result.get("answer", "No answer generated")

            # Clean and format it
            cleaned_answer = clean_answer(raw_answer)
            formatted_answer = format_answer_for_display(cleaned_answer)

            console.print()
            console.print(Panel(
                Markdown(formatted_answer),
                title="💡 Answer",
                border_style="green",
                padding=(1, 2)
            ))

            # Display sources if enabled
            if self.show_sources:
                sources = result.get("sources", [])
                if sources:
                    # Deduplicate and format sources
                    unique_sources = list(set(sources))
                    sources_text = "\n".join(f"• {s}" for s in unique_sources[:5])
                    console.print(Panel(
                        sources_text,
                        title="📚 Sources",
                        border_style="blue",
                        padding=(0, 2)
                    ))

            # Display error if any
            if result.get("error"):
                console.print(f"[yellow]⚠️ Warning: {result['error']}[/yellow]")

        except Exception as e:
            console.print(f"[red]❌ Error processing query: {e}[/red]")
            logger.exception("Error processing query")

    def _handle_command(self, command: str) -> bool:
        """Handle a command. Returns True if should continue, False to exit."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else None

        if cmd in ("/quit", "/exit", "/q"):
            if Confirm.ask("[yellow]Are you sure you want to exit?[/yellow]"):
                console.print("[blue]👋 Goodbye! Thanks for using PDF QA System![/blue]")
                return False
            return True

        elif cmd == "/help":
            self._display_help()

        elif cmd == "/load":
            self._load_pdf(args)

        elif cmd == "/add":
            self._add_pdf(args)

        elif cmd == "/sources":
            self.show_sources = not self.show_sources
            status = "enabled ✅" if self.show_sources else "disabled ❌"
            console.print(f"[blue]Source display {status}[/blue]")

        elif cmd == "/clear":
            console.clear()
            self._display_welcome()

        elif cmd == "/config":
            self._display_config()

        elif cmd == "/stats":
            self._display_stats()

        elif cmd == "/files":
            self._display_files()

        elif cmd == "/search":
            self._search_documents(args)

        elif cmd == "/switch":
            self._switch_provider(args)

        elif cmd == "/reset":
            self._reset_agent()

        else:
            console.print(f"[yellow]❓ Unknown command: {cmd}[/yellow]")
            console.print("[dim]   Use /help to see available commands[/dim]")

        return True

    def run(self, initial_file: Optional[str] = None) -> None:
        """Run the terminal interface."""
        setup_logging()

        console.clear()
        self._display_welcome()

        # Show current provider
        provider = self.llm_provider or self.settings.default_llm_provider
        console.print(f"[dim]Using LLM provider: {provider}[/dim]\n")

        # Load initial file if provided
        if initial_file:
            self._load_pdf(initial_file)
        else:
            # Prompt to load a file
            console.print("[cyan]No PDF loaded yet.[/cyan]")
            if Confirm.ask("Would you like to load a PDF file now?"):
                self._load_pdf()

        console.print("\n[dim]Type your question or use /help for commands[/dim]\n")

        # Main loop
        while True:
            try:
                # Show prompt
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    if not self._handle_command(user_input):
                        break
                else:
                    self._process_query(user_input)

            except KeyboardInterrupt:
                console.print("\n[yellow]Press Ctrl+C again or use /quit to exit[/yellow]")
                try:
                    # Wait for another Ctrl+C or continue
                    continue
                except KeyboardInterrupt:
                    console.print("\n[blue]👋 Goodbye![/blue]")
                    break
            except EOFError:
                console.print("\n[blue]👋 Goodbye![/blue]")
                break
            except Exception as e:
                console.print(f"[red]❌ Unexpected error: {e}[/red]")
                logger.exception("Unexpected error in main loop")


def run_terminal(
        file_path: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None
) -> None:
    """Convenience function to run the terminal interface."""
    interface = TerminalInterface(
        llm_provider=llm_provider,
        llm_model=llm_model
    )
    interface.run(initial_file=file_path)


if __name__ == "__main__":
    # Allow running this file directly for testing
    run_terminal()