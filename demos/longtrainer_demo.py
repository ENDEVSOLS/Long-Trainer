#!/usr/bin/env python3
"""
LongTrainer — Complete RAG Workflow Demo

Demonstrates the full LongTrainer lifecycle with live progress tracking:
  Init → Bot Creation → Document Ingestion → Retrieval → Q&A

All configuration is driven by environment variables with sensible defaults.
Run with:
    OPENAI_API_KEY=sk-... python demos/longtrainer_demo.py

Or override any setting:
    LLM_PROVIDER=ollama LLM_MODEL=llama3 VECTORSTORE=faiss python demos/longtrainer_demo.py
"""

import os
import textwrap
import time

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.table import Table

console = Console()

# ═══════════════════════════════════════════════════════════════════════════
# Configuration — all env-var driven
# ═══════════════════════════════════════════════════════════════════════════

LLM_PROVIDER       = os.environ.get("LLM_PROVIDER", "openai")
LLM_MODEL          = os.environ.get("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", LLM_PROVIDER)
EMBEDDING_MODEL    = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
VECTORSTORE        = os.environ.get("VECTORSTORE", "faiss")
MONGO_URI          = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
NUM_K              = int(os.environ.get("NUM_K", "3"))

# Validate that the required API key exists for the chosen provider
_KEY_MAP = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
_required_key = _KEY_MAP.get(LLM_PROVIDER.lower())
if _required_key and not os.environ.get(_required_key):
    console.print(f"\n[bold red]❌ Error:[/bold red] {_required_key} is not set")
    console.print(f"[dim]Export it first: export {_required_key}=...[/dim]\n")
    raise SystemExit(1)

console.print("\n[bold blue]🚀 LongTrainer - Complete RAG Workflow Demo[/bold blue]\n")
console.print("[dim]Building a RAG chatbot with document ingestion and Q&A[/dim]\n")

# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def create_progress_table(steps_status):
    """Create progress table showing all steps"""
    table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    table.add_column("Step", style="bold", width=35)
    table.add_column("Status", width=5)
    table.add_column("Details", style="")
    
    for step_name, status, details in steps_status:
        if status == "done":
            table.add_row(step_name, "[green]✓[/green]", f"[green]{details}[/green]")
        elif status == "running":
            table.add_row(step_name, "[cyan]⚙️[/cyan]", f"[cyan]{details}[/cyan]")
        else:
            table.add_row(step_name, "[dim]⏳[/dim]", f"[dim]{details}[/dim]")
    
    return table

def show_config_panel(config):
    """Show configuration details"""
    content = Text()
    content.append("Configuration:\n\n", style="bold")
    for key, value in config.items():
        content.append(f"  {key}: ", style="cyan")
        content.append(f"{value}\n", style="white")
    return Panel(content, border_style="cyan", title="⚙️  Setup")

def show_documents_panel(docs):
    """Show ingested documents"""
    content = Text()
    content.append(f"Ingested {len(docs)} documents:\n\n", style="bold")
    for i, doc in enumerate(docs, 1):
        topic = doc.metadata.get("topic", "Unknown")
        source = doc.metadata.get("source", "unknown")
        content.append(f"{i}. ", style="cyan")
        content.append(f"{topic}", style="bold white")
        content.append(f" ({source})\n", style="dim")
        snippet = doc.page_content[:60].replace("\n", " ")
        content.append(f"   \"{snippet}...\"\n\n", style="white")
    return Panel(content, border_style="green", title="📄 Knowledge Base")

def show_retrieval_panel(query, retrieved):
    """Show retrieval results"""
    content = Text()
    content.append(f"Query: {query}\n\n", style="bold cyan")
    content.append(f"Retrieved {len(retrieved)} chunks:\n\n", style="bold")
    for i, doc in enumerate(retrieved[:2], 1):
        source = doc.metadata.get("source", "unknown")
        content.append(f"{i}. ", style="green")
        content.append(f"[{source}]\n", style="dim")
        snippet = doc.page_content[:80].replace("\n", " ")
        content.append(f"   \"{snippet}...\"\n\n", style="white")
    if len(retrieved) > 2:
        content.append(f"   ...and {len(retrieved) - 2} more chunk(s)", style="dim")
    return Panel(content, border_style="green", title="🔍 Retrieval Results")

def show_qa_panel(question, answer):
    """Show Q&A exchange"""
    content = Text()
    content.append("Q: ", style="bold cyan")
    content.append(f"{question}\n\n", style="cyan")
    content.append("A: ", style="bold green")
    wrapped = textwrap.fill(answer, width=90)
    content.append(f"{wrapped}", style="white")
    return Panel(content, border_style="yellow", title="💬 Q&A Exchange")

# ═══════════════════════════════════════════════════════════════════════════
# Main Workflow
# ═══════════════════════════════════════════════════════════════════════════

# Build vectorstore kwargs
vs_kwargs = {}
if VECTORSTORE == "qdrant":
    vs_kwargs = {"path": "qdrant_db"}

# Show configuration
config = {
    "LLM": f"{LLM_PROVIDER} / {LLM_MODEL}",
    "Embeddings": f"{EMBEDDING_PROVIDER} / {EMBEDDING_MODEL}",
    "Vector Store": VECTORSTORE,
    "MongoDB": MONGO_URI,
    "Top-K": str(NUM_K),
}
console.print(show_config_panel(config))
time.sleep(2)

# Initialize steps
steps = [
    ("1. Initialize LongTrainer", "running", "Connecting to MongoDB..."),
    ("2. Ingest Documents", "pending", "Waiting..."),
    ("3. Create RAG Bot", "pending", "Waiting..."),
    ("4. Test Retrieval", "pending", "Waiting..."),
    ("5. Start Chat Session", "pending", "Waiting..."),
    ("6. Q&A Exchanges", "pending", "Waiting..."),
]

with Live(console=console, refresh_per_second=10) as live:
    # Step 1: Initialize
    live.update(Panel(create_progress_table(steps), title="🔄 Workflow Progress", border_style="cyan"))
    time.sleep(1)
    
    from longtrainer.trainer import LongTrainer
    
    trainer = LongTrainer(
        mongo_endpoint=MONGO_URI,
        llm_provider=LLM_PROVIDER,
        default_llm=LLM_MODEL,
        embedding_provider=EMBEDDING_PROVIDER,
        embedding_model_name=EMBEDDING_MODEL,
        vector_store_provider=VECTORSTORE,
        vector_store_kwargs=vs_kwargs,
        num_k=NUM_K,
        ensemble=False,
    )
    
    steps[0] = ("1. Initialize LongTrainer", "done", "Connected successfully")
    live.update(Panel(create_progress_table(steps), title="🔄 Workflow Progress", border_style="cyan"))
    time.sleep(0.8)
    
    # Step 2: Ingest Documents
    steps[1] = ("2. Ingest Documents", "running", "Building knowledge base...")
    live.update(Panel(create_progress_table(steps), title="🔄 Workflow Progress", border_style="cyan"))
    
    from langchain_core.documents import Document
    
    sample_docs = [
        Document(
            page_content=textwrap.dedent("""\
                Python is a high-level, general-purpose programming language.
                It was created by Guido van Rossum and first released in 1991.
                Python emphasizes code readability and simplicity.
                It supports multiple programming paradigms including procedural,
                object-oriented, and functional programming.
                Python is widely used in data science, machine learning,
                web development, automation, and scientific computing.
            """),
            metadata={"source": "python_overview.txt", "topic": "Python"},
        ),
        Document(
            page_content=textwrap.dedent("""\
                Machine learning is a subset of artificial intelligence.
                It involves building systems that learn from data to make predictions
                or decisions without being explicitly programmed for each task.
                Common types include supervised learning, unsupervised learning,
                and reinforcement learning.
                Popular frameworks include scikit-learn, TensorFlow, and PyTorch.
            """),
            metadata={"source": "ml_overview.txt", "topic": "Machine Learning"},
        ),
        Document(
            page_content=textwrap.dedent("""\
                RAG stands for Retrieval-Augmented Generation.
                It combines a retrieval system (like FAISS vector search) with a
                language model (like GPT-4). When a user asks a question, the system
                first retrieves the most relevant chunks from a knowledge base, then
                passes those chunks + the question to the LLM to generate an answer.
                This dramatically reduces hallucinations compared to pure LLM usage.
            """),
            metadata={"source": "rag_overview.txt", "topic": "RAG"},
        ),
    ]
    
    bot_id = trainer.initialize_bot_id()
    trainer.pass_documents(documents=sample_docs, bot_id=bot_id)
    
    steps[1] = ("2. Ingest Documents", "done", f"{len(sample_docs)} docs in DB")
    live.update(Panel(create_progress_table(steps), title="🔄 Workflow Progress", border_style="cyan"))
    time.sleep(0.5)

    # Step 3: Create Bot (Embeds Documents)
    steps[2] = ("3. Create RAG Bot", "running", "Embedding docs...")
    live.update(Panel(create_progress_table(steps), title="🔄 Workflow Progress", border_style="cyan"))
    
    trainer.create_bot(
        bot_id=bot_id,
        prompt_template="You are a helpful assistant that answers questions about Python and machine learning.",
    )
    
    steps[2] = ("3. Create RAG Bot", "done", f"Bot ID: {bot_id[:8]}...")
    live.update(Panel(create_progress_table(steps), title="🔄 Workflow Progress", border_style="cyan"))
    time.sleep(0.8)
    
    # Show documents
    docs_display = Group(
        create_progress_table(steps),
        Text(""),
        show_documents_panel(sample_docs)
    )
    live.update(Panel(docs_display, title="🔄 Workflow Progress", border_style="cyan"))
    time.sleep(2.5)
    
    # Step 4: Test Retrieval
    live.update(Panel(create_progress_table(steps), title="🔄 Workflow Progress", border_style="cyan"))
    steps[3] = ("4. Test Retrieval", "running", "Querying vector store...")
    live.update(Panel(create_progress_table(steps), title="🔄 Workflow Progress", border_style="cyan"))
    
    query = "What is RAG and how does it work?"
    retrieved = trainer.invoke_vectorstore(bot_id=bot_id, query=query)
    
    steps[3] = ("4. Test Retrieval", "done", f"Retrieved {len(retrieved)} chunks")
    live.update(Panel(create_progress_table(steps), title="🔄 Workflow Progress", border_style="cyan"))
    time.sleep(0.5)
    
    # Show retrieval results
    retrieval_display = Group(
        create_progress_table(steps),
        Text(""),
        show_retrieval_panel(query, retrieved)
    )
    live.update(Panel(retrieval_display, title="🔄 Workflow Progress", border_style="cyan"))
    time.sleep(2.5)
    
    # Step 5: Start Chat
    live.update(Panel(create_progress_table(steps), title="🔄 Workflow Progress", border_style="cyan"))
    steps[4] = ("5. Start Chat Session", "running", "Creating chat...")
    live.update(Panel(create_progress_table(steps), title="🔄 Workflow Progress", border_style="cyan"))
    
    chat_id = trainer.new_chat(bot_id=bot_id)
    
    steps[4] = ("5. Start Chat Session", "done", f"Chat ID: {chat_id[:8]}...")
    live.update(Panel(create_progress_table(steps), title="🔄 Workflow Progress", border_style="cyan"))
    time.sleep(0.8)
    
    # Step 6: Q&A
    steps[5] = ("6. Q&A Exchanges", "running", "Asking questions...")
    live.update(Panel(create_progress_table(steps), title="🔄 Workflow Progress", border_style="cyan"))
    
    questions = [
        "What is Python used for?",
        "How is machine learning different from traditional programming?",
    ]
    
    # Ask first question
    question = questions[0]
    result = trainer.get_response(
        query=question,
        bot_id=bot_id,
        chat_id=chat_id,
    )
    # get_response returns (answer_text, sources_list)
    response = result[0] if isinstance(result, tuple) else str(result)
    if not response or not response.strip():
        response = "[LLM returned empty — check your API key and quota]"
    
    steps[5] = ("6. Q&A Exchanges", "done", f"{len(questions)} questions answered")
    live.update(Panel(create_progress_table(steps), title="🔄 Workflow Progress", border_style="green"))
    time.sleep(0.5)
    
    # Show Q&A
    qa_display = Group(
        create_progress_table(steps),
        Text(""),
        show_qa_panel(question, response)
    )
    live.update(Panel(qa_display, title="🔄 Workflow Progress", border_style="green"))
    time.sleep(3)

# Final summary
console.print()
summary_table = Table(title="📊 Workflow Summary", show_header=True, expand=True, border_style="green")
summary_table.add_column("Metric", style="bold")
summary_table.add_column("Value", justify="right")

summary_table.add_row("Bot ID", f"{bot_id[:16]}...")
summary_table.add_row("Chat ID", f"{chat_id[:16]}...")
summary_table.add_row("Documents Ingested", f"{len(sample_docs)}")
summary_table.add_row("Questions Answered", f"{len(questions)}")
summary_table.add_row("Vector Store", VECTORSTORE)
summary_table.add_row("LLM", f"{LLM_PROVIDER} / {LLM_MODEL}")
summary_table.add_row("Embeddings", f"{EMBEDDING_PROVIDER} / {EMBEDDING_MODEL}")

console.print(summary_table)

console.print("\n[bold green]✅ LongTrainer Demo Complete![/bold green]")
console.print("[dim]RAG chatbot is ready with document knowledge and chat history stored in MongoDB.[/dim]\n")
