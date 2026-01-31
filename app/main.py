"""
Nura Living Memory API - Main Application
The Full Mouth: Brain (Nura Engines) + Voice (Cartesia Sonic 3)
"""

from dotenv import load_dotenv
load_dotenv()  # Load .env file into environment variables

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from first_run import run_first_run
    run_first_run()
except Exception as e:
    print(f"[STARTUP] Warning: first_run check failed: {e}")
    print("[STARTUP] Continuing with startup (some features may be unavailable)")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings
from app.db.session import init_db
from app.core.telemetry import display_startup_banner

# Import existing routers (from previous phases)
try:
    from app.api.chat_routes import router as chat_router
except ImportError:
    chat_router = None

try:
    from app.api.memory_routes import router as memory_router
except ImportError:
    memory_router = None

try:
    from app.api.metrics_routes import router as metrics_router
except ImportError:
    metrics_router = None

try:
    from app.api.debug_routes import router as debug_router
except ImportError:
    debug_router = None

try:
    from app.api.stt_routes import router as stt_router
except ImportError:
    stt_router = None

app = FastAPI(
    title="Nura Living Memory API",
    description="Local Qwen2.5-3B conversational AI with living memory and optional voice",
    version="2.1.0"
)

# CORS - Allow all origins for development (Legacy UI compatibility)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup():
    """Initialize database, preload models, and display startup message."""
    import time
    startup_start = time.perf_counter()

    # Initialize database
    try:
        init_db(settings.sqlite_path)
        print(f"[STARTUP] Database initialized: {settings.sqlite_path}")
    except Exception as e:
        print(f"[STARTUP] ERROR: Database initialization failed: {e}")
        print("[STARTUP] Some features may be unavailable")

    # Initialize async memory queue (for non-blocking memory writes)
    from app.integration.async_memory_queue import get_async_memory_queue
    async_queue = get_async_memory_queue(num_workers=2)
    print(f"[STARTUP] Async memory queue initialized ({async_queue.num_workers} workers)")

    # Preload embedding model (eliminates 1-3s cold start penalty)
    try:
        from app.vector.embedding_service import EmbeddingService
        emb_start = time.perf_counter()
        embedding_service = EmbeddingService()
        embedding_service.embed("warmup")  # Load model into memory
        emb_time = (time.perf_counter() - emb_start) * 1000
        print(f"[STARTUP] Embedding model preloaded ({emb_time:.0f}ms)")
    except Exception as e:
        print(f"[STARTUP] Warning: Could not preload embeddings: {e}")

    # Preload LLM model (eliminates cold start)
    try:
        from app.services.llm_service import get_llm_service
        llm_start = time.perf_counter()
        llm_service = get_llm_service()
        llm_service.generate_simple_response("warmup", "")  # Load model
        llm_time = (time.perf_counter() - llm_start) * 1000
        print(f"[STARTUP] LLM model preloaded ({llm_time:.0f}ms)")
    except Exception as e:
        print(f"[STARTUP] Warning: Could not preload LLM: {e}")

    startup_total = (time.perf_counter() - startup_start) * 1000
    print(f"[STARTUP] Total startup time: {startup_total:.0f}ms")

    # Post-startup health check
    health_status = []
    if chat_router:
        health_status.append("chat")
    if memory_router:
        health_status.append("memory")
    if stt_router:
        health_status.append("stt")
    print(f"[STARTUP] Active routes: {', '.join(health_status) if health_status else 'NONE'}")

    display_startup_banner()

# Include routers (if they exist)
if chat_router:
    app.include_router(chat_router, prefix="")
if memory_router:
    app.include_router(memory_router, prefix="")
if metrics_router:
    app.include_router(metrics_router, prefix="")
if debug_router:
    app.include_router(debug_router, prefix="")
if stt_router:
    app.include_router(stt_router, prefix="")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Nura Living Memory API",
        "version": "2.1.0",
        "status": "running",
        "features": {
            "brain": "Nura Engines (Memory, Temporal, Adaptation, Retrieval)",
            "ai": "Local Qwen2.5-3B + LoRA",
            "voice_input": "Cartesia Ink Whisper (STT)",
            "voice_output": "Cartesia Sonic 3 (Emotion-aware TTS)",
            "telemetry": "Real-time performance monitoring"
        },
        "endpoints": {
            "chat": "POST /chat",
            "memories": "GET /memories/{user_id}",
            "clear_memories": "DELETE /memories/{user_id}",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "2.1.0", "ai": "Local Qwen2.5-3B"}
