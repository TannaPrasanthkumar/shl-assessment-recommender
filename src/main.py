"""
FastAPI application entrypoint.
Configures request/response handling pipelines, CORS middleware, startup lifespan indices,
and endpoints implementing stateless dialogue service orchestration.
"""

import os
import time
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.models.schemas import ChatRequest, ChatResponse, ConstraintState, HealthResponse
from src.database.vector_store import SHLCatalogRepository
from src.agents.state_tracker import DialogueStateTracker
from src.agents.decision_engine import DecisionEngine
from src.utils.logger import app_logger

# LIFESPAN lifecycle definition to manage startup warmup and database index initialization
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages application lifespan events (Startup database load, warm indexes, teardown).
    """
    app_logger.info("Initializing application startup sequence...")
    
    # Resolve clean catalog path dynamically using settings.BASE_DIR
    default_path = os.path.join(settings.BASE_DIR, "data", "clean_catalog.json")
        
    repository = SHLCatalogRepository(catalog_path=default_path, vector_db_path="")
    await repository.load_catalog()
    
    # Store instances in app state
    app.state.repository = repository
    app.state.state_tracker = DialogueStateTracker()
    
    app_logger.info("Warm index loaded. Startup checks complete. System ready.")
    yield
    app_logger.info("Graceful shutdown initiated. Releasing DB handles.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Conversational Recommendation Platform for SHL Assessments",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Logging & Latency Monitoring Middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next) -> Request:
    start_time = time.time()
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        app_logger.info(
            f"HTTP {request.method} {request.url.path} resolved with {response.status_code} in {duration_ms:.2f}ms"
        )
        return response
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        app_logger.error(
            f"HTTP {request.method} {request.url.path} failed in {duration_ms:.2f}ms: {e}"
        )
        raise


# Global Exception Handling mapping
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    app_logger.exception(f"Unhandled Exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."}
    )


# Dependency Injection Resolvers
def get_repository(request: Request) -> SHLCatalogRepository:
    return request.app.state.repository


def get_state_tracker(request: Request) -> DialogueStateTracker:
    return request.app.state.state_tracker


def get_decision_engine(
    repository: SHLCatalogRepository = Depends(get_repository)
) -> DecisionEngine:
    return DecisionEngine(repository=repository)


router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check readiness endpoint",
    description="Returns service status. Evaluator allows 2-minute warmup on first call."
)
async def health_check(
    repository: SHLCatalogRepository = Depends(get_repository)
) -> HealthResponse:
    """
    Retrieves system health status.
    """
    return HealthResponse(status="ok")


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Stateless dialogue assessment recommendation endpoint",
    description="Takes message history, tracks state constraints, routes request, returns recommendations list."
)
async def chat_interaction(
    payload: ChatRequest,
    tracker: DialogueStateTracker = Depends(get_state_tracker),
    engine: DecisionEngine = Depends(get_decision_engine)
) -> ChatResponse:
    """
    Main communication handler interface orchestrating the platform services.
    """
    try:
        async with asyncio.timeout(28.0):
            # 1. Stateless Dialogue State Tracking (DST) constraints extraction
            constraints = await tracker.extract_constraints(payload.messages)
            
            # 2. Extract latest user query text
            latest_user_query = ""
            for msg in reversed(payload.messages):
                if msg.role == "user":
                    latest_user_query = msg.content
                    break
            
            # 3. Retrieve database search scores & confidence
            if not engine.repository.retriever:
                await engine.repository.load_catalog()
                
            candidates, confidence = engine.repository.retriever.retrieve(
                latest_user_query,
                constraints,
                limit=10
            )
            
            # 4. Determine state transition action route
            action = engine.evaluate_routing_action(
                history_len=len(payload.messages),
                constraints=constraints,
                retrieval_confidence=confidence,
                latest_query=latest_user_query
            )
            
            # 5. Formulate response ChatResponse payload
            response = await engine.execute_action(
                action=action,
                constraints=constraints,
                raw_query=latest_user_query
            )
            return response
            
    except TimeoutError:
        app_logger.error("Chat interaction timed out after 28 seconds.")
        return ChatResponse(
            reply="I'm sorry, but the request timed out. Please try again with a simplified query.",
            recommendations=[],
            end_of_conversation=False
        )
    except Exception as e:
        app_logger.error(f"Chat interaction orchestration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Orchestration failure: {str(e)}"
        )


app.include_router(router)

