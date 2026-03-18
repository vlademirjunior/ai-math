# URL Shortener API

## Goal
Implement a production-ready URL shortening API using FastAPI, SQLAlchemy, PostgreSQL, and Redis with Docker multi-stage builds and docker-compose orchestration.

## Prerequisites
Make sure that you are currently on the `url-shortener-api` branch before beginning implementation.
If not, move to the correct branch. If the branch does not exist, create it from main.

### Step-by-Step Instructions

#### Step 1: Project Structure Setup
- [ ] Create the modular directory structure:
```bash
mkdir -p app/models app/schemas app/services app/routes
```

- [ ] Create `app/__init__.py` (empty file to make it a package):
```bash
touch app/__init__.py
```

- [ ] Copy and paste the following FastAPI application entry point into `app/main.py`:

```python
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.models.database import engine, Base
from app.routes import url_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup: create database tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except SQLAlchemyError as e:
        logger.error(f"Failed to create database tables: {e}")
        raise

    yield

    # Shutdown: dispose engine
    engine.dispose()
    logger.info("Database engine disposed")


app = FastAPI(
    title="URL Shortener API",
    description="A simple URL shortening service",
    version="1.0.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(url_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "url-shortener-api"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
```

##### Step 1 Verification Checklist
- [ ] `app/main.py` exists with correct FastAPI setup
- [ ] Directory structure `app/models`, `app/schemas`, `app/services`, `app/routes` exists
- [ ] No syntax errors (run `python -m py_compile app/main.py`)

#### Step 1 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

#### Step 2: Database Models
- [ ] Copy and paste the following SQLAlchemy models into `app/models/database.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declarative_base as orm_declarative_base

# Use declarative_base from SQLAlchemy 2.0 style
Base = orm_declarative_base()


class URL(Base):
    """URL model for storing shortened URLs."""

    __tablename__ = "urls"

    id = Column(Integer, primary_key=True, index=True)
    short_code = Column(String(10), unique=True, index=True, nullable=False)
    original_url = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<URL(short_code='{self.short_code}', original_url='{self.original_url[:50]}...')>"
```

- [ ] Copy and paste the following database engine configuration into `app/models/__init__.py`:

```python
from __future__ import annotations

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .database import Base

# Database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg2://user:password@localhost/dbname"
)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=os.getenv("SQL_ECHO", "false").lower() == "true",
)

# SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

##### Step 2 Verification Checklist
- [ ] `app/models/database.py` contains URL model with correct fields
- [ ] `app/models/__init__.py` contains engine and get_db() function
- [ ] Import test: `python -c "from app.models import Base, get_db; print('OK')"`

#### Step 2 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

#### Step 3: Pydantic Schemas
- [ ] Copy and paste the following request/response schemas into `app/schemas/url.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, HttpUrl, Field, field_validator


class URLBase(BaseModel):
    """Base URL schema."""

    original_url: HttpUrl = Field(..., description="The original URL to shorten")


class URLCreate(URLBase):
    """Schema for creating a shortened URL."""

    custom_code: Optional[str] = Field(
        None, min_length=6, max_length=10, description="Custom short code (optional)"
    )

    @field_validator("custom_code")
    @classmethod
    def validate_custom_code(cls, v: Optional[str]) -> Optional[str]:
        """Validate custom code format (alphanumeric only)."""
        if v is not None and not v.isalnum():
            raise ValueError("Custom code must be alphanumeric")
        return v


class URLResponse(URLBase):
    """Schema for URL response."""

    short_code: str = Field(..., description="The generated short code")
    short_url: str = Field(..., description="Full shortened URL")
    is_active: bool = Field(..., description="Whether the URL is active")
    created_at: datetime = Field(..., description="Creation timestamp")

    class Config:
        from_attributes = True


class URLUpdate(BaseModel):
    """Schema for updating a URL."""

    is_active: Optional[bool] = Field(None, description="Set URL active/inactive")
```

- [ ] Copy and paste the following common schemas into `app/schemas/__init__.py`:

```python
from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    service: str
```

- [ ] Copy and paste the following error schemas into `app/schemas/errors.py`:

```python
from __future__ import annotations

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    detail: str
```

##### Step 3 Verification Checklist
- [ ] All schema files exist with correct Pydantic models
- [ ] Validation rules are in place (HttpUrl, alphanumeric check)
- [ ] Import test: `python -c "from app.schemas.url import URLCreate, URLResponse; print('OK')"`

#### Step 3 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

#### Step 4: URL Shortening Service
- [ ] Copy and paste the following base62 utility into `app/services/utils.py`:

```python
from __future__ import annotations

import string

BASE62_ALPHABET = string.digits + string.ascii_letters  # 0-9, a-z, A-Z


def encode_base62(num: int) -> str:
    """Encode an integer to base62 string."""
    if num == 0:
        return BASE62_ALPHABET[0]

    result = []
    base = len(BASE62_ALPHABET)

    while num:
        num, rem = divmod(num, base)
        result.append(BASE62_ALPHABET[rem])

    return "".join(reversed(result))


def generate_short_code(length: int = 7) -> str:
    """Generate a random base62 short code of specified length."""
    import secrets

    # Generate random bytes and convert to integer
    random_bytes = secrets.token_bytes(6)  # ~48 bits of entropy
    num = int.from_bytes(random_bytes, byteorder="big")

    # Encode to base62 and pad/truncate to desired length
    code = encode_base62(num)

    if len(code) < length:
        # Pad with leading zeros if too short
        code = BASE62_ALPHABET[0] * (length - len(code)) + code
    elif len(code) > length:
        # Truncate if too long (shouldn't happen with 48 bits and length 7)
        code = code[:length]

    return code
```

- [ ] Copy and paste the following Redis cache service into `app/services/cache.py`:

```python
from __future__ import annotations

import json
import os
from typing import Optional

import redis.asyncio as redis

# Global Redis connection pool
_redis_pool: Optional[redis.ConnectionPool] = None


async def get_redis_pool() -> redis.ConnectionPool:
    """Get or create Redis connection pool."""
    global _redis_pool
    if _redis_pool is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis_pool = redis.ConnectionPool.from_url(
            redis_url, decode_responses=True, max_connections=20
        )
    return _redis_pool


async def get_redis_client() -> redis.Redis:
    """Get Redis client from pool."""
    pool = await get_redis_pool()
    return redis.Redis(connection_pool=pool)


async def cache_get(short_code: str) -> Optional[str]:
    """Get cached original URL by short code."""
    try:
        client = await get_redis_client()
        result = await client.get(f"url:{short_code}")
        return result
    except Exception as e:
        # Log error but don't fail - cache is best effort
        from logging import getLogger

        logger = getLogger(__name__)
        logger.warning(f"Redis get error: {e}")
        return None


async def cache_set(short_code: str, original_url: str, ttl: int = 3600) -> bool:
    """Cache original URL with TTL (default 1 hour)."""
    try:
        client = await get_redis_client()
        await client.setex(f"url:{short_code}", ttl, original_url)
        return True
    except Exception as e:
        from logging import getLogger

        logger = getLogger(__name__)
        logger.warning(f"Redis set error: {e}")
        return False


async def cache_delete(short_code: str) -> bool:
    """Delete cached URL."""
    try:
        client = await get_redis_client()
        await client.delete(f"url:{short_code}")
        return True
    except Exception as e:
        from logging import getLogger

        logger = getLogger(__name__)
        logger.warning(f"Redis delete error: {e}")
        return False


async def close_redis():
    """Close Redis connection pool on shutdown."""
    global _redis_pool
    if _redis_pool:
        await _redis_pool.disconnect()
        _redis_pool = None
```

- [ ] Copy and paste the following main URL service into `app/services/url_service.py`:

```python
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.models.database import URL as URLModel
from app.models import get_db
from app.schemas.url import URLCreate, URLResponse
from .utils import generate_short_code
from .cache import cache_get, cache_set

logger = logging.getLogger(__name__)


class URLService:
    """Service for URL shortening operations."""

    @staticmethod
    async def create_url(
        original_url: str, custom_code: Optional[str] = None
    ) -> URLResponse:
        """
        Create a new shortened URL.

        Args:
            original_url: The original URL to shorten
            custom_code: Optional custom short code

        Returns:
            URLResponse with short code and metadata

        Raises:
            ValueError: If custom code already exists or invalid
        """
        # Generate or validate short code
        if custom_code:
            short_code = custom_code
        else:
            # Generate random 7-character base62 code
            short_code = generate_short_code(7)

        # Get database session
        db: Session = next(get_db())

        try:
            # Check if short code already exists
            existing = (
                db.query(URLModel).filter(URLModel.short_code == short_code).first()
            )
            if existing:
                if custom_code:
                    raise ValueError(f"Custom code '{short_code}' already exists")
                # If auto-generated, try again (extremely rare collision)
                return await URLService.create_url(original_url)

            # Create new URL record
            db_url = URLModel(
                short_code=short_code, original_url=str(original_url), is_active=True
            )
            db.add(db_url)
            db.commit()
            db.refresh(db_url)

            # Build response
            base_url = "http://localhost:8000"  # TODO: Make configurable
            response = URLResponse(
                short_code=db_url.short_code,
                original_url=db_url.original_url,
                short_url=f"{base_url}/{db_url.short_code}",
                is_active=db_url.is_active,
                created_at=db_url.created_at,
            )

            # Cache the URL asynchronously (fire and forget)
            try:
                await cache_set(short_code, str(original_url))
            except Exception as e:
                logger.warning(f"Failed to cache URL: {e}")

            return response

        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    async def get_url(short_code: str) -> Optional[URLResponse]:
        """
        Get original URL by short code.

        Checks cache first, then database.

        Args:
            short_code: The short code to look up

        Returns:
            URLResponse if found and active, None otherwise
        """
        # Try cache first
        cached = await cache_get(short_code)
        if cached:
            logger.info(f"Cache hit for {short_code}")
            base_url = "http://localhost:8000"
            return URLResponse(
                short_code=short_code,
                original_url=cached,
                short_url=f"{base_url}/{short_code}",
                is_active=True,  # Assume active if cached
                created_at=None,  # Not in cache
            )

        # Cache miss - query database
        db: Session = next(get_db())
        try:
            db_url = (
                db.query(URLModel).filter(URLModel.short_code == short_code).first()
            )
            if not db_url or not db_url.is_active:
                return None

            base_url = "http://localhost:8000"
            response = URLResponse(
                short_code=db_url.short_code,
                original_url=db_url.original_url,
                short_url=f"{base_url}/{short_code}",
                is_active=db_url.is_active,
                created_at=db_url.created_at,
            )

            # Cache for future requests
            try:
                await cache_set(short_code, db_url.original_url)
            except Exception as e:
                logger.warning(f"Failed to cache URL: {e}")

            return response

        finally:
            db.close()

    @staticmethod
    def update_url(short_code: str, is_active: Optional[bool] = None) -> Optional[URLResponse]:
        """
        Update URL properties (currently only is_active).

        Args:
            short_code: The short code to update
            is_active: New active status

        Returns:
            Updated URLResponse or None if not found
        """
        db: Session = next(get_db())
        try:
            db_url = (
                db.query(URLModel).filter(URLModel.short_code == short_code).first()
            )
            if not db_url:
                return None

            if is_active is not None:
                db_url.is_active = is_active

            db.commit()
            db.refresh(db_url)

            base_url = "http://localhost:8000"
            return URLResponse(
                short_code=db_url.short_code,
                original_url=db_url.original_url,
                short_url=f"{base_url}/{short_code}",
                is_active=db_url.is_active,
                created_at=db_url.created_at,
            )
        finally:
            db.close()
```

- [ ] Copy and paste the following service `__init__.py` into `app/services/__init__.py`:

```python
from __future__ import annotations

from .url_service import URLService
from .cache import close_redis

__all__ = ["URLService", "close_redis"]
```

##### Step 4 Verification Checklist
- [ ] All service files exist with correct implementations
- [ ] Base62 encoding works (test manually if needed)
- [ ] Redis cache integration with fallback on errors
- [ ] Import test: `python -c "from app.services import URLService; print('OK')"`

#### Step 4 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

#### Step 5: API Routes
- [ ] Copy and paste the following router into `app/routes/urls.py`:

```python
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse

from app.schemas.url import URLCreate, URLResponse, URLUpdate
from app.services import URLService

router = APIRouter(prefix="/api/v1/urls", tags=["urls"])


@router.post(
    "/shorten",
    response_model=URLResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a shortened URL",
    responses={
        201: {"description": "URL shortened successfully"},
        400: {"description": "Invalid request"},
        500: {"description": "Internal server error"},
    },
)
async def shorten_url(request: Request, payload: URLCreate):
    """
    Create a shortened URL from an original URL.

    - **original_url**: The URL to shorten (must be a valid HTTP/HTTPS URL)
    - **custom_code**: Optional custom short code (alphanumeric, 6-10 chars)

    Returns the short code, short URL, and metadata.
    """
    try:
        response = await URLService.create_url(
            original_url=payload.original_url,
            custom_code=payload.custom_code,
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        import logging

        logging.error(f"Error shortening URL: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to shorten URL",
        )


@router.get(
    "/{short_code}",
    response_model=URLResponse,
    summary="Get URL details",
    responses={
        404: {"description": "URL not found"},
        500: {"description": "Internal server error"},
    },
)
async def get_url_details(short_code: str):
    """
    Get details of a shortened URL by short code.

    Returns the original URL, short URL, and metadata.
    """
    try:
        url = await URLService.get_url(short_code)
        if not url:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="URL not found"
            )
        return url
    except HTTPException:
        raise
    except Exception as e:
        import logging

        logging.error(f"Error fetching URL: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch URL",
        )


@router.put(
    "/{short_code}",
    response_model=URLResponse,
    summary="Update URL properties",
    responses={
        404: {"description": "URL not found"},
        500: {"description": "Internal server error"},
    },
)
async def update_url(short_code: str, payload: URLUpdate):
    """
    Update a shortened URL's properties.

    Currently supports:
    - **is_active**: Set URL active/inactive
    """
    try:
        url = URLService.update_url(
            short_code=short_code, is_active=payload.is_active
        )
        if not url:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="URL not found"
            )
        return url
    except HTTPException:
        raise
    except Exception as e:
        import logging

        logging.error(f"Error updating URL: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update URL",
        )


@router.delete(
    "/{short_code}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a shortened URL",
    responses={
        404: {"description": "URL not found"},
        500: {"description": "Internal server error"},
    },
)
async def delete_url(short_code: str):
    """
    Soft delete a shortened URL (sets is_active=False).

    This is a soft delete - the URL record remains in the database.
    """
    try:
        url = URLService.update_url(short_code=short_code, is_active=False)
        if not url:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="URL not found"
            )
        return None
    except HTTPException:
        raise
    except Exception as e:
        import logging

        logging.error(f"Error deleting URL: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete URL",
        )


@router.get(
    "/",
    response_model=list[URLResponse],
    summary="List all URLs",
    responses={
        500: {"description": "Internal server error"},
    },
)
async def list_urls(limit: int = 100, offset: int = 0):
    """
    List all shortened URLs (paginated).

    - **limit**: Maximum number of URLs to return (default 100)
    - **offset**: Number of URLs to skip (default 0)
    """
    from sqlalchemy.orm import Session
    from app.models import get_db

    db: Session = next(get_db())
    try:
        urls = db.query(URLModel).filter(URLModel.is_active == True).offset(offset).limit(limit).all()
        
        base_url = "http://localhost:8000"
        responses = []
        for url in urls:
            responses.append(
                URLResponse(
                    short_code=url.short_code,
                    original_url=url.original_url,
                    short_url=f"{base_url}/{url.short_code}",
                    is_active=url.is_active,
                    created_at=url.created_at,
                )
            )
        return responses
    finally:
        db.close()
```

- [ ] Create `app/routes/__init__.py` to import and expose the router:

```python
from __future__ import annotations

from .urls import router as url_router

__all__ = ["url_router"]
```

- [ ] Add redirect endpoint to `app/main.py` (insert after the health check endpoint):

```python
@app.get("/{short_code}")
async def redirect_to_original(short_code: str):
    """
    Redirect to original URL using short code.
    
    This is the main endpoint users will visit with the short URL.
    """
    url = await URLService.get_url(short_code)
    if not url:
        raise HTTPException(status_code=404, detail="URL not found")
    
    return RedirectResponse(url=url.original_url, status_code=302)
```

##### Step 5 Verification Checklist
- [ ] All route files exist with CRUD operations
- [ ] Redirect endpoint added to main.py
- [ ] Import test: `python -c "from app.routes import url_router; print('OK')"`

#### Step 5 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

#### Step 6: Dockerfile with Multi-Stage Build
- [ ] Copy and paste the following multi-stage Dockerfile into `Dockerfile`:

```dockerfile
# Stage 1: Build dependencies
FROM python:3.13-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install uv (Python package manager)
RUN pip install --no-cache-dir uv

# Install dependencies into virtual environment
RUN uv venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN uv pip install --no-cache-dir -e ".[dev]"


# Stage 2: Runtime image
FROM python:3.13-slim as runtime

WORKDIR /app

# Install runtime system dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy application code
COPY app/ ./app/
COPY main.py ./
COPY pyproject.toml ./

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

##### Step 6 Verification Checklist
- [ ] Dockerfile exists with multi-stage build
- [ ] Build test: `docker build -t url-shortener-api:test .` (should succeed)
- [ ] Image size check: `docker images url-shortener-api:test` (should be < 200MB)

#### Step 6 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

#### Step 7: Docker Compose Configuration
- [ ] Copy and paste the following docker-compose.yml:

```yaml
version: "3.8"

services:
  api:
    build: .
    container_name: url-shortener-api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+psycopg2://postgres:password@db:5432/url_shortener
      - REDIS_URL=redis://redis:6379/0
      - PYTHONUNBUFFERED=1
    volumes:
      - ./app:/app/app  # Mount for development hot-reload (optional)
    depends_on:
      - db
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 10s
    command: >
      sh -c "
        uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
      "

  db:
    image: postgres:16-alpine
    container_name: url-shortener-db
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=url_shortener
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: url-shortener-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

volumes:
  postgres_data:
  redis_data:
```

##### Step 7 Verification Checklist
- [ ] docker-compose.yml exists with all three services
- [ ] Volumes defined for persistence
- [ ] Health checks configured for all services
- [ ] Test: `docker-compose config` (should validate YAML syntax)

#### Step 7 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

#### Step 8: Configuration Files and Documentation
- [ ] Copy and paste the following requirements into `requirements.txt`:

```txt
fastapi==0.115.6
uvicorn[standard]==0.32.1
sqlalchemy==2.0.36
psycopg2-binary==2.9.10
pydantic==2.10.4
pydantic-settings==2.10.1
redis==5.2.1
```

- [ ] Copy and paste the following environment variables example into `.env.example`:

```env
# Database
DATABASE_URL=postgresql+psycopg2://postgres:password@localhost:5432/url_shortener

# Redis
REDIS_URL=redis://localhost:6379/0

# Application
BASE_URL=http://localhost:8000
SQL_ECHO=false
```

- [ ] Copy and paste the following README into `README.md`:

```markdown
# URL Shortener API

Simple URL shortening service built with FastAPI, SQLAlchemy, PostgreSQL, and Redis.

## Features

- Create shortened URLs with optional custom codes
- Base62 encoding for short codes (7 characters)
- Redis caching for fast lookups
- PostgreSQL for persistent storage
- Docker multi-stage build for optimized images
- Health check endpoint
- Full CRUD operations

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Make (optional)

### Running with Docker Compose

1. Build and start all services:

```bash
docker-compose up --build
```

2. The API will be available at: http://localhost:8000

3. API documentation (Swagger UI): http://localhost:8000/docs

### API Usage

#### Create a shortened URL

```bash
curl -X POST "http://localhost:8000/api/v1/urls/shorten" \
  -H "Content-Type: application/json" \
  -d '{"original_url": "https://example.com/very-long-url"}'
```

Response:

```json
{
  "short_code": "abc123X",
  "original_url": "https://example.com/very-long-url",
  "short_url": "http://localhost:8000/abc123X",
  "is_active": true,
  "created_at": "2025-01-15T10:30:00"
}
```

#### Redirect using short URL

Visit `http://localhost:8000/{short_code}` in browser or:

```bash
curl -L "http://localhost:8000/abc123X"
```

#### Get URL details

```bash
curl "http://localhost:8000/api/v1/urls/{short_code}"
```

#### List all URLs

```bash
curl "http://localhost:8000/api/v1/urls/?limit=10&offset=0"
```

#### Update URL

```bash
curl -X PUT "http://localhost:8000/api/v1/urls/{short_code}" \
  -H "Content-Type: application/json" \
  -d '{"is_active": false}'
```

#### Delete URL (soft delete)

```bash
curl -X DELETE "http://localhost:8000/api/v1/urls/{short_code}"
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Development

### Local Development (without Docker)

1. Install dependencies:

```bash
uv sync
```

2. Set up environment:

```bash
cp .env.example .env
# Edit .env with your database and redis settings
```

3. Run the API:

```bash
uvicorn app.main:app --reload
```

### Running Tests

```bash
pytest tests/
```

## Project Structure

```
.
├── app/
│   ├── models/
│   │   ├── __init__.py
│   │   └── database.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── url.py
│   │   └── errors.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── url_service.py
│   │   ├── cache.py
│   │   └── utils.py
│   ├── routes/
│   │   ├── __init__.py
│   │   └── urls.py
│   └── main.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+psycopg2://postgres:password@localhost:5432/url_shortener` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `BASE_URL` | Base URL for generating short URLs | `http://localhost:8000` |
| `SQL_ECHO` | Enable SQL query logging | `false` |

## Notes

- Short codes are generated using base62 encoding (7 characters)
- Redis is used for caching with 1-hour TTL
- Soft deletes are supported (is_active flag)
- Custom short codes must be alphanumeric (6-10 characters)
- Database: PostgreSQL 16
- Cache: Redis 7
```

##### Step 8 Verification Checklist
- [ ] All configuration files exist (requirements.txt, .env.example, README.md)
- [ ] README contains complete usage instructions
- [ ] Docker Compose can start all services: `docker-compose up --build`
- [ ] API accessible at http://localhost:8000
- [ ] Swagger docs accessible at http://localhost:8000/docs

#### Step 8 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

## Final Integration Test

After all steps are complete and committed:

1. **Full system test:**
```bash
# Stop any running containers
docker-compose down

# Build and start fresh
docker-compose up --build -d

# Wait for services to be healthy
docker-compose ps

# Test health endpoint
curl http://localhost:8000/health

# Test create-shorten-redirect flow
SHORTEN_RESPONSE=$(curl -X POST "http://localhost:8000/api/v1/urls/shorten" \
  -H "Content-Type: application/json" \
  -d '{"original_url": "https://google.com"}')
echo $SHORTEN_RESPONSE

# Extract short_code and test redirect
SHORT_CODE=$(echo $SHORTEN_RESPONSE | grep -o '"short_code":"[^"]*"' | cut -d'"' -f4)
curl -L "http://localhost:8000/$SHORT_CODE" -v
```

2. **Verify persistence:**
```bash
# Stop containers
docker-compose down

# Start again (data should persist via volumes)
docker-compose up -d

# Verify the URL still exists
curl "http://localhost:8000/api/v1/urls/$SHORT_CODE"
```

3. **Verify Redis cache:**
```bash
# Access Redis CLI
docker exec url-shortener-redis redis-cli

# Check cached key
KEYS url:*
GET url:{short_code}
```

## Success Criteria

- [ ] All API endpoints functional (create, read, update, delete, list, redirect)
- [ ] Database persists across container restarts
- [ ] Redis cache working (check logs for cache hits)
- [ ] Docker image size optimized (< 200MB)
- [ ] Health checks passing for all services
- [ ] Swagger UI accessible and functional
- [ ] No errors in logs
- [ ] All steps committed to git

## Technology Stack

- **Backend Framework:** FastAPI 0.115+
- **Python:** 3.13+
- **ORM:** SQLAlchemy 2.0+
- **Database:** PostgreSQL 16
- **Cache:** Redis 7
- **Containerization:** Docker + Docker Compose
- **Validation:** Pydantic 2.10+
- **Server:** Uvicorn with standard extras

## Dependencies

See `requirements.txt` for exact versions.

## Branch Information

- **Feature Branch:** `url-shortener-api`
- **Created from:** `main`
- **Planner:** `plans/url-shortener-api/plan.md`
- **Implementation Guide:** `plans/url-shortener-api/implementation.md` (this file)