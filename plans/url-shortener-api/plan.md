# URL Shortener API

**Branch:** `feat/url-shortener-api`
**Description:** Implementa uma API de encurtador de URLs usando FastAPI, SQLAlchemy e PostgreSQL com Redis para cache.

## Goal
Criar uma API RESTful de encurtador de URLs que recebe uma URL longa e retorna uma versão curta, com redirecionamento 301 e métricas de acesso. A solução usa FastAPI, SQLAlchemy ORM, PostgreSQL para persistência, Redis para cache e contador de acessos, seguindo padrão Modular Layout.

## Implementation Steps

### Step 1: Estrutura do Projeto e Dependências
**Files:** `poc/requirements.txt`, `poc/.env.example`
**What:** Criar estrutura modular `poc/app/` com subpastas `models/`, `schemas/`, `services/`, `routes/` e arquivos de configuração. Definir dependências no requirements.txt.
**Testing:** Verificar que `poc/` contém a estrutura de pastas e `requirements.txt` lista: fastapi, uvicorn, sqlalchemy, psycopg2-binary, redis, pydantic-settings, python-dotenv.

### Step 2: Modelos de Banco de Dados e Configuração
**Files:** `poc/app/models/base.py`, `poc/app/models/models.py`, `poc/app/core/database.py`, `poc/app/core/config.py`
**What:** Definir modelo SQLAlchemy `URL` com campos: id, short_code (único), original_url, created_at, access_count. Criar configuração Pydantic Settings para variáveis de ambiente. Implementar conexão com PostgreSQL.
**Testing:** Testar conexão com banco executando `python -c "from poc.app.core.database import engine; engine.connect()"` dentro do container.

### Step 3: Schemas Pydantic
**Files:** `poc/app/schemas/url.py`
**What:** Criar schemas para validação: `URLCreate` (original_url), `URLResponse` (short_code, original_url, created_at, access_count), `URLStats` (access_count).
**Testing:** Validar que schemas rejeitam URLs inválidas e aceitam URLs válidas com testes manuais ou pytest.

### Step 4: Serviço de Encurtamento
**Files:** `poc/app/services/url_service.py`
**What:** Implementar lógica de geração de short_code (base62 ou hash), verificar colisões, salvar no banco, e incrementar contador de acessos. Integrar com Redis para cache de short_code → original_url.
**Testing:** Testar geração de códigos únicos e colisões com função unitária.

### Step 5: Rotas da API
**Files:** `poc/app/routes/url_routes.py`, `poc/app/main.py`
**What:** Criar endpoints: `POST /shorten` (cria encurtamento), `GET /{short_code}` (redireciona 301), `GET /{short_code}/stats` (retorna estatísticas). Configurar FastAPI com middlewares e CORS.
**Testing:** Usar curl ou pytest-httpserver para testar: `curl -X POST http://localhost:8000/shorten -d '{"original_url": "https://example.com"}'`.

### Step 6: Dockerfile Multi-stage
**Files:** `poc/Dockerfile`
**What:** Criar Dockerfile com multi-stage: stage builder instala dependências, stage runner copia apenas necessário. Usar imagem slim (python:3.13-slim). Expor porta 8000.
**Testing:** Construir imagem: `docker build -t url-shortener -f poc/Dockerfile .` e verificar tamanho reduzido.

### Step 7: Docker Compose
**Files:** `poc/docker-compose.yml`
**What:** Definir serviços: `api` (com build do Dockerfile), `postgres` (imagem oficial, volume para dados), `redis` (imagem oficial). Configurar variáveis de ambiente, networks, healthchecks. Volumes: `postgres_data`, `redis_data`.
**Testing:** Executar `docker-compose -f poc/docker-compose.yml up --build` e verificar que todos os serviços sobem.

### Step 8: README e Documentação
**Files:** `poc/README.md`
**What:** Documentar: pré-requisitos, como executar com docker-compose, endpoints da API, variáveis de ambiente, troubleshooting.
**Testing:** Seguir instruções do README em ambiente limpo para validar.

## Validation Checkpoints

1. **Estrutura**: `poc/app/{models,schemas,services,routes,core}/` existe com arquivos __init__.py
2. **Dependências**: `pip install -r poc/requirements.txt` instala sem erros
3. **Banco**: `docker-compose up postgres` permite conexão via SQLAlchemy
4. **API**: `POST /shorten` retorna short_code válido e salva no banco
5. **Redirecionamento**: `GET /{short_code}` retorna 301 com Location header correto
6. **Cache**: Redis armazena e recupera URLs sem consultar PostgreSQL repetidamente
7. **Docker**: `docker-compose up --build` sobe todos os serviços sem erros
8. **Persistência**: Parar e remover containers, recriar com `docker-compose up` mantém dados
