# Helo CLI (AI)

CLI para orquestrar um pipeline de 3 fases com roles especializados:

- planner
- generator
- implementer

A implementacao principal fica em `main.py` (one-file). Testes ficam em `tests/`.

## Pipeline

Fluxo de engenharia (quando acionado):

1. planner
2. generator
3. implementer

Arquivos esperados em `plans/{feature-name}/` (criados pelos roles via tools do deep agent):

- `plan.md` (saida do planner)
- `implementation.md` (saida do generator)

O implementer le `implementation.md` dessa pasta para executar o plano.

## Modos de Execucao

### 1. Orquestrado (padrao)

Executa planner -> generator -> implementer automaticamente quando o prompt indica tarefa de engenharia.

Exemplo de conversa natural:

- `oi` -> resposta natural, sem acionar roles.

Exemplo de tarefa de engenharia:

- `implementar feature de auth` -> aciona pipeline.

### 2. Manual por role

Permite chamar um role especifico:

- comando `role`
- slash command no chat (`/planner`, `/generator`, `/implementer`)
*O comando `role` fica em modo interativo por padrão, permitindo responder perguntas de clarificação do agente durante a execução.*
*Use `--no-interactive-followup` para executar apenas uma vez sem aguardar respostas.*

## Implementer: HITL e Auto

Sem `--auto`:

- o implementer pausa em checkpoints de STOP & COMMIT
- pede para o usuario validar, commitar e fazer push manualmente
- continua somente quando o usuario envia `continue`

Com `--auto`:

- remove o modo human-in-the-loop
- executa 100% continuo ate finalizar

Importante:

- o agente nao executa git automaticamente
- apenas orienta quando pausar/continuar

## Fluxo de Arquivos (plans)

1. Planner deve criar `plans/{feature-name}/plan.md`.
2. Generator deve criar `plans/{feature-name}/implementation.md`.
3. Implementer sempre usa `implementation.md` para implementar.

O nome da pasta deve vir da estrategia do planner/skills e nao de criacao manual no runtime.

## MCP Integration

O CLI agora detecta servidores MCP automaticamente a partir destes arquivos do repositorio:

- `.agents/mcp.json`
- `.vscode/mcp.json`

Tambem e possivel configurar via variaveis de ambiente com nested settings:

```env
MCP_SERVERS__DOCS__TRANSPORT=streamable_http
MCP_SERVERS__DOCS__URL=https://docs.langchain.com/mcp
MCP_SERVERS__DOCS__TIMEOUT=30
```

Config suportada por servidor:

- `transport`: `stdio`, `streamable_http`, `sse`, `websocket`
- `command` + `args` para `stdio`
- `url` + `headers` para transportes de rede
- `timeout` em segundos

Exemplo `stdio`:

```json
{
	"servers": {
		"local-tools": {
			"transport": "stdio",
			"command": "python",
			"args": ["tools/mcp_server.py"],
			"timeout": 30
		}
	}
}
```

Exemplo `streamable_http`:

```json
{
	"servers": {
		"Docs by LangChain": {
			"transport": "streamable_http",
			"url": "https://docs.langchain.com/mcp",
			"timeout": 30
		}
	}
}
```

Prioridade de contexto aplicada no prompt:

1. Arquivos/pastas referenciados com `#context` (limite `MAX_CONTEXT_FILE_CHARS=12000`)
2. Skills em `.agents/**/SKILL.md` (mesmo budget do item anterior)
3. Recursos MCP (limite separado `MAX_MCP_CONTEXT_CHARS=6000`)

### Autenticação de servidores MCP

Alguns servidores MCP (ex: GitHub Copilot) exigem **Autorização (Bearer token)**.
Se você receber `HTTP 401 Unauthorized`, adicione o cabeçalho `Authorization` no seu `mcp.json`:

```json
{
  "servers": {
    "github": {
      "transport": "streamable_http",
      "url": "https://api.githubcopilot.com/mcp/",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN_HERE"
      }
    }
  }
}
```

Também é possível passar o header via variável de ambiente (exemplo genérico):

```bash
export MCP_SERVERS__github__HEADERS__Authorization="Bearer YOUR_TOKEN_HERE"
```

Mensagens de status MCP:

- `✓ MCP servers: ...` quando ha servidores online
- `⚠ MCP servers: ... (offline)` quando parte dos servidores esta indisponivel
- `[MCP] All MCP servers offline - skipping MCP context` quando nao ha disponibilidade

Erros de conexao MCP nao interrompem execucao. O runtime faz fallback automatico e segue sem contexto MCP.

## Requisitos

- Python 3.13+
- uv
- chave de modelo (OpenRouter/OpenAI, ou provedor configurado)

## Setup Rapido

```bash
uv sync
cp .env.example .env  # se existir no projeto
```

Exemplo de variaveis:

```env
OPENROUTER_API_KEY=your_key
PLANNER__MODEL=stepfun/step-3.5-flash:free
GENERATOR__MODEL=stepfun/step-3.5-flash:free
IMPLEMENTER__MODEL=stepfun/step-3.5-flash:free
PROJECT_ROOT=.
```

## Comandos (Python)

```bash
# Chat interativo (pipeline padrao)
uv run python main.py chat

# Pipeline em uma execucao
uv run python main.py chat --prompt "Plan refactor for auth service"

# Pipeline totalmente automatico (sem HITL no implementer)
uv run python main.py chat --prompt "Plan refactor for auth service" --auto

# Pipeline com log detalhado (verbose)
uv run python main.py chat --prompt "Plan refactor for auth service" --verbose

# Execucao single-shot
uv run python main.py run "Implement tests for parser"

# Execucao single-shot em auto
uv run python main.py run "Implement tests for parser" --auto

# Execucao single-shot em modo verbose
uv run python main.py run "Implement tests for parser" --verbose

# Manual por role
uv run python main.py role planner "Criar plano da feature XPTO"
uv run python main.py role planner "Criar plano da feature XPTO" --no-interactive-followup
uv run python main.py role generator "Gerar implementation guide com checkpoints"
uv run python main.py role implementer "Executar implementation atual"
uv run python main.py role implementer "Executar implementation atual" --auto
uv run python main.py role implementer "Executar implementation atual" --verbose

# Slash command no chat (manual por role)
uv run python main.py chat --prompt "/planner implementar feature xpto"
uv run python main.py chat --prompt "/generator gerar guia de implementacao"
uv run python main.py chat --prompt "/implementer executar plano atual"

# Conversa natural (nao aciona roles)
uv run python main.py chat --prompt "oi"

# Utilitarios
uv run python main.py doctor
uv run python main.py models
uv run python main.py skills
```

## Tracing (LangSmith)

Tracing fica desabilitado por padrao.

Para habilitar:

```bash
export ENABLE_LANGSMITH=true
export LANGSMITH_API_KEY=your_key
export LANGSMITH_PROJECT=deep-agents-cli
uv run python main.py doctor
```

## Testes

```bash
uv run pytest
```

## Lint & Typecheck

```bash
uv run ruff check .
uv run mypy --no-incremental . # execução limpa, sem cache!
```

## Executavel Standalone

Uso do binario pre-build esta documentado em `EXECUTABLE.md`.
