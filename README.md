# Deep Agents CLI (One-File Runtime)

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

Arquivos salvos em `plans/{feature-name}/`:

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

1. Planner roda e cria `plans/{feature-name}/plan.md`.
2. Generator roda e cria `plans/{feature-name}/implementation.md`.
3. Implementer sempre usa `implementation.md` para implementar.

O `feature-name` eh derivado do prompt (slug) e mantido por `thread_id` para as proximas etapas.

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

# Execucao single-shot
uv run python main.py run "Implement tests for parser"

# Execucao single-shot em auto
uv run python main.py run "Implement tests for parser" --auto

# Manual por role
uv run python main.py role planner "Criar plano da feature XPTO"
uv run python main.py role generator "Gerar implementation guide com checkpoints"
uv run python main.py role implementer "Executar implementation atual"
uv run python main.py role implementer "Executar implementation atual" --auto

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

## Executavel Standalone

Uso do binario pre-build esta documentado em `EXECUTABLE.md`.
