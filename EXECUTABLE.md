# Executavel Deep Agents

Guia para uso do binario `deep-agents` sem Python local.

## Requisitos

- Linux x86_64 (ou Windows se o binario foi buildado em Windows)
- Variaveis de ambiente configuradas

## Leitura de .env

Ordem de busca:

1. diretorio atual
2. `PROJECT_ROOT/.env` (se definido)
3. pasta do executavel (quando empacotado)

## Variaveis de Ambiente

Obrigatorias (dependendo do provider):

- `OPENROUTER_API_KEY` (provider default)
- `OPENAI_API_KEY` (fallback para OpenRouter)
- `OLLAMA_BASE_URL` (quando usar ollama)

Opcionais:

- `PROJECT_ROOT`
- `THREAD_ID_DEFAULT` (default: `helo-ai-cli-session`)
- `ENABLE_LANGSMITH`
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`

Config por role:

- `PLANNER__PROVIDER`, `PLANNER__MODEL`, `PLANNER__TEMPERATURE`
- `GENERATOR__PROVIDER`, `GENERATOR__MODEL`, `GENERATOR__TEMPERATURE`
- `IMPLEMENTER__PROVIDER`, `IMPLEMENTER__MODEL`, `IMPLEMENTER__TEMPERATURE`

## Fluxos Suportados

### 1. Pipeline automatico por orquestracao

- planner -> generator -> implementer
- planner/generator devem criar `plans/{feature-name}/plan.md` e `plans/{feature-name}/implementation.md`
- implementer le `implementation.md` para executar

### 2. Conversa natural

- prompts casuais (ex: `oi`) respondem normalmente
- sem acionar roles

### 3. Manual por role

- chamada direta por role (`role planner|generator|implementer`)
- slash command no chat (`/planner ...`, `/generator ...`, `/implementer ...`)

*O comando `role` fica em modo interativo por padrão para responder perguntas de clarificação do agente.*
*Use `--no-interactive-followup` para executar apenas uma vez sem aguardar respostas.*

## Implementer: comportamento

Modo padrao (sem `--auto`):

- para em checkpoints de STOP & COMMIT
- solicita commit/push manual do usuario
- so continua com `continue`

Modo `--auto`:

- remove pausas HITL
- executa continuo ate concluir

Observacao:

- o agente nao faz git automaticamente

## Comandos com Executavel

Assumindo binario em `./dist/deep-agents`:

```bash
# Ajuda e diagnostico
./dist/deep-agents --help
./dist/deep-agents doctor
./dist/deep-agents models
./dist/deep-agents skills

# Chat (pipeline)
./dist/deep-agents chat
./dist/deep-agents chat --prompt "Plan refactor for auth service"
./dist/deep-agents chat --prompt "Plan refactor for auth service" --auto
./dist/deep-agents chat --prompt "Plan refactor for auth service" --verbose

# Conversa natural (sem roles)
./dist/deep-agents chat --prompt "oi"

# Slash command no chat
./dist/deep-agents chat --prompt "/planner implementar feature xpto"
./dist/deep-agents chat --prompt "/generator gerar guia de implementacao"
./dist/deep-agents chat --prompt "/implementer executar plano atual"

# Single-shot
./dist/deep-agents run "Implement tests for parser"
./dist/deep-agents run "Implement tests for parser" --auto

# Manual por role
./dist/deep-agents role planner "Criar plano da feature XPTO"
./dist/deep-agents role planner "Criar plano da feature XPTO" --no-interactive-followup
./dist/deep-agents role generator "Gerar implementation guide com checkpoints"
./dist/deep-agents role implementer "Executar implementation atual"
./dist/deep-agents role implementer "Executar implementation atual" --auto
./dist/deep-agents role implementer "Executar implementation atual" --verbose
```

## Build do Executavel

```bash
uv sync
.venv/bin/pyinstaller --onefile --name deep-agents main.py
```

Saida esperada:

- `dist/deep-agents`

## Lint & Typecheck (local)

Mesmo em modo "executavel", voce pode checar a qualidade do codigo antes de buildar:

```bash
uv run ruff check .
uv run mypy --no-incremental .
```

## Troubleshooting

Permissao:

```bash
chmod +x dist/deep-agents
```

Erro de API key:

- confira `OPENROUTER_API_KEY` ou `OPENAI_API_KEY`

Erro de modelo/provider:

- valide `*_PROVIDER` e `*_MODEL`
