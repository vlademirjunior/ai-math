# Executável Helo

Este diretório contém o executável `deep-agents` (ou `deep-agents.exe` no Windows) que permite executar a CLI Deep Agents sem necessidade de Python instalado.

## 📋 Requisitos

- Sistema operacional Linux (ou Windows se construído no Windows)
- Variáveis de ambiente configuradas (veja abaixo)

## 🔧 Variáveis de Ambiente

O executável lê variáveis de ambiente do sistema ou de um arquivo `.env` no diretório atual.

Ordem de busca do `.env`:

1. Diretório atual de execução (`pwd`)
2. `PROJECT_ROOT/.env` (se `PROJECT_ROOT` estiver definido)
3. Pasta do executável (quando empacotado com PyInstaller)

### Variáveis Obrigatórias

Pelo menos uma das seguintes chaves API deve ser configurada, dependendo do provedor usado:

- `OPENROUTER_API_KEY` - Para usar o provedor OpenRouter (padrão)
- `OPENAI_API_KEY` - Também funciona como fallback para o provedor OpenRouter
- `OLLAMA_BASE_URL` - Para usar o provedor Ollama local (padrão: http://localhost:11434)

### Variáveis Opcionais

- `PROJECT_ROOT` - Caminho do projeto (padrão: diretório atual)
- `THREAD_ID_DEFAULT` - ID da thread de conversa (padrão: "deep-agents-session")
- `ENABLE_LANGSMITH` - Habilitar tracing do LangSmith (true/false)
- `LANGSMITH_API_KEY` - API key do LangSmith (obrigatória se ENABLE_LANGSMITH=true)
- `LANGSMITH_PROJECT` - Nome do projeto no LangSmith

### Configuração de Modelos

As configurações de modelos podem ser sobrescritas com variáveis de ambiente usando o delimitador `__`:

- `PLANNER__PROVIDER` - Provedor do planner (openrouter, ollama, litellm)
- `PLANNER__MODEL` - Nome do modelo para o planner
- `PLANNER__TEMPERATURE` - Temperatura do planner (0.0-1.0)
- `GENERATOR__PROVIDER` - Provedor do generator
- `GENERATOR__MODEL` - Modelo do generator
- `GENERATOR__TEMPERATURE` - Temperatura do generator
- `IMPLEMENTER__PROVIDER` - Provedor do implementer
- `IMPLEMENTER__MODEL` - Modelo do implementer
- `IMPLEMENTER__TEMPERATURE` - Temperatura do implementer

Exemplo:
```bash
export PLANNER__MODEL="gpt-4o"
export GENERATOR__TEMPERATURE=0.2
```

## 📁 Arquivo .env (Alternativa)

Crie um arquivo `.env` no diretório onde o executável será rodado:

```env
OPENROUTER_API_KEY=sua_chave_aqui
# ou use OPENAI_API_KEY=sua_chave_aqui
PLANNER__MODEL=stepfun/step-3.5-flash:free
GENERATOR__MODEL=stepfun/step-3.5-flash:free
IMPLEMENTER__MODEL=stepfun/step-3.5-flash:free
PROJECT_ROOT=.
```

## 🚀 Como Usar

### 1. Comandos Básicos

```bash
# Ver ajuda
./deep-agents --help

# Testar configuração
./deep-agents doctor

# Listar modelos disponíveis
./deep-agents models

# Iniciar chat interativo
./deep-agents chat

# Executar tarefa única
./deep-agents run "Implement a simple calc.py"

# Gerenciar skills
./deep-agents skills
```

### 2. Usando com Variáveis de Ambiente

```bash
# Definir variáveis diretamente na linha de comando
OPENROUTER_API_KEY=sua_chave ./deep-agents doctor
# ou
OPENAI_API_KEY=sua_chave ./deep-agents doctor

# Usar arquivo .env
cp .env.example .env
./deep-agents doctor

# Verificar qual .env foi carregado
# (campo "dotenv_path" no JSON)
```

### 3. Modo de Tracing (Debug)

Para habilitar tracing no LangSmith:

```bash
export ENABLE_LANGSMITH=true
export LANGSMITH_API_KEY=sua_chave_langsmith
export LANGSMITH_PROJECT=deep-agents-cli
./deep-agents doctor
```

## 📦 Distribuição

O executável está localizado em `/workspace/dist/deep-agents`.

Para distribuir:
1. Copie o arquivo `deep-agents` para o sistema alvo
2. Garanta que tenha permissão de execução: `chmod +x deep-agents`
3. Configure as variáveis de ambiente ou arquivo `.env`
4. Execute: `./deep-agents [COMANDO]`

## 🔨 Construção

Para reconstruir o executável:

```bash
# Instalar dependências
uv sync

# Build com PyInstaller
.venv/bin/pyinstaller --onefile --name deep-agents main.py
```

O executável será gerado em `dist/deep-agents`.

## ⚠️ Notas

- O executável é específico para Linux x86_64
- O tamanho do executável é aproximadamente 64MB devido às dependências
- Todas as variáveis de ambiente do sistema são respeitadas
- O arquivo `.env` é lido do diretório de trabalho atual
- Para Windows, o build deve ser feito no Windows

## 🐛 Troubleshooting

### Erro de permissão
```bash
chmod +x dist/deep-agents
```

### Erro de API key
Certifique-se de que a variável de ambiente `OPENROUTER_API_KEY` ou `OPENAI_API_KEY` está definida.

### Erro de modelo não encontrado
Verifique se o nome do modelo está correto e se o provedor está configurado adequadamente.

### Executável muito grande
Isso é normal - o PyInstaller empacota todo o Python e dependências. Considere usar um servidor com o Python instalado se o tamanho for problema.
