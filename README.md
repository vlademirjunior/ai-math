Para dominar o **uv** por completo, você precisa conhecer não só os comandos de instalação, mas também os de manutenção e limpeza. O `uv` é uma "canivete suíço" que substitui o `pip`, `pip-tools`, `pipx`, `poetry`, `pyenv` e até o `virtualenv`.

Aqui está a lista exaustiva dos comandos mais úteis dividida por categoria:

---

### 1. Gestão de Fluxo de Trabalho (Project Workflow)

Estes são os comandos modernos para quem trabalha com o `pyproject.toml`.

* **`uv init`**: Inicializa um novo projeto na pasta atual.
* **`uv init --lib`**: Inicializa um projeto focado em biblioteca (cria um `src/`).
* **`uv add [pacote]`**: Adiciona dependência ao projeto.
* `--dev`: Adiciona como dependência de desenvolvimento.
* `--editable`: Adiciona o pacote atual em modo editável.


* **`uv remove [pacote]`**: Remove a dependência e limpa o ambiente.
* **`uv sync`**: Garante que o `.venv` está **exatamente** igual ao `uv.lock`. Se você deletar a pasta `.venv`, esse comando a recria em milissegundos.
* **`uv tree`**: Exibe a árvore de dependências do seu projeto (ótimo para achar conflitos).

### 2. Gestão de Versões do Python

O `uv` baixa binários oficiais do Python (da fonte *indygreg*) automaticamente.

* **`uv python install`**: Instala a versão mais recente do Python.
* **`uv python install 3.10 3.11 3.12`**: Instala múltiplas versões de uma vez.
* **`uv python list`**: Mostra todas as versões instaladas no seu PC e as disponíveis para download.
* **`uv python find`**: Localiza onde está o executável do Python que o uv está usando.
* **`uv python pin 3.12`**: Trava o projeto atual em uma versão específica do Python.

### 3. Execução de Ferramentas e Scripts (O estilo "pipx")

Útil para rodar ferramentas sem instalá-las globalmente no sistema.

* **`uvx [ferramenta]`**: Atalho para `uv tool run`. Roda algo como `ruff`, `black` ou `httpie` de forma isolada.
* **`uv run [script.py]`**: Roda um script Python garantindo que as dependências do projeto estejam presentes.
* **`uv run --with requests script.py`**: Roda um script único, criando um ambiente temporário com `requests` apenas para essa execução.
* **`uv tool install [pacote]`**: Instala uma ferramenta globalmente em um ambiente isolado (ex: `uv tool install yt-dlp`).
* **`uv tool list`**: Lista as ferramentas instaladas via `uv tool`.

### 4. Compatibilidade com Pip (Low-level)

Para quando você precisa de controle granular ou está em servidores CI/CD.

* **`uv pip install .`**: Instala o pacote do diretório atual.
* **`uv pip compile requirements.in -o requirements.txt`**: Transforma uma lista de dependências soltas em um arquivo travado (estilo `pip-tools`).
* **`uv pip sync requirements.txt`**: Faz o ambiente virtual espelhar exatamente o arquivo TXT (remove o que não estiver lá).
* **`uv pip list`**: Lista o que está instalado no ambiente virtual atual.

### 5. Manutenção e Performance

O `uv` é rápido porque usa cache e links inteligentes. Às vezes você quer limpar isso.

* **`uv cache clean`**: Remove todo o cache de pacotes e downloads de Python.
* **`uv cache prune`**: Remove apenas entradas de cache antigas ou não utilizadas.
* **`uv cache dir`**: Mostra onde o `uv` guarda os gigabytes de pacotes.
* **`uv self update`**: Atualiza o próprio executável do `uv` para a versão mais nova.

---

### Resumo de Ouro

| Se você quer... | Use este comando: |
| --- | --- |
| **Começar um projeto** | `uv init` |
| **Instalar um pacote** | `uv add nome-do-pacote` |
| **Rodar seu código** | `uv run main.py` |
| **Rodar um CLI (ex: ruff)** | `uvx ruff` |
| **Trocar a versão do Python** | `uv python install 3.13` |    
## Tracing (default off)

By default, tracing is disabled and the process enforces:

- `LANGSMITH_TRACING=false`

To enable tracing:

```bash
export ENABLE_LANGSMITH=true
export LANGSMITH_API_KEY=your_key
export LANGSMITH_PROJECT=deep-agents-cli
```

Then run:

```bash
uv run python main.py doctor
```

You should see `"langsmith_enabled": true` and `"langsmith_env": "true"`.

## Commands

```bash
uv run python main.py chat
uv run python main.py chat --prompt "Plan refactor for auth service"
uv run python main.py run "Implement tests for parser"
uv run python main.py doctor
uv run python main.py models
uv run python main.py skills
```

## Non-interactive usage

Use `run` or `chat --prompt` in CI scripts for deterministic single-shot execution.

## 🚀 Executável Standalone

Você também pode usar o executável pré-construído que não requer Python instalado:

```bash
# Usar o executável diretamente
./dist/deep-agents doctor
./dist/deep-agents chat
./dist/deep-agents run "Sua tarefa aqui"
```

O executável lê variáveis de ambiente ou arquivo `.env` da mesma forma que a versão Python. Veja `EXECUTABLE.md` para instruções completas.

### Configuração Rápida

```bash
# Copie o exemplo de .env
cp .env.example .env

# Edite o arquivo .env com sua API key do OpenRouter
# Em seguida, execute:
./dist/deep-agents doctor
```

Para mais detalhes sobre como usar o executável, consulte [EXECUTABLE.md](EXECUTABLE.md).
