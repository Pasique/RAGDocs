# Operação do RAGDocs

## Ambiente e comandos

Use Python 3.11 e um ambiente `.venv` exclusivo deste projeto. A raiz Git pode estar
no diretório pai; execute comandos do aplicativo nesta pasta.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m ragdocs.cli build-index
streamlit run app.py
pytest
ruff check .
python -m ragdocs.cli evaluate --live
```

## Estrutura e convenções

Código em `src/ragdocs`, interface em `app.py`, corpus atribuído em `corpus` e testes
em `tests`. Contratos compartilhados ficam em `contracts.py`. Preserve a separação
entre falhas técnicas e contexto documental insuficiente. Nunca execute exemplos
de código carregados como documentação. Não habilite memória de conversa no RAG.

## Ferramentas e credenciais

Tarefas, prompts de desenvolvimento, handoffs e planejamento ficam exclusivamente
em `documentacao_projeto/`, ignorada pelo Git. Estado dos agentes, worktrees,
ambientes e resultados locais também são ignorados. Prompts do produto são código.
Nunca registre chaves ou corpos de erros do provedor contendo dados sensíveis.
Carregue somente o `.env` desta pasta; não procure credenciais de outros projetos.

Testes comuns são offline. Avaliação paga exige `--live` e chave dedicada com limite
total de US$ 1 no OpenRouter, sem renovação. Não execute chamadas reais em paralelo,
não altere o teto financeiro e não repita automaticamente avaliações concluídas.

## Verificação e Git

Execute testes pertinentes e `ruff check .` antes da integração. Use commits
significativos no padrão Conventional Commits e adicione somente arquivos deste
projeto. Verifique `git diff --cached --name-only` e regras com `git check-ignore`.
Não versione planejamento local, índices, caches ou credenciais. Não altere arquivos
de outros projetos no repositório pai. A extração do histórico usa `git subtree split`
a partir da raiz Git e deve ser validada em um clone fora do repositório pai.
