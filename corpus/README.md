# Coleção documental congelada

Seis fontes oficiais, em inglês, distribuídas sob MIT. O manifesto registra as
URLs imutáveis, revisões Git e SHA-256 dos textos efetivamente indexados. Licenças
originais completas em `licenses/`; preserve os avisos dos autores.

Os textos LangChain pertencem a LangChain (2025); os textos FAISS pertencem a Meta
Platforms, Inc., com os avisos dos exemplos preservados.

## Preparação

MDX foi convertido em Markdown: front matter e imports de componentes foram
removidos; blocos JavaScript/TypeScript, diagramas Mermaid e tags de apresentação
foram omitidos. A prosa compartilhada e os blocos Python foram preservados.
Componentes externos de seleção de embeddings/vector store não foram expandidos.
Os tutoriais Python FAISS foram envelopados em cercas de código, sem execução.

Na revisão congelada, `src/oss/langchain/retrieval.mdx` redireciona para
`src/oss/deepagents/retrieval.mdx`. A fonte de recuperação usa esse destino na
mesma revisão, cuja URL está registrada no manifesto.

Exemplos podem descrever APIs, ferramentas e modelos além do MVP. Eles são apenas
documentação consultada e nunca executados pela aplicação. Nenhuma atualização
de fontes ocorre durante perguntas.
