"""Erros operacionais com mensagens apropriadas para a interface."""


class RagDocsError(Exception):
    """Falha técnica; nunca equivale a conteúdo documental insuficiente."""


class ConfigurationError(RagDocsError):
    pass


class CorpusError(RagDocsError):
    pass


class DocumentError(RagDocsError):
    """PDF inválido ou coleção fora dos limites, antes das chamadas de embeddings."""

    pass


class IndexMissingError(RagDocsError):
    pass


class IndexIncompatibleError(RagDocsError):
    pass


class ProviderError(RagDocsError):
    pass


class AuthenticationError(ProviderError):
    pass


class CreditError(ProviderError):
    pass


class ProviderTimeoutError(ProviderError):
    pass


class ProviderUnavailableError(ProviderError):
    pass


class InvalidResponseError(RagDocsError):
    pass


class QuestionError(RagDocsError):
    pass
