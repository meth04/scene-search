class RetrievalError(Exception):
    code = "RETRIEVAL_ERROR"
    status_code = 422

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class NotFoundError(RetrievalError):
    code = "NOT_FOUND"
    status_code = 404


class ForbiddenError(RetrievalError):
    code = "FORBIDDEN"
    status_code = 403


class ConflictError(RetrievalError):
    code = "VERSION_CONFLICT"
    status_code = 409


class ContractError(RetrievalError):
    code = "MODEL_INDEX_INCOMPATIBLE"
    status_code = 409


class IdempotencyConflictError(ConflictError):
    code = "IDEMPOTENCY_KEY_REUSED"


class ExportValidationError(RetrievalError):
    code = "EXPORT_NOT_ELIGIBLE"


class EmbeddingContractMismatchError(ContractError):
    code = "EMBEDDING_CONTRACT_MISMATCH"


class EncoderNotReadyError(RetrievalError):
    code = "ENCODER_NOT_READY"
    status_code = 503


class SearchBusyError(RetrievalError):
    code = "SEARCH_BUSY"
    status_code = 503
