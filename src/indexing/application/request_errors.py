"""Application rejection details translated to HTTP only at the API boundary."""


class IndexingRequestError(Exception):
    """Carry a stable status and message for existing ingestion endpoints.

    Attributes:
        status_code: Existing endpoint rejection status.
        detail: Existing user-facing error message.
    """

    def __init__(self, *, status_code, detail):
        """Record an ingestion rejection without depending on FastAPI.

        Args:
            status_code: Response status to preserve at the HTTP boundary.
            detail: Description explaining why the request was rejected.
        """
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
