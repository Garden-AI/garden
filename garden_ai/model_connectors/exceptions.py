class ConnectorInvalidRevisionError(Exception):
    """Raised when the connector can't infer the commit revision."""

    def __init__(self, original_exception, explanation=None):
        super().__init__(
            f"Failed to fetch commmit hash: {original_exception}"
            + (f" - {explanation}" if explanation else "")
        )
