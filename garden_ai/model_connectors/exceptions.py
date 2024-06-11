class ConnectorRevisionError(Exception):
    """Raised when there is an issue with the revision"""

    def __init__(self, original_exception, explanation=None):
        super().__init__(
            f"Error with git revision: {original_exception}"
            + (f" - {explanation}" if explanation else "")
        )


class ConnectorInvalidRevisionError(ConnectorRevisionError):
    """Raised when the connector can't infer the commit revision."""

    def __init__(self, original_exception, explanation=None):
        super().__init__(
            f"Failed to fetch commmit hash: {original_exception}"
            + (f" - {explanation}" if explanation else "")
        )


class ConnectorLFSError(Exception):
    """Raised when a git-lfs file is detected in the repo"""

    def __init__(self):
        super().__init__(
            "git-lfs detected. We do not support git-lfs from GitHub. Please consider using HuggingFace for large files."
        )


class ConnectorAPIError(Exception):
    """Raised when a connector has an issue communicating with a remote API."""

    def __init__(self, original_exception, explanation=None):
        super().__init__(
            f"Failed to communicate with API: {original_exception}"
            + (f" - {explanation}" if explanation else "")
        )


class ConnectorStagingError(Exception):
    """Raised when a model connector fails to stage."""

    def __init__(self, connected_repository, original_exception, explanation=None):
        super().__init__(
            f"Failed to download model from {connected_repository}: {original_exception}"
            + (f" - {explanation}" if explanation else "")
        )


class UnsupportedConnectorError(Exception):
    """Raised when trying to connect to an unsupported repository type."""

    pass


class ConnectorInvalidRepoIdError(Exception):
    """Raised when a connector is constructed with an invalid repo_id."""

    pass


class ConnectorInvalidUrlError(Exception):
    """Raised when a connector is constructed with an invalid URL."""

    pass
