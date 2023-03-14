from mlflow.tracking.request_header.abstract_request_header_provider import (  # type: ignore
    RequestHeaderProvider,
)


class BinaryContentTypeHeaderProvider(RequestHeaderProvider):
    """
    Adds a Content-Type header so that we can ask AWS API Gateway not to mangle our requests.
    See https://github.com/Garden-AI/garden-backend/issues/11 for details on the bug that prompted this.
    We have made a feature request to MLFlow to try and find a stabler fix.
    Issue link: https://github.com/mlflow/mlflow/issues/8026
    """

    def in_context(self):
        """
        A required method for subclasses of RequestHeaderProvider. Returning True makes it so that
        the headers provided by this provider will be included in every request.
        """
        return True

    def request_headers(self):
        """
        Another method expected from a RequestHeaderProvider. The dictionary it returns
        is combined with the headers from all other active providers.
        """
        return {"Content-Type": "application/octet-stream"}
