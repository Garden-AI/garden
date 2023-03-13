from mlflow.tracking.request_header.abstract_request_header_provider import (  # type: ignore
    RequestHeaderProvider,
)


class BinaryContentTypeHeaderProvider(RequestHeaderProvider):
    """
    Adds a Content-Type header so that we can ask AWS API Gateway not to mangle our requests.
    """

    def in_context(self):
        return True

    def request_headers(self):
        return {"Content-Type": "application/octet-stream"}
