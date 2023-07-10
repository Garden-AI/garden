import importlib.metadata  # type: ignore

try:
    __version__ = importlib.metadata.version("garden-ai")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
