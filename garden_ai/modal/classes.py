from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any
from garden_ai.modal.functions import ModalFunction

from ..schemas.modal import ModalFunctionMetadata


if TYPE_CHECKING:
    from garden_ai.client import GardenClient
else:
    GardenClient = TypeVar("GardenClient")


class ModalClassWrapper:
    """
    A wrapper for Modal class objects that allows method access and invocation through Garden.
    """

    def __init__(
        self,
        class_name: str,
        methods: list[ModalFunction],
    ):
        self.class_name = class_name
        # Create method lookup internally
        self._methods = {
            method.metadata.function_name.split(".")[-1]: method for method in methods
        }

    def __getattr__(self, method_name: str) -> Any:
        """Allow accessing methods as attributes of the class wrapper."""
        if method_name in self._methods:
            return self._methods[method_name]
        raise AttributeError(f"'{self.class_name}' has no method '{method_name}'")

    def __dir__(self) -> list[str]:
        """Return list of valid attributes for tab completion."""
        return list(super().__dir__()) + list(  # Get standard attributes
            self._methods.keys()
        )  # Add all method names

    @classmethod
    def from_metadata(
        cls,
        class_name: str,
        methods_metadata: list[ModalFunctionMetadata],
        client: GardenClient | None = None,
    ) -> ModalClassWrapper:
        """Create a ModalClassWrapper from metadata"""
        methods = [ModalFunction(metadata, client) for metadata in methods_metadata]
        return cls(class_name, methods)

    def __repr__(self) -> str:
        method_list = ", ".join(self._methods.keys())
        return f"<ModalClass '{self.class_name}' with methods: {method_list}>"
