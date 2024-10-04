from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING

import logging

from tabulate import tabulate

from garden_ai.schemas.entrypoint import RegisteredEntrypointMetadata
from garden_ai.schemas.garden import GardenMetadata
from garden_ai.schemas.modal import ModalFunctionMetadata
from garden_ai.modal.functions import ModalFunction

from .entrypoints import Entrypoint

logger = logging.getLogger()

if TYPE_CHECKING:
    from garden_ai.client import GardenClient
else:
    GardenClient = TypeVar("GardenClient")


class Garden:
    """
    Represents a collection of related [Entrypoints][garden_ai.Entrypoint], providing a way to organize and invoke machine learning models and functions.

    A Garden serves as a container for multiple Entrypoints, allowing users to group related models or functions together. It provides metadata about the collection and enables easy access to the contained Entrypoints.

    This class is geared towards users wishing to access a published Garden, and is meant to be instantiated by the client's [get_garden][garden_ai.GardenClient.get_garden] method.

    Attributes:
        metadata (GardenMetadata): The Garden's published metadata, including information such as title, authors, description, and DOI.
        entrypoints (list[Entrypoint]): The callable Entrypoints associated with this Garden. Individual entrypoints are also accessible like attributes on this object.

    Note:
        When an Entrypoint is called through a Garden, it is executed remotely on the specified Globus Compute endpoint. The default is the free garden demo endpoint.

    Example:
        Entrypoints can be accessed as attributes of the Garden instance, allowing for intuitive calling of the associated functions:
        ```python
        client = garden_ai.GardenClient()
        garden = client.get_garden("my_garden_doi")
        result = garden.my_entrypoint(data, endpoint="endpoint_uuid")
        ```
    """  # noqa: E501

    def __init__(
        self,
        metadata: GardenMetadata,
        entrypoints: list[Entrypoint] | None = None,
        modal_functions: list[ModalFunction] | None = None,
    ):
        entrypoints = entrypoints or []
        modal_functions = modal_functions or []

        if set(metadata.entrypoint_ids) ^ set([ep.metadata.doi for ep in entrypoints]):
            raise ValueError(
                "Expected `entrypoints` DOIs to match `metadata.entrypoint_ids`. "
                f"Got: {[ep.metadata.doi for ep in entrypoints]} != {metadata.entrypoint_ids}"
            )
        # TODO: update to use DOIs when modal functions have them
        if set(metadata.modal_function_ids) ^ set(
            [mf.metadata.function_name for mf in modal_functions]
        ):
            raise ValueError(
                "Expected `modal_functions` to match `metadata.modal_function_ids`. "
                f"Got: {[mf.metadata.function_name for mf in modal_functions]} != {metadata.modal_function_ids}"
            )
        self.metadata = metadata
        self.entrypoints = entrypoints
        self.modal_functions = modal_functions
        return

    def __getattr__(self, name):
        # enables method-like syntax for calling entrypoints from this garden.
        # note: this is only called as a fallback when __getattribute__ raises an exception,
        # existing attributes are not affected by overriding this
        message_extra = ""
        for entrypoint in self.entrypoints:
            short_name = entrypoint.metadata.short_name
            alias = (
                self.metadata.entrypoint_aliases.get(entrypoint.metadata.doi)
                or short_name
            )
            if name == alias:
                return entrypoint
            elif name == short_name:
                message_extra = f" Did you mean {alias}?"

        for modal_function in self.modal_functions:
            if name == modal_function.metadata.function_name:
                return modal_function

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'."
            + message_extra
        )

    def __dir__(self):
        # this gets us jupyter/ipython/repl tab-completion of entrypoint names
        entrypoint_names = []
        for entrypoint in self.entrypoints:
            name = self.metadata.entrypoint_aliases.get(entrypoint.metadata.doi)
            if name is None:
                name = entrypoint.metadata.short_name
            entrypoint_names += [name]

        modal_function_names = [
            mf.metadata.function_name for mf in self.modal_functions
        ]

        return list(super().__dir__()) + entrypoint_names + modal_function_names

    def _repr_html_(self) -> str:
        data = self.metadata.model_dump(
            exclude={"owner_identity_id", "id", "language", "publisher"}
        )
        data["entrypoints"] = [ep.metadata.model_dump() for ep in self.entrypoints]
        data["modal_functions"] = [
            mf.metadata.model_dump() for mf in self.modal_functions
        ]

        style = "<style>th {text-align: left;}</style>"
        title = f"<h2>{data['title']}</h2>"
        details = f"<p>Authors: {', '.join(data['authors'])}<br>DOI: {data['doi']}</p>"
        entrypoints = "<h3>Entrypoints</h3>" + tabulate(
            [
                {
                    key.title(): str(entrypoint[key])
                    for key in ("short_name", "title", "authors", "doi")
                }
                for entrypoint in data["entrypoints"]
            ],
            headers="keys",
            tablefmt="html",
        )
        modal_functions = "<h3>Modal Functions</h3>" + tabulate(
            [
                {
                    key.title(): str(entrypoint[key])
                    for key in ("short_name", "title", "authors", "doi")
                }
                for entrypoint in data["modal_functions"]
            ],
            headers="keys",
            tablefmt="html",
        )
        optional = "<h3>Additional data</h3>" + tabulate(
            [
                (field, str(val))
                for field, val in data.items()
                if field not in ("title", "authors", "doi", "short_name")
                and "entrypoint" not in field
                and val
            ],
            tablefmt="html",
        )
        return style + title + details + entrypoints + modal_functions + optional

    @classmethod
    def _from_nested_metadata(cls, data: dict, client: GardenClient | None = None):
        """helper: instantiate from search index-style payload with nested entrypoint metadata.

        Note: `client` is generally fine to omit outside of tests
        """
        metadata = GardenMetadata(**data)
        entrypoints = []
        modal_functions = []
        for entrypoint_data in data["entrypoints"]:
            entrypoints += [Entrypoint(RegisteredEntrypointMetadata(**entrypoint_data))]
            metadata.entrypoint_ids += [entrypoint_data["doi"]]

        if "modal_functions" in data:
            for modal_fn_data in data["modal_functions"]:
                modal_functions += [
                    ModalFunction(ModalFunctionMetadata(**modal_fn_data), client)
                ]
                # TODO: use DOIs once modal functions have them
                metadata.modal_function_ids += [modal_fn_data["function_name"]]

        return cls(metadata, entrypoints, modal_functions)
