from __future__ import annotations

import logging

from tabulate import tabulate

from garden_ai.schemas.entrypoint import RegisteredEntrypointMetadata
from garden_ai.schemas.garden import GardenMetadata

from .entrypoints import Entrypoint

logger = logging.getLogger()


class Garden:
    def __init__(self, metadata: GardenMetadata, entrypoints: list[Entrypoint]):
        if set(metadata.entrypoint_ids) != set([ep.metadata.doi for ep in entrypoints]):
            raise ValueError(
                "Expected `entrypoints` DOIs to match `metadata.entrypoint_ids`. "
                f"Got: {[ep.metadata.doi for ep in entrypoints]} != {metadata.entrypoint_ids}"
            )
        self.metadata = metadata
        self.entrypoints = entrypoints
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

        return list(super().__dir__()) + entrypoint_names

    def _repr_html_(self) -> str:
        data = self.metadata.model_dump(
            exclude={"owner_identity_id", "id", "language", "publisher"}
        )
        data["entrypoints"] = [ep.metadata.model_dump() for ep in self.entrypoints]

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
        return style + title + details + entrypoints + optional

    @classmethod
    def _from_nested_metadata(cls, data: dict):
        """helper: instantiate from search index-style payload with nested entrypoint metadata."""
        metadata = GardenMetadata(**data)
        entrypoints = []
        for entrypoint_data in data["entrypoints"]:
            entrypoints += [Entrypoint(RegisteredEntrypointMetadata(**entrypoint_data))]
            metadata.entrypoint_ids += [entrypoint_data["doi"]]

        return cls(metadata, entrypoints)
