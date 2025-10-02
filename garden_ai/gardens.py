from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from tabulate import tabulate

from garden_ai.hpc.functions import HpcFunction
from garden_ai.modal.classes import ModalClassWrapper
from garden_ai.modal.functions import ModalFunction
from garden_ai.schemas.garden import GardenMetadata
from garden_ai.schemas.hpc import HpcFunctionMetadata
from garden_ai.schemas.modal import ModalFunctionMetadata

logger = logging.getLogger()


@dataclass
class JobStatus:
    """Status information for an HPC batch job."""

    status: str  # "pending" | "running" | "completed" | "failed" | "unknown"
    results_available: bool = False
    error: str | None = None
    stdout: str = ""
    stderr: str = ""


if TYPE_CHECKING:
    from garden_ai.client import GardenClient
else:
    GardenClient = TypeVar("GardenClient")


class Garden:
    """
    Represents a collection of related functions, providing a way to organize and invoke machine learning models.

    This class is geared towards users wishing to access a published Garden, and is meant to be instantiated by the client's [get_garden][garden_ai.GardenClient.get_garden] method.

    Attributes:
        metadata (GardenMetadata): The Garden's published metadata, including information such as title, authors, description, and DOI.
        modal_functions (list[ModalFunction]): The callable functions associated with this Garden. Individual functions are also accessible like attributes on this object.

    Example:
        Functions can be accessed as attributes of the Garden instance, allowing for intuitive calling of the associated functions:
        ```python
        client = garden_ai.GardenClient()
        garden = client.get_garden("my_garden_doi")
        result = garden.my_function(data)
        ```
    """  # noqa: E501

    def __init__(
        self,
        metadata: GardenMetadata,
        modal_functions: list[ModalFunction] | None = None,
        modal_classes: list[ModalClassWrapper] | None = None,
        hpc_functions: list[HpcFunction] | None = None,
    ):
        modal_functions = modal_functions or []
        modal_classes = modal_classes or []
        hpc_functions = hpc_functions or []

        expected_modal_ids = set(metadata.modal_function_ids)
        actual_modal_ids = set(mf.metadata.id for mf in modal_functions)
        for modal_class in modal_classes:
            actual_modal_ids.update(
                method.metadata.id for method in modal_class._methods.values()
            )

        if expected_modal_ids ^ actual_modal_ids:
            raise ValueError(
                "Expected `modal_functions` to match `metadata.modal_function_ids`. "
                f"Got: {actual_modal_ids} != {expected_modal_ids}"
            )

        self.metadata = metadata
        self.modal_functions = modal_functions
        self.modal_classes = modal_classes
        self.hpc_functions = hpc_functions

    def __getattr__(self, name):
        # enables method-like syntax for calling Modal functions from this garden.
        # note: this is only called as a fallback when __getattribute__ raises an exception,
        # existing attributes are not affected by overriding this
        message_extra = ""

        for modal_function in self.modal_functions:
            if name == modal_function.metadata.function_name:
                return modal_function

        for modal_class in self.modal_classes:
            if name == modal_class.class_name:
                return modal_class

        for hpc_function in self.hpc_functions:
            if name == hpc_function.metadata.function_name:
                return hpc_function

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'."
            + message_extra
        )

    def __dir__(self):
        # this gets us jupyter/ipython/repl tab-completion of function names
        modal_function_names = [
            mf.metadata.function_name for mf in self.modal_functions
        ]

        modal_class_names = [mc.class_name for mc in self.modal_classes]

        hpc_function_names = [
            hpcf.metadata.function_name for hpcf in self.hpc_functions
        ]

        return (
            list(super().__dir__())
            + modal_function_names
            + modal_class_names
            + hpc_function_names
        )

    def _repr_html_(self) -> str:
        data = self.metadata.model_dump(
            exclude={"owner_identity_id", "id", "language", "publisher"}
        )
        data["modal_functions"] = [
            mf.metadata.model_dump() for mf in self.modal_functions
        ]

        style = "<style>th {text-align: left;}</style>"
        title = f"<h2>{data['title']}</h2>"
        details = f"<p>Authors: {', '.join(data['authors'])}<br>DOI: {data['doi']}</p>"
        modal_functions = "<h3>Modal Functions</h3>" + tabulate(
            [
                {
                    key.title(): str(modal_function[key])
                    for key in ("function_name", "title", "authors", "doi")
                }
                for modal_function in data["modal_functions"]
            ],
            headers="keys",
            tablefmt="html",
        )

        modal_classes = ""
        if self.modal_classes:
            classes_data = []
            for cls in self.modal_classes:
                for method in cls._methods.values():
                    classes_data.append(
                        {
                            "Class": cls.class_name,
                            "Method": method.metadata.function_name.split(".")[-1],
                            "Title": str(method.metadata.title),
                            "Authors": ", ".join(method.metadata.authors),
                            "DOI": str(method.metadata.doi or ""),
                        }
                    )

            modal_classes = "<h3>Modal Class Methods</h3>" + tabulate(
                classes_data,
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
        return style + title + details + modal_functions + modal_classes + optional

    @classmethod
    def _from_nested_metadata(cls, data: dict, client: GardenClient | None = None):
        """helper: instantiate from search index-style payload with nested function metadata.

        Note: `client` is generally fine to omit outside of tests
        """
        metadata = GardenMetadata(**data)
        modal_functions = []
        class_methods: dict[str, list[ModalFunctionMetadata]] = {}
        hpc_functions: list[HpcFunction] = []

        # Process modal functions and organize into classes
        if "modal_functions" in data:
            for modal_fn_data in data["modal_functions"]:
                fn_metadata = ModalFunctionMetadata(**modal_fn_data)
                metadata.modal_function_ids += [fn_metadata.id]

                # Check if this is a class method
                if "." in fn_metadata.function_name:
                    class_name, _ = fn_metadata.function_name.split(".", 1)
                    if class_name not in class_methods:
                        class_methods[class_name] = []
                    class_methods[class_name].append(fn_metadata)
                else:
                    modal_functions.append(ModalFunction(fn_metadata, client))

        modal_classes = [
            ModalClassWrapper.from_metadata(class_name, methods, client)
            for class_name, methods in class_methods.items()
        ]

        for hpc_function_data in data.get("hpc_functions", []):
            fn_metadata = HpcFunctionMetadata(**hpc_function_data)
            metadata.hpc_function_ids += [fn_metadata.id]
            hpc_functions.append(HpcFunction(fn_metadata))

        return cls(metadata, modal_functions, modal_classes, hpc_functions)

    def get_job_status(self, job_id: str) -> "JobStatus":
        """
        Get status information for a submitted HPC job.

        Args:
            job_id: Globus Compute task ID returned by HpcFunction.submit()

        Returns:
            JobStatus object with current status and outputs

        Example:
            >>> status = garden.get_job_status(job_id)
            >>> print(status.status)  # "completed"
            >>> if status.status == "completed":
            ...     results = garden.get_results(job_id)
        """
        from globus_compute_sdk import Client as GlobusComputeClient

        from garden_ai.hpc.utils import decode_if_base64

        gc_client = GlobusComputeClient()
        task_info = gc_client.get_task(job_id)

        # Check if still pending
        if task_info.get("pending", True):
            return JobStatus(status="pending")

        # Try to get result
        try:
            job_result = gc_client.get_result(job_id)

            # Check for error in result
            if isinstance(job_result, dict) and "error" in job_result:
                return JobStatus(
                    status="failed",
                    error=job_result["error"],
                    stdout=decode_if_base64(job_result.get("stdout", "")),
                    stderr=decode_if_base64(job_result.get("stderr", "")),
                )

            # Success - results available
            return JobStatus(
                status="completed",
                results_available=True,
                stdout=decode_if_base64(
                    job_result.get("stdout", "") if isinstance(job_result, dict) else ""
                ),
                stderr=decode_if_base64(
                    job_result.get("stderr", "") if isinstance(job_result, dict) else ""
                ),
            )

        except Exception as e:
            return JobStatus(
                status="unknown",
                error=f"Failed to get job status: {str(e)}",
            )

    def get_results(
        self,
        job_id: str,
        output_path: str | Path | None = None,
    ) -> Any:
        """
        Retrieve results from a completed HPC job.

        Args:
            job_id: Globus Compute task ID
            output_path: Optional local path to save results (for file-based results)

        Returns:
            Job results (type depends on the function)

        Raises:
            RuntimeError: If job is not completed or failed

        Example:
            >>> results = garden.get_results(job_id)
            >>> # Or save to file:
            >>> results = garden.get_results(job_id, output_path="./results.xyz")
        """
        from pathlib import Path

        from globus_compute_sdk import Client as GlobusComputeClient

        # Check status first
        status_info = self.get_job_status(job_id)

        if status_info.status == "pending":
            raise RuntimeError(f"Job {job_id} is still pending")
        elif status_info.status == "running":
            raise RuntimeError(f"Job {job_id} is still running")
        elif status_info.status == "failed":
            raise RuntimeError(f"Job {job_id} failed: {status_info.error}")
        elif status_info.status != "completed":
            raise RuntimeError(f"Job {job_id} has unknown status")

        if not status_info.results_available:
            raise RuntimeError(f"No results available for job {job_id}")

        # Get the actual result
        gc_client = GlobusComputeClient()
        job_result = gc_client.get_result(job_id)

        # Handle encoded data if present (from subproc_wrapper)
        if isinstance(job_result, dict) and "raw_data" in job_result:
            import base64
            import pickle

            actual_result = pickle.loads(base64.b64decode(job_result["raw_data"]))
        else:
            actual_result = job_result

        # Save to file if requested
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(actual_result, str):
                with open(output_path, "w") as f:
                    f.write(actual_result)
            else:
                # For non-string results, try to pickle or stringify
                import pickle

                try:
                    with open(output_path, "wb") as f:
                        pickle.dump(actual_result, f)
                except Exception:
                    with open(output_path, "w") as f:
                        f.write(str(actual_result))

            print(f"✅ Results saved to {output_path}")

        return actual_result
