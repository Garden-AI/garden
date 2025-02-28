from uuid import UUID

from ..gardens import Garden
from ..entrypoints import Entrypoint
from garden_ai.schemas.entrypoint import RegisteredEntrypointMetadata
from garden_ai.schemas.garden import GardenMetadata


class AlphaFoldGarden(Garden):
    """A Garden that uses a specific endpoint and function."""

    def __init__(self, client, doi: str):
        self.client = client

        function_ids = [
            "b3b3ffba-6fbb-46c9-8631-c8d3c7d6033b",
            "d6558962-2877-4ba7-a193-c1ab43cdde6d",
            "49f01f32-2574-47a7-b536-44008d15c58e",
        ]
        function_names = ["predict", "check_prediction_status", "retrieve_results"]
        endpoint_id = "b1be97db-6d56-4ba7-8ef3-d5f77581e87c"

        stub_entrypoints = []
        for function_id, function_name in zip(function_ids, function_names):
            entrypoint_metadata = RegisteredEntrypointMetadata(
                doi=f"{doi}/{function_name}",
                authors=["DeepMind"],
                title="AlphaFold2 Prediction",
                description="Predict protein structure using AlphaFold2",
                short_name=function_name,
                func_uuid=function_id,
                container_uuid=UUID("00000000-0000-0000-0000-000000000000"),
                base_image_uri="N/A",
                full_image_uri="N/A",
                notebook_url="https://thegardens.ai/",
                function_text="",
                entrypoint_ids=[],
                doi_is_draft=False,
            )
            stub_entrypoints.append(Entrypoint(entrypoint_metadata))

        metadata = GardenMetadata(
            doi=doi,
            authors=["DeepMind"],
            title="AlphaFold2",
            description="AlphaFold2 protein structure prediction",
            entrypoint_ids=[ep.metadata.doi for ep in stub_entrypoints],
            entrypoint_aliases={},
            doi_is_draft=False,
        )

        super().__init__(metadata=metadata, entrypoints=stub_entrypoints)
        self.endpoint_id = endpoint_id

    def submit(self, fasta_string: str):
        """Main prediction method that invokes the custom endpoint."""
        print(f"Submitting HPC job to predict structure of {fasta_string}")
        resp = self.entrypoints[0](fasta_string, endpoint=self.endpoint_id)
        if type(resp) is dict:
            return resp
        raise ValueError(resp)

    def check_prediction_status(self, task_id: str):
        """Check the status of a submitted task."""
        return self.entrypoints[1](task_id, endpoint=self.endpoint_id)

    def retrieve_results(self, pdb_id: str):
        """Retrieve the results of a submitted task."""
        return self.entrypoints[2](pdb_id, endpoint=self.endpoint_id)
