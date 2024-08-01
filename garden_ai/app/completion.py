from garden_ai import GardenClient


def complete_entrypoint(incomplete: str):
    client = GardenClient()
    user_id = client.get_user_identity_id()
    entrypoints = client.backend_client.get_entrypoints(owner_uuid=user_id)
    completions = [
        (
            entrypoint.metadata.doi,
            f"{entrypoint.metadata.short_name}, {entrypoint.metadata.title}",
        )
        for entrypoint in entrypoints
        if entrypoint.metadata.doi.startswith(incomplete)
    ]

    return completions


def complete_garden(incomplete: str):
    client = GardenClient()
    user_id = client.get_user_identity_id()
    gardens = client.backend_client.get_gardens(owner_uuid=user_id)
    completions = [
        (garden.metadata.doi, garden.metadata.title)
        for garden in gardens
        if garden.metadata.doi.startswith(incomplete)
    ]
    return completions
