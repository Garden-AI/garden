from garden_ai import local_data


def complete_entrypoint(incomplete: str):
    completion = []
    entrypoints = local_data.get_all_local_entrypoints()
    all_entrypoints = (
        [(entrypoint.doi, entrypoint.title) for entrypoint in entrypoints]
        if entrypoints
        else []
    )
    if not incomplete:
        return all_entrypoints
    for name, help_text in all_entrypoints:
        if name.startswith(incomplete):
            completion.append((name, help_text))
    return completion


def complete_garden(incomplete: str):
    completion = []
    gardens = local_data.get_all_local_gardens()
    all_gardens = [(garden.doi, garden.title) for garden in gardens] if gardens else []
    if not incomplete:
        return all_gardens
    for name, help_text in all_gardens:
        if name.startswith(incomplete):
            completion.append((name, help_text))
    return completion
