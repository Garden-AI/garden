from garden_ai import local_data


def complete_pipeline(incomplete: str):
    completion = []
    pipelines = local_data.get_all_local_pipelines()
    all_pipelines = (
        [(pipeline.doi, pipeline.title) for pipeline in pipelines] if pipelines else []
    )
    if not incomplete:
        return all_pipelines
    for name, help_text in all_pipelines:
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
