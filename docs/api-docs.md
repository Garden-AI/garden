## Client

::: garden_ai.GardenClient
    options:
        show_source: false
        members:
            - get_garden
            - get_entrypoint

## Garden objects

::: garden_ai.gardens
    options:
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - Garden
            - GardenMetadata

## Entrypoint decorators and objects

The `EntrypointMetadata` class along with the `@entrypoint` and `@entrypoint_test` decorators are used to create and publish new entrypoints.

::: garden_ai.entrypoints
    options:
        show_root_heading: false
        show_root_full_path: false
        show_object_full_path: false
        show_root_toc_entry: false
        members:
            - entrypoint
            - entrypoint_test

::: garden_ai.entrypoints
    options:
        show_root_heading: false
        show_root_full_path: false
        show_object_full_path: false
        show_root_toc_entry: false
        members:
            - EntrypointMetadata


Next, the `Entrypoint` and `RegisteredEntrypointMetadata` classes are for users of already-published entrypoints.

::: garden_ai.entrypoints
    options:
        show_root_heading: false
        show_root_full_path: false
        show_object_full_path: false
        show_root_toc_entry: false
        members:
            - Entrypoint


::: garden_ai.entrypoints
    options:
        show_root_heading: false
        show_root_full_path: false
        show_object_full_path: false
        show_root_toc_entry: false
        members:
            - RegisteredEntrypointMetadata

## Metadata for "Related Work"

::: garden_ai.schemas.entrypoint
    options:
        show_root_heading: false
        show_root_full_path: false
        show_object_full_path: false
        show_root_toc_entry: false
        members:
            - DatasetMetadata
            - PaperMetadata
            - RepositoryMetadata
            - ModelMetadata

## Model Connector objects

::: garden_ai.model_connectors
    options:
        show_root_heading: false
        show_root_full_path: false
        show_object_full_path: false
        show_root_toc_entry: false
        members:
            - create_connector

::: garden_ai.model_connectors
    options:
        show_root_heading: false
        show_root_full_path: false
        show_object_full_path: false
        show_root_toc_entry: false
        members:
            - ModelConnector
            - GitHubConnector
            - HFConnector
