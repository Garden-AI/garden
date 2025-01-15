---
alias: Architecture Overview
---
## Garden Project Conceptual Overview

The Garden project is structured around two "proper noun" concepts: `Garden` and `Entrypoint`.

`Garden` and `Entrypoint` objects define the *citable* and *reproducible* end products, and enable users to discover and share each other's scientific work.
### Core Concepts
1. **Gardens:**

    - A `Garden` is a user-curated collection or portfolio of related `Entrypoint`s, aimed at promoting discoverability.
    - `Garden`s are the primary way to discover and run published `Entrypoint`s.
    - An `Entrypoint` from a different `Garden` can be added to your `Garden` simply by adding the other `Entrypoint`'s DOI to your `Garden` and publishing.
    - A `Garden` is typically created and published from the [web UI](https://thegardens.ai/#/garden/create)
    - Alternatively, you can create one from CLI, e.g.:
    ```bash
	garden-ai garden create \
		--title "Garden of Live Flowers" \
		--author "The Red Queen" --year 1871
	```
    - See also: [Gardens API Reference](api-docs.md#garden-objects)

2. **Entrypoints:**

    - An `Entrypoint` is a plain Python function, enriched with citation metadata and registered with our service for reproducibility and remote execution (via Globus Compute).
    - Users can execute an `Entrypoint` remotely via any `Garden` it has been published to.
    - An `Entrypoint` is typically defined in a regular Jupyter notebook, and functions like an "entrypoint" to a saved notebook session when registered and published to a `Garden`.
    - An `Entrypoint` is registered when its notebook is published.
    - When a notebook is published, all the `Entrypoint`s in the notebook are published to their respective gardens -- `Entrypoint`s defined in the same notebook need not be published to the same `Garden`.
    - See also: [Entrypoints API Reference](api-docs.md#entrypoint-decorators-and-objects)
> [!NOTE]
> The `Entrypoint`s attached to a `Garden` are callable and accessible like attributes -- for example, if I've registered `my_entrypoint` and added it to `some_garden`, executing it remotely might look like `some_garden.my_entrypoint(*args, endpoint="...")`.


### The `@entrypoint` Decorator:

- This decorator is used to designate which functions in the notebook should be registered as public/published `Entrypoint`s.
- It distinguishes between functions meant for public use and those that are simply part of the execution context.
- This is also how users attach citation metadata to an `Entrypoint`, such as authors or other related published work like papers or datasets

### Notebook Workflow - Defining and Developing Entrypoints

1. **Development in Jupyter Notebooks:**

    - The process begins by writing a Jupyter notebook containing functions marked as entrypoints using the `@entrypoint` decorator.
    - The `garden-ai` CLI provides a `garden-ai notebook start path/to/my.ipynb` command to open a notebook in an isolated local Docker container (conceptually similar to a Google Colab notebook, but running locally and with a choice of prebuilt base images).
	- See [installation](user_guide/installation.md) for Docker-specific prerequisites

2. **Publishing Notebooks:**

    - Once the notebook is complete and defines one or more `Entrypoint`s, the `garden-ai notebook publish path/to/my.ipynb` command is used to finalize and register each `Entrypoint` in the notebook for remote execution with Globus Compute.

See the [tutorial](user_guide/tutorial.md) for a more detailed walkthrough of the notebook publication flow.


### Containerization and Execution Context:

- Upon publishing with the `garden-ai notebook publish` command, the entire notebook is run like a script in the specified base image, and the Python interpreter state is "baked in" to the final registered container using the `dill` library.

- On the remote end, executing a registered `Entrypoint` entails spinning up its respective container, loading the saved interpreter session, then calling the decorated function.

### Publishing

1. **Garden Publishing:**
    - A `Garden` is created, manipulated, and published using the CLI, making it accessible to other users.
2. **Entrypoint Publishing:**
    - An `Entrypoint` is published by attaching it to a published `Garden`. This can be done either manually from the CLI, or automatically by specifying a particular `Garden` DOI in the `@entrypoint` decorator.
