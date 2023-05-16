---
alias: Architecture Overview
---
## Architecture Overview

The Garden project is structured around four core concepts: `Gardens`, `Pipelines`, `steps`, and `Models`. Each of these is represented by one or more classes in our SDK.

Broadly speaking, `step` and `Model` are lower-level abstractions for users to bring existing work into our system (but won't necessarily be made easily available to other users). The higher-level  `Garden` and `Pipeline` objects describe the citable end product enabling users to discover/share scientific work.

### [Models](Models.md)

The `Model` class represents a pre-trained machine learning model registered with our service. It includes information about the model itself, such as its flavor (framework used to develop the model, e.g. `sklearn`), architecture, parameters, and state, as well as metadata such as its training history and performance metrics.

Models in Garden are registered through our hosted MLflow registry, using the Garden CLI:

```bash
garden-ai model register path/to/model.pkl --flavor=sklearn
```

The output of this command gives you a *full model name*, like `"me@institution.edu-my-model-name/1"`, which can then be referenced by steps (see example below). Repeatedly registering the same model under the same model name will increment the version suffix.

Currently, we support models in the following flavors: `sklearn`, `pytorch`, and `tensorflow`.

Once a model has been registered, it can be used in a `Step` by referencing its name as a _**default argument**_, not in the body of the step's function. This is necessary because the `Step` collects information about the model's dependencies to pass along to the eventual container specification that the pipeline will run in.

Here's an example of a step that uses a registered model to perform inference:

```python
@step
def run_inference(
    cleaned_data: pd.DataFrame,
    model=Model("me@institution.edu-my-model-name/1"),
) -> np.ndarray:
    """running some inference"""
    results = model.predict(cleaned_data)
    return results
```

This way, your pre-trained models can be easily integrated into any pipeline and executed as part of your workflows, while ensuring that all necessary dependencies are captured and included in the runtime environment.

### [Steps](Steps.md)

A step is the smallest "unit of code" in the Garden framework. In other words, the only user code a `Pipeline` can "see" is the code contained in its steps. The `Step` class (typically via the `@step` decorator) represents a Python function or callable that performs a specific task, such as pre-processing data or running inference. A `Step` is a Python callable wrapped with additional metadata, such as input/output types, a human-readable description, etc.

Here is an example of a step using the decorator:

```python
@step
def preprocessing(input_data: pd.DataFrame) -> pd.DataFrame:
    """doing some preprocessing"""
    smiles = input_data["smiles"]
    return smiles
```

You may define a pipeline which uses as many steps as you like, as long as there is at least one. A step requires standard python type annotations, with the caveat that `Any` and `None` are not allowed as hints, since they don't help us verify that steps would compose together (no type checking is done at runtime). Type annotations are also used to populate input/output metadata for pipelines, if not explicitly provided.

### [Pipelines](Pipelines.md)

If a step is the least "unit of code", a `Pipeline` is the least "unit of computation". In other words, a `Pipeline` is _callable_, and a `RegisteredPipeline` is _callable anywhere_. Moreover, the final `RegisteredPipeline` is merely metadata -- it's like a citation you can run.

A `Pipeline` in Garden represents a series of Steps composed together to perform a more complex task, typically running inference with an AI/ML model. The steps in a pipeline are composed in such a way that the output of one step is passed directly as the input to the subsequent step. This composition requires the users to use standard Python type hints to annotate the functions they'd like to use as steps. The pipeline validates that these annotations agree before it composes the steps together.

#### Creating a Pipeline

The easiest way to create a new pipeline is with the CLI:

```bash
garden-ai pipeline create my_new_pipeline \
        --author "Dee, Tweedle" \
        --contributor "Dum, Tweedle" \
        --description "..." # etc
```

Which will generate a templated python module which defines a pipeline for the user to complete. Once finished, the CLI can register the pipeline defined in that python file (see below).

Here's what a typical completed `Pipeline` might look like:

```python
#  ionization_potential_predictor.py
...
ionization_potential_predictor: Pipeline = client.create_pipeline(
    title="Ionization Potential Predictor",
    steps=(preprocessing, run_inference),  # functions decorated above with @step
    requirements_file=REQUIREMENTS_FILE,
    authors=["Ward, Logan", "et al."],
    contributors=["The Garden Gang"],
    description="Predicts ionization potential from SMILES strings, given as a 'smiles' column in a pandas DataFrame.",
    version="0.0.1",
    year=2023,
    tags=["example", "pipeline"],
    uuid="50039c98-b6c6-415a-b11b-9a0845c0a9b8",
)
```

A plain `Pipeline` is meant for local execution and can be directly called to invoke the composed steps. Unlike a `RegisteredPipeline`, a `Pipeline` object is tightly coupled to its source file.

#### Registering a Pipeline

To register a pipeline with Globus Compute for remote execution, and make it reproducible, containerized, and universally executable, you use the Garden CLI:

```bash
garden-ai pipeline register /path/to/my_pipeline.py
```

After registration, the pipeline's metadata, including its Globus Compute UUID, is simply stored locally at `~/.garden/data.json`. This metadata is sufficient to create an instance of `RegisteredPipeline`, which represents a pipeline that has been registered with Globus Compute (and so is no longer tied to any particular source file, just a DOI. This was the inspiration for our unofficial slogan: "If you can DOI it, you can DO it!").

Once associated with a DOI, a `RegisteredPipeline` can be added to a known Garden, creating a curated collection of reproducible pipelines:

```bash
garden-ai garden add-pipeline --garden='10.garden/doi' --pipeline='10.registered/pipeline'
```

### [Gardens](Gardens.md)

Finally, a `Garden` is the "unit of discoverability": a user-curated collection of related `Pipelines` (more specifically, pipeline IDs/DOIs), each potentially associated with a scientific ML model. All that a `Garden` "really is" is a set of `Pipeline` citations that you can conveniently run.

Here's how a `Garden` is typically created using the Garden AI CLI:

```bash
garden-ai garden create --author "Queen of Hearts" --title "Rose Garden" --year 1865
```

Once a `Garden` is created, a `RegisteredPipeline`-- from any user -- can be added to a `Garden` using their respective DOIs:

```bash
garden-ai garden add-pipeline --garden='10.garden/doi' --pipeline='10.registered/pipeline'
```

Important note: despite not being directly tied to a source file, a `RegisteredPipeline` _does_ have a `short_name` attribute which must be a valid python identifier -- if none is specified, the variable name from the original source file is used, and the pipeline would be accessed from the garden like `garden.pipeline_short_name(...)`. If this isn't desirable, an alias can be specified when adding the pipeline, which will be specific to that garden.

Also note that `RegisteredPipeline` can only be executed remotely on Globus Compute -- it's still callable, but needs to be called with the keyword argument `endpoint=...` specifying a valid Globus Compute endpoint, like: `garden_instance.pipeline_name(*args, endpoint=...)`.

Finally, a `Garden` can be published with the CLI, minting its DOI and making it findable/accessible to others. After completing the development of a Garden and adding any number of pipelines, it can be published like so:

```bash
garden-ai garden publish --garden='10.garden/doi'
```

Which enables other users to fetch that garden and call any of its pipelines with their own input and on their own Globus Compute endpoint. This might look like:

```python
>>> gc = GardenClient()
>>> other_garden = gc.get_garden_by_doi('10.garden/doi')
>>> my_data = pd.DataFrame(...)
>>> results = other_garden.curated_pipeline(my_data, endpoint="...")
>>> print(results)  # cool!
```
