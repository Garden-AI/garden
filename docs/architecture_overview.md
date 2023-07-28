---
alias: Architecture Overview
---
## Overview

The Garden project is structured around four core concepts: `Gardens`, `Pipelines`, `steps`, and `Models`. Each of these is represented by one or more classes in our SDK.

Broadly speaking, `step` and `Model` are lower-level abstractions for users to bring existing work into our system (but won't necessarily be made easily available to other users). The higher-level  `Garden` and `Pipeline` objects define the *citable* and *reproducible* end products, and enable users to discover/share scientific work.

### [Steps](Steps.md)

A `Step` is the smallest "unit of code" usable to the Garden framework. A step is just a single function or callable that performs a specific task - such as pre-processing data or running inference - wrapped in some additional metadata (such as input/output type annotations) so that it can be composed with other steps in a Pipeline.

In other words, the only user code a `Pipeline` can "see" is the code contained in its steps, so its steps need to contain *all* of the code necessary to run it.

Rather than instantiating the `Step` class directly, steps are typically defined using the `@step` decorator. For example:

```python
@step
def preprocessing(input_data: pd.DataFrame) -> pd.DataFrame:
    """do some preprocessing"""
    filtered_data = input_data["important column"]
    return filtered_data
```

You may define a pipeline which uses as many steps as you like, as long as there is at least one. A step **requires** standard python type annotations (or a default value which can be used to infer a type) for all of its input arguments and return type, which we check before attempting to compose mismatched steps together in a pipeline.

> [!NOTE]
> `Any` and `None` are not allowed as hints, since they don't help us verify that steps would compose together(no type checking is done at runtime). Type annotations will also used to populate input/output metadata for pipelines, if not explicitly provided.

### [Models](Models.md)

The `Model` class represents a pre-trained machine learning model registered with our service. It includes information about the model itself, such as its flavor (framework used to develop the model, e.g. `sklearn`), serialization format, architecture, parameters, and state, as well as metadata such as links to training datasets. Currently, we support models in the following flavors: `sklearn`, `pytorch`, and `tensorflow`. 

. 

Models in Garden are registered using the Garden CLI:

```bash
garden-ai model register path/to/model.pkl sklearn
```

Optionally you can pass in a serialization format:
```bash
garden-ai model register path/to/model.pkl sklearn --serialize-type joblib
```
>[!NOTE]
> For serialization types we currently support `pickle` and `joblib` for `sklearn`. For `tensorflow` and `pytorch` we use the default save/serialization methods, but if you deisre to be explicit these can be entered in the CLI via `--serialize-type keras` for tensorflow and `--serialize-type torch` for pytorch. If no serialize-type an attempt at using a flavor-compatible default will be made.

> [!NOTE]
> The output of this command gives you a *full model name*, like `"me@institution.edu-my-model-name"`, which can then be referenced by steps (see example below).


Once a model has been registered, it should be used in a `Step` by referencing its name as a _**default argument**_, not in the body of the step's function.

Here's an example of a step that names a registered model to perform inference:

```python
@step
def run_inference(
    cleaned_data: pd.DataFrame,
    model=Model("me@institution.edu-my-model-name"),
) -> np.ndarray:
    """running some inference"""
    results = model.predict(cleaned_data)
    return results
```

This way, your pre-trained models can be easily integrated into any pipeline and executed as part of your workflows.

### [Pipelines](Pipelines.md)

The primary purpose of a Pipeline in the Garden framework is to make the code and models contained in its steps _citable_ and _reproducible_. Pipelines collect enough metadata for us to do two things: mint a DOI, and build a container spec in which its code could run.

There are two relevant classes: `Pipeline` and `RegisteredPipeline`, corresponding to a user's local "work in progress" and a user's finished citable product, respectively. Moreover, the final `RegisteredPipeline` is "mere metadata" -- it's effectively a citation you can run.

Unlike a `RegisteredPipeline`, a `Pipeline` object has direct references to its steps and is tightly coupled to its source file. In other words, a `Pipeline` is callable locally, and a `RegisteredPipeline` is callable anywhere else.


> [!NOTE]
> Both are callable objects, but while calling a `Pipeline` calls its steps directly, calling a `RegisteredPipeline` calls a function which was registered with Globus Compute. This means that when calling a registered pipeline, you must specify a valid Globus Compute endpoint with the `endpoint="..."` keyword argument.

#### Creating a Pipeline

The easiest way to create a new pipeline is with the CLI:

```bash
garden-ai pipeline create looking_glass_pipeline \
        --author "Dee, Tweedle" \
        --contributor "Dum, Tweedle" \
        --description "..." # etc
```

Which will generate a templated `looking_glass_pipeline.py` file, which defines a pipeline for the user to fill in. Once finished, the CLI can register the pipeline defined in that python file (see below).

Here's what a typical user-completed `Pipeline` might look like:

```python
#  looking_glass_pipeline.py
...
client = GardenClient()
looking_glass_pipeline: Pipeline = client.create_pipeline(
    title="Looking Glass Pipeline",
    steps=(preprocessing, run_inference),  # functions decorated above with @step
    requirements_file="/path/to/requirements.txt",
    authors=["Dee, Tweedle", "et al."],
    contributors=["Dum, Tweedle"],
    description="Makes you feel a little sorry for the poor oysters",
    tags=["Carpentry", "Walrus Studies"],
)
```

A plain `Pipeline` is meant for local execution and can be directly called to invoke the composed steps, by e.g. importing it in a notebook/REPL.

#### Registering a Pipeline

To register a pipeline with Globus Compute for remote execution, making it reproducible, containerized, and universally executable, you use the CLI like so:

```bash
garden-ai pipeline register /path/to/my_pipeline.py
```

This completes the pipeline development process by minting a DOI (if necessary) and registering the pipeline's composed steps with Globus Compute as a single function. After registration, the pipeline's metadata, including its new Globus Compute UUID, is simply stored locally at `~/.garden/data.json`. This metadata is sufficient to create an instance of `RegisteredPipeline` which is no longer tied to any particular source file, just a DOI.

> [!NOTE]
> The pipeline you want to register should be the ONLY pipeline object instantiated in the file given to the CLI.

Once associated with a DOI, a `RegisteredPipeline` can be added to a known Garden, creating a curated collection of reproducible pipelines:

```bash
garden-ai garden add-pipeline \
	--garden='10.garden/doi' \
	--pipeline='10.registered/pipeline'
```

> [!NOTE]
> Despite not being directly tied to python source, a `RegisteredPipeline` _does_ have a `short_name` attribute which must be a valid python identifier -- if none is specified, the variable name from the original source file is used, and the pipeline would be accessed from the garden like `garden.pipeline_short_name(...)`. If this isn't desirable, an alias can be specified when adding the pipeline, which will be specific to that garden.

### [Gardens](Gardens.md)

Finally, a `Garden` is how we make all this work _discoverable_: a garden is user-curated collection of related pipelines, each potentially associated with a scientific ML model. All that a `Garden` "really is" is a set of `Pipeline` citations (more specifically, `RegisteredPipeline` DOIs) that you can conveniently run.


> [!NOTE]
> A garden's pipelines are still callable, and are accessible as attributes - for example, if I've registered `my_pipeline` and added it to `my_garden`, I can execute it remotely like `my_garden.my_pipeline(*args, endpoint="...")`


Here's how a `Garden` is typically created using the Garden CLI:

```bash
garden-ai garden create \
	--title "Garden of Live Flowers" \
	--author "The Red Queen" --year 1871
```



Also note that `RegisteredPipeline` can only be executed remotely on Globus Compute -- it's still callable, but needs to be called with the keyword argument `endpoint=...` specifying a valid Globus Compute endpoint, like: `garden_instance.pipeline_name(*args, endpoint=...)`.

Finally, a `Garden` can be published with the CLI, minting its DOI and making it findable/accessible to others. After completing the development of a Garden and adding any number of pipelines, it can be published like so:

```bash
garden-ai garden publish --garden='10.garden/doi'
```

Which enables other users to fetch that garden and call any of its pipelines with their own input and on their own Globus Compute endpoint. This might look like:

```python
>>> gc = GardenClient()
>>> other_garden = gc.get_garden_by_doi('10.garden/doi') # someone else's doi
>>> my_data = pd.DataFrame(...)
>>> results = other_garden.their_pipeline_name(my_data, endpoint="...")
>>> print(results)  # neat!
```
