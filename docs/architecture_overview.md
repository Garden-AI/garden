## Architecture Overview

The Garden project is structured around four core concepts: Gardens, Pipelines, Steps, and Models. Each of these is represented by one or more classes (typically Pydantic models) in our SDK.

### Steps

A `Step` is a Python function that performs a specific task, such as pre-processing data or running inference. Steps are the smallest building blocks in a Pipeline. In Garden, a `Step` is represented as a Python callable wrapped with additional metadata, such as type signatures, provided by the `Step` class/decorator.

Here is an example of a step using the decorator:

```python
@step
def preprocessing(input_data: pd.DataFrame) -> pd.DataFrame:
    """doing some preprocessing"""
    smiles = input_data["smiles"]
    return smiles
```

### Models

The `Model` class represents a pre-trained machine learning model registered with our service. It includes information about the model itself, such as its architecture, parameters, and state, as well as metadata such as its training history and performance metrics.

Models in Garden are registered through our hosted MLflow registry, using the Garden CLI:

```bash
garden-ai model register path/to/model.pkl --flavor=sklearn
```

Currently, we support models in the following flavors: `sklearn`, `pytorch`, and `tensorflow`.

Once a model has been registered, it can be used in a `Step` by referencing its name as a _*default argument*_, not in the body of the step's function. This is an essential aspect because the `Step` collects information about the model's dependencies for the eventual container specification that the pipeline will run in.

Here's an example of a step that uses a registered model to perform inference:

```python
@step
def run_inference(
    smiles: pd.DataFrame,
    model=Model("owenpriceskelly@uchicago.edu-real-demo-model/8"),
) -> np.ndarray:
    """running some inference"""
    results = model.predict(smiles)
    return results
```

This way, your pre-trained models can be easily integrated into any pipeline and executed as part of your workflows, while ensuring that all necessary dependencies are captured and included in the runtime environment.
### Pipelines

A `Pipeline` in Garden represents a series of Steps composed together to perform a more complex task. The steps in a pipeline are composed in such a way that the output of one step is passed directly as the input to the subsequent step. This composition requires the users to use standard Python type hints to annotate the functions they'd like to use as steps. The pipeline validates that these annotations agree before it composes the steps together.

Here's how a `Pipeline` is typically defined:

```python
ionization_potential_predictor: Pipeline = client.create_pipeline(
    title="Ionization Potential Predictor",
    steps=ALL_STEPS,
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

A `Pipeline` is meant for local execution and can be directly called to invoke the composed steps.

To register a pipeline with Globus Compute for remote execution, and make it reproducible, containerized, and universally executable, you use the Garden AI CLI:

```bash
garden-ai pipeline register /path/to/my_pipeline.py
```

After registration, the pipeline's metadata, including its Globus Compute UUID, is stored locally at `~/.garden/data.json`. This metadata is sufficient to create an instance of `RegisteredPipeline`, which represents a pipeline that has been registered with Globus Compute.

It's important to note that a `RegisteredPipeline` can only be executed remotely on Globus Compute -- a `RegisteredPipeline` is also callable, but needs to be called with the keyword argument `endpoint=...` specifying a Globus Compute endpoint.

Once associated with a DOI, a `RegisteredPipeline` can be added to a known Garden, creating a curated collection of reproducible pipelines:

```bash
garden-ai garden add-pipeline --garden='10.garden/doi' --pipeline='10.registered/pipeline'
```

By using `Pipeline` and `RegisteredPipeline`, you can develop and test your workflows locally, register them for remote execution, and curate them in Gardens to make them widely available and executable on any hardware.

### Gardens

A `Garden` is a collection of Pipelines and associated Models. It includes metadata that links the Pipelines and Models to various resources, such as scientific papers, testing metrics, known model limitations, and source code. The `Garden` class provides methods for managing Gardens, including adding new Pipelines and Models, updating metadata, and managing links to associated resources.

## Globus Compute Integration

Garden leverages Globus Compute (formerly funcx), a high-performance computing resource, for executing the Pipelines. The execution of the Pipelines is handled within the `Pipeline` class itself, which prepares the Pipeline for execution, submits it to Globus Compute, and handles the results.
