
## Tutorial: Develop a Garden and Pipeline from Scratch

This tutorial will walk you through a simple end-to-end example of developing a single `Pipeline` from scratch in a notebook; publishing it to a new `Garden`; and finally running the Pipeline remotely via that `Garden`.

This means doing the following:
- Create a new Garden using the CLI (`garden-ai garden create`)
- Open a notebook using the CLI (`garden-ai notebook start`)
- Define and decorate a function in the notebook to make it a `Pipeline`
	- Optional: debug the completed notebook (`garden-ai notebook debug`)
- Publish the notebook using the CLI, attaching the `Pipeline` to the `Garden` in the process (`garden-ai notebook publish`)
- Test remote execution on our tutorial endpoint
#### Prerequisites
- You'll need the `garden-ai` CLI as well as a local install of `docker` logged in to a Dockerhub account with a public image repository. See [installation](installation.md) for more details.
- We've used huggingface to host the [sample model weights](https://huggingface.co/Garden-AI/sklearn-seedling/tree/main) for this tutorial. If you're following along with your own pretrained model weights, you will likely also want a huggingface account to do the same.

### Step 0: Train Your Model
Garden is geared towards publishing and sharing models which have already been trained. If you don't have one handy, we'd recommend training a small model using one of the scikit-learn [toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html) and uploading the weights to a public huggingface repo in order to follow along.

The code to train the toy model used in this tutorial is available [here](https://huggingface.co/Garden-AI/sklearn-seedling/blob/main/Train_Model.ipynb).

### Step 1: Creating a New Garden
First, we're going to make a new `Garden` using the CLI so that we can eventually share our `Pipeline` with others. Before we've added any `Pipeline`s, a brand new `Garden` is going to be nothing more than some citation metadata, which is at least a title, one or more authors, and a year (the minimum needed mint a DOI).

Here's what this might look like:
```bash
garden-ai garden create \
	--title "Garden of Live Flowers" \
	--author "The Red Queen" --year 1871
```

To see the full list of metadata fields you can specify from the CLI, try `garden-ai garden create --help`.

In the output of that command, you should see something like:
```bash
...
Garden 'Garden of Live Flowers' created with DOI: 10.23677/z2b3-3p02
```

Make a note of this DOI for later, so we can publish our `Pipeline` to this specific `Garden` (see also: `garden-ai garden list` to list all local `Garden`s).

### Step 2: Starting the Notebook

We're going to define our `Pipeline` in a regular Jupyter notebook file (`.ipynb`) on our local filesystem, which we can open and edit with the `garden-ai notebook start [path/to/my.ipynb]` command. We're using a scikit-learn model and python 3.10 for this tutorial, so we'll choose the `3.10-sklearn` base image from the CLI, like so:
```bash
garden-ai notebook start tutorial_notebook.ipynb --base-image=3.10-sklearn
```


> [!NOTE] Note
> If you open the same notebook again later, you can omit the `--base-image` argument to default to the most recently used base image.

This might take a little while to download the base image (if necessary) but should automatically open the notebook in your default browser, where you can edit the notebook. You should eventually see output like this:

```bash
$ garden-ai notebook start tutorial_notebook.ipynb --base-image=3.10-sklearn
Using base image: gardenai/base:python-3.10-jupyter-sklearn
Notebook started! Opening http://127.0.0.1:8888/notebooks/tutorial_notebook.ipynb?token=791fb91ea2175a1bbf15e1c9606930ebdf6c5fe6a0c3d5bd in your default browser (you may need to refresh the page)

[stream of jupyter logs]
```

Because `tutorial_notebook.ipynb` didn't already exist, this command created and opened a [template notebook](https://github.com/Garden-AI/garden/blob/main/garden_ai/notebook_templates/sklearn.ipynb) for us. Any changes we make from there will still persist in `tutorial_notebook.ipynb` after the container has stopped.


>[!NOTE] `garden-ai notebook start` vs `jupyter notebook`
> Editing a notebook opened with `garden-ai notebook start` should feel very similar to opening that same notebook with `jupyter notebook`. However, Garden opens the notebook in a **fully isolated** Docker container (instead of whichever local environment happens to have Jupyter installed).
>
> This is a similar execution model to Google Colab, but with the container running locally instead of on Google servers. Like Colab, you'll need to `%pip install` any dependencies not already found in your selected base image at the top of your notebook.


#### Step 3: Developing a Pipeline

The notebook we've just opened is what will set up the context in which our Pipeline will run, including side-effects like installing `pip` (or `conda`) dependencies.

Depending on the base image you chose, some dependencies might already be present.

The first cell of our notebook is often just a few `pip install`s:
```ipython
%pip install garden-ai==0.6.1
%pip install scikit-learn==1.3.0
%pip install joblib==1.3.2
# %pip install some-other-library etc
```

Best practice is to pin exact versions of your dependencies wherever possible, but this is especially important for the library used to train the model and/or originally save the model weights (`scikit-learn==1.3.0` and `joblib==1.3.2` in this case).

Next, we fill out the citation metadata for our `Pipeline`:
```ipython
from garden_ai.pipelines import PipelineMetadata
my_pipeline_meta = PipelineMetadata(
    title="Tutorial Pipeline",
    authors=["me", "you", "our collaborator"],
    tags=["example", "science", "cool"],
)
```
This metadata is used to mint a DOI for the `Pipeline` automatically, and is key for the discoverability of your work. At least a title and authors are required; for a full list of allowed metadata fields, see: [Pipelines](../Pipelines.md).

Next, we'll want to make sure our `Pipeline` function will have access to the model weights we're storing on huggingface, which we do with a "Model Connector" like so:
```ipython
from garden_ai.model_connectors import HFConnector
tutorial_hf_repo = HFConnector("Garden-AI/sklearn-seedling")
```

>[!NOTE] Note
> Our `Pipeline` function could simply download the weights from huggingface directly, but by using a Model Connector like the `HFConnector` above we can automatically extract model metadata provided by huggingface and associate it with the `Pipeline` that uses the model.

We can now define a function for our pipeline which uses the `tutorial_hf_repo` connector to download the model (as well as a helper function to clean the input data):

```ipython
import pandas as pd
import joblib

def preprocess(input_df: pd.DataFrame) -> pd.DataFrame:
    input_df.fillna(0, inplace=True)
    return input_df


def run_tutorial_model(input_df: pd.DataFrame) -> pd.DataFrame:
    """Clean input data, load pretrained model weights, and run inference."""
    cleaned_df = preprocess(input_df)
    # use our model connector's `.stage()` method to download model weights
    download_path = tutorial_hf_repo.stage()
    model = joblib.load(f"{download_path}/model.joblib")
    return model.predict(cleaned_df)
```

But our `run_tutorial_model` function isn't a proper `Pipeline` yet -- we still need to mark it as a `Pipeline` and link the metadata to the function. We do this with the `@garden_pipeline` decorator:
```ipython
from garden_ai import garden_pipeline

import pandas as pd
import joblib

# helper, no need to decorate
def preprocess(input_df: pd.DataFrame) -> pd.DataFrame:
    input_df.fillna(0, inplace=True)
    return input_df


@garden_pipeline(
	metadata=my_pipeline_meta,            # link PipelineMetadata from above
	model_connectors=[tutorial_hf_repo],  # link HFConnector metadata
	garden_doi="10.23677/z2b3-3p02",      # publish to the Garden we created earlier
)
def run_tutorial_model(input_df: pd.DataFrame) -> pd.DataFrame:
    """Clean input data, load pretrained model weights, and run inference."""
    cleaned_df = preprocess(input_df)
    # use our model connector's `.stage()` method to download model weights
    download_path = tutorial_hf_repo.stage()
    model = joblib.load(f"{download_path}/model.joblib")
    return model.predict(cleaned_df)
```

Now that we've finished developing our `Pipeline`, it's a good idea to restart the kernel and run all cells to sanity-check that your pipeline function is working as expected before moving on to the publication step.

#### Step 4: Optional Debugging Step

If you're finding that a pipeline function works correctly following a "restart kernel and run all cells" when run from within a `garden-ai notebook start` session but fails on remote execution, the `garden-ai notebook debug` command is the best place to start troubleshooting.

Unfortunately, the notebook session you had active when manually running your notebook won't always be *exactly* equivalent to the session ultimately saved for remote inference. This could be for a couple reasons:
	- The remote execution context is a session which has been saved and reloaded, which may not be a perfect re-creation of the session before it was saved.
	- Your notebook is converted to a plain python script before being executed, and there may be unexpected discrepancies between the notebook and the notebook-as-script causing problems.

The `garden-ai notebook debug` command accepts the same arguments as `garden-ai notebook start`, but instead of opening the notebook in a base image, it opens a debugging notebook in an image with your notebook session saved (i.e. the one that remote inference will use).

The debugging notebook only has a snippet of code to reload your saved session -- this is as close to the remote execution context as possible without actually executing remotely.

If calling your pipeline function in the `garden-ai notebook debug` session behaves the same as calling your pipeline function in a `garden-ai notebook start` session, that likely indicates a problem with the remote endpoint.

If calling your pipeline function in the `garden-ai notebook debug` session behaves differently than in a `garden-ai notebook start` session, that indicates a problem with serializing or deserializing your notebook state. If this is the case, please open an issue on our [Github](https://github.com/Garden-AI/garden/issues), including the notebook and any additional context that might be useful so we can reproduce the bug.

#### Step 5: Publishing the Notebook

- `garden-ai notebook publish` command and successful output
- Explain how this attaches the `Pipeline` to the `Garden`.
- Emphasize that pipelines can still attach to other gardens

#### Step 6: Remote Execution
[OLD CONTENT]

Now that your Garden is published, let's see how you (or others) can find and use published Gardens.

### Discover a Garden

You can find a published Garden by searching for it using the CLI. This would list all published Gardens with `"Dee, Tweedle"` listed as an author (substitute your own name to find your garden from part 1):

```bash
garden-ai garden search --author "Dee, Tweedle"
```

Grabbing just the DOI from the output of that command (or from anywhere else this Garden may have been cited), we have everything we need to execute the pipeline on a choice Globus Compute endpoint:

```python
>>> gc = GardenClient()
>>> found_garden = gc.get_garden_by_doi('10.garden/doi')
```

### Remotely Execute a Pipeline

Once you have a Garden, you can execute any of its pipelines remotely. Make sure to specify a valid Globus Compute endpoint (or use the default tutorial endpoint):

```python
>>> my_data = pd.DataFrame(...)
>>> tutorial_endpoint = "86a47061-f3d9-44f0-90dc-56ddc642c000"
>>> results = found_garden.looking_glass_pipeline(my_data, endpoint=tutorial_endpoint)
# ... executing remotely on endpoint 86a47061-f3d9-44f0-90dc-56ddc642c000
>>> print(results)  # neat!
```

That's all there is to it! You've just developed and published your own Garden, and learned how to use published Gardens to remotely execute and reproduce your work. Happy Gardening! 🌱
