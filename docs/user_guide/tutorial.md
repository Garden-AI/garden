
## Tutorial: Develop a Garden and Entrypoint from Scratch

This tutorial will walk you through a simple end-to-end example of developing a single `Entrypoint` from scratch in a notebook; publishing it to a new `Garden`; and finally running the `Entrypoint` remotely via that `Garden`.

This means doing the following:
- Create a new Garden using the CLI (`garden-ai garden create`)
- Open a notebook using the CLI (`garden-ai notebook start`)
- Define and decorate a function in the notebook to make it an `Entrypoint`
- Publish the notebook using the CLI, attaching the `Entrypoint` to the `Garden` in the process (`garden-ai notebook publish`)
- Test remote execution on our tutorial endpoint
#### Prerequisites
- We've used huggingface to host the [sample model weights](https://huggingface.co/Garden-AI/sklearn-seedling/tree/main) for this tutorial. If you're following along with your own pretrained model weights, you will likely also want a huggingface account to do the same.

### Step 0: Train Your Model
Garden is geared towards publishing and sharing models which have already been trained. If you don't have one handy, we'd recommend training a small model using one of the scikit-learn [toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html) and uploading the weights to a public huggingface repo in order to follow along.

The code to train the toy model used in this tutorial is available [here](https://huggingface.co/Garden-AI/sklearn-seedling/blob/main/Train_Model.ipynb).

### Step 1: Creating a New Garden
First, we're going to make a new `Garden` using the CLI so that we can eventually share our `Entrypoint` with others. Before we've added an `Entrypoint`, a brand new `Garden` is going to be nothing more than some citation metadata, which is at least a title, one or more authors, and a year (the minimum needed mint a DOI).

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

Make a note of this DOI for later, so we can publish our `Entrypoint` to this specific `Garden` (see also: `garden-ai garden list` to list all local `Gardens`).

### Step 2: Starting the Notebook

We're going to define our `Entrypoint` in a regular Jupyter notebook file (`.ipynb`) on our local filesystem, which we can open and edit with the `garden-ai notebook start [path/to/my.ipynb]` command. We're using a scikit-learn model and python 3.10 for this tutorial, so we'll choose the `3.10-sklearn` base image from the CLI, like so:
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

Because `tutorial_notebook.ipynb` didn't already exist, this command created and opened a [template notebook](https://github.com/Garden-AI/garden/blob/main/garden_ai/notebook_templates/sklearn.ipynb) for us. Any changes we make from here will still persist in `tutorial_notebook.ipynb` after the container has stopped.


>[!NOTE] `garden-ai notebook start` vs `jupyter notebook`
> Editing a notebook opened with `garden-ai notebook start` should feel very similar to opening that same notebook with `jupyter notebook`. However, Garden opens the notebook in a **fully isolated** Docker container (instead of whichever local environment happens to have Jupyter installed).
>
> This is a similar execution model to Google Colab, but with the container running locally instead of on Google servers. Like Colab, you can `%pip install` any dependencies not already found in your selected base image at the top of your notebook.

For managing dependencies that aren't included in the Garden base images, we recommend passing a `requirements` file. You can provide a pip-style (requirements.txt) or conda-style (conda.yml) requirements file with the `requirements` option, and Garden will install the requirements into the container before starting your notebook session.

```bash
$ garden-ai notebook start tutorial_notebook.ipynb --base-image=3.10-sklearn --requirements=conda.yml

```

>[!NOTE] Note
> Garden does not remember your `requirements` file across commands, so if you have one be sure to specify it every time.


### Step 3: Developing the Entrypoint

The notebook is where we define the entrypoint function, but it is also how we set up the context in which our `Entrypoint` will run, including side-effects like installing `pip` (or `conda`) dependencies.


> [!NOTE] Note
> Any side-effects like installing packages or downloading files will be "baked in" to the image we build when publishing the notebook. This means that the container used for remote inference will already have these installed, even if they weren't part of the base image.
>

The first cell of our notebook is often just a few `pip install`s:
```ipython
%pip install garden-ai==0.6.1
%pip install scikit-learn==1.3.0
%pip install joblib==1.3.2
# %pip install some-other-library etc
```

Best practice is to pin exact versions of your dependencies wherever possible, but this is especially important for the library used to train the model and/or originally save the model weights (`scikit-learn==1.3.0` and `joblib==1.3.2` in this case).

> [!NOTE] Note
> Doing `%pip install`s in the notebook is often very helpful for interactively experimenting with packages while coding. But once you've figured out all the packages and versions you need, we recommend pulling them out into a `requirements` file that you can pass to Garden with the `--requirements` flag.

Next, we fill out the citation metadata for our `Entrypoint`:
```ipython
from garden_ai.entrypoints import EntrypointMetadata
my_entrypoint_meta = EntrypointMetadata(
    title="Irises, Classified through the Looking-Glass",
    authors=["Dee, Tweedle", "Dum, Tweedle", "Alice"],
    tags=["fiction", "science"],
    short_name="looking_glass_entrypoint",
)
```
This metadata is used to mint a DOI for the `Entrypoint` automatically, and is key for the discoverability of your work. A title and list of at least one author are required; for a full list of allowed metadata fields, see: [Entrypoints](../Entrypoints.md).


> [!NOTE] the `short_name` field
> Calling the `Entrypoint` from a `Garden` object uses method-like syntax, e.g. `my_garden.some_entrypoint(*args, **kwargs)`. The `some_entrypoint` identifier is set by the Entrypoint's `short_name` field, and by default is just the underlying function's name.


Next, we'll want to make sure our `Entrypoint` function will have access to the model weights we're storing on huggingface, which we do with a "Model Connector" like so:
```ipython
from garden_ai.model_connectors import HFConnector
tutorial_hf_repo = HFConnector("Garden-AI/sklearn-seedling")
```

>[!NOTE] Note
> Our `Entrypoint` function could simply download the weights from huggingface directly, but by using a Model Connector, like the `HFConnector` above, we can automatically extract any model metadata provided by huggingface and associate it with the `Entrypoint` that uses the model.

We can now define a function for our `Entrypoint` which uses the `tutorial_hf_repo` connector to download the model (as well as a helper function to clean the input data):

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

But our `run_tutorial_model` function isn't a proper `Entrypoint` yet -- we still need to mark it as an `Entrypoint` and link the `EntrypointMetadata` to the function. We do this with the `@garden_entrypoint` decorator:

```ipython
from garden_ai import garden_entrypoint

import pandas as pd
import joblib

# helper, no need to decorate
def preprocess(input_df: pd.DataFrame) -> pd.DataFrame:
    input_df.fillna(0, inplace=True)
    return input_df


@garden_entrypoint(
	metadata=my_entrypoint_meta,            # link EntrypointMetadata from above
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

Now that we've finished developing our `Entrypoint`, it's a good idea to restart the kernel and run all cells to sanity-check that your entrypoint function is working as expected before moving on to the publication step.


### Step 4: Publishing the Notebook

Finally, we're ready to finalize and publish our `Entrypoint`, making it reproducible and discoverable as part of a `Garden`.


The only thing we need to do now is call `garden-ai notebook publish` with our notebook path and our image repository, like so:

```bash
$ garden-ai notebook publish tutorial_notebook.ipynb --requirements=conda.yml
```

Your output should look something like this:
```bash
$ garden-ai notebook publish tutorial_notebook.ipynb
Using base image: gardenai/base:python-3.10-jupyter-sklearn
Using image repository: johntenniel/garden-images
Building image ...
Built image: <Image: ''>
Pushing image to repository: johntenniel/garden-images
Successfully pushed image to: docker.io/johntenniel/garden-images:tutorial_notebook-20231130-135145
Added entrypoint 10.23677/stg7-cr32 (looking_glass_entrypoint) to garden 10.23677/z2b3-3p02 (Garden of Live Flowers)!
(Re-)publishing garden 10.23677/z2b3-3p02 (Garden of Live Flowers) ...
```

> [!NOTE] Adding to other Gardens
> Because we specified a `Garden` we wanted to publish to in the notebook itself (via an argument to the decorator), this automatically adds the `Entrypoint` to and re-publishes that `Garden` so others can discover and invoke it.
>
> If you want to add an `Entrypoint` to a `Garden` that wasn't specified when its notebook was first published, you can do so from the CLI with `garden-ai garden add-entrypoint`, then `garden-ai garden publish` to re-publish the `Garden`.

### Step 6: Remote Execution

Now that your Garden is published, let's see how you (or others) can find and use published Gardens.

#### Discover a Garden

If you don't already have the `Garden` DOI, you can find a published `Garden` by searching for it using the CLI. For example, this would list all published `Gardens` with "The Red Queen" listed as an author:

```bash
garden-ai garden search --author "The Red Queen"
```

Once we have the `Garden` DOI, we have everything we need to remotely execute any of its `Entrypoints` on a choice Globus Compute endpoint:

```python
>>> gc = GardenClient()
>>> live_flower_garden = gc.get_published_garden('10.23677/z2b3-3p02')
```

#### Remotely Execute an Entrypoint

Once you have the `Garden` object, you can execute any of its `Entrypoints` remotely. Make sure to specify a valid Globus Compute endpoint (or use the tutorial endpoint below):

```python
>>> my_data = pd.DataFrame(...)
>>> tutorial_endpoint = "86a47061-f3d9-44f0-90dc-56ddc642c000"
>>> results = live_flower_garden.looking_glass_entrypoint(my_data, endpoint=tutorial_endpoint)
# ... executing remotely on endpoint 86a47061-f3d9-44f0-90dc-56ddc642c000
>>> print(results)  # neat!
```

That's all there is to it! Happy Gardening 🌱
