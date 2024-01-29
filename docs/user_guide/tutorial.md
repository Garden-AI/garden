
## Publish Your First Model in 15 Minutes

In this tutorial you will publish a trained ML model with Garden. Then you can invoke the model remotely on a Garden demo server.

![Box and arrow diagram showing the steps to publish a Garden entrypoint.](./images/Clean-Garden-Diagram.jpg)

For this walkthrough, we will deploy a simple Scikit-Learn model that classifies irises. We already have the trained model [here in a public Hugging Face repository](https://huggingface.co/Garden-AI/sklearn-seedling/tree/main). You do not need to train a model yourself for this tutorial.

### Prerequisites

- You need the Garden CLI (`garden-ai`) installed on your computer.
    - We recommend installing with [pipx](https://github.com/pypa/pipx?tab=readme-ov-file#pipx--install-and-run-python-applications-in-isolated-environments) like `pipx install garden-ai`.
    - [See here for more instructions on installing garden-ai.](user_guide/installation.md)
- You need Docker installed on your computer.
    - [See here for instructions on installing Docker.](user_guide/docker.md)
- You need a Globus account. (TODO: explain and link)

Confirm you have Garden and Docker installed.

```bash
garden-ai docker check
```

If you see a flower you're good to go.

```bash
Docker is running and accessible. Happy Gardening!

     /\^/`\
    | \/   |
    | |    |
    \ \    /
     '\\//'
       ||
       ||
       ||
       ||  ,
   |\  ||  |\
   | | ||  | |
   | | || / /
    \ \||/ /
jgs  `\//`
    ^^^^^^^^
```

!!! info

    If `garden-ai docker check` reports issues with Docker, [look at our Docker troubleshooting guide.](docker.md#troubleshooting-your-installation)

### Step 1: Create a Garden

Gardens are collections of related machine learning models that you can curate. A chemist might create a garden of interatomic potential predictiors to compare different frameworks and architectures.

Create a simple garden so you can add your model to it later.

```
garden-ai garden create \
    --title "Tutorial Garden" \
	--author "Your Name"
```

In the output of that command, you should see something like:
```bash
...
Garden 'Tutorial Garden' created with DOI: 10.23677/z2b3-3p02
```

Make note of this DOI for later.

!!! info

    You can also use `garden-ai garden list` to list all of your gardens and see their DOIs.


### Step 2: Start a Notebook in an Isolated Environment

Garden creates an isolated Docker environment for you to write and test code that executes your model. This helps Garden save and recreate the exact environment you need to run your model on remote servers. Garden opens a Jupyter notebook from within the isolated Docker environment that you can edit from your web browser.

Start your notebook in an isolated environment.

```bash
garden-ai notebook start tutorial_notebook.ipynb --base-image=3.10-sklearn --tutorial
```

!!! info
    The `base-image` option lets you pick a premade environment, or "image" in Docker-speak. This lets you start off with the Python version and ML framework that you need. You can still install more packages within this base environment.

    We will be serving a Scikit-Learn model trained and serialized with Python 3.10, so we will use the `3.10-sklearn` base image. Use `garden-ai notebook list-premade-images` to see other base images you can choose from.

Garden will ask you to confirm if you want to do this. Type `y` and hit enter.

```bash
This will create a new notebook foo.ipynb and open it in Docker image gardenai/base:python-3.10-jupyter-sklearn.
Do you want to proceed? [y/N]: y
```

You should see output like this:

```bash
Using base image: gardenai/base:python-3.10-jupyter-sklearn
Notebook started! Opening http://127.0.0.1:8888/notebooks/tutorial_notebook.ipynb in your default browser (you may need to refresh the page)

[stream of jupyter logs]
```

You'll notice that your notebook already has some code. When Garden creates a notebook from scratch, it includes instructions and sample code to help get you started. You won't need the sample code for this tutorial so feel free to delete these premade cells.

!!! note
    The Jupyter notebook file (tutorial_notebook.ipynb) is shared across your computer's filesystem and the container filesystem. So when you stop the containerized Jupyter server you will still have your notebook.

!!! warning
    Be sure to save your notebook manually as you go! The local Jupyter server doesn't auto-save as frequently as Colab.

### Step 3: Write a Function That Invokes Your Model

Now it's time to actually write the code that executes your model. The notebook you've opened will serve two big roles in publishing your model.

1. **Defining the "entrypoint" function that runs your model.** The notebook will contain a function called an entrypoint. This is the function that will run on a remote server when someone wants to invoke your model.
2. **Defining the context that your entrypoint function will run in.** Other code in your notebook - helper functions, import statements, etc - will be available to your entrypoint function when it is called. You can also include code that installs extra packages or edits files on the container.

Because you passed the `--tutorial` flag to `garden-ai notebook start`, you should be looking at a notebook that's already populated with code and instructions on how to run the iris model. From here, just work through the notebook. Come back when you're done.

### Step 4: Publish the Function That Invokes Your Model

Now you will publish your model execution function on Garden. This will make it discoverable and usable by other researchers.

To do this, you just need to use `garden-ai notebook publish` like so.

```bash
$ garden-ai notebook publish tutorial_notebook.ipynb --base-image="3.10-sklearn"
```

Your output should look something like this:
```bash
$ garden-ai notebook publish tutorial_notebook.ipynb --base-image="3.10-sklearn"
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
