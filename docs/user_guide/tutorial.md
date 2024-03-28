
## Publish Your First Model in 15 Minutes

In this tutorial you will publish a trained ML model with Garden. Then you can invoke the model remotely on a Garden demo server.

![Box and arrow diagram showing the steps to publish a Garden entrypoint.](./images/Clean-Garden-Diagram.jpg)

For this walkthrough, we will deploy a simple Scikit-Learn model that classifies irises. We already have the trained model [here in a public Hugging Face repository](https://huggingface.co/Garden-AI/sklearn-seedling/tree/main). You do not need to train a model yourself for this tutorial.

### Prerequisites

- You need the Garden CLI (`garden-ai`) installed on your computer.
    - We recommend installing with [pipx](https://github.com/pypa/pipx?tab=readme-ov-file#pipx--install-and-run-python-applications-in-isolated-environments) like `pipx install garden-ai`.
    - [See here for more instructions on installing garden-ai.](installation.md)
- You need Docker installed on your computer.
    - [See here for instructions on installing Docker.](docker.md)
- You need a free Globus account.
    - [Globus](https://www.globus.org/what-we-do) is a research computing platform that Garden builds on top of. Garden uses Globus to link you to academic computing clusters that you have access to.
    - [Go here to create a free Globus account.](https://app.globus.org/) If you are a researcher at a university, we recommend logging in with your institution's SSO. You can also log in with a Google or GitHub account.
- You need to join the "Garden Users" Globus Group.
	- This is necessary in order to run code on our demo Globus Compute endpoint.
	- Join the group [here](https://app.globus.org/groups/53952f8a-d592-11ee-9957-193531752178/about).

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

This will prompt Garden to log you in with Globus if you haven't logged in from your terminal yet. You need to be logged in to create gardens. Follow the instructions and paste the authorization code in your terminal.

Then in the output of the `create` command, you should see something like:
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
This will create a new notebook foo.ipynb and open it in Docker image gardenai/base:python-3.10-sklearn.
Do you want to proceed? [y/N]: y
```

You should see output like this:

```bash
Using base image: gardenai/base:python-3.10-sklearn
Notebook started! Opening http://127.0.0.1:9188/notebooks/tutorial_notebook.ipynb in your default browser (you may need to refresh the page)

[stream of jupyter logs]
```

!!! note
    The Jupyter notebook file (tutorial_notebook.ipynb) is shared across your computer's filesystem and the container filesystem. So when you stop the containerized Jupyter server you will still have your notebook.

!!! warning
    Be sure to save your notebook manually as you go! The local Jupyter server doesn't auto-save as frequently as Colab.

### Step 3: Write a Function That Invokes Your Model

Now it's time to actually write the code that executes your model. The notebook you've opened will serve two big roles in publishing your model.

1. **Defining the "entrypoint" function that runs your model.** The notebook will contain a function called an entrypoint. This is the function that will run on a remote server when someone wants to invoke your model.
2. **Defining the context that your entrypoint function will run in.** Other code in your notebook - helper functions, import statements, etc - will be available to your entrypoint function when it is called. You can also include code that installs extra packages or edits files on the container.

Because you passed the `--tutorial` flag to `garden-ai notebook start`, you should be looking at a notebook that's already populated with code and instructions on how to run the iris model. If you don't see **Garden Tutorial Notebook** near the top of the notebook, double check your shell command and try again.

From here, just work through the tutorial notebook. Come back when you're done.

### Step 4: Publish the Function That Invokes Your Model

Now you will publish your model execution function on Garden. This will make it discoverable and usable by other researchers.

To do this, use `garden-ai notebook publish`.

```bash
$ garden-ai notebook publish tutorial_notebook.ipynb --base-image="3.10-sklearn"
```

Your output should look something like this:
```bash
Using base image: gardenai/base:python-3.10-sklearn
Preparing image ...
Building image ...
Built image: <Image: ''>
Pushing image to repository: public.ecr.aws/x2v7f8j4/garden-containers-dev
Successfully pushed image to: docker.io/public.ecr.aws/x2v7f8j4/garden-containers-dev:tutorial-20240129-101040
Added entrypoint 10.23677/58gx-0515 (classify_irises) to garden 10.23677/g5fd-3d33 (Tutorial Garden)!
(Re-)publishing garden 10.23677/g5fd-3d33 (Tutorial Garden) ...
```

Notice how Garden published your entrypoint to your tutorial garden. Now anyone can discover your garden, find your entrypoint function inside of the garden, and run the function to use your model.

### Step 5: Test Your Published Model

Now that you've published your first garden and first entrypoint function, you should invoke it remotely like a user would.

> [!NOTE] IMPORTANT
> You will need to be part of the "Garden Users" Globus Group in order to run your code remotely on our demo endpoint.
>
> Join the group [here](https://app.globus.org/groups/53952f8a-d592-11ee-9957-193531752178/about).


**The tutorial will continue in a separate notebook.** [Click here to continue in a Google Colab notebook that walks you through running your model like an end user would](https://colab.research.google.com/drive/1VM_SjYFnY1pxxac9ILQuqBT0fl3JADu0?usp=sharing). If you don't want to use Colab, you can also start a new notebook locally with `garden-ai notebook start` and follow the steps from the Colab notebook.

### Epilogue

If you've been following along, you should now have ...

1. Created your first garden
2. Written an entrypoint function that downloads and executes a simple ML model
3. Published the entrypoint function to a garden
4. Executed the model remotely via your published garden and entrypoint

Congrats! Now you know the basics of gardening. 🪴

This tutorial walked you through the bare minimum to get a model working on Garden. There are many important topics we didn't cover to keep the introductory tutorial simple. To learn how to provide custom requirements to a container, run your entrypoints on different computing endpoints, and much more, please consult [the rest of the documentation](https://garden-ai.readthedocs.io/en/latest/).
