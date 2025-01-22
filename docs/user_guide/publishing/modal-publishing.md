# Publishing a Garden with Modal

!!! Warning
    Use of this feature requires that your Globus account is authorized by the Garden Team. Please [contact](mailto:thegardens@uchicago.edu) us if you are interested in using this feature.
    For another publishing flow that doesn't require special authorization, see: [Publishing with Docker](tutorial_docker.md).

## 1. Introduction

This guide will walk you through the process of publishing a Garden from a Modal App.

## Prerequisites

### Create a GitHub account

Modal uses GitHub for authentication.

If you already have a GitHub account, you're good to move to the next step. Otherwise, create a free [GitHub](https://github.com/) account before continuing.

### Create a Modal Account

Create a free Modal account by linking your GitHub: [signup](https://modal.com/signup).

!!! Note
    You'll get $30/month of free compute credit for your personal Modal account. It's totally safe to spend those credits developing and debugging your modal functions before you publish them with garden -- modal functions published through garden won't charge your personal account.


### Install the Modal CLI
Create a python virtual env and install the current modal CLI and SDK along
with the Garden CLI and SDK:
    
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install modal
    pip install garden-ai
    ```

You need to log in into modal using your GitHub account and create a token:
```bash
modal setup
```
## 2. Author a Modal App
!!! Tip
    Skip the details and see the [complete modal file](#the-complete-modal-file).

### Create a python file for your Modal App

Modal files are regular python scripts that define a modal app and decorate one 
or more functions. Create a new python file in your project:

```bash
$ touch my_app.py
```

You can also use modal's [interactive playground](https://modal.com/playground/get_started) to develop your modal function.

### Add Imports

```python
# my_app.py

# Import the Modal SDK
import modal
```

### Define the Modal App

Create a `modal.App` object where your custom functions will be registered.
Modal apps need to be assigned to a variable named `app` and at the top-level (global) scope.

```
app = modal.App("my-cool-app")
```

### Customize the container your code will run in (optional)

To make python packages available to your Modal functions, declare a `modal.Image` with your requirements.

Check out this [playground](https://modal.com/playground/custom_container) to see how it works.

```python
image = modal.Image.debian_slim().pip_install("numpy", "pandas")
```

See Modal's [Image docs](https://modal.com/docs/reference/modal.Image) for more information.

### Define Modal functions

Modal functions are regular python functions that have been decorated with `@app.function()`.
The `@app.function()` decorator registers the function with the `modal.App` created above.
Like Modal Apps, Modal functions need to be defined in the top-level (global) scope.

```python
# Define a function and register it with the app
# Functions can be named anything you like
@app.function()
def my_awesome_function(data):
  result = sum(data)
  return result


# You can register multiple functions to the same app
@app.function(image=image)
def my_other_cool_function(data):
  # import from custom image inside the modal function
  import numpy as np
  import pandas as pd

  data = np.array(data)
  result = max(sum(data), 42)
  return result

```

### The Complete Modal File

This is all we need to do to define a Modal App that can be published with Garden! See the complete modal file below:

```python
# my_app.py

# Import the Modal SDK
import modal

# Define your App -- this is the top-level entity that holds references to functions
# It must be assigned to a variable named 'app' for Garden to extact it properly
app = modal.App("my-cool-app")

# Define a custom container for your functions
image = modal.Image.debian_slim().pip_install("numpy", "pandas")

# Define a function and register it with the app
# Functions can be named anything you like
@app.function()
def my_awesome_function(data):
  result = sum(data)
  return result

# You can register multiple functions to the same app
@app.function(image=image)
def my_other_cool_function(data):
  # import from custom image inside the modal function
  import numpy as np
  import pandas as pd

  data = np.array(data)
  result = max(sum(data), 42)
  return result
```

## 3. Testing and Debugging
Garden provides a citable, findable, wrapper around the modal function. Before we
register the function with Garden it can be useful to test it with the raw modal 
runtime tools. This makes for a faster feedback loop when developing and debugging.

To enable this, we have to add a `local_entrypoint` to our python script. This is only
used by the modal CLI and will be ignored by Garden.

Add this to your python script:

```python
@app.local_entrypoint()
def main():
    # run the function on the modal servers
    print(my_awesome_function.remote([1, 2, 3, 4, 5]))
```

Ask modal to run this with:
    
```bash
modal run my_app.py
```

## 4. Specify Dependencies and Cache Model
Modal allows you to define an _image_ in which the function will run. See the modal 
docs on [images](https://modal.com/docs/guide/images) for full information. Here's
an example of how to define an image that includes some pip dependencies, installs
wget into the debian image, and downloads a model from figshare:

```python
image = modal.Image.debian_slim(python_version="3.12").apt_install("wget") \
.pip_install(
    "pandas==2.2.3",
    "torch==2.2.2",
    "torchvision==0.17.2",
    "requests==2.32.3",
) \
    .run_commands("wget -O model.pth https://figshare.com/ndownloader/files/51767333", "pwd", "ls -lht")
```

In this example, the model will be available to the function in the container at `/model.pth`.

### Caching Hugging Face Models
If you are using one of HuggingFace's published transformers, you can take advantage 
of the library's caching strategy. You need to call the python functions to load the models
during the `run_commands` phase of the image build. Here is an example of how to cache
the `distilbert-base-uncased` model:


```python
image = modal.Image.debian_slim(python_version="3.12") \
.pip_install(
    "transformers",
    "sentencepiece",
    "torch",
    "datasets") \
    .run_commands('python -c "from transformers import T5Tokenizer, T5ForConditionalGeneration; model_name = \'t5-small\'; T5Tokenizer.from_pretrained(model_name); T5ForConditionalGeneration.from_pretrained(model_name) "')
```

For models that have been published to the HuggingFace model hub, you can use the
a `git clone` to download the model and cache it in the image. Here is an example
```python
image = modal.Image.debian_slim(python_version="3.12").apt_install("wget") \
.apt_install("git") \
.pip_install(
    "torch",
    "datasets") \
    .run_commands('git clone https://huggingface.co/Arigadam/img2float', 'pwd', 'ls -lht')

```


## 5. Upload your App to Garden
Now that you have a python file defining your Modal App, you can upload it to Garden.

### Create a new Garden

Create a Garden and upload your Modal App on the [Create Garden](https://thegardens.ai/#/garden/create) page.

Or Click 'Create a Garden' from the [Garden Home Page](https://thegardens.ai)

![Create a Garden](./images/modal_publishing/create_a_garden.png)

Then click 'Get Started' on the option to create a garden from a Modal App.

![Create a Garden from Modal App](./images/modal_publishing/create_garden_from_modal_app.png)

### Fill in Garden Details

Fill in the general details about your garden including a title, description of the Garden, and any tags you want to add.

![Fill in Garden Details](./images/modal_publishing/garden_general_info.png)

### Upload the Modal App

Upload the Modal App by clicking 'Browse' and selecting the python file defining your Modal App.

![Upload Modal App](./images/modal_publishing/garden_modal_app.png)

### Add Contributors

Fill in information about the authors and any contributors to the Garden.

![Add Contributors](./images/modal_publishing/garden_contributors_and_submit.png)

### Submit the form

When the information is correct click 'Create Garden'!

It may take a few minutes for the deployment process to finish.

Check the 'My Gardens' tab on your [Profile](https://thegardens.ai/#/user) page for the new Garden. Note the DOI for the next step.

## 6. Invoke your published functions using Garden

After uploading your Modal App to Garden, you should have a new DOI referencing the Garden you created.

You can run your Modal functions like any other Entrypoint published on Garden:

```python
from garden_ai import GardenClient

# the doi of the Garden created in step 3
garden_doi = "10.1234/567-8910f"


data = [1, 2, 3, 4, 5]

client = GardenClient()
garden = client.get_garden(garden_doi)

# Function is executed remotely on Modal!
result = garden.my_awesome_function(data)
```
