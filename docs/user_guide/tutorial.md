
## Tutorial: Develop a Garden from Scratch

This tutorial will guide you through developing a Garden from scratch: creating and registering a Pipeline, adding the Pipeline to your Garden, and publishing your Garden. It will then show you how to use published Gardens, including how to find a published Garden and execute its pipeline(s) on a remote Globus Compute endpoint of choice.

#### Prerequisites

1. The Garden CLI [installed](../installation) on your system.


### Create a Pipeline

> [!NOTE] Note
> Pipeline creation is currently under construction. Check back soon. 👷🏽


### Register a Pipeline

> [!NOTE] Note
> Pipeline registration is currently under construction. Check back soon. 👷🏽

### Create Your Garden

With your pipeline registered, it's time to create a Garden to house your pipeline(s). Use the Garden CLI to create your Garden:

```bash
garden-ai garden create \
	--title "Garden of Live Flowers" \
	--author "The Red Queen" --year 1871
```

The output of this will give you a DOI you can use to reference this garden in other commands.

### Add Pipeline to Your Garden

You can add your registered pipeline to your newly created garden using the `garden add-pipeline` subcommand, like so:

```bash
garden-ai garden add-pipeline \
	--garden='10.garden/doi' \
	--pipeline='10.pipeline/doi'
```


> [!NOTE] Note
> If adding the pipeline to an already-published Garden, you specify the Garden by its DOI. You'll need to publish the Garden again for others to see the new Pipeline, however.

### Publish Your Garden

Finally, after creating your Garden and adding the pipeline, it's time to publish your Garden:

```bash
garden-ai garden publish --garden='10.garden/doi'
```

The output of this command will contain a DOI, which you can use to cite and share your Garden, as shown in the next section of this tutorial.

Congratulations! You've just developed a Garden from scratch. Now your work is findable, accessible, interoperable and reusable 🌱.

## Using Published Gardens

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
