---
alias: Architecture Overview
---
## Overview

The Garden project is structured around two core concepts: `Gardens` and `Pipelines`. Each of these is represented by one or more classes in our SDK.

`Garden` and `Pipeline` objects define the *citable* and *reproducible* end products, and enable users to discover/share scientific work.

### [Pipelines](Pipelines.md)

The primary purpose of a Pipeline in the Garden framework is to make the code and models it contains _citable_ and _reproducible_. Pipelines collect enough metadata for us to do two things: mint a DOI, and build a container spec in which its code could run.

#### Creating a Pipeline


> [!NOTE] Note
> Pipeline creation is currently under construction. Check back soon. 👷🏽

#### Registering a Pipeline

> [!NOTE] Note
> Pipeline registration is currently under construction. Check back soon. 👷🏽o

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
