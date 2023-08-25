# Remote Pipelines without `garden-ai` as a dependency
This is a write-up aimed at core garden devs/maintainers to motivate and explain the `dill` trick that enables pipelines to run remotely without needing `garden-ai` to be installed in the container.

## Problem / Context

The user python code and pre-trained models we want to host, like anything in the python AI/ML world, is extremely prone to dependency conflicts.

The `garden-ai` package has a nontrivial set of dependencies that are liable to conflict with a user's actual dependencies. However, our dependencies are really only necessary for the user to *register* their code/models with our service, not to *run* the code once registered.

For example: the first time we ran into this ([GH issue reference](https://github.com/Garden-AI/garden/issues/131)) it was conflicting versions of the `pyyaml` package, which was only ever necessary for us to collect dependency information for the container, but totally unnecessary to have on the remote endpoint.

Taking that a step further, if we can make the function we register with Globus Compute a "garden-agnostic" function, then most of `garden-ai` itself is actually also unnecessary to have on the remote endpoint. All we really need to be able to do is download the right pre-trained model from our backend, the rest is up to the user's code.

## Solution(s)

### skinny-garden solution
One option was the "skinny-garden" (like DLHub's "home_run") approach -- identify the subset of our SDK that would be necessary on the other end, and release that as a distinct PyPI package with fewer dependencies. But managing a secondary package involves an extra layer of maintenance (and technically still leaves room for future breaking changes to affect the hosted code).

### zero-dependency solution
tl;dr:

- if `dill` sees that it will need an un-importable class in order to deserialize, it will serialize the entire definition for that class instead of just a reference.
- `dill` knows that anything defined in `__main__` (e.g. in a REPL) won't likely be importable on the other end.
- We can tactically use `exec` to make the class appear un-importable, and `dill` is forced to include the definition with the function at the time it's registered.

All we really needed on the other end was to instantiate a single class from our SDK -- at the time of writing this, `garden_ai._model._Model` encapsulates the logic to download a pre-trained model from our backend and run inference with a `.predict()` method.

`dill`, the serialization library used by Globus Compute, prefers to dump classes defined in modules "by reference" (i.e. by name/import) wherever possible, to keep the size of the serialized payload small. It also doesn't serialize an actual object instance, just a `dict` containing the object's instance attributes.

This means that `dill` would always prefer to do something like the following in order to serialize/de-serialize an instance of a `_Model`:

1. (serializing) save instance attributes: `old_dict = model.__dict__`
2. (deserializing) import the class def: `from garden_ai._model import _Model`
3. create a new empty instance: `new_obj = _Model.__new__()`
4. copy the attributes over: `new_obj.__dict__ = old_dict`

The problematic step is 2, because that needs the definition to be present and importable on the other end. But! `dill` is smart enough to know that not every class will be importable on the other end -- namely, classes defined interactively in a REPL session won't be there unless you happen to be deserializing in the exact same session, which isn't very useful.

In other words, `dill` would always try (and fail) to lookup `garden_ai._Model` on the remote endpoint, but knows better than to try and lookup `__main__._Model`. So if we can disguise `garden_ai.mlmodel._Model` as `__main__._Model`, then we persuade dill to include the `_Model` class definition as an implicit part of the payload when registered by Globus Compute.

#### References:

- [SO - why dill dumps by reference if possible](https://stackoverflow.com/a/32364403)
- [GH issue tracking discussion from SO post](https://github.com/uqfoundation/dill/issues/128)
- [author mentions that he's implemented the behavior we exploit](https://stackoverflow.com/questions/16269071/pickle-dealing-with-updated-class-definitions#comment29540363_19362849)
- [good answer for a semi-related edge case](https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory/25244948#25244948)
- [SO - mentions treating `__main__` as a module](https://stackoverflow.com/a/28094722)
- [python docs on behavior of `import __main__`](https://docs.python.org/3/library/__main__.html#import-main)
## Implementation


To force dill to include the class definition, it needs to look like it was defined in `__main__`. Luckily, this turned out to be really simple:

```python
def redef_in_main(obj):
    """Re-define an object in __main__ module namespace."""
	import __main__             # get a reference to the __main__ module
	s = inspect.getsource(obj)  # get source defining object as string
	exec(s, __main__.__dict__)  # exec that string, and store the result in __main__'s globals
```

So after running something like `redef_in_main(garden_ai._model._Model)`, we could instantiate an equivalent object two equivalent ways:

- normal way: `mymodel1 = garden_ai._model._Model(...)`
- tricky way: `mymodel2 = __main__._Model(...)`

And because they are using the exact same definition verbatim, the only actual difference is the namespace they're defined in, which is just enough for dill to include that definition in the payload. (note that python still treats them as different classes, so `isinstance` checks etc have to match)

We also "mainify" the function that we register with Globus Compute when we build it (see `utils._meta.make_func_to_serialize`), because dill would pick up a reference to a `garden_ai` namespace. Otherwise that fails for the same reason as `garden_ai._Model` would on the other end.

### sub-hack: loading pipeline obj from user code
After implementing the trick above, I found myself able to register a pipeline function no problem directly from a test script which just called `client.register_pipeline(...)`, but when I tried to register the same exact pipeline using the CLI like `garden-ai pipeline register pipeline.py` it would spectacularly fail to even serialize the function at all.

For reasons I'm honestly still pretty hazy on, our existing helper to dynamically load a pipeline from a user's python file caused dill to accidentally pick up a reference to global stuff like `rich.Console` objects which aren't serializable. I am pretty sure this was because the user's code would itself `import garden_ai`, and some of whatever python internals the builtin `importlib` manipulate combined with our hack above confused `dill` too far in the wrong direction, and it wanted to serialize *everything*.

Fortunately, we didn't actually need to interact with the user's module as a robust `ModuleType` object in order to extract the single `Pipeline` object, we just wanted everything to be in an appropriate namespace. The less-robust but much simpler alternative is to use exec to define a dummy class that just contains all the user's code as class attributes. Executing the user's `import garden_ai` line (inside the body of the dummy class) is now a noop, since we're clearly already imported at the top level.

This was a little bit more involved than `redef_in_main` but is a basically similar idea:
```python
...
code_str = f"""
class _USER_PIPELINE_MODULE:
{textwrap.indent(pipeline_code, '    ')}
"""

    # run user's code as if defining `__main__._USER_PIPELINE_MODULE` class
    local_namespace: dict = {}
    exec(code_str, __main__.__dict__, local_namespace)
    cls = local_namespace["_USER_PIPELINE_MODULE"]

    # Now, one of those class attributes is going to be a Pipeline instance
    for name, value in vars(cls).items():
        if isinstance(value, Pipeline):
            return value

```

### possible other/related workarounds:
This is a relatively recent GH issue that asks for exactly the feature our workaround accomplishes (forcing by-value serialization for specific classes) and seems like there's some good discussion ongoing: https://github.com/uqfoundation/dill/issues/424

it also might be worth looking into some of the builtin `runpy` module helpers as an `importlib` alternative that doesn't require the `"class _USER_PIPELINE_MODULE:..."` hack.

## misc future debugging notes

If this breaks in the future, it's likely because we just don't have very much control over dill's behavior, which could change due to either changes in dill itself and/or globus compute's use of dill.

I'm not *too* worried though, because the behavior we're exploiting is a pretty high priority for dill as a library (being able to dump to/from interactive sessions) and we don't make any assumptions about how Globus Compute configures it (currently they use default dill behavior). Maybe Globus Compute switches to `cloudpickle` or something instead of `dill`, but I don't think that's likely either.

If you've taken the REPL-pill (talking to you, future me) for testing small changes you need to be extra mindful -- remember that the crux of this hack is that dill's behavior in `__main__` is different than its typical behavior.

Specifically, in interactive sessions or top-level scripts, we always have `__name__ == '__main__'`, but in general that won't be the case anywhere our library code is running on behalf of a user / from the CLI. (This is what made the problem behind the sub-hack above so tricky to diagnose)

Extremely useful is the combo of chatgpt and the dill helper function, `dill.detect.trace`, which prints out what's dill is trying to serialize any time it pickles an object. This is already really handy, but copy-pasting that output and asking chatgpt to guess what it means was orders of magnitude better. Likewise with the output of `pickletools.dis`. The big thing to watch out for is stray references to `garden_ai` and/or unusually large output from either.
