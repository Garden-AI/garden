import inspect
import linecache
import logging
import textwrap

logger = logging.getLogger()


def redef_in_main(obj):
    """Helper: redefine an object in __main__, e.g. garden_ai._Model -> __main__._Model.

    This has the effect of coaxing dill into serializing both the definition and
    the instance of an object together (in the case of a class), so it can be
    deserialized without needing the definition to be available for import on
    the other side. We do this for the "real" function we register, too (see below)
    in order to guarantee that there are no intra-garden references that dill might
    try to import on the other end.

    This works because dill is smart enough to know that if you defined a class
    interactively (like in a repl) then it can't expect that definition to be
    available after the session exits, so has to include it.

    The upshot is that we can embed stuff like the _Model class as an implicit part of
    the "function" that we register with globus compute, so there won't be any
    garden-specific dependencies users need to worry about in the container.
    """

    # make sure it's not already in main
    if obj.__module__ != "__main__":
        import __main__

        s = inspect.getsource(obj)
        exec(s, __main__.__dict__)


def make_func_to_serialize(pipeline):
    """
    Ensure that the composed function we register with globus compute is
    "mainified", so that it has no references to the `garden_ai` namespace.
    """
    import __main__

    redef_in_main(garden_ai_compose_functions)

    functions = [step.func for step in pipeline.steps]
    return __main__.garden_ai_compose_functions(*functions)


def garden_ai_compose_functions(*functions):
    """helper: compose functions together and inject special `_env_vars` kwarg.

    Defines other helper functions locally to prevent polluting the __main__
    namespace (see: `make_func_to_serialize`).

    Any future "pipeline middleware" (like `inject_env_kwarg`) should also be
    defined locally inside this function.
    """

    def compose(*functions):
        func, *funcs = functions

        def comp_inner(*args, **kwargs):
            result = func(*args, **kwargs)
            for f in funcs:
                result = f(result)
            return result

        return comp_inner

    def inject_env_kwarg(func):
        """
        Helper: modify a function so that it will accept an ``_env_vars`` keyword argument.

        This can be used to dynamically set environment variables before executing the
        original function, particularly useful if the function is executing remotely.
        """

        def env_inner(*args, _env_vars=None, **kwargs):
            if _env_vars:
                import os

                for k, v in _env_vars.items():
                    os.environ[k] = v
            return func(*args, **kwargs)

        return env_inner

    return inject_env_kwarg(compose(*functions))


def exec_getsource(source, globals=None, locals=None):
    """
    helper: same as built in exec, but in a way that inspect.getsource can
    still extract source code (i.e. for steps we parsed from a pipeline.py)
    """
    # https://stackoverflow.com/a/69668959
    getlines = linecache.getlines

    def monkey_patch(filename, module_globals=None):
        if filename == "<string>":
            return source.splitlines(keepends=True)
        else:
            return getlines(filename, module_globals)

    linecache.getlines = monkey_patch

    try:
        exec(source, globals, locals)
        # you can now use inspect.getsource() on the result of exec() here
    finally:
        linecache.getlines = getlines


def _load_pipeline_from_python_file(python_file):
    from garden_ai import Pipeline
    from garden_ai.mlmodel import Model
    from garden_ai.utils.filesystem import PipelineLoadException

    with open(python_file, "r") as file:
        pipeline_code = file.read()

    # dynamically create a class that we're using exclusively for the sake of a
    # simpler namespace than a ModuleType object.
    #
    # This is necessary because user code also imports garden_ai, and the more
    # robust importlib equivalent goes too far for our needs -- we want `import
    # garden_ai` in the user's code to refer to the already-imported garden_ai,
    # but something about `exec_module`'s internals does a "true" re-import of
    # garden_ai, with the unfortunate side effect of causing dill to think it
    # needs to trace/serialize far more than it actually should.  (i.e.
    # basically everything, including unserializable objects like the CLI's rich
    # console).
    code_str = f"""
class _USER_PIPELINE_MODULE:
{textwrap.indent(pipeline_code, '    ')}
"""

    # exec the user's code in a fresh namespace to define a fake `_USER_PIPELINE_MODULE` class
    local_namespace: dict = {}
    exec_getsource(code_str, {}, local_namespace)
    cls = local_namespace["_USER_PIPELINE_MODULE"]

    # Now, one of those class attributes is going to be a Pipeline instance
    for name, value in vars(cls).items():
        if isinstance(value, Pipeline):
            if not Model.has_been_called:
                logger.warning(
                    f"No model exists in file. If this pipeline includes a model, please input the model in {python_file} "
                    "with the name of a registered Garden model."
                    "\nFor more information on how to use Garden, please read our docs: "
                    "https://garden-ai.readthedocs.io/en/latest/"
                )
            value.short_name = value.short_name or name
            return value
    raise PipelineLoadException(
        "Pipeline couldn't be loaded from the user's module. Could not find pipeline object in file."
    )
