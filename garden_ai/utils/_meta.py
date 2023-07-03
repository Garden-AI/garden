import inspect
import linecache
import textwrap

from garden_ai import Pipeline


def __main__ify(cls):
    """Helper: redefine a class in __main__, e.g. garden_ai._Model -> __main__._Model.

    This has the effect of coaxing dill into serializing both the definition and
    the instance of this object together, so it can be deserialized without
    needing the definition to be available for import on the other side.

    This works because dill is smart enough to know that if you defined a class
    interactively (like in a repl) then it can't expect that definition to be
    available after the session exits, so has to include it.

    The upshot is that we can embed stuff like the _Model class as an implicit part of
    the "function" that we register with globus compute, so there won't be any
    garden-specific dependencies users need to worry about in the container.
    """

    # make sure it's not already in main
    if cls.__module__ != "__main__":
        import __main__

        s = inspect.getsource(cls)
        exec(s, __main__.__dict__)


def make_func_to_serialize(pipeline):
    """
    Ensure that the composed function we register with globus compute is
    "mainified", so that it has no references to the `garden_ai` namespace.
    """
    import __main__

    __main__ify(garden_ai_compose_functions)

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

        def inner(*args, **kwargs):
            result = func(*args, **kwargs)
            for f in funcs:
                result = f(result)
            return result

        return inner

    def inject_env_kwarg(func):
        """
        Helper: modify a function so that it will accept an ``_env_vars`` keyword argument.

        This can be used to dynamically set environment variables before executing the
        original function, particularly useful if the function is executing remotely.
        """

        def inner(*args, _env_vars=None, **kwargs):
            if _env_vars:
                import os

                for k, v in _env_vars.items():
                    os.environ[k] = v
            return func(*args, **kwargs)

        return inner

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
    import __main__

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

    # namespace the user's code at `__main__._USER_PIPELINE_MODULE`
    # dill behaves most predictably when it's serializing something in
    # __main__
    local_namespace: dict = {}
    exec(code_str, __main__.__dict__, local_namespace)
    # exec(code_str, {}, local_namespace)  # TODO see if this could work
    cls = local_namespace["_USER_PIPELINE_MODULE"]

    # Now, one of those class attributes is going to be a Pipeline instance
    for name, value in vars(cls).items():
        if isinstance(value, Pipeline):
            return value
    raise Exception(f"Did not find top-level pipeline object in {python_file}.")
