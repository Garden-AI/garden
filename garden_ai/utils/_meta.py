import inspect
import linecache
import logging

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

        source = inspect.getsource(obj)
        exec(source, __main__.__dict__)


def exec_getsource(source, globals=None, locals=None):
    """
    helper: same as built in exec, but in a way that inspect.getsource can
    still extract the source code of objects defined in the appropriate namespace.

    This lets us use the default `DillCodeTextInspect` serialization
    strategy with Globus Compute.
    """
    # https://stackoverflow.com/a/69668959
    getlines = linecache.getlines

    def monkey_patch(filename, module_globals=None):
        if filename == "<string>":
            return source.splitlines(keepends=True)
        else:
            return getlines(filename, module_globals)

    # can now use inspect.getsource() on the result of exec() here
    linecache.getlines = monkey_patch
    exec(source, globals, locals)


def make_function_to_register(func_name: str):
    """Dynamically define a simple `call_<func_name>` function to be registered with Globus Compute.

    The generated function naively calls whatever function the given `func_name`
    resolves to in the `__main__` namespace after loading a saved dill session.

    Note that `__main__` is the namespace dill targets by default because when
    we save the session from a script, we're already executing as the `__main__`
    namespace. So we don't need to explicitly `import __main__` in the serialization script (scripts.),
    just the deserialization side (which is not going to be running as `__main__`).

    Parameters:
    - func_name (str): The name of the saved function to be called by the proxy function we actually register.

    Returns:
    - The proxy function to be registered with Globus Compute. Should be serialized with
        DillCodeTextInspect strategy, not DillCode.
    """
    source = f"""
def call_{func_name}(*args, **kwargs):
    import __main__
    import dill

    dill.load_session("/garden/session.pkl")

    return __main__.{func_name}(*args, **kwargs)
"""

    # use exec_getsource instead of exec so that we can register plain source code instead of code objects
    exec_getsource(source, globals(), locals())

    return locals()[f"call_{func_name}"]
