import inspect
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

        s = inspect.getsource(obj)
        exec(s, __main__.__dict__)
