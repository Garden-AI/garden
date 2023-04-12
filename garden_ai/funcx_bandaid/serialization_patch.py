# type: ignore

from globus_compute_sdk.serialize.base import SerializerError
from globus_compute_sdk.serialize.concretes import DillCodeSource, DillCodeTextInspect

"""
This is a temporary hack until this PR gets merged: https://github.com/funcx-faas/funcX/pull/1083
Basically FuncX is using the wrong serialization method by default on our pipeline functions.
The PR and this monkeypatched version of it make sure FuncX supports the kind of composed functions
that we submit to FuncX.
"""


def monkeypatch_method(cls):
    def decorator(func):
        original_func = getattr(cls, func.__name__, None)
        setattr(cls, func.__name__, func)
        func.original_func = original_func
        return func

    return decorator


@monkeypatch_method(DillCodeSource)
def serialize(self, *args, **kwargs):
    if args[0].__closure__ is not None:
        raise SerializerError("Payload non-local variables ignored by `getsource`.")
    return self.original_func(*args, **kwargs)


@monkeypatch_method(DillCodeTextInspect)
def serialize(self, *args, **kwargs):  # noqa: F811
    if args[0].__closure__ is not None:
        raise SerializerError("Payload non-local variables ignored by `getsource`.")
    return self.original_func(*args, **kwargs)
