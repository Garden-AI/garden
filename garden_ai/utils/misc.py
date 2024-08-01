import functools
import logging

from typing_extensions import TypeAlias

JSON: TypeAlias = str

logger = logging.getLogger()


def trackcalls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # note: attribute set only after func completes execution
        # so that func itself can determine if it's been called
        try:
            return func(*args, **kwargs)
        finally:
            wrapper.has_been_called = True

    wrapper.has_been_called = False
    return wrapper
