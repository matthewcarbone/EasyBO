from contextlib import contextmanager
import sys

# from unittest.mock import MagicMock

from loguru import logger


SIMPLE_LOGGER_FMT = (
    # "<fg #808080>"
    "<fg #808080>{time:YYYY-MM-DD HH:mm:ss} "
    "{name}:{function}:{line}</> "
    "[<lvl>{level}</>] {message}"
    # "[<lvl>{level}</>] <lvl>{message}</>"
)

LOGGER_FMT = (
    "<fg #808080>{time:YYYY-MM-DD HH:mm:ss} "
    "{name}:{function}:{line}</> "
    # "|<lvl>{level: <10}</>| {message}"
    "[<lvl>{level}</>] {message}"
)


def generic_filter(names):
    def f(record):
        return record["level"].name in names

    return f


def set_logger_style(
    debug=False,
    debug_simple=False,
    stdout=True,
    stdout_simple=True,
    stderr=True,
    stderr_simple=False,
):

    logger.remove(None)

    if debug:
        logger.add(
            sys.stdout,
            colorize=True,
            filter=generic_filter(["DEBUG"]),
            format=SIMPLE_LOGGER_FMT if debug_simple else LOGGER_FMT,
        )

    if stdout:
        logger.add(
            sys.stdout,
            colorize=True,
            filter=generic_filter(["INFO", "SUCCESS"]),
            format=SIMPLE_LOGGER_FMT if stdout_simple else LOGGER_FMT,
        )

    if stderr:
        logger.add(
            sys.stdout,
            colorize=True,
            filter=generic_filter(["WARNING", "ERROR", "CRITICAL"]),
            format=SIMPLE_LOGGER_FMT if stderr_simple else LOGGER_FMT,
        )


@contextmanager
def mode(**kwargs):
    set_logger_style(**kwargs)
    try:
        yield None
    finally:
        set_logger_style()


set_logger_style()


# class LimitedLogger:

#     def __getattr__(self, attr):
#         if self._limit is None or self._count < self._limit:
#             self._count += 1
#             return logger.__getattribute__(attr)
#         else:
#             return MagicMock()

#     def __init__(self, limit=None):
#         self._limit = limit
#         self._count = 0


# logger2 = LimitedLogger(limit=2)
