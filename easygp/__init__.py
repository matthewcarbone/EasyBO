import sys

from loguru import logger as logger

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


def generic_filter(names):
    if names == "all":
        return None

    def f(record):
        return record["level"].name in names

    return f


# Remove the default logger
logger.remove(0)

FMT = (
    "<fg #808080>{name}</>:<fg #808080>{function}</> [<lvl>{level}</>] "
    "<lvl>{message}</>"
)


def set_stdout_logger(filters=["DEBUG", "INFO", "SUCCESS"], fmt=FMT):
    global STDOUT_LOGGER_ID
    STDOUT_LOGGER_ID = logger.add(
        sys.stdout,
        colorize=True,
        filter=generic_filter(filters),
        format=FMT,
    )


def set_stderr_logger(filters=["WARNING", "ERROR", "CRITICAL"], fmt=FMT):
    global STDERR_LOGGER_ID
    STDERR_LOGGER_ID = logger.add(
        sys.stderr,
        colorize=True,
        filter=generic_filter(filters),
        format=FMT,
    )


def remove_stdout_logger():
    logger.remove(STDOUT_LOGGER_ID)


def remove_stderr_logger():
    logger.remove(STDERR_LOGGER_ID)


set_stdout_logger()
set_stderr_logger()
