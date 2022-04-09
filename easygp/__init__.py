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

FMT = "<k>{name}</>:<k>{function}</> [<lvl>{level}</>] <lvl>{message}</>"


STDOUT_LOGGER_ID = logger.add(
    sys.stdout,
    colorize=True,
    filter=generic_filter(["DEBUG", "INFO", "SUCCESS"]),
    format=FMT,
)

STDERR_LOGGER_ID = logger.add(
    sys.stderr,
    colorize=True,
    filter=generic_filter(["WARNING", "ERROR", "CRITICAL"]),
    format=FMT,
)
