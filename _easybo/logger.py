from contextlib import contextmanager
from functools import wraps
import sys
from warnings import catch_warnings

from loguru import logger


SIMPLE_LOGGER_FMT = (
    # "<fg #808080>"
    # "<fg #808080>{time:YYYY-MM-DD HH:mm:ss} "
    # "<fg #808080>{name}:{function}:{line}</> "
    "<lvl>{level: <8}</> {message}"
    # "[<lvl>{level}</>] <lvl>{message}</>"
)

LOGGER_FMT = (
    "<lvl>{level: <8}</> "
    "<fg #808080>({time:YYYY-MM-DD HH:mm:ss} "
    "{name}:{function}:{line})</> "
    # "|<lvl>{level: <10}</>| {message}"
    "{message}"
)


def generic_filter(names):
    def f(record):
        return record["level"].name in names

    return f


def set_logger_style(
    debug=False,
    debug_simple=False,
    info=True,
    info_simple=True,
    success=True,
    success_simple=True,
    warning=True,
    warning_simple=True,
    error=True,
    error_simple=True,
    critical=True,
    critical_simple=False,
):

    logger.remove(None)

    if debug:
        logger.add(
            sys.stdout,
            colorize=True,
            filter=generic_filter(["DEBUG"]),
            format=SIMPLE_LOGGER_FMT if debug_simple else LOGGER_FMT,
        )

    if info:
        logger.add(
            sys.stdout,
            colorize=True,
            filter=generic_filter(["INFO"]),
            format=SIMPLE_LOGGER_FMT if info_simple else LOGGER_FMT,
        )

    if success:
        logger.add(
            sys.stdout,
            colorize=True,
            filter=generic_filter(["SUCCESS"]),
            format=SIMPLE_LOGGER_FMT if success_simple else LOGGER_FMT,
        )

    if warning:
        logger.add(
            sys.stdout,
            colorize=True,
            filter=generic_filter(["WARNING"]),
            format=SIMPLE_LOGGER_FMT if warning_simple else LOGGER_FMT,
        )

    if error:
        logger.add(
            sys.stdout,
            colorize=True,
            filter=generic_filter(["ERROR"]),
            format=SIMPLE_LOGGER_FMT if error_simple else LOGGER_FMT,
        )

    if critical:
        logger.add(
            sys.stdout,
            colorize=True,
            filter=generic_filter(["CRITICAL"]),
            format=SIMPLE_LOGGER_FMT if critical_simple else LOGGER_FMT,
        )


def _log_warnings(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        with catch_warnings(record=True) as w:
            output = f(*args, **kwargs)
        for warning in w:
            klass = warning.category.__name__
            message = str(warning.message)
            v = vars(warning)
            logger.warning(f"{klass}: {message} | {v}")
        return output

    return wrapper


@contextmanager
def logging_mode(**kwargs):
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
