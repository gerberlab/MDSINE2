import os
import sys
import errno
import logging
import logging.config
import logging.handlers

from pathlib import Path
from typing import Callable


__env_key__ = "MDSINE2_LOG_INI"
__name__ = "MDSINELogger"
__ini__ = os.getenv(__env_key__, os.path.abspath("log_config.ini"))


class LoggingLevelFilter(logging.Filter):
    def __init__(self, levels):
        super().__init__()
        self.levels = levels

    def filter(self, rec):
        return rec.levelno in self.levels


class MakeDirTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """
    A class which calls makedir() on the specified file path.
    """
    def __init__(self,
                 filename,
                 when='D',
                 interval=1,
                 backupCount=0,
                 encoding=None,
                 delay=False,
                 utc=False,
                 atTime=None):
        path = Path(filename).resolve()
        MakeDirTimedRotatingFileHandler.mkdir_path(path.parent)
        super().__init__(filename=filename,
                         when=when,
                         interval=interval,
                         backupCount=backupCount,
                         encoding=encoding,
                         delay=delay,
                         utc=utc,
                         atTime=atTime)

    @staticmethod
    def mkdir_path(path):
        """http://stackoverflow.com/a/600612/190597 (tzot)"""
        try:
            os.makedirs(path, exist_ok=True)  # Python>3.2
        except TypeError:
            try:
                os.makedirs(path)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and Path(path).is_dir():
                    pass
                else:
                    raise


def default_logger(name: str):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.addFilter(LoggingLevelFilter([logging.INFO, logging.DEBUG]))
    stdout_handler.setLevel(logging.DEBUG)
    stdout_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    stdout_handler.setFormatter(stdout_formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.addFilter(LoggingLevelFilter([logging.ERROR, logging.WARNING, logging.CRITICAL]))
    stderr_handler.setLevel(logging.WARNING)
    stderr_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    stderr_handler.setFormatter(stderr_formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    return logger


def create_logger(ini_path: Path) -> logging.Logger:
    logging.handlers.MakeDirTimedRotatingFileHandler = MakeDirTimedRotatingFileHandler
    if not ini_path.exists():
        def __create_logger(module_name):
            return default_logger(name=module_name)

        this_logger = __create_logger(__name__)
        this_logger.debug("Using default logger (stdout, stderr).")
    else:
        def __create_logger(module_name):
            return logging.getLogger(name=module_name)

        logging.config.fileConfig(ini_path)
        this_logger = __create_logger(__name__)
        this_logger.debug("Using logging configuration {}".format(
            str(ini_path)
        ))

    return this_logger


# ============= Create logger instance. Execute once globally. ===========
logger = create_logger(Path(__ini__))
