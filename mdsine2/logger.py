import os
import sys
import errno
import logging
import logging.config
import logging.handlers


__env_key__ = "MDSINE2_LOG_INI"
__name__ = "MDSINELogger"
__ini__ = os.getenv(__env_key__, "log_config.ini")


class InfoFilter(logging.Filter):
    def filter(self, rec):
        return rec.levelno == logging.INFO


def mkdir_path(path):
    """http://stackoverflow.com/a/600612/190597 (tzot)"""
    try:
        os.makedirs(path, exist_ok=True)  # Python>3.2
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise


class MakeDirTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """
    A class which calls makedir() on the specified file path.
    """
    def __init__(self,
                 filename,
                 when='h',
                 interval=1,
                 backupCount=0,
                 encoding=None,
                 delay=False,
                 utc=False,
                 atTime=None):
        mkdir_path(os.path.dirname(filename))
        super().__init__(filename=filename,
                         when=when,
                         interval=interval,
                         backupCount=backupCount,
                         encoding=encoding,
                         delay=delay,
                         utc=utc,
                         atTime=atTime)


def default_loggers():
    # Default behavior: direct all INFO/WARNING/ERROR to stdout.
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.addFilter(InfoFilter())
    stdout_handler.setLevel(logging.INFO)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)

    logger = logging.getLogger("DefaultLogger")
    logger.setLevel(logging.INFO)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    return logger


# ============= Create logger instance. Execute once globally. ===========
__config_loaded__ = False
if not __config_loaded__:
    logging.handlers.MakeDirTimedRotatingFileHandler = MakeDirTimedRotatingFileHandler
    if not os.path.exists(__ini__):
        print("[logger.py] No logging INI file found. "
              "Create a `log_config.ini` file, "
              "or set the `{}` environment variable to point to the right configuration.".format(__env_key__))
        print("[logger.py] Loading default settings (stdout, stderr).")
        logger = default_loggers()
    else:
        try:
            logging.config.fileConfig(__ini__)
            logger = logging.getLogger(__name__)
            print("[logger.py] Loaded logging configuration from {}.".format(__ini__))
        except KeyError as e:
            print("[logger.py] Key error while looking for loggers. "
                  "Make sure INI file defines logger with key `{}` .".format(__name__))
            raise e
    __config_loaded__ = True
