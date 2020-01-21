import logging
import sys
import tqdm


# disable shitty tensorflow logging
import tensorflow as tf  # pylint: disable = unused-import
import absl.logging

logging.root.removeHandler(absl.logging._absl_handler)  # pylint: disable=protected-access
absl.logging._warn_preinit_stderr = False  # pylint: disable=protected-access


class TQDMHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def get_logger(name, level=logging.INFO, with_tqdm=True):
    if with_tqdm:
        handler = TQDMHandler()
    else:
        handler = logging.StreamHandler(stream=sys.stderr)
    formatter = logging.Formatter(
        fmt="%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def progress_bar(iterable, total, **kwargs):
    return tqdm.tqdm(iterable=iterable, total=total, ascii=True, **kwargs)
