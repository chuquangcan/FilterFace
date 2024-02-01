import logging
from pytorch_lightning.utilities import rank_zero_only

def get_logger(name = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")

    for l in logging_levels:
        setattr(logger, l, rank_zero_only(getattr(logger, l)))

    return logger