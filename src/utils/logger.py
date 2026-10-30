from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

# src/utils/logger.py -> parents[2] is the project root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
LOG_DIR: Path = PROJECT_ROOT / "logs"

LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

CONSOLE_LEVEL: int = logging.INFO
FILE_LEVEL: int = logging.DEBUG

# Attribute set on handlers created here, so repeat calls can detect them.
_HANDLER_TAG: str = "_efims_handler"

# Ensures the "file logging unavailable" warning is only emitted once.
_file_logging_warned: bool = False


def _has_handler(logger: logging.Logger, kind: str) -> bool:
    """Return True if ``logger`` already has an efims handler of ``kind``."""
    return any(getattr(h, _HANDLER_TAG, None) == kind for h in logger.handlers)


def _build_file_handler(formatter: logging.Formatter) -> logging.FileHandler | None:
    """Create the dated file handler, or return None if the file is unwritable."""
    global _file_logging_warned

    log_path = LOG_DIR / f"efims_{date.today():%Y%m%d}.log"
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
    except OSError as error:
        if not _file_logging_warned:
            _file_logging_warned = True
            sys.stderr.write(
                f"[efims logger] File logging disabled, could not open "
                f"'{log_path}': {error}. Continuing with console output only.\n"
            )
        return None

    handler.setLevel(FILE_LEVEL)
    handler.setFormatter(formatter)
    setattr(handler, _HANDLER_TAG, "file")
    return handler


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to the console and to today's log file.

    Safe to call any number of times with the same name: handlers are only
    attached once, so messages are never duplicated.

    Args:
        name: Logger name, normally ``__name__`` of the calling module
            (e.g. ``"src.models.severity.trainer"``).

    Returns:
        A configured ``logging.Logger``. The console handler emits INFO and
        above; the file handler (if the log file could be opened) emits DEBUG
        and above to ``logs/efims_YYYYMMDD.log`` under the project root.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # Stop records bubbling up to the root logger, which would print them twice
    # if any library (or notebook) has configured root handlers.
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if not _has_handler(logger, "console"):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(CONSOLE_LEVEL)
        console_handler.setFormatter(formatter)
        setattr(console_handler, _HANDLER_TAG, "console")
        logger.addHandler(console_handler)

    if not _has_handler(logger, "file"):
        file_handler = _build_file_handler(formatter)
        if file_handler is not None:
            logger.addHandler(file_handler)

    return logger
