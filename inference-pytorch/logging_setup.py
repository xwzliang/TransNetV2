import logging
import os
import time
from logging import LogRecord

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "server.log")
MAX_BYTES = 5 * 1024 * 1024  # 5MB


class TimestampRotatingFileHandler(logging.Handler):
    def __init__(self, filename=LOG_FILE):
        super().__init__()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.filename = filename
        self._handler = logging.FileHandler(self.filename, encoding="utf-8")
        self._handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )

    def emit(self, record: LogRecord):
        try:
            self._rotate_if_needed()
            self._handler.emit(record)
        except Exception:
            self.handleError(record)

    def _rotate_if_needed(self):
        try:
            if (
                os.path.exists(self.filename)
                and os.path.getsize(self.filename) >= MAX_BYTES
            ):
                timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                backup_name = f"{self.filename}.{timestamp}"
                # close current handler before renaming
                self._handler.close()
                os.replace(self.filename, backup_name)
                # recreate handler
                self._handler = logging.FileHandler(self.filename, encoding="utf-8")
                self._handler.setFormatter(
                    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
                )
        except Exception:
            pass  # rotation is best-effort


def get_logger(name="video_summarizer") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(logging.INFO)
    handler = TimestampRotatingFileHandler()
    logger.addHandler(handler)
    # also log to stdout
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console)
    # also capture Uvicorn's logs
    for uv_logger in ("uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(uv_logger)
        lg.addHandler(handler)
    return logger
