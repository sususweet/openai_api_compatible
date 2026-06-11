import logging

from dify_plugin.config.logger_format import plugin_logger_handler


def setup_plugin_logging() -> None:
    """Route plugin code loggers to Dify stdout JSON format for daemon visibility."""
    for name in ("models", "provider"):
        log = logging.getLogger(name)
        log.setLevel(logging.INFO)
        if plugin_logger_handler not in log.handlers:
            log.addHandler(plugin_logger_handler)
        log.propagate = False
