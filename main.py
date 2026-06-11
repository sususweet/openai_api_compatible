from models.logging_config import setup_plugin_logging
from dify_plugin import Plugin, DifyPluginEnv

setup_plugin_logging()

plugin = Plugin(DifyPluginEnv())

if __name__ == "__main__":
    plugin.run()
