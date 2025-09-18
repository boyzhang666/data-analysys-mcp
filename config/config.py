import os
import json


with open("config/config.json", "r") as f:
    config = json.load(f)


def get_env_or_config(key, default):
    return os.environ.get(key, default)


config_host = get_env_or_config("REDIS_HOST", config["Redis"]["host"])
config_port = int(get_env_or_config("REDIS_PORT", config["Redis"]["port"]))
config_database = int(get_env_or_config("REDIS_DB", config["XXX"]["database"]))
config_password = get_env_or_config("REDIS_PASSWORD", config["XXX"]["password"])
