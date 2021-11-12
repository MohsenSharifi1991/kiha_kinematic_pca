import json
# import yaml


def get_config():
    with open('./configs/config.json') as f:
    # with open('F:/codes/python/opensim_trial/configs/config.json') as f:
        config = json.load(f)
    return config

def get_config_hip():
    with open('./configs/config_hip.json') as f:
    # with open('F:/codes/python/opensim_trial/configs/config.json') as f:
        config = json.load(f)
    return config

def get_iterconfig():
    with open('./configs/iterconfig.json') as f:
        config = json.load(f)
    return config


def get_iterconfig_hip():
    with open('./configs/iterconfig_hip.json') as f:
        config = json.load(f)
    return config


def get_sweep_config():
    with open('./configs/sweep_config.yaml') as f:
        config = yaml.load(f)
    return config


def get_sweep_config_hip():
    with open('./configs/sweep_config_hip.yaml') as f:
        config = yaml.load(f)
    return config


def get_sweep_iterconfig():
    with open('./configs/sweep_iterconfig.yaml') as f:
        config = yaml.load(f)
    return config


def get_sweep_iterconfig_hip():
    with open('./configs/sweep_iterconfig_hip.yaml') as f:
        config = yaml.load(f)
    return config

def get_model_config(model_config):
    with open(f'./configs/{model_config}.json') as f:
        config = json.load(f)
    return config