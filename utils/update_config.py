

def update_config(config, sweep_config, wandb_config):
    '''
    update config file based on sweep config (parameters) and wandb config
    :param config: main config file .json
    :param sweep_config: sweep config file
    :param wandb_config: wandb config file after wandb init  by default config
    :return: updated main config file based on sweep
    '''
    for key in sweep_config['parameters']:
        config[key] = wandb_config.__getitem__(key)
    return config


def update_model_config(config, model_config):
    '''
    update config file based on sweep config (parameters) and wandb config
    :param config: main config file .json
    :param sweep_config: sweep config file
    :param wandb_config: wandb config file after wandb init  by default config
    :return: updated main config file based on sweep
    '''
    for key in model_config:
        if key in config:
            model_config[key] = config.__getitem__(key)
    return model_config