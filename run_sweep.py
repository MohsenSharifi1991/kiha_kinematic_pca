import wandb
from config import get_sweep_config
from regression_activity_to_activity import run_regressor

sweep_config = get_sweep_config()
sweep_id = wandb.sweep(sweep_config, project="kia_kinematic_pca_regression")
wandb.agent(sweep_id, function=run_regressor)