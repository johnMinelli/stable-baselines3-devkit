"""Script for offline agent's training using Stable Baselines3."""

import os
from copy import deepcopy

import torch

from algos import *  # noqa: F401,F403
from common.callbacks import CheckpointCallback, ValidationCallback
from common.cfg_helpers import get_args, get_cfg
from common.datasets.dataloader import DataLoader, make_dataset
from common.logger import Logger
from common.utils import get_checkpoint_path

# Resume policies making sure to update/freeze the loaded policy args
OVERWRITE_POLICY_ARGS = True


def main():
    global _args_cli, _agent_cfg
    args_cli, agent_cfg = deepcopy(_args_cli), deepcopy(_agent_cfg)

    root_dir = os.path.join(
        "save",
        (
            f"{args_cli.task}_{args_cli.experiment_name}"
            if args_cli.task and args_cli.experiment_name
            else args_cli.task or args_cli.experiment_name or ""
        ),
    )
    log_folder = None
    if args_cli.resume:
        checkpoint_path = (
            args_cli.checkpoint if args_cli.checkpoint else get_checkpoint_path(root_dir, ".*", "model_.*.zip")
        )
        root_dir, log_folder = os.path.split(os.path.dirname(checkpoint_path))

    logger = Logger(args_cli, log_root=root_dir, log_folder=log_folder)

    if args_cli.sweep_id and args_cli.wandb:
        import wandb

        for k, v in wandb.run.config.items():
            k = k.split(".")
            cfg = agent_cfg
            for subk in k[:-1]:
                cfg = cfg[subk]
            cfg[k[-1]] = v

    if (args_cli.resume and OVERWRITE_POLICY_ARGS) or not args_cli.resume:
        logger.log_hp(agent_cfg, os.path.join(logger.log_dir, "params", "agent.yaml"))
    checkpoint_callback = CheckpointCallback(
        save_freq=args_cli.save_interval, save_path=logger.log_dir, name_prefix="model", verbose=2
    )
    sim_validation_callback = ValidationCallback(
        val_simulation_script=f"predict.py --task {agent_cfg['env_cfg']['task']} --envsim {agent_cfg['env_cfg']['name']} --num_envs 1 --agent Maniskill/maniskill_sl_inference_cfg --device cuda --sim_device cuda --seed -1",
        validation_freq=args_cli.val_interval,
        best_model_save_path=logger.log_dir,
        val_episodes=20,
        non_blocking_validation=False,  # if validation is slower than training, set this as False to wait the validation of the previous checkpoint before submitting another one
        verbose=1,
    )

    # dataset
    dataset_cfg = agent_cfg.pop("env_cfg")  # 'env_cfg' is the dataset config in offline training
    offline_dataset = make_dataset(dataset_cfg)
    # train_dataset, val_dataset = torch.utils.data.random_split(offline_dataset, [0.9, 0.1])  # We validate in simulation instead of on the dataset
    agent_cfg["preprocessor_kwargs"].update({"stats": offline_dataset.meta.stats})
    demonstrations_data_loader = DataLoader(
        dataset=offline_dataset,
        batch_size=agent_cfg["batch_size"],
        shuffle=True,
        num_workers=args_cli.num_workers,
        seed=agent_cfg["seed"],
        device=args_cli.device if args_cli.num_workers == 0 else torch.device("cpu"),
    )
    # val_data_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=True, num_workers=args_cli.num_workers, seed=agent_cfg["seed"], device=args_cli.device if args_cli.num_workers==0 else torch.device("cpu"))

    agent_class = eval(agent_cfg.pop("agent_class"))
    tot_timesteps = agent_cfg.pop("n_timesteps", None)

    # run agent
    if args_cli.resume:
        logger.log(f"Resuming from checkpoint {checkpoint_path}")
        agent = agent_class.load(
            checkpoint_path,
            None,
            ds_input_shapes=dataset_cfg["input_shapes"],
            ds_output_shapes=dataset_cfg["output_shapes"],
            demonstrations_data_loader=demonstrations_data_loader,
            val_data_loader=None,
            overwrite_policy_arguments=OVERWRITE_POLICY_ARGS,
            **agent_cfg,
        )
    else:
        agent = agent_class(
            ds_input_shapes=dataset_cfg["input_shapes"],
            ds_output_shapes=dataset_cfg["output_shapes"],
            demonstrations_data_loader=demonstrations_data_loader,
            val_data_loader=None,
            verbose=1,
            **agent_cfg,
        )
    agent.set_logger(logger)

    agent.learn(
        total_timesteps=(int(tot_timesteps) if tot_timesteps is not None else agent.n_epochs * len(offline_dataset))
        - agent.num_timesteps,
        callback=[checkpoint_callback, sim_validation_callback],
        log_interval=args_cli.log_interval,
        val_interval=0,
        reset_num_timesteps=False,
        progress_bar=True,
    )

    agent.save(os.path.join(logger.log_dir, "model"))


if __name__ == "__main__":
    # Get arguments
    global _args_cli, _agent_cfg
    _args_cli = get_args()

    _agent_cfg = get_cfg(_args_cli)

    # launch script
    if _args_cli.sweep_id is not None:
        import wandb

        if not _args_cli.wandb:
            print("Since `sweep_id` has been specified WandB logging is enabled in automatic!")
            _args_cli.wandb = True
        wandb.agent(_args_cli.sweep_id, main)
    else:
        main()
