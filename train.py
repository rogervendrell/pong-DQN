"""
Part 1 — Train a DQN agent on PongNoFrameskip-v4.

Reads all settings from config.yml (infrastructure) and hyperparams/dqn.yml
(algorithm hyperparameters), then delegates to rl_zoo3 for training.

Usage:
    python train.py
"""

import sys
import yaml


def main() -> None:
    with open("config.yml") as f:
        cfg = yaml.safe_load(f)

    # Build the argument list consumed by rl_zoo3's internal argparse.
    sys.argv = [
        "train",
        "--algo",          cfg["algo"],
        "--env",           cfg["env"],
        "--n-timesteps",   str(cfg["n_timesteps"]),
        "--seed",          str(cfg["seed"]),
        "--eval-freq",     str(cfg["eval_freq"]),
        "--eval-episodes", str(cfg["eval_episodes"]),
        "--save-freq",     str(cfg["save_freq"]),
        "--log-folder",    cfg["log_folder"],
        "--conf-file",     cfg.get("conf_file", "hyperparams/dqn.yml"),
        "--device",        cfg.get("device", "auto"),
        "--tensorboard-log", f"{cfg['log_folder']}/tensorboard",
        "--progress",
        "--verbose", "1",
    ]

    if cfg.get("use_wandb"):
        sys.argv += ["--track", "--wandb-project-name", cfg["wandb_project"]]
        if cfg.get("wandb_entity"):
            sys.argv += ["--wandb-entity", cfg["wandb_entity"]]

    from rl_zoo3.train import train
    train()


if __name__ == "__main__":
    main()
