# Pong DQN — RL Activity 1

Training and evaluation of a **Deep Q-Network (DQN)** agent on the
[Pong](https://ale.farama.org/environments/pong/) Atari environment using
[RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) and
[Weights & Biases](https://wandb.ai) for experiment tracking.

---

## Repository structure

```
pong-DQN/
├── hyperparams/
│   └── dqn.yml        # Algorithm hyperparameters (read by rl_zoo3)
├── config.yml         # Training infrastructure settings (seed, eval_freq, WandB, etc.)
├── train.py           # Part 1 — reads config.yml and launches rl_zoo3
├── evaluate.py        # Part 2 — evaluation + GIF export
├── requirements.txt   # Python dependencies
└── README.md
```

---

## Setup

### 1. Create and activate the virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Atari ROMs

`autorom` (included in requirements) downloads and installs the ROMs automatically:

```bash
AutoROM --accept-license
```

### 4. Log in to Weights & Biases (optional but recommended)

```bash
wandb login
```

---

## Part 1 — Training

```bash
# Default: 10 M steps, WandB enabled, seed 42
python train.py

# Quick smoke-test (no WandB):
python train.py --timesteps 100000 --no-wandb

# Custom WandB project and seed:
python train.py --wandb-project pong-dqn --seed 0
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--timesteps` | `10000000` | Total training steps |
| `--seed` | `42` | Random seed |
| `--eval-freq` | `50000` | Steps between evaluations |
| `--eval-episodes` | `10` | Episodes per evaluation |
| `--save-freq` | `200000` | Steps between checkpoints |
| `--log-folder` | `logs` | Root output directory |
| `--wandb-project` | `pong-dqn` | WandB project name |
| `--no-wandb` | — | Disable WandB (TensorBoard only) |

Hyperparameters are in [`hyperparams/dqn.yml`](hyperparams/dqn.yml) and are
commented to explain each choice and document experimented alternatives.

After training, the final model is saved at:
```
logs/dqn/PongNoFrameskip-v4_<run_id>/PongNoFrameskip-v4.zip
```

### TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

---

## Part 2 — Evaluation

```bash
python evaluate.py --model-path logs/dqn/PongNoFrameskip-v4_1/PongNoFrameskip-v4.zip
```

This will:
1. Run **100 independent episodes** and print mean reward ± std dev.
2. Re-run the best- and worst-performing episodes and save them as GIFs in
   `results/`.

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | *(required)* | Path to `.zip` model file |
| `--n-episodes` | `100` | Number of evaluation episodes |
| `--output-dir` | `results` | Directory for GIF output |
| `--gif-fps` | `30` | Frames per second in exported GIFs |
| `--no-gif` | — | Report stats only, skip GIF export |

---

## Environment wrappers

The standard Atari preprocessing pipeline is applied via SB3's `AtariWrapper`:

| Wrapper | Purpose |
|---------|---------|
| NOOP reset (max 30) | Randomises the starting state for diversity |
| Frame skip ×4 | Agent acts every 4 frames; reduces temporal redundancy |
| Max-pool over last 2 frames | Removes flickering artefacts in Atari rendering |
| Resize to 84×84 grayscale | Reduces input dimensionality; removes colour redundancy |
| Reward clipping to {−1, 0, +1} | Stabilises training across different score scales |
| Life-loss terminal | Treats each life as a separate episode during training |
| Frame stack ×4 | Gives the agent velocity/direction information |
