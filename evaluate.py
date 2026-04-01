"""
Part 2 — Evaluate a trained DQN agent on PongNoFrameskip-v4.

This script:
  1. Loads a trained SB3 DQN model (.zip) produced by train.py.
  2. Runs 100 independent episodes and reports the mean reward +/- std dev.
  3. Exports GIF animations of the best- and worst-performing episodes.

Usage
    python evaluate.py --model-path logs/dqn/PongNoFrameskip-v4_1/best_model.zip
"""

import argparse
import os

import ale_py
import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Register Atari environments — newer ale-py versions don't do this automatically
gym.register_envs(ale_py)


# Environment factory
def make_env(n_stack: int = 4, render_mode: str | None = None) -> VecFrameStack:
    """
    Build a vectorised PongNoFrameskip-v4 env that matches the training setup:
      - make_atari_env: applies AtariWrapper (NOOP reset, frame-skip x4,
        84x84 grayscale, reward clip, life-loss terminal) inside a DummyVecEnv.
      - VecFrameStack: stacks n_stack consecutive frames on the channel axis,
        producing observations of shape (84, 84, n_stack).

    Using make_atari_env + VecFrameStack is important: it replicates exactly
    what rl_zoo3 does during training, so observation shapes match the model.

    Args:
        n_stack:     Number of frames to stack (must match training config).
        render_mode: "rgb_array" to capture frames, None for headless eval.

    Returns:
        VecFrameStack-wrapped environment.
    """
    env = make_atari_env(
        "PongNoFrameskip-v4",
        n_envs=1,
        env_kwargs={"render_mode": render_mode} if render_mode else {},
    )
    return VecFrameStack(env, n_stack=n_stack)


# Episode runner
def run_episode(env: VecFrameStack, model: DQN, record: bool = False) -> tuple[float, list]:
    """
    Run a single episode and return (total_reward, frames).

    Uses the VecEnv API (reset returns obs directly; step returns arrays).

    Args:
        env:    VecFrameStack-wrapped Pong environment (n_envs=1).
        model:  Loaded SB3 DQN model.
        record: If True, collect RGB frames via env.render() at each step.

    Returns:
        total_reward: Cumulative (unclipped) reward for the episode.
        frames:       List of RGB numpy arrays (empty list when record=False).
    """
    obs = env.reset()
    done = False
    total_reward = 0.0
    frames: list = []

    while not done:
        # deterministic=True: always pick the greedy action (no exploration)
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, _ = env.step(action)
        done = bool(dones[0])
        total_reward += float(rewards[0])

        if record:
            frame = env.render()
            # VecEnv render() returns a list of images (one per env)
            if isinstance(frame, list):
                frame = frame[0]
            if frame is not None:
                frames.append(frame)

    return total_reward, frames


# GIF export
def save_gif(frames: list, path: str, fps: int = 30) -> None:
    """
    Save a list of RGB numpy arrays as an animated GIF.

    Args:
        frames: List of H x W x 3 uint8 arrays.
        path:   Output file path (should end in .gif).
        fps:    Playback frame rate.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # duration is in milliseconds per frame for the imageio GIF writer
    imageio.mimwrite(path, frames, duration=int(1000 / fps), loop=0)
    print(f"  Saved: {path}  ({len(frames)} frames @ {fps} fps)")


# Argument parsing
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN on PongNoFrameskip-v4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to the trained model .zip file.",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=100,
        help="Number of independent evaluation episodes.",
    )
    parser.add_argument(
        "--n-stack", type=int, default=4,
        help="Frame stack size (must match the value used during training).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save GIF animations.",
    )
    parser.add_argument(
        "--gif-fps", type=int, default=30,
        help="Frame rate for exported GIFs.",
    )
    parser.add_argument(
        "--no-gif", action="store_true",
        help="Skip GIF export (report stats only).",
    )
    return parser.parse_args()


# Main
def main() -> None:
    args = parse_args()

    # Load model
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    print(f"Loading model from: {args.model_path}")
    model = DQN.load(args.model_path)
    print("Model loaded successfully.\n")

    # Single-pass evaluation with rendering
    # We render every episode but only keep the frames for the running best and
    # worst episodes, so memory usage is bounded to ~2 episodes at any time.
    print(f"Running {args.n_episodes} evaluation episodes...")
    env = make_env(n_stack=args.n_stack, render_mode=None if args.no_gif else "rgb_array")

    rewards: list[float] = []
    best_reward, worst_reward = -float("inf"), float("inf")
    best_frames: list = []
    worst_frames: list = []

    for ep in range(args.n_episodes):
        reward, frames = run_episode(env, model, record=not args.no_gif)
        rewards.append(reward)

        if not args.no_gif:
            if reward > best_reward:
                best_reward = reward
                best_frames = frames
            if reward < worst_reward:
                worst_reward = reward
                worst_frames = frames

        if (ep + 1) % 10 == 0:
            print(f"  Episodes done: {ep + 1}/{args.n_episodes}  "
                  f"(last reward: {reward:+.0f})")

    env.close()

    # Statistic
    rewards_arr = np.array(rewards)
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"  Episodes     : {args.n_episodes}")
    print(f"  Mean reward  : {rewards_arr.mean():+.2f}")
    print(f"  Std dev      : {rewards_arr.std():.2f}")
    print(f"  Min reward   : {rewards_arr.min():+.0f}")
    print(f"  Max reward   : {rewards_arr.max():+.0f}")
    print("=" * 50 + "\n")

    if args.no_gif:
        return

    # Export GIFs
    for label, frames, reward in [
        ("best",  best_frames,  best_reward),
        ("worst", worst_frames, worst_reward),
    ]:
        gif_path = os.path.join(args.output_dir, f"{label}_episode_reward{reward:+.0f}.gif")
        print(f"Saving {label} episode (reward {reward:+.0f})...")
        save_gif(frames, gif_path, fps=args.gif_fps)

    print("\nDone.")


if __name__ == "__main__":
    main()
