"""Pong DQN pipeline for ALE/Pong-v5. The default is a CPU smoke run."""
from __future__ import annotations

import argparse
from statistics import fmean, stdev
import random

import ale_py
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dqn_cartpole import ReplayBuffer, Transition, greedy_action, minibatch_loss


class PongQ(nn.Module):
    def __init__(self, actions: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.Flatten(), nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, actions),
        )

    def forward(self, frames):
        return self.layers(frames.to(torch.float32) / 255)


def preprocess(observation: np.ndarray) -> np.ndarray:
    """Convert RGB to 84×84 grayscale, preserving uint8 storage."""
    rgb = torch.from_numpy(observation).to(torch.float32)
    luminance = rgb @ torch.tensor([0.299, 0.587, 0.114])
    resized = F.interpolate(luminance[None, None], size=(84, 84), mode="area")
    return resized[0, 0].round().clamp(0, 255).to(torch.uint8).numpy()


def push_frame(stack: np.ndarray, observation: np.ndarray) -> np.ndarray:
    return np.concatenate([stack[1:], preprocess(observation)[None]], axis=0)


def make_env():
    gym.register_envs(ale_py)
    return gym.make("ALE/Pong-v5", frameskip=4, repeat_action_probability=0.25)


def reset_stack(env, seed: int) -> np.ndarray:
    observation, _ = env.reset(seed=seed)
    return np.repeat(preprocess(observation)[None], 4, axis=0)


def evaluate(model, episodes: int, seed: int) -> list[float]:
    env = make_env()
    scores = []
    try:
        for index in range(episodes):
            episode_seed = seed + index
            state = reset_stack(env, episode_seed)
            policy_rng = random.Random(episode_seed + 80_000)
            score = 0.0
            while True:
                action = (policy_rng.randrange(env.action_space.n) if model is None
                          else greedy_action(model, state))
                observation, reward, terminated, truncated, _ = env.step(action)
                score += reward
                if terminated or truncated:
                    scores.append(score)
                    break
                state = push_frame(state, observation)
    finally:
        env.close()
    return scores


def run(steps: int, seed: int, eval_episodes: int = 0) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    policy_rng = random.Random(seed)
    replay_rng = random.Random(seed + 90_000)
    env = make_env()
    replay = ReplayBuffer(2048)
    batch_size, warmup = (4, 8) if steps <= 64 else (32, 512)
    actions = env.action_space.n
    online, target = PongQ(actions), PongQ(actions)
    target.load_state_dict(online.state_dict())
    target.requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
    losses = []
    episodes_finished = 0
    score = 0.0
    try:
        state = reset_stack(env, seed)
        initial_stack = state.copy()
        for step in range(1, steps + 1):
            epsilon = 1.0 - 0.9 * min((step - 1) / 25_000, 1.0)
            action = (policy_rng.randrange(actions) if policy_rng.random() < epsilon
                      else greedy_action(online, state))
            observation, reward, terminated, truncated, _ = env.step(action)
            next_state = push_frame(state, observation)
            replay.append(Transition(state, action, reward, next_state, terminated, truncated))
            state = next_state
            score += reward
            if len(replay) >= warmup and step % 4 == 0:
                loss = minibatch_loss(online, target, replay.sample(batch_size, replay_rng), 0.99)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                if len(losses) % 1000 == 0:
                    target.load_state_dict(online.state_dict())
            if terminated or truncated:
                episodes_finished += 1
                if step < steps:
                    state = reset_stack(env, seed + episodes_finished)
        with torch.no_grad():
            output_shape = tuple(online(torch.from_numpy(state)[None]).shape)
    finally:
        env.close()

    print(
        f"Gymnasium {gym.__version__}, ALE {ale_py.__version__}, torch {torch.__version__}, "
        f"frameskip=4, sticky=0.25"
    )
    print(
        f"steps={steps}, episodes_finished={episodes_finished}, replay={len(replay)}, "
        f"updates={len(losses)}, stack={state.shape} {state.dtype}, Q={output_shape}, "
        f"training_reward_sum={score:.1f}"
    )
    if losses:
        print(f"first_loss={losses[0]:.8f}, last_loss={losses[-1]:.8f}")
    result = {"steps": steps, "episodes_finished": episodes_finished, "losses": losses,
              "q_shape": output_shape, "stack": initial_stack}
    if eval_episodes:
        eval_seed = seed + 100_000
        result["random_scores"] = evaluate(None, eval_episodes, eval_seed)
        result["test_scores"] = evaluate(online, eval_episodes, eval_seed)
        for key in ("random_scores", "test_scores"):
            scores = result[key]
            spread = f", sd={stdev(scores):.2f}" if len(scores) >= 2 else ""
            print(f"{key}: mean={fmean(scores):.2f}{spread}, episodes={len(scores)}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--eval-episodes", type=int, default=0)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.eval_episodes < 0:
        parser.error("expected positive --steps and nonnegative --eval-episodes")
    return run(args.steps, args.seed, args.eval_episodes)


if __name__ == "__main__":
    main()