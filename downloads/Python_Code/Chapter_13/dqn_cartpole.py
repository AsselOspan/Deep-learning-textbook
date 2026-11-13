"""DQN for Gymnasium CartPole-v1 with paired held-out evaluation."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from statistics import fmean, stdev
import random

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool


class ReplayBuffer:
    """Bounded FIFO buffer with uniform sampling without replacement."""

    def __init__(self, capacity: int):
        self.items = deque(maxlen=capacity)

    def append(self, transition: Transition) -> None:
        self.items.append(
            Transition(
                transition.state.copy(), transition.action, transition.reward,
                transition.next_state.copy(), transition.terminated, transition.truncated,
            )
        )

    def sample(self, size: int, rng: random.Random) -> list[Transition]:
        return rng.sample(self.items, size)

    def __len__(self) -> int:
        return len(self.items)


@torch.no_grad()
def td_targets(
    rewards: torch.Tensor, next_values: torch.Tensor,
    terminated: torch.Tensor, gamma: float,
) -> torch.Tensor:
    continuation = (~terminated).to(rewards.dtype)
    return rewards + gamma * continuation * next_values


def minibatch_loss(online, target, batch, gamma):
    states = torch.from_numpy(np.stack([item.state for item in batch]))
    next_states = torch.from_numpy(np.stack([item.next_state for item in batch]))
    actions = torch.tensor([item.action for item in batch], dtype=torch.long)
    rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32)
    terminated = torch.tensor([item.terminated for item in batch], dtype=torch.bool)
    estimates = online(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_values = target(next_states).max(dim=1).values
        targets = td_targets(rewards, next_values, terminated, gamma)
    return F.smooth_l1_loss(estimates, targets, beta=1.0)


def build_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 2),
    )


@torch.no_grad()
def greedy_action(model: nn.Module, state: np.ndarray) -> int:
    return int(model(torch.from_numpy(state)[None]).argmax(dim=1).item())


def evaluate(model: nn.Module | None, seeds: list[int]) -> list[float]:
    env = gym.make("CartPole-v1")
    returns = []
    try:
        for seed in seeds:
            state, _ = env.reset(seed=seed)
            random_actions = random.Random(seed + 700_000)
            episode_return = 0.0
            while True:
                action = random_actions.randrange(2) if model is None else greedy_action(model, state)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward
                if terminated or truncated:
                    break
            returns.append(episode_return)
    finally:
        env.close()
    return returns


def train(seed: int, episodes: int, validation_every: int) -> dict:
    torch.manual_seed(seed)
    policy_rng = random.Random(seed)
    replay_rng = random.Random(seed + 90_000)
    env = gym.make("CartPole-v1")
    online = build_model()
    target = build_model()
    target.load_state_dict(online.state_dict())
    target.requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
    replay = ReplayBuffer(10_000)
    batch_size, gamma = 64, 0.99
    interactions = updates = 0
    best_score = float("-inf")
    best_epoch = 0
    best_state = None
    training_returns = []
    validation = []
    val_seeds = list(range(20_000, 20_005))

    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset(seed=1_000 * seed + episode)
            episode_return = 0.0
            while True:
                epsilon = 1.0 - 0.95 * min(interactions / 5_000, 1.0)
                if policy_rng.random() < epsilon:
                    action = policy_rng.randrange(2)
                else:
                    action = greedy_action(online, state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                replay.append(Transition(state, action, reward, next_state, terminated, truncated))
                state = next_state
                episode_return += reward
                interactions += 1
                if len(replay) >= 256 and interactions % 4 == 0:
                    loss = minibatch_loss(online, target, replay.sample(batch_size, replay_rng), gamma)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
                    optimizer.step()
                    updates += 1
                    if updates % 200 == 0:
                        target.load_state_dict(online.state_dict())
                if terminated or truncated:
                    break
            training_returns.append(episode_return)
            if episode % validation_every == 0 or episode == episodes:
                score = fmean(evaluate(online, val_seeds))
                validation.append((episode, score))
                if score > best_score:
                    best_score, best_epoch = score, episode
                    best_state = {key: value.clone() for key, value in online.state_dict().items()}
    finally:
        env.close()

    online.load_state_dict(best_state)
    return {
        "model": online, "best_episode": best_epoch, "validation": validation,
        "training_returns": training_returns, "interactions": interactions, "updates": updates,
    }


def experiment(profile: str) -> dict:
    torch.set_num_threads(1)
    if profile == "short":
        seeds, episodes, every, test_count = [11], 30, 10, 10
    else:
        seeds, episodes, every, test_count = [11, 22, 33], 180, 20, 30
    test_seeds = list(range(30_000, 30_000 + test_count))
    random_returns = evaluate(None, test_seeds)
    result = {"profile": profile, "episodes": episodes, "test_seeds": test_seeds,
              "random_returns": random_returns, "runs": {}}
    print(f"Gymnasium {gym.__version__}, torch {torch.__version__}, CPU, profile={profile}")
    print(f"Random: mean={fmean(random_returns):.2f}, sd={stdev(random_returns):.2f}")
    for seed in seeds:
        run = train(seed, episodes, every)
        test_returns = evaluate(run["model"], test_seeds)
        run["test_returns"] = test_returns
        result["runs"][seed] = run
        print(
            f"seed={seed} best_episode={run['best_episode']} "
            f"test_mean={fmean(test_returns):.2f} test_sd={stdev(test_returns):.2f} "
            f"at_500={test_returns.count(500.0)}/{test_count} "
            f"steps={run['interactions']} updates={run['updates']}"
        )
        print("validation=" + ",".join(f"{episode}:{score:.1f}" for episode, score in run["validation"]))
    means = [fmean(run["test_returns"]) for run in result["runs"].values()]
    print(f"Mean across training seeds={fmean(means):.2f}")
    if len(means) > 1:
        print(f"Sample SD across training seeds={stdev(means):.2f}")
    return result


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("short", "full"), default="short")
    args = parser.parse_args(argv)
    return experiment(args.profile)


if __name__ == "__main__":
    main()