"""Табличный FrozenLake 4×4, Q-learning и независимая оценка."""

import argparse
from random import Random
from statistics import mean, stdev


GRID = ("SFFF", "FHFH", "FFFH", "HFFG")
TILES = "".join(GRID)
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
ARROWS = ("←", "↓", "→", "↑")
GAMMA, ALPHA, MAX_STEPS = 0.99, 0.1, 100
PROFILES = {
    "short": (2000, 200, (11, 22, 33)),
    "full": (16000, 500, (11, 22, 33, 44, 55)),
}


def transitions(state, action, slippery):
    """Результаты (вероятность, состояние, награда, завершение MDP)."""
    if TILES[state] in "HG":
        return [(1.0, state, 0, True)]
    moves = ((action - 1) % 4, action, (action + 1) % 4) if slippery else (action,)
    row, col = divmod(state, 4)
    outcomes = []
    for move in moves:
        dy, dx = DIRECTIONS[move]
        new_row = min(3, max(0, row + dy))
        new_col = min(3, max(0, col + dx))
        successor = 4 * new_row + new_col
        tile = TILES[successor]
        outcomes.append((1 / len(moves), successor, int(tile == "G"), tile in "HG"))
    return outcomes


def transition_table(slippery):
    return [[transitions(state, a, slippery) for a in range(4)] for state in range(16)]


def q_update(old, reward, next_best, terminated, alpha, gamma):
    target = reward if terminated else reward + gamma * next_best
    return old + alpha * (target - old)


def greedy_action(row):
    return max(range(4), key=row.__getitem__)


def train(seed, episodes, slippery=True):
    rng = Random(seed)
    dynamics = transition_table(slippery)
    q = [[0.0] * 4 for _ in range(16)]
    for episode in range(episodes):
        epsilon = max(0.05, 1 - 0.95 * episode / max(1, 0.8 * episodes))
        state = 0
        for _ in range(MAX_STEPS):
            if rng.random() < epsilon:
                action = rng.randrange(4)
            else:
                best = max(q[state])
                ties = [a for a, value in enumerate(q[state]) if value == best]
                action = rng.choice(ties)
            _, new_state, reward, terminated = rng.choice(dynamics[state][action])
            # Истинный конец убирает bootstrap. Лимит шагов лишь завершает сбор эпизода.
            q[state][action] = q_update(
                q[state][action], reward, max(q[new_state]), terminated, ALPHA, GAMMA
            )
            state = new_state
            if terminated:
                break
    return q


def evaluate(policy, seed, episodes, slippery=True):
    dynamics = transition_table(slippery)
    successes = holes = timeouts = 0
    lengths = []
    for episode in range(episodes):
        env_rng = Random(seed + episode)
        policy_rng = Random(1_000_000 + seed + episode)
        state = 0
        for step in range(1, MAX_STEPS + 1):
            action = policy(state, policy_rng)
            _, state, reward, terminated = env_rng.choice(dynamics[state][action])
            if terminated:
                successes += reward
                holes += 1 - reward
                break
        else:
            timeouts += 1
        lengths.append(step)
    return {
        "success": successes / episodes,
        "goals": successes,
        "holes": holes,
        "timeouts": timeouts,
        "mean_length": mean(lengths),
    }


def value_iteration(slippery, gamma=GAMMA):
    dynamics = transition_table(slippery)
    values = [0.0] * 16

    def action_values(state, reference):
        return [
            sum(p * (r + (0.0 if done else gamma * reference[ns]))
                for p, ns, r, done in outcomes)
            for outcomes in dynamics[state]
        ]

    for _ in range(10000):
        new_values = [max(action_values(state, values)) for state in range(16)]
        if max(abs(a - b) for a, b in zip(new_values, values)) < 1e-11:
            return new_values, [greedy_action(action_values(s, new_values)) for s in range(16)]
        values = new_values
    raise RuntimeError("Итерация ценности не сошлась при заданном числе итераций")


def experiment(profile="full", slippery=True):
    episodes, evaluation_episodes, seeds = PROFILES[profile]
    values, reference_policy = value_iteration(slippery)
    rows, tables = [], {}
    print(f"profile={profile} slippery={slippery} train={episodes} eval={evaluation_episodes}")
    print(f"gamma={GAMMA} alpha={ALPHA} max_steps={MAX_STEPS} V_star_start={values[0]:.6f}")
    for seed in seeds:
        q = train(seed, episodes, slippery)
        tables[seed] = q
        eval_seed = 100_000 + seed * 1000
        random_result = evaluate(lambda _, rng: rng.randrange(4), eval_seed,
                                 evaluation_episodes, slippery)
        greedy_result = evaluate(lambda s, _: greedy_action(q[s]), eval_seed,
                                 evaluation_episodes, slippery)
        reference = evaluate(lambda s, _: reference_policy[s], eval_seed,
                             evaluation_episodes, slippery)
        rows.append(dict(seed=seed, random=random_result, q=greedy_result, dp=reference))
        print(f"seed={seed} random={random_result['success']:.3f} "
              f"q={greedy_result['success']:.3f} dp={reference['success']:.3f} "
              f"q_goals={greedy_result['goals']} q_holes={greedy_result['holes']} "
              f"q_timeouts={greedy_result['timeouts']} q_mean_steps={greedy_result['mean_length']:.2f}")
    for name in ("random", "q", "dp"):
        rates = [row[name]["success"] for row in rows]
        print(f"{name}: success_mean={mean(rates):.4f} success_sd={stdev(rates):.4f}")
    return rows, tables, values, reference_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="full")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    experiment(args.profile, slippery=not args.deterministic)


if __name__ == "__main__":
    main()
