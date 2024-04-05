import gym
import time
from tqdm import tqdm

from src.envs import InfiniteSimpleBlackjack
from src.explorations import EpsilonGreedy, UCB
from src.agents import MonteCarloAgent
from src.utils.general import state_to_index, get_num_states
from src.utils.visualization import plot_policy_simple_blackjack
from src.utils.data_struct import MonteCarloParameters

def play_episode(env: gym.Env, agent: MonteCarloAgent, render: bool = False):
    state = env.reset()
    if render:
        env.render()
    state = state_to_index(state, env.observation_space)
    terminated = False
    while not terminated:
        action = agent.get_best_action(state)
        next_state, reward, terminated, _, _ = env.step(action)
        if render:
            env.render()
        next_state = state_to_index(next_state, env.observation_space)
        state = next_state
    return reward


def main(env: gym.Env, agent: MonteCarloAgent, n_episodes: int):
    start = time.time()
    n_wins = 0
    n_draws = 0
    for _ in tqdm(range(n_episodes)):
        episode = agent.generate_episode(env)
        reward = episode[-1][2]
        agent.step(episode)

        if reward == 1:
            n_wins += 1
        elif reward == 0:
            n_draws += 1
    print(f"Win rate: {n_wins / n_episodes:.2f}")
    print(f"Draw rate: {n_draws / n_episodes:.2f}")
    print(f"\nTime taken: {time.time() - start:.2f} seconds")


if __name__ == "__main__":

    n_episodes = 500_000
    env = InfiniteSimpleBlackjack(seed=42)
    # exploration = EpsilonGreedy(epsilon=0.8, decay=0.999999, seed=42)
    exploration = UCB(num_states=get_num_states(env.observation_space), num_actions=env.action_space.n, seed=42)
    monte_carlo_parameters = MonteCarloParameters()
    agent = MonteCarloAgent(monte_carlo_parameters, exploration)

    main(env, agent, n_episodes)

    policy = agent.get_policy()
    plot_policy_simple_blackjack(policy, env.observation_space)

    n_test_episodes = 10000
    rewards = [play_episode(env, agent) for _ in range(n_test_episodes)]
    n_wins = sum(reward == 1 for reward in rewards)
    n_draws = sum(reward == 0 for reward in rewards)
    print(f"Win rate: {n_wins / n_test_episodes:.4f}")
    print(f"Draw rate: {n_draws / n_test_episodes:.4f}")
