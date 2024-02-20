import gym
import time
import argparse

from src.agents import Agent, RandomAgent
from src.envs import SimpleBlackjack

def main(env: gym.Env, agent: Agent, n_episodes: int):
    start = time.time()
    n_wins = 0
    n_draws = 0
    for _ in range(n_episodes):
        state = env.reset()
        agent.reset()
        terminated = False
        while not terminated:
            action = agent.act(state)
            state, reward, terminated, _, _ = env.step(action)
            agent.step(state, reward, terminated)
        if reward == 1:
            n_wins += 1
        elif reward == 0:
            n_draws += 1

    print(f"Win rate: {n_wins / n_episodes:.2f}")
    print(f"Draw rate: {n_draws / n_episodes:.2f}")

    print(f"\nTime taken: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main script.")
    parser.add_argument("--n_episodes", type=int, default=200000, help="The number of episodes to run.")
    parser.add_argument("--env", type=str, default="SimpleBlackjack", help="The environment to use.")
    parser.add_argument("--agent", type=str, default="RandomAgent", help="The agent to use.")
    args = parser.parse_args()

    env = SimpleBlackjack(seed=42)
    agent = RandomAgent(env.action_space, seed=42)
    n_episodes = 200000

    main(env, agent, n_episodes)
