import gym
import time
from tqdm import tqdm

from src.envs import SimpleBlackjack
from src.agents import RandomAgent

def main(env: gym.Env, agent: RandomAgent, n_episodes: int):
    start = time.time()
    n_wins = 0
    n_draws = 0

    for _ in tqdm(range(n_episodes)):
        state = env.reset()
        terminated = False
        while not terminated:
            action = agent.act(state)
            state, reward, terminated, _, _ = env.step(action)

        if reward == 1:
            n_wins += 1
        elif reward == 0:
            n_draws += 1

    print(f"Win rate: {n_wins / n_episodes:.2f}")
    print(f"Draw rate: {n_draws / n_episodes:.2f}")

    print(f"\nTime taken: {time.time() - start:.2f} seconds")

if __name__ == "__main__":

    n_episodes = 500_000
    env = SimpleBlackjack(seed=42)
    agent = RandomAgent(env.action_space, seed=42)

    main(env, agent, n_episodes)
