import gym
import time

from src.envs import SimpleBlackjack
from src.agents import HumanAgent

def main(env: gym.Env, agent: HumanAgent, n_episodes: int):
    start = time.time()
    n_wins = 0
    n_draws = 0

    for _ in range(n_episodes):
        print("=====================================")
        state = env.reset()
        env.render("hidden")
        terminated = False
        while not terminated:
            action = agent.act(state)
            state, reward, terminated, _, _ = env.step(action)
            env.render("hidden")

        print(f"Your reward: {reward}")
        if reward == 1:
            n_wins += 1
        elif reward == 0:
            n_draws += 1

    print(f"Win rate: {n_wins / n_episodes:.2f}")
    print(f"Draw rate: {n_draws / n_episodes:.2f}")

    print(f"\nTime taken: {time.time() - start:.2f} seconds")

if __name__ == "__main__":

    n_episodes = 30
    env = SimpleBlackjack(seed=42)
    agent = HumanAgent(seed=42)
    main(env, agent, n_episodes)
