import gym
import time
from tqdm import tqdm

from src.envs import SimpleBlackjack
from src.explorations import EpsilonGreedy, UCB
from src.agents import QlearningAgent
from src.utils.general import state_to_index, get_num_states
from src.utils.visualization import plot_policy_simple_blackjack
from src.utils.data_struct import Transition, QlearningParameters

def play_episode(env: gym.Env, agent: QlearningAgent, render: bool = False):
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

def main(env: gym.Env, agent: QlearningAgent, n_episodes: int):
    start = time.time()
    n_wins = 0
    n_draws = 0
    for _ in tqdm(range(n_episodes)):
        state = env.reset()
        state = state_to_index(state, env.observation_space)
        terminated = False
        while not terminated:
            action = agent.act(state)
            next_state, reward, terminated, _, _ = env.step(action)
            next_state = state_to_index(next_state, env.observation_space)
            
            transition = Transition(state=state, action=action, next_state=next_state, reward=reward, done=terminated)
            
            state = next_state
            agent.step(transition)

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
    # exploration = EpsilonGreedy(epsilon=0.8, decay=0.999999, seed=42)
    exploration = UCB(num_states=get_num_states(env.observation_space), num_actions=env.action_space.n, seed=42)
    qlearning_parameters = QlearningParameters(num_states=get_num_states(env.observation_space), num_actions=env.action_space.n)
    agent = QlearningAgent(qlearning_parameters, exploration)

    main(env, agent, n_episodes)

    policy = agent.get_policy()
    plot_policy_simple_blackjack(policy, env.observation_space)


    n_test_episodes = 10000
    rewards = [play_episode(env, agent) for _ in range(n_test_episodes)]
    n_wins = sum(reward == 1 for reward in rewards)
    n_draws = sum(reward == 0 for reward in rewards)
    print(f"Win rate: {n_wins / n_test_episodes:.2f}")
    print(f"Draw rate: {n_draws / n_test_episodes:.2f}")
