import numpy as np
import gym
import time
import torch
from tqdm import tqdm

from src.envs import Blackjack
from src.agents.dqn import DQN
from src.utils.data_struct import Transition, DQNParameters
from src.explorations import EpsilonGreedy
from src.utils.buffer import ReplayBuffer
from src.utils.general import state_to_array_encoding, get_input_dim_encoding
from src.networks import MLP

def play_episode(env: gym.Env, agent: DQN, render: bool = False):
    state = env.reset()
    if render:
        env.render()
    state = state_to_array_encoding(state, env.observation_space)
    terminated = False
    while not terminated:
        action = agent.get_best_action(state)
        next_state, reward, terminated, _, _ = env.step(action)
        if render:
            env.render()
        next_state = state_to_array_encoding(next_state, env.observation_space)
        state = next_state
    return reward

def main(env: gym.Env, agent: DQN, n_episodes: int):
    start = time.time()
    n_wins = 0
    n_draws = 0
    for _ in tqdm(range(n_episodes)):
        state = env.reset()
        state = state_to_array_encoding(state, env.observation_space)
        terminated = False
        while not terminated:
            action = agent.act(state)
            next_state, reward, terminated, _, _ = env.step(action)

            next_state = state_to_array_encoding(next_state, env.observation_space)
            action = torch.Tensor([action]).long()
            reward = torch.Tensor([reward]).float()
            terminated = torch.Tensor([terminated]).float()

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
    n_episodes = 50_000
    env = Blackjack(seed=42)

    params = DQNParameters()
    q_network = MLP(get_input_dim_encoding(env.observation_space), 64, env.action_space.n).to(params.device)
    target_network = MLP(get_input_dim_encoding(env.observation_space), 64, env.action_space.n).to(params.device)
    target_network.load_state_dict(q_network.state_dict())
    replay_buffer = ReplayBuffer(10_000)
    optimizer = torch.optim.AdamW(q_network.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = torch.nn.MSELoss()
    exploration = EpsilonGreedy(0.3, 0.9)

    agent = DQN(q_network, target_network, replay_buffer, optimizer, criterion, exploration, params)
    main(env, agent, n_episodes)

    n_test_episodes = 10000
    rewards = [play_episode(env, agent) for _ in range(n_test_episodes)]
    n_wins = sum(reward == 1 for reward in rewards)
    n_draws = sum(reward == 0 for reward in rewards)
    print(f"Win rate: {n_wins / n_test_episodes:.2f}")
    print(f"Draw rate: {n_draws / n_test_episodes:.2f}")
