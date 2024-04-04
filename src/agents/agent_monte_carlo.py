from .agent import Agent
from ..explorations.exploration_policy import ExplorationPolicy
from ..utils.data_struct import MonteCarloParameters
from src.utils.general import state_to_index

import numpy as np

class MonteCarloAgent(Agent):
    def __init__(self,params: MonteCarloParameters,exploration_policy: ExplorationPolicy,):
        self.params = params
        self.exploration_policy = exploration_policy

        self.q_values = np.zeros((self.params.num_states, self.params.num_actions))


    def generate_episode(self,env):
        """generates an instance of one episode with current policy"""
        episode = []
        state = state_to_index(env.reset(),env.observation_space)

        policy = self.get_policy()
        while True:
            action = self.act(state)
            next_state, reward, terminated, _, _ =  env.step(action)
            next_state = state_to_index(next_state, env.observation_space)
            episode.append((state, action, reward))
            state = next_state
            if terminated:
                break
        return episode
    
    def step(self, episode):
        """updates Q based on episode"""
        states, actions, rewards = zip(*episode)
        discounts = np.array([self.params.gamma**i for i in range(len(rewards)+1)])
        for i, state in enumerate(states):
            self.q_values[state][actions[i]] = self.q_values[state][actions[i]] + self.params.alpha * (sum(rewards[i:] * discounts[:-(1+i)]) - self.q_values[state][actions[i]])

    def act(self, state: int) -> int:
        return self.exploration_policy(self.q_values[state], state)
    def get_best_action(self, state: int) -> np.int64:
        return np.argmax(self.q_values[state])
    def get_policy(self) -> np.ndarray:
        return np.argmax(self.q_values, axis=1)
    def get_value_function(self) -> np.ndarray:
        return np.max(self.q_values, axis=1)
