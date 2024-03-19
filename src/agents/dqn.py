from ..utils.data_struct import Transition, DQNParameters
from ..utils.buffer import ReplayBuffer
import torch

class DQN:
    def __init__(self, q_network, target_network, replay_buffer: ReplayBuffer, optimizer, criterion, exploration, params: DQNParameters):
        self.q_network = q_network
        self.target_network = target_network
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.criterion = criterion
        self.exploration = exploration
        self.params = params
        self.n_steps = 0

    def step(self, transition: Transition) -> None:
        self.replay_buffer.push(transition)
        self.n_steps += 1
        self.update()

    def update(self):
        if len(self.replay_buffer) < self.params.batch_size:
            return

        transitions = self.replay_buffer.sample(self.params.batch_size)
        state_batch = torch.stack(transitions.state).to(self.params.device)
        next_state_batch = torch.stack(transitions.next_state).to(self.params.device)
        action_batch = torch.stack(transitions.action).to(self.params.device)
        reward_batch = torch.cat(transitions.reward).to(self.params.device)
        non_final_mask = 1 - torch.cat(transitions.done).to(self.params.device)

        q_values = self.q_network(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1).values * non_final_mask
            expected_q_values = reward_batch + self.params.gamma * next_q_values
        loss = self.criterion(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.n_steps % self.params.freq_target_update == 0:
            self.update_target()
            self.n_steps = 0

    def update_target(self):
        self.q_network.load_state_dict(self.target_network.state_dict())
        self.target_network.eval()

    def act(self, state: torch.Tensor) -> int:
        q_values = self.q_network(state.float().to(self.params.device)).detach().cpu().numpy()
        return self.exploration(q_values, state)
    
    def get_best_action(self, state: torch.Tensor) -> int:
        return self.q_network(state.float().to(self.params.device)).max(-1).indices.item()
