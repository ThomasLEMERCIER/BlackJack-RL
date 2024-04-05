import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int=1):
        super(MLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)

class MLPEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, input_range: int, depth: int=0):
        super(MLPEmbedding, self).__init__()
        self.opp_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        layers = []
        layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.main = nn.Sequential(*layers)

        self.embedding = nn.Embedding(input_range, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        player = x[:, :-1].long()
        opp = x[:, -1:]

        player = self.embedding(player)
        player = player.sum(dim=1)

        opp = self.opp_net(opp)

        x = torch.cat([player, opp], dim=-1)
        return self.main(x)
