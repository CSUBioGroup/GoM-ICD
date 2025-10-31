from torch import nn as nn
from torch.nn import functional as F
import torch, time, os, random
import numpy as np
from collections import OrderedDict
from torch import Tensor

expert_list_all = []
class SwitchGate(nn.Module):
    def __init__(
            self,
            dim,
            num_experts: int,
            capacity_factor: float = 1.0,
            epsilon: float = 1e-6,
            topk_expert=1,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.topk_expert = topk_expert
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor):
        gate_scores = F.softmax(self.w_gate(x), dim=-1)
        capacity = int(self.capacity_factor)
        top_k_scores, top_k_indices = gate_scores.topk(self.topk_expert, dim=-1)
        mask = torch.zeros_like(gate_scores).scatter_(
            -1, top_k_indices, 1
        )
        masked_gate_scores = gate_scores * mask
        denominators = (
                masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        )
        gate_scores = (masked_gate_scores / denominators) * capacity
        return gate_scores


class SwitchMoE(nn.Module):

    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            num_experts: int,
            capacity_factor: float = 1.0,
            topk_expert: int = 1,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim)
            ) for _ in range(num_experts)
        ])

        self.gate = SwitchGate(
            dim,
            num_experts,
            capacity_factor,
            1e-6,
            topk_expert,
        )

    def forward(self, x: Tensor):
        gate_scores = self.gate(x)
        expert_outputs = [expert(x) for expert in self.experts]
        if torch.isnan(gate_scores).any():
            print("NaN in gate scores")
            gate_scores[torch.isnan(gate_scores)] = 0

        stacked_expert_outputs = torch.stack(
            expert_outputs, dim=-1
        )
        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[
                torch.isnan(stacked_expert_outputs)
            ] = 0

        moe_output = torch.sum(
            gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )
        return moe_output
