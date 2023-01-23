import math
from typing import Optional, Sequence, cast

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from d3rlpy.gpu import Device
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.preprocessing import RewardScaler, Scaler
from d3rlpy.torch_utility import TorchMiniBatch
from d3rlpy.algos.torch.dqn_impl import DoubleDQNImpl
from d3rlpy.models.torch import (
    Encoder,
)
from d3rlpy.models.torch.q_functions.utility import compute_huber_loss, compute_reduce, pick_value_by_action
from d3rlpy.models.builders import create_discrete_q_function



class DiscreteMeanDiscrimanator(nn.Module):  # type: ignore
    _action_size: int
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int,
        discriminator_clip_ratio: float = None,
        discriminator_temp: float = None):
        super().__init__()
        self._action_size = action_size
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), action_size)
        self._discriminator_clip_ratio = discriminator_clip_ratio
        self._discriminator_temp = discriminator_temp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self._fc(self._encoder(x)))

    def compute_sa_normalized_ratio(self, x: torch.Tensor, actions: torch.Tensor):
        one_hot = F.one_hot(actions.long().view(-1), num_classes=self.action_size)
        z = (one_hot * self.forward(x)).sum(dim=-1, keepdim=True)
        clipped_x = torch.clip(z,
          -(1 + self._discriminator_clip_ratio),
          (1 + self._discriminator_clip_ratio))
        normalized_ratio = F.softmax(clipped_x / self._discriminator_temp, dim=0)
        return normalized_ratio

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class DiscreteMeanFlowConserveDiscrimanator(nn.Module):  # type: ignore
    _action_size: int
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int,
        discriminator_clip_ratio: float = None,
        discriminator_temp: float = None):
        super().__init__()
        self._action_size = action_size
        self._encoder = encoder
        self._state_action_fc = nn.Linear(encoder.get_feature_size(), action_size)
        self._state_fc = nn.Linear(encoder.get_feature_size(), 1)
        self._discriminator_clip_ratio = discriminator_clip_ratio
        self._discriminator_temp = discriminator_temp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self._state_action_fc(self._encoder(x))), cast(torch.Tensor, self._state_fc(self._encoder(x)))

    def compute_normalized_ratio_and_logits(self, observation: torch.Tensor, actions: torch.Tensor, next_observation: torch.Tensor):
        one_hot = F.one_hot(actions.long().view(-1), num_classes=self.action_size)
        logits_sa, logits_s = self.forward(observation)
        _, logits_next_s = self.forward(next_observation)
        logits = (one_hot * logits_sa).sum(dim=-1, keepdim=True) + logits_s

        clipped_logits = torch.clip(logits,
          -(1 + self._discriminator_clip_ratio),
          (1 + self._discriminator_clip_ratio))
        normalized_ratio = F.softmax(clipped_logits / self._discriminator_temp, dim=0)

        return normalized_ratio, logits_sa, logits_s, logits_next_s

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class RDDiscreteCQLImpl(DoubleDQNImpl):
    _alpha: float

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        n_critics: int,
        alpha: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        reward_scaler: Optional[RewardScaler],

        discriminator_kl_penalty_coef = 0.001,
        discriminator_clip_ratio = 1.0,
        discriminator_weight_temp = 1.0,
        discriminator_lr = 3e-4,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            n_critics=n_critics,
            use_gpu=use_gpu,
            scaler=scaler,
            reward_scaler=reward_scaler,
        )
        self._alpha = alpha

        self.discriminator_kl_penalty_coef = discriminator_kl_penalty_coef
        self.discriminator_clip_ratio = discriminator_clip_ratio
        self.discriminator_weight_temp = discriminator_weight_temp
        self.discriminator_lr = discriminator_lr

    def build(self) -> None:
        # setup torch models
        self._build_network()

        # setup target network
        self._targ_q_func = copy.deepcopy(self._q_func)

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_optim()

    def _build_network(self) -> None:
        self._q_func = create_discrete_q_function(
            self._observation_shape,
            self._action_size,
            self._encoder_factory,
            self._q_func_factory,
            n_ensembles=self._n_critics,
        )
        encoder = self._encoder_factory.create(self._observation_shape)
        self._discriminator = DiscreteMeanDiscrimanator(
            encoder, self._action_size,
            self.discriminator_clip_ratio,
            self.discriminator_weight_temp)

    def _build_optim(self) -> None:
        assert self._q_func is not None
        self._optim = self._optim_factory.create(
            self._q_func.parameters(), lr=self._learning_rate
        )
        self._dicriminator_optim = self._optim_factory.create(
            self._discriminator.parameters(), lr=self.discriminator_lr,
        )

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        assert self._q_func is not None
        assert q_tpn.ndim == 2

        normalized_ratios = self._discriminator.compute_sa_normalized_ratio(
          batch.observations, batch.actions)
        weights = (normalized_ratios).detach()

        td_loss = torch.tensor(
            0.0, dtype=torch.float32, device=batch.observations.device
        )

        for q_func in self._q_func._q_funcs:
            qloss = q_func.compute_error(
                observations=batch.observations,
                actions=batch.actions.long(),
                rewards=batch.rewards,
                target=q_tpn,
                terminals=batch.terminals,
                gamma=self._gamma**batch.n_steps,
                reduction="none",
            )
            td_loss += (qloss * weights).sum()

        conservative_loss = self._compute_conservative_loss(
            batch.observations, batch.actions.long(), weights
        )

        one_hot = F.one_hot(batch.actions.long().view(-1), num_classes=self.action_size)
        discriminator_loss = -((normalized_ratios * one_hot).sum(dim=-1, keepdim=True) * batch.rewards).sum() + \
            self.discriminator_kl_penalty_coef * (normalized_ratios * torch.log(normalized_ratios)).sum()

        return td_loss + self._alpha * conservative_loss + discriminator_loss

    def _compute_conservative_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        # compute logsumexp
        policy_values = self._q_func(obs_t)
        logsumexp = torch.logsumexp(policy_values, dim=1, keepdim=True)

        # estimate action-values under data distribution
        one_hot = F.one_hot(act_t.view(-1), num_classes=self.action_size)
        data_values = (self._q_func(obs_t) * one_hot).sum(dim=1, keepdim=True)
        return ((logsumexp - data_values) * weights).sum()


class FlowConserveRDDiscreteCQLImpl(DoubleDQNImpl):

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        n_critics: int,
        alpha: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        reward_scaler: Optional[RewardScaler],

        discriminator_kl_penalty_coef = 0.001,
        discriminator_clip_ratio = 1.0,
        discriminator_weight_temp = 1.0,
        discriminator_lr = 3e-4,
        discriminator_flow_coef = 1.0,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            n_critics=n_critics,
            use_gpu=use_gpu,
            scaler=scaler,
            reward_scaler=reward_scaler,
        )
        self._alpha = alpha

        self.discriminator_kl_penalty_coef = discriminator_kl_penalty_coef
        self.discriminator_clip_ratio = discriminator_clip_ratio
        self.discriminator_weight_temp = discriminator_weight_temp
        self.discriminator_lr = discriminator_lr
        self.discriminator_flow_coef = discriminator_flow_coef

    def _build_network(self) -> None:
        self._q_func = create_discrete_q_function(
            self._observation_shape,
            self._action_size,
            self._encoder_factory,
            self._q_func_factory,
            n_ensembles=self._n_critics,
        )
        encoder = self._encoder_factory.create(self._observation_shape)
        self._discriminator = DiscreteMeanFlowConserveDiscrimanator(
            encoder, self._action_size,
            self.discriminator_clip_ratio,
            self.discriminator_weight_temp)

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        assert self._q_func is not None
        assert q_tpn.ndim == 2

        normalized_ratios, logits_sa, logits_s, logits_next_s = self._discriminator.compute_normalized_ratio_and_logits(
          batch.observations, batch.actions, batch.next_observations)
        weights = (normalized_ratios).detach()

        td_loss = torch.tensor(
            0.0, dtype=torch.float32, device=batch.observations.device
        )

        for q_func in self._q_func._q_funcs:
            qloss = q_func.compute_error(
                observations=batch.observations,
                actions=batch.actions.long(),
                rewards=batch.rewards,
                target=q_tpn,
                terminals=batch.terminals,
                gamma=self._gamma**batch.n_steps,
                reduction="none",
            )
            td_loss += (qloss * weights).sum()

        conservative_loss = self._compute_conservative_loss(
            batch.observations, batch.actions.long(), weights
        )

        one_hot = F.one_hot(batch.actions.long().view(-1), num_classes=self.action_size)
        discriminator_loss = -((normalized_ratios * one_hot).sum(dim=-1, keepdim=True) * batch.rewards).sum() + \
            self.discriminator_kl_penalty_coef * (normalized_ratios * torch.log(normalized_ratios)).sum() + \
            self.discriminator_flow_coef * torch.square(self._gamma * torch.exp((logits_sa + logits_s) / self.discriminator_weight_temp) - torch.exp(logits_next_s / self.discriminator_weight_temp)).mean()

        return td_loss + self._alpha * conservative_loss + discriminator_loss


    def build(self) -> None:
        # setup torch models
        self._build_network()

        # setup target network
        self._targ_q_func = copy.deepcopy(self._q_func)

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_optim()

    def _build_optim(self) -> None:
        assert self._q_func is not None
        self._optim = self._optim_factory.create(
            self._q_func.parameters(), lr=self._learning_rate
        )
        self._dicriminator_optim = self._optim_factory.create(
            self._discriminator.parameters(), lr=self.discriminator_lr,
        )

    def _compute_conservative_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        # compute logsumexp
        policy_values = self._q_func(obs_t)
        logsumexp = torch.logsumexp(policy_values, dim=1, keepdim=True)

        # estimate action-values under data distribution
        one_hot = F.one_hot(act_t.view(-1), num_classes=self.action_size)
        data_values = (self._q_func(obs_t) * one_hot).sum(dim=1, keepdim=True)
        return ((logsumexp - data_values) * weights).sum()