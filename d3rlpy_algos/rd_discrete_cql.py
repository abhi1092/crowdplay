from typing import Any, Optional, Sequence

from d3rlpy.argument_utility import (
    ActionScalerArg,
    EncoderArg,
    QFuncArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
)
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from d3rlpy.algos.dqn import DoubleDQN
from .rd_cql_impl import RDDiscreteCQLImpl
from .rd_cql_impl import FlowConserveRDDiscreteCQLImpl


class RDDiscreteCQL(DoubleDQN):

    _alpha: float
    _impl: Optional[RDDiscreteCQLImpl]

    def __init__(
        self,
        *,
        learning_rate: float = 6.25e-5,
        optim_factory: OptimizerFactory = AdamFactory(),
        encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 32,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        n_critics: int = 1,
        target_update_interval: int = 8000,
        alpha: float = 1.0,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[RDDiscreteCQLImpl] = None,
        
        # For reward discriminator
        discriminator_kl_penalty_coef = 0.001,
        discriminator_clip_ratio = 1.0,
        discriminator_weight_temp = 1.0,
        discriminator_lr = 3e-4,

        **kwargs: Any,
    ):
        super().__init__(
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            q_func_factory=q_func_factory,
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            n_critics=n_critics,
            target_update_interval=target_update_interval,
            use_gpu=use_gpu,
            scaler=scaler,
            reward_scaler=reward_scaler,
            impl=impl,
            **kwargs,
        )
        self._alpha = alpha
        self.discriminator_kl_penalty_coef = discriminator_kl_penalty_coef
        self.discriminator_clip_ratio = discriminator_clip_ratio
        self.discriminator_weight_temp = discriminator_weight_temp
        self.discriminator_lr = discriminator_lr

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = RDDiscreteCQLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            n_critics=self._n_critics,
            alpha=self._alpha,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            reward_scaler=self._reward_scaler,
            
            discriminator_kl_penalty_coef=self.discriminator_kl_penalty_coef,
            discriminator_clip_ratio=self.discriminator_clip_ratio,
            discriminator_weight_temp=self.discriminator_weight_temp,
            discriminator_lr=self.discriminator_lr,
        )
        self._impl.build()




class FlowConserveRDDiscreteCQL(DoubleDQN):

    _alpha: float
    _impl: Optional[FlowConserveRDDiscreteCQLImpl]

    def __init__(
        self,
        *,
        learning_rate: float = 6.25e-5,
        optim_factory: OptimizerFactory = AdamFactory(),
        encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 32,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        n_critics: int = 1,
        target_update_interval: int = 8000,
        alpha: float = 1.0,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[FlowConserveRDDiscreteCQLImpl] = None,
        
        # For reward discriminator
        discriminator_kl_penalty_coef = 0.001,
        discriminator_clip_ratio = 1.0,
        discriminator_weight_temp = 1.0,
        discriminator_lr = 3e-4,
        discriminator_flow_coef = 1.0,

        **kwargs: Any,
    ):
        super().__init__(
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            q_func_factory=q_func_factory,
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            n_critics=n_critics,
            target_update_interval=target_update_interval,
            use_gpu=use_gpu,
            scaler=scaler,
            reward_scaler=reward_scaler,
            impl=impl,
            **kwargs,
        )
        self._alpha = alpha
        self.discriminator_kl_penalty_coef = discriminator_kl_penalty_coef
        self.discriminator_clip_ratio = discriminator_clip_ratio
        self.discriminator_weight_temp = discriminator_weight_temp
        self.discriminator_lr = discriminator_lr
        self.discriminator_flow_coef = discriminator_flow_coef

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = FlowConserveRDDiscreteCQLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            n_critics=self._n_critics,
            alpha=self._alpha,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            reward_scaler=self._reward_scaler,
            
            discriminator_kl_penalty_coef=self.discriminator_kl_penalty_coef,
            discriminator_clip_ratio=self.discriminator_clip_ratio,
            discriminator_weight_temp=self.discriminator_weight_temp,
            discriminator_lr=self.discriminator_lr,
            discriminator_flow_coef=self.discriminator_flow_coef
        )
        self._impl.build()