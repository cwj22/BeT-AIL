

import abc
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Type, cast

import gym
import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common import preprocessing
from torch import nn
from torch.nn import functional as F
from imitation.util import networks, util




class OptionRewardNet(nn.Module, abc.ABC):
    """Minimal abstract reward network.

    Only requires the implementation of a forward pass (calculating rewards given
    a batch of states, actions, next states and dones).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        option_space: gym.Space,
        latent_dim: int,
        normalize_images: bool = True,
    ):
        """Initialize the RewardNet.

        Args:
            observation_space: the observation space of the environment
            action_space: the action space of the environment
            normalize_images: whether to automatically normalize
                image observations to [0, 1] (from 0 to 255). Defaults to True.
        """
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.option_space = option_space
        self.latent_dim = latent_dim
        self.normalize_images = normalize_images

    @abc.abstractmethod
    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        option: th.Tensor,
        next_option: th.Tensor
    ) -> th.Tensor:
        """Compute rewards for a batch of transitions and keep gradients."""

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        option: np.ndarray,
        next_option: np.ndarray,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Preprocess a batch of input transitions and convert it to PyTorch tensors.

        The output of this function is suitable for its forward pass,
        so a typical usage would be ``model(*model.preprocess(transitions))``.

        Args:
            state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            action: The action input. Its shape is
                `(batch_size,) + action_space.shape`. The None dimension is
                expected to be the same as None dimension from `obs_input`.
            next_state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            done: Whether the episode has terminated. Its shape is `(batch_size,)`.

        Returns:
            Preprocessed transitions: a Tuple of tensors containing
            observations, actions, next observations and dones.
        """
        state_th = util.safe_to_tensor(state).to(self.device)
        action_th = util.safe_to_tensor(action).to(self.device)
        next_state_th = util.safe_to_tensor(next_state).to(self.device)
        done_th = util.safe_to_tensor(done).to(self.device)
        option_th = util.safe_to_tensor(option).to(self.device)
        next_option_th = util.safe_to_tensor(next_option).to(self.device)


        del state, action, next_state, done, option, next_option  # unused

        # preprocess
        # we only support array spaces, so we cast
        # the observation to torch tensors.
        state_th = cast(
            th.Tensor,
            preprocessing.preprocess_obs(
                state_th,
                self.observation_space,
                self.normalize_images,
            ),
        )
        action_th = cast(
            th.Tensor,
            preprocessing.preprocess_obs(
                action_th,
                self.action_space,
                self.normalize_images,
            ),
        )
        next_state_th = cast(
            th.Tensor,
            preprocessing.preprocess_obs(
                next_state_th,
                self.observation_space,
                self.normalize_images,
            ),
        )
        done_th = done_th.to(th.float32)
        option_th = cast(
            th.Tensor,
            preprocessing.preprocess_obs(
                option_th,
                self.option_space,
                self.normalize_images,
            ),
        )
        next_option_th = cast(
            th.Tensor,
            preprocessing.preprocess_obs(
                next_option_th,
                self.option_space,
                self.normalize_images,
            ),
        )

        n_gen = len(state_th)
        assert state_th.shape == next_state_th.shape
        assert len(action_th) == n_gen

        return state_th, action_th, next_state_th, done_th, option_th, next_option_th

    def predict_th(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        option: np.ndarray,
        next_option: np.ndarray,
    ) -> th.Tensor:
        """Compute th.Tensor rewards for a batch of transitions without gradients.

        Preprocesses the inputs, output th.Tensor reward arrays.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.

        Returns:
            Computed th.Tensor rewards of shape `(batch_size,`).
        """
        with networks.evaluating(self):
            # switch to eval mode (affecting normalization, dropout, etc)

            state_th, action_th, next_state_th, done_th, option_th, next_option_th = self.preprocess(
                state,
                action,
                next_state,
                done,
                option,
                next_option,
            )
            with th.no_grad():
                rew_th = self(state_th, action_th, next_state_th, done_th, option_th, next_option_th)

            assert rew_th.shape == state.shape[:1]
            return rew_th

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        option: np.ndarray,
        next_option: np.ndarray,
    ) -> np.ndarray:
        """Compute rewards for a batch of transitions without gradients.

        Converting th.Tensor rewards from `predict_th` to NumPy arrays.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.

        Returns:
            Computed rewards of shape `(batch_size,)`.
        """
        rew_th = self.predict_th(state, action, next_state, done, option, next_option)
        return rew_th.detach().cpu().numpy().flatten()

    def predict_processed(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        option: np.ndarray,
        next_option: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Compute the processed rewards for a batch of transitions without gradients.

        Defaults to calling `predict`. Subclasses can override this to normalize or
        otherwise modify the rewards in ways that may help RL training or other
        applications of the reward function.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.
            kwargs: additional kwargs may be passed to change the functionality of
                subclasses.

        Returns:
            Computed processed rewards of shape `(batch_size,`).
        """
        del kwargs
        return self.predict(state, action, next_state, done, option, next_option)

    @property
    def device(self) -> th.device:
        """Heuristic to determine which device this module is on."""
        try:
            first_param = next(self.parameters())
            return first_param.device
        except StopIteration:
            # if the model has no parameters, we use the CPU
            return th.device("cpu")

    @property
    def dtype(self) -> th.dtype:
        """Heuristic to determine dtype of module."""
        try:
            first_param = next(self.parameters())
            return first_param.dtype
        except StopIteration:
            # if the model has no parameters, default to float32
            return th.get_default_dtype()

class OptionBaseRewardNet(OptionRewardNet):
    """MLP that takes as input the state, action, next state and done flag.

     These inputs are flattened and then concatenated to one another. Each input
     can enabled or disabled by the `use_*` constructor keyword arguments.
     """

    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            latent_dim: int,
            use_state: bool = True,
            use_action: bool = True,
            use_next_state: bool = False,
            use_done: bool = False,
            use_option = True,
            use_next_option=True,
            is_discriminator_shared: bool = False,
            use_spectral_norm: bool = False,
            spectral_norm_kwargs: Optional[Dict] = None,
            **kwargs,
    ):
        """Builds reward MLP.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            use_state: should the current state be included as an input to the MLP?
            use_action: should the current action be included as an input to the MLP?
            use_next_state: should the next state be included as an input to the MLP?
            use_done: should the "done" flag be included as an input to the MLP?
            kwargs: passed straight through to `build_mlp`.
        """
        option_space = spaces.Box(low = observation_space.low[0],
                                      high = observation_space.high[0],
                                      shape = (latent_dim,),
                                      dtype=observation_space.dtype)
        super().__init__(observation_space, action_space, option_space, latent_dim)
        combined_size_in = 0
        combined_size_out = 1


        if not use_option == use_next_option:
            raise NotImplemented # not implemented yet, must use both options or none

        self.use_state = use_state
        if self.use_state:
            combined_size_in += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_action = use_action
        if self.use_action:
            combined_size_in += preprocessing.get_flattened_obs_dim(action_space)

        self.use_next_state = use_next_state
        if self.use_next_state:
            combined_size_in += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_done = use_done
        if self.use_done:
            combined_size_in += 1

        self.use_option = use_option
        self.use_next_option = use_next_option
        if self.use_option and self.use_next_option:
            combined_size_out += latent_dim
            combined_size_out *= latent_dim

        self.latent_dim = latent_dim
        self.is_discrimintator_shared = is_discriminator_shared

        full_build_mlp_kwargs: Dict[str, Any] = {
            "hid_sizes": (32, 32),
            **kwargs,
            # we do not want the values below to be overridden
            "in_size": combined_size_in,
        }
        self.use_spectral_norm = use_spectral_norm
        self.spectral_norm_kwargs = spectral_norm_kwargs

        if self.is_discrimintator_shared or not self.use_option:
            full_build_mlp_kwargs["squeeze_output"] = False
            if self.use_spectral_norm:
                # build one mlp with c*(c+1) outputs
                full_build_mlp_kwargs["out_size"] = combined_size_out if self.use_option else 1
                self.mlp = networks.build_mlp_spectral_norm(**full_build_mlp_kwargs, spectral_norm_kwargs=spectral_norm_kwargs)
            else:
                # build one mlp with c*(c+1) outputs
                full_build_mlp_kwargs["out_size"] = combined_size_out if self.use_option else 1
                self.mlp = networks.build_mlp(**full_build_mlp_kwargs)
        else:
            full_build_mlp_kwargs["squeeze_output"] = True
            if self.use_spectral_norm:
                # build c*(c+1) mlps with one output each
                full_build_mlp_kwargs["out_size"] = 1
                self.mlp = th.nn.ModuleList(
                    [networks.build_mlp_spectral_norm(**full_build_mlp_kwargs,
                                                      spectral_norm_kwargs=spectral_norm_kwargs
                                                      ) for _ in range(combined_size_out)]
                )

            else:
                # build c*(c+1) mlps with one output each
                full_build_mlp_kwargs["out_size"] = 1
                self.mlp = th.nn.ModuleList(
                    [networks.build_mlp(**full_build_mlp_kwargs) for _ in range(combined_size_out)]
                )

    def forward(self, state, action, next_state, done, option, next_option):
        inputs = []
        if self.use_state:
            inputs.append(th.flatten(state, 1))
        if self.use_action:
            inputs.append(th.flatten(action, 1))
        if self.use_next_state:
            inputs.append(th.flatten(next_state, 1))
        if self.use_done:
            inputs.append(th.reshape(done, [-1, 1]))

        inputs_concat = th.cat(inputs, dim=1)
        # if self.use_option:
        #     inputs.append(th.flatten(option, 1))
        # if self.use_next_option:
        #     inputs.append(th.flatten(next_option, 1))
        if self.use_next_option and self.use_option:
            # get c*(c+1) outputs from discriminator network
            if self.is_discrimintator_shared:
                outputs = self.mlp(inputs_concat)
            else:
                outputs = th.cat([m(inputs_concat)[..., None] for m in self.mlp], dim=1)

            # gather outputs from the current and next option
            outputs = outputs.view(-1, self.latent_dim + 1, self.latent_dim)
            option = option.view(-1, 1, 1).expand(-1, 1, self.latent_dim).type(th.int64)
            next_option = next_option.view(-1, 1).type(th.int64)
            outputs = outputs.gather(dim=-2, index=option).squeeze(dim=-2).gather(dim=-1, index=next_option).squeeze()
        else:
            # use regular discriminator
            outputs = self.mlp(inputs_concat).squeeze()
        assert outputs.shape == state.shape[:1]




        return outputs


class OptionsRewardNetFromDiscriminatorLogit(OptionRewardNet):
    def __init__(self, base: OptionRewardNet):
        """Builds LogSigmoidRewardNet to wrap `reward_net`."""
        # TODO(adam): make an explicit RewardNetWrapper class?
        assert isinstance(base, OptionRewardNet) # make sure reward net takes options
        super().__init__(
            observation_space=base.observation_space,
            action_space=base.action_space,
            normalize_images=base.normalize_images,
            latent_dim=base.latent_dim,
            option_space=base.option_space,
        )
        self.base = base

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        option: th.Tensor,
        next_option: th.Tensor
    ) -> th.Tensor:
        logits = self.base.forward(state, action, next_state, done, option, next_option)
        return -F.logsigmoid(-logits)