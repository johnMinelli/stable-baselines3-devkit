"""
Wrapper to configure an environment instance to Stable-Baselines3 vectorized environment that is common to environments and datasets.
The original class is based on the IsaacLab SB3 wrapper. Every other environment will necessitate a dedicated wrapper
that converts the output data of the specific environment to the standard format used by IsaacLab.

"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn  # noqa: F401
from gymnasium import spaces
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
)

"""
Configuration Parser.
"""


def process_sb3_cfg(cfg: dict) -> dict:
    """Convert simple YAML types to Stable-Baselines classes/components.

    Args:
        cfg: A configuration dictionary.

    Returns:
        A dictionary containing the converted configuration.

    Reference:
        https://github.com/DLR-RM/rl-baselines3-zoo/blob/0e5eb145faefa33e7d79c7f8c179788574b20da5/utils/exp_manager.py#L358
    """

    def update_dict(hyperparams: dict[str, Any]) -> dict[str, Any]:
        for key, value in hyperparams.items():
            if isinstance(value, dict):
                update_dict(value)
            else:
                if key in ["policy_kwargs"]:
                    hyperparams[key] = eval(value) if isinstance(value, str) else value
                elif key in ["clip_range", "clip_range_vf", "delta_std"]:
                    if isinstance(value, str):
                        _, initial_value = value.split("_")
                        initial_value = float(initial_value)
                        hyperparams[key] = lambda progress_remaining: progress_remaining * initial_value
                    elif isinstance(value, (float, int)):
                        # Negative value: ignore (ex: for clipping)
                        if value < 0:
                            continue
                        hyperparams[key] = constant_fn(float(value))
                    elif value is None:
                        continue
                    else:
                        raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")

        return hyperparams

    # parse agent configuration and convert to classes
    return update_dict(cfg)


"""
Vectorized environment wrapper.
"""


class Sb3EnvStdWrapper(VecEnv):
    """Wraps around Isaac Lab environment for Stable Baselines3.

    Isaac Sim internally implements a vectorized environment. However, since it is
    still considered a single environment instance, Stable Baselines tries to wrap
    around it using the :class:`DummyVecEnv`. This is only done if the environment
    is not inheriting from their :class:`VecEnv`. Thus, this class thinly wraps
    over the environment from :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.

    Note: While Stable-Baselines3 supports Gym 0.26+ API, their vectorized environment
    still uses the old API (i.e. it is closer to Gym 0.21). Thus, we implement the old
    API for the vectorized environment.

    We also add monitoring functionality that computes the un-discounted episode
    return and length. This information is added to the info dicts under key `episode`.

    In contrast to the Isaac Lab environment, stable-baselines expect the following:

    1. numpy datatype for MDP signals
    2. a list of info dicts for each sub-environment (instead of a dict)
    3. when environment has terminated, the observations from the environment should correspond
       to the one after reset. The "real" final observation is passed using the info dicts
       under the key ``terminal_obs``.

    .. warning::

        By the nature of physics stepping in Isaac Sim, it is not possible to forward the
        simulation buffers without performing a physics step. Thus, reset is performed
        inside the :meth:`step()` function after the actual physics step is taken.
        Thus, the returned observations for terminated environments is the one after the reset.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:

    1. https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
    2. https://stable-baselines3.readthedocs.io/en/master/common/monitor.html

    """

    def __init__(self, env, backand_device=None):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # initialize the wrapper
        self.env = env
        # collect common information
        self.num_envs = getattr(self.unwrapped, "num_envs", 1)
        self.sim_device = getattr(self.unwrapped, "device", "cpu")
        self.backand_device = backand_device or self.sim_device
        self.render_mode = self.unwrapped.render_mode

        # obtain gym spaces
        # Note: stable-baselines3 does not like when we have unbounded action space so
        #   we set it to some high value here. Maybe this is not general but something to think about.

        # 'policy' key becomes state and everything else is allegedly a camera.
        observation_space = spaces.Dict({
            "state": self.env.single_observation_space["policy"],
            "images": spaces.Dict({k:v for k, v in self.env.single_observation_space.items() if k!="policy"})
        })
        self.image_keys = [k for k in self.env.single_observation_space.keys() if k!="policy"]

        action_space = self.env.single_action_space if hasattr(self.env, "single_action_space") else self.env.action_space
        if isinstance(action_space, spaces.Box) and not action_space.is_bounded("both"):
            action_space = spaces.Box(low=-100, high=100, shape=action_space.shape)

        # initialize vec-env
        VecEnv.__init__(self, self.num_envs, observation_space, action_space)
        # add buffer for logging episodic information
        self._ep_rew_buf = torch.zeros(self.num_envs, device=self.backand_device)
        self._ep_len_buf = torch.zeros(self.num_envs, device=self.backand_device)

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self):
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def get_episode_rewards(self) -> list[float]:
        """Returns the rewards of all the episodes."""
        return self._ep_rew_buf.cpu().tolist()

    def get_episode_lengths(self) -> list[int]:
        """Returns the number of time-steps of all the episodes."""
        return self._ep_len_buf.cpu().tolist()

    """
    Operations - MDP
    """

    def seed(self, seed: int | None = None) -> list[int | None]:  # noqa: D102
        return [self.env.seed(seed)] * self.env.num_envs

    def reset(self) -> VecEnvObs:  # noqa: D102
        obs_dict, _ = self.env.reset()
        # reset episodic information buffers
        self._ep_rew_buf.zero_()
        self._ep_len_buf.zero_()
        # convert data types to numpy depending on backend
        return self._process_obs(obs_dict)

    def step_async(self, actions):  # noqa: D102
        # convert input to numpy array
        if isinstance(actions, dict):
            actions = {k: v.to(device=self.sim_device) for k, v in actions.items()}
        elif isinstance(actions, torch.Tensor):
            actions = actions.to(device=self.sim_device, dtype=torch.float32)
        else:
            actions = np.asarray(actions)
            actions = torch.from_numpy(actions).to(device=self.sim_device, dtype=torch.float32)
        # convert to tensor
        self._async_actions = actions

    def step_wait(self) -> VecEnvStepReturn:  # noqa: D102
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(self._async_actions)
        # compute reset ids
        dones = terminated | truncated
        reset_ids = (dones > 0).nonzero(as_tuple=False)

        # convert data types to numpy depending on backend
        # Note: ManagerBasedRLEnv uses torch backend (by default).
        obs = self._process_obs(obs_dict)
        rew = rew.detach().to(self.backand_device)
        terminated = terminated.detach().to(self.backand_device)
        truncated = truncated.detach().to(self.backand_device)
        dones = dones.detach().to(self.backand_device)

        # update episode un-discounted return and length
        self._ep_rew_buf += rew
        self._ep_len_buf += 1
        # convert extra information to list of dicts
        infos = self._process_extras(obs, terminated, truncated, extras, reset_ids)
        # reset info for terminated environments
        self._ep_rew_buf[reset_ids] = 0
        self._ep_len_buf[reset_ids] = 0

        return obs, rew, dones, infos

    def close(self):  # noqa: D102
        self.env.close()

    def get_attr(self, attr_name, indices=None):  # noqa: D102
        # resolve indices
        if indices is None:
            indices = slice(None)
            num_indices = self.num_envs
        else:
            num_indices = len(indices)
        # obtain attribute value
        attr_val = getattr(self.env, attr_name)
        # return the value
        if not isinstance(attr_val, torch.Tensor):
            return [attr_val] * num_indices
        return attr_val[indices].detach().cpu().numpy()

    def set_attr(self, attr_name, value, indices=None):  # noqa: D102
        raise NotImplementedError("Setting attributes is not supported.")

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):  # noqa: D102
        if method_name == "render":
            # gymnasium does not support changing render mode at runtime
            if hasattr(self.env, "render_human"):
                return self.env.render_human()
            return self.env.render()
        # this isn't properly implemented but it is not necessary.
        # mostly done for completeness.
        env_method = getattr(self.env, method_name)
        return env_method(*method_args, indices=indices, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):  # noqa: D102
        raise NotImplementedError("Checking if environment is wrapped is not supported.")

    def get_images(self):  # noqa: D102
        raise NotImplementedError("Getting images is not supported.")

    """
    Helper functions.
    """

    def _process_obs(self, obs_dict: torch.Tensor | dict[str, torch.Tensor]) -> np.ndarray | dict[str, np.ndarray]:
        """Semantic grouping of the output of the simulation environment."""

        if not isinstance(obs_dict, dict):
            raise NotImplementedError(f"Unsupported data type: {type(obs_dict)}")

        obs = {}
        for k, v in obs_dict.items():
            if k == "policy":
                obs["state"] = v.detach().to(self.backand_device)
            elif k in self.image_keys:  # **customize at necessity**
                imgs = obs.get("images", {})
                imgs.update({k: v.detach().to(self.backand_device)})
                obs["images"] = imgs
            else:
                extras = obs.get("extra", {})
                extras.update({k: v.detach().to(self.backand_device)})
                obs["extra"] = extras

        return obs

    def _process_extras(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        extras: dict[str, torch.Tensor],
        reset_ids: torch.Tensor,
    ) -> dict[str, Any]:
        """Vectorized version function that returns a dict of tensors."""
        infos = {"truncated": truncated & ~terminated}
        # Handle TimeLimit.truncated - vectorized
        # Handle episode information
        episode_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.backand_device)
        episode_mask[reset_ids] = True
        # Initialize episode dict with rewards and lengths
        infos["episode"] = {
            "rew": torch.where(episode_mask, self._ep_rew_buf, torch.full_like(self._ep_rew_buf, float("nan"))),
            "len": torch.where(episode_mask, self._ep_len_buf, torch.full_like(self._ep_len_buf, float("nan"))),
        }
        # Handle terminal observations
        if isinstance(obs, dict):
            infos["terminal_obs"] = {}
            for key, value in obs.items():
                if isinstance(value, dict):
                    # Handle nested dictionaries
                    infos["terminal_obs"][key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            if sub_value.dtype in [torch.uint8, torch.float32, torch.float64]:
                                infos["terminal_obs"][key][sub_key] = torch.full_like(sub_value.float(), float("nan"))
                                infos["terminal_obs"][key][sub_key][episode_mask] = sub_value[episode_mask].float()
                            else:
                                infos["terminal_obs"][key][sub_key] = torch.where(
                                    episode_mask.unsqueeze(-1).expand_as(sub_value), sub_value,
                                    torch.full_like(sub_value, -1))
                        else:
                            infos["terminal_obs"][key][sub_key] = [sub_value[i] if mask else None for i, mask in
                                                                   enumerate(episode_mask.cpu().tolist())]
                elif isinstance(value, torch.Tensor):
                    if value.dtype in [torch.float32, torch.float64]:
                        infos["terminal_obs"][key] = torch.where(episode_mask.unsqueeze(-1).expand_as(value),
                                                                 value.float(), torch.full_like(value.float(), float("nan")), )
                    else:
                        infos["terminal_obs"][key] = torch.where(episode_mask.unsqueeze(-1).expand_as(value),
                                                                 value, torch.full_like(value, -1))
                else:
                    infos["terminal_obs"][key] = [value[i] if mask else None for i, mask in
                                                  enumerate(episode_mask.cpu().tolist())]
        else:
            if obs.dtype in [torch.float32, torch.float64]:
                infos["terminal_obs"] = torch.where(episode_mask.unsqueeze(-1).expand_as(obs), obs.float(),
                                                    torch.full_like(obs.float(), float("nan")))
            else:
                infos["terminal_obs"] = torch.where(episode_mask.unsqueeze(-1).expand_as(obs), obs,
                                                    torch.full_like(obs, -1))
        # Handle remaining extras
        for key, value in extras.items():
            if key != "log":
                if isinstance(value, torch.Tensor):
                    if value.dim() == 0:  # scalar tensor
                        expanded_value = value.expand(self.num_envs)
                    else:  # batched tensor
                        expanded_value = value

                    if expanded_value.dtype in [torch.float32, torch.float64]:
                        infos[key] = expanded_value.float()
                    else:
                        infos[key] = expanded_value
                else:  # non-tensor value
                    try:
                        infos[key] = torch.full((self.num_envs,), value, device=self.backand_device, dtype=torch.float32)
                    except (ValueError, TypeError):
                        infos[key] = [value for _ in range(self.num_envs)]
            else:
                for sub_key, sub_value in extras["log"].items():
                    if isinstance(sub_value, torch.Tensor):
                        if sub_value.dim() == 0:  # scalar tensor
                            expanded_value = sub_value.expand(self.num_envs).to(self.backand_device)
                        else:  # batched tensor
                            expanded_value = sub_value
                    else:  # non-tensor value
                        try:
                            expanded_value = torch.full(
                                (self.num_envs,), sub_value, device=self.backand_device, dtype=torch.float32
                            )
                        except (ValueError, TypeError):
                            # For non-numeric values, create an object array
                            expanded_value = sub_value
                            infos["episode"][sub_key] = [
                                sub_value if mask else None for mask in episode_mask.cpu().tolist()
                            ]
                            continue
                    # Only process tensor values
                    if isinstance(expanded_value, torch.Tensor):
                        if expanded_value.dtype in [torch.float32, torch.float64]:
                            infos["episode"][sub_key] = torch.where(
                                episode_mask,
                                expanded_value.float(),
                                torch.full_like(expanded_value.float(), float("nan")),
                            )
                        else:
                            # For non-float tensors, use a sentinel value (e.g., -1) or create a masked tensor
                            infos["episode"][sub_key] = torch.where(
                                episode_mask, expanded_value, torch.full_like(expanded_value, -1)
                            )

        return infos
