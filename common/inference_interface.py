from typing import Optional, Tuple

from stable_baselines3.common.vec_env import VecEnv

from common.callbacks import BaseCallback


class InferenceInterface:
    def predict_episodes(
        self,
        env: VecEnv,
        n_eval_episodes: int,
        deterministic: bool = True,
        callback: Optional[BaseCallback] = None,
    ) -> Tuple[float, int]:
        """
        Run prediction/evaluation for multiple episodes.

        :param env: The evaluation environment
        :param n_eval_episodes: Number of episodes to evaluate
        :param deterministic: Whether to use deterministic actions
        :param callback: Callback that will be called at each step
        :return: Tuple of (episode_rewards, episode_lengths)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        if callback is not None:
            callback.on_rollout_start()

        results = self.predict_rollout(env, callback=callback, n_episodes=n_eval_episodes)

        if callback is not None:
            callback.on_rollout_end()

        self.policy.set_training_mode(True)

        return results
