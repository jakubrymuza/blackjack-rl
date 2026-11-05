import torch
from sb3_contrib import MaskablePPO

from blackjack_rl.config import Config


class ModelAgent:
    def __init__(self, model: MaskablePPO):
        self.model = model
        self.name = "ppo"

    def probabilities(self, observation, public, mask, actions):
        with torch.no_grad():
            obs, _ = self.model.policy.obs_to_tensor(observation)
            distribution = self.model.policy.get_distribution(obs, action_masks=mask)
            probs = distribution.distribution.probs.cpu().numpy().reshape(-1).astype(float)
        probs[~mask] = 0
        return probs / probs.sum()


def make_model(config: Config, env, seed: int, log_dir: str):
    t = config.training
    if t is None:
        raise ValueError("Training configuration required")
    kwargs = {
        k: getattr(t, k)
        for k in (
            "n_steps",
            "batch_size",
            "n_epochs",
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "device",
        )
    }
    return MaskablePPO(
        "MultiInputPolicy",
        env,
        seed=seed,
        tensorboard_log=log_dir,
        policy_kwargs={"net_arch": {"pi": t.hidden_sizes, "vf": t.hidden_sizes}},
        verbose=1,
        **kwargs,
    )
