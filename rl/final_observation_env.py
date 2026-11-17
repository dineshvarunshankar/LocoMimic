"""Terminal observations for mjlab 1.2's automatically resetting environment."""

from collections.abc import Mapping

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.noise.noise_cfg import NoiseModelCfg


class FinalObservationEnv(ManagerBasedRlEnv):
    """Keep pre-reset actor/critic observations in extras without a second reset.

    The capture uses the current command, before mjlab resets and advances it.
    History, delay buffers and stateful noise are unsupported: computing those
    groups twice would change the observation manager's state. Stateless noise,
    clipping and scaling use the normal observation-manager pipeline.
    """

    def __init__(self, cfg, *args, **kwargs):
        for name in ("actor", "critic"):
            group = cfg.observations.get(name)
            if group is None:
                continue
            for term in group.terms.values():
                history = group.history_length
                if history is None:
                    history = term.history_length
                if history or term.delay_max_lag or (
                    group.enable_corruption and isinstance(term.noise, NoiseModelCfg)
                ):
                    raise ValueError(
                        f"FinalObservationEnv requires {name} observations without "
                        "history, delay, or stateful noise"
                    )
        self._capture_final_observation = False
        super().__init__(cfg, *args, **kwargs)

    def _clear_final_observation(self):
        self.extras.pop("final_observation", None)
        self.extras.pop("final_env_ids", None)

    def reset(self, **kwargs):
        self._clear_final_observation()
        return super().reset(**kwargs)

    def step(self, action):
        self._clear_final_observation()
        self._capture_final_observation = True
        try:
            return super().step(action)
        finally:
            self._capture_final_observation = False

    def _reset_idx(self, env_ids=None):
        if self._capture_final_observation:
            # Upstream resets before its forward/sense calls. Refresh terminal
            # kinematics first, then bypass compute()'s previous-step cache.
            self.sim.forward()
            self.sim.sense()
            final = {}
            for name in ("actor", "critic"):
                if name not in self.cfg.observations:
                    continue
                obs = self.observation_manager.compute_group(name, update_history=False)
                final[name] = (
                    {key: value.clone() for key, value in obs.items()}
                    if isinstance(obs, Mapping) else obs.clone()
                )
            self.extras["final_observation"] = final
            self.extras["final_env_ids"] = env_ids.clone()
        super()._reset_idx(env_ids)
