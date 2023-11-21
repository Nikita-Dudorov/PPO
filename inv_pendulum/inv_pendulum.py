import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box

class InvertedPendulumEnv(MujocoEnv, utils.EzPickle):
    MAX_EP_LEN = 1000
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            # "single_rgb_array",
            # "single_depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            "/home/nikita/Projects/RL/PPO/inv_pendulum/model.xml",
            2,
            observation_space=observation_space,
            **kwargs
        )
        self.last_ob = None
        self._timestep = 0 

    def step(self, a):
        self._timestep += 1
        reward = 1.0  # change it to angle diff with vertical pos
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        terminated = bool(not np.isfinite(ob).all())
        truncated = self._timestep >= self.MAX_EP_LEN

        if self.render_mode == "human":
            self.render()
        return ob, reward, terminated, truncated, {}

    def reset_model(self):
        self._timestep = 0
        qpos = self.init_qpos
        qvel = self.init_qvel
        qpos[1] = 3.14 # Set the pole to be facing down
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent