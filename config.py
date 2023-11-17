class Args:
    def __init__(self):
        self.gym_id = 'CartPole-v1'
        self.device = 'cpu'
        self.seed = 123

        # train
        self.n_env_steps = 1e6
        self.rollout_len = 128
        self.batch_size = self.rollout_len // 4
        self.n_epochs = 4
        self.lr = 2.5e-4
        self.adam_eps = 1e-5

        # log
        self.log_every = 1

        # eval
        self.eval_every = 10
        self.n_eval_episodes = 100

        # agent
        self.n_hidden = 64
        self.ppo_eps = 0.2
        self.gamma = 0.99