class Args:
    def __init__(self):
        self.gym_id = 'CartPole-v1'
        self.device = 'cpu'
        self.seed = 123

        # train
        self.n_env_steps = 1e5
        self.rollout_len = 256
        self.batch_size = 64
        self.n_epochs = 1  # 20
        self.lr = 1e-5  # 1e-3
        self.lr_decay = True
        self.grad_clip = 0.5

        # log
        self.log_every = 10

        # eval
        self.eval_every = 100
        self.n_eval_episodes = 100

        # agent
        self.n_hidden = 64
        self.ppo_eps = 0.2
        self.ppo_eps_decay = True
        self.gamma = 0.98
        self.gae_lam = 0.8
        self.c_val_loss = 0.5
        self.c_entr_loss = 0.0