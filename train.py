import torch
import gymnasium as gym
import wandb

from config import Args
from agent import ActorCritic

def get_rollout(agent, env, init_obs, rollout_len, gamma):
    observations = torch.zeros()
    action_probs = torch.zeros()
    rewards = torch.zeros()
    dones = torch.zeros()  # taking action at this step terminates the episode
    rewards_to_go = torch.zeros()
    values = torch.zeros()
    advantages = torch.zeros()

    # collect policy rollout
    with torch.no_grad():

        act, act_prob, act_entropy = agent.get_action(init_obs)
        val = agent.get_value(init_obs)
        observations[0] = init_obs
        action_probs[0] = act_prob
        values[0] = val

        for t in range(1, rollout_len):
            obs, rwd, done, truncated, info = env.step(act)
            terminated = done or truncated
            rewards[t-1] = rwd
            dones[t-1] = terminated
            if terminated:
                obs, info = env.reset()
            act, act_prob, act_entropy = agent.get_action(obs)
            val = agent.get_value(obs)
            observations[t] = obs
            action_probs[t] = act_prob 
            values[t] = val

        last_obs, rwd, done, truncated, info = env.step(act)
        terminated = done or truncated
        rewards[rollout_len-1] = rwd
        dones[rollout_len-1] = terminated
        if terminated:
            last_obs, info = env.reset()
    
        for t in reversed(range(rollout_len)):
            if dones[t]:
                rewards_to_go[t] = rewards[t]
            else:
                rewards_to_go[t] = rewards[t] + gamma * values[t+1]
        
        advantages = rewards_to_go - values  # TODO normalize advantages?

    return observations, action_probs, rewards_to_go, advantages, last_obs 


def eval(agent, env, n_eval_episodes):
    with torch.no_grad():
        scores = torch.zeros(n_eval_episodes)
        for ep in range(n_eval_episodes):
            obs, info = env.reset()
            terminated = False
            ep_ret = 0
            while not terminated:
                act, *_ = agent.get_action(obs)
                obs, rwd, done, truncated, info = env.step(act)
                terminated = done or truncated
                ep_ret += rwd
            scores[ep] = ep_ret
    return scores
            

if __name__ == "__main__":
    args = Args()
    ppo_eps = args.ppo_eps
    n_env_steps = args.n_env_steps
    rollout_len = args.rollout_len
    batch_size = args.batch_size
    device = args.device

    wandb.init(
        project = 'PPO',
        name = args.gym_id,
        config = args,
        monitor_gym=True,
        mode = 'offline',
    )

    # setup env
    env = gym.make(args.gym_id)
    eval_env = gym.make(args.gym_id)
    eval_env = gym.wrappers.RecordVideo(eval_env, video_folder='videos', episode_trigger=lambda k: k % 100 == 0)

    # set seed for reproducibility
    env.seed(args.seed)
    eval_env.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = ActorCritic(
        n_hidden=args.n_hidden, 
        act_dim=env.action_space.shape, 
        obs_dim=env.observation_space.shape
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=args.adam_eps)

    init_obs, info = env.reset()
    for iter in range(n_env_steps // rollout_len):

        rollout, init_obs = get_rollout(agent, env, init_obs, rollout_len, args.gamma)
        observations, action_probs, rewards_to_go, advantages = rollout

        for epoch in range(args.n_epochs):
            inds = torch.arange(0, rollout_len)
            inds = torch.randperm(inds)
            for step in range(rollout_len // batch_size):
                b_inds = inds[step*batch_size, (step+1)*batch_size]
                b_obs = observations[b_inds]
                b_act_prob = action_probs[b_inds]
                b_rwd_to_go = rewards_to_go[b_inds]
                b_adv = advantages[b_inds]

                b_pred_val = agent.get_value(b_obs)
                b_pred_act, b_pred_act_prob, b_pred_act_entropy = agent.get_action(b_obs)
                b_prob_ratio = b_pred_act_prob / b_act_prob

                ppo_loss = -(b_prob_ratio * b_adv, b_prob_ratio.clip(1-ppo_eps, 1+ppo_eps) * b_adv).min(axis=0)  # TODO use mean()? 
                value_loss = ((b_pred_val - b_rwd_to_go)**2).sum()  # TODO use mean?
                entropy_loss = -b_pred_act_entropy.sum()  # TODO use mean?
                loss = ppo_loss + value_loss + entropy_loss  # TODO add loss coefficients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        env_steps_trained = (iter+1)*rollout_len
        if iter % args.log_every == 0:
            print(f"| iter: {iter+1} | env steps trained: {env_steps_trained} |  loss: {loss.item()} |")
            wandb.log('loss', loss.item(), step=env_steps_trained)
        if iter % args.eval_every == 0:
            scores = eval(agent, eval_env, args.n_eval_episodes)
            score_mean = scores.mean().item()
            score_std = scores.std().item()
            print(f"| iter: {iter+1} | env steps trained: {env_steps_trained} | episodic return mean: {score_mean} | episodic return std: {score_std} |")
            wandb.log('return_mean', score_mean, step = env_steps_trained)
            wandb.log('return_std', score_std, step = env_steps_trained)

