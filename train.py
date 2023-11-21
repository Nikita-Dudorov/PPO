import random
import torch
import wandb
import gymnasium as gym

from config import Args
from agent import ActorCritic

def get_rollout(agent, env, init_obs, rollout_len, gamma, device):
    obs_dim = env.observation_space.shape[0]
    observations = torch.zeros((rollout_len, obs_dim)).to(device)
    action_probs = torch.zeros((rollout_len, 1)).to(device)
    rewards = torch.zeros((rollout_len, 1)).to(device)
    dones = torch.zeros((rollout_len, 1)).to(device)  # taking action at this step terminates the episode
    rewards_to_go = torch.zeros((rollout_len, 1)).to(device)
    values = torch.zeros((rollout_len, 1)).to(device)
    advantages = torch.zeros((rollout_len, 1)).to(device)
    time_intervals = []  # first and last steps of episodes within rollout
    start = 0

    # collect policy rollout
    with torch.no_grad():

        init_obs = torch.from_numpy(init_obs).to(device).float()
        act, act_prob, act_entropy = agent.get_action(init_obs)
        val = agent.get_value(init_obs)
        observations[0] = init_obs
        action_probs[0] = act_prob
        values[0] = val

        for t in range(1, rollout_len):
            obs, rwd, done, truncated, info = env.step(act.item())
            terminated = done or truncated
            rewards[t-1] = rwd
            dones[t-1] = terminated
            if terminated:
                finish = t-1
                time_intervals.append((start, finish))
                start = t
                obs, info = env.reset()
            obs = torch.from_numpy(obs).to(device).float()
            act, act_prob, act_entropy = agent.get_action(obs)
            val = agent.get_value(obs)
            observations[t] = obs
            action_probs[t] = act_prob 
            values[t] = val

        t = rollout_len - 1
        last_obs, rwd, done, truncated, info = env.step(act.item())
        last_val = agent.get_value(torch.from_numpy(last_obs).to(device).float())
        terminated = done or truncated
        rewards[t] = rwd
        dones[t] = terminated
        finish = t
        time_intervals.append((start, finish))
        start = rollout_len
        if terminated:
            last_obs, info = env.reset()

        # calculate rewards-to-go
        for interval in time_intervals:
            start, finish = interval
            for t in reversed(range(start, finish+1)):
                if dones[t]:
                    rewards_to_go[t] = rewards[t]
                else:
                    if t + 1 <= rollout_len - 1:
                        rewards_to_go[t] = rewards[t] + gamma * rewards_to_go[t+1]
                    else:
                        rewards_to_go[t] = rewards[t] + gamma * last_val
        # rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + 1e-8)  # TODO normalize rewards-to-go?
        
        # calculate advantages
        advantages = rewards_to_go - values
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # TODO normalize advantages?
        # TODO gae + lambda

        only_complete_episodes = False
        if only_complete_episodes: 
            final_step = torch.where(dones == 1)[0].max().item()
            last_obs, info = env.reset()
        else:
            final_step = rollout_len - 1

    return observations[:final_step+1], action_probs[:final_step+1], rewards_to_go[:final_step+1], values[:final_step+1], advantages[:final_step+1], last_obs 


def eval(agent, env, n_eval_episodes, device):
    with torch.no_grad():
        scores = torch.zeros(n_eval_episodes)
        for ep in range(n_eval_episodes):
            obs, info = env.reset()
            terminated = False
            ep_ret = 0
            while not terminated:
                act, *_ = agent.get_action(torch.from_numpy(obs).to(device).float())
                obs, rwd, done, truncated, info = env.step(act.item())
                terminated = done or truncated
                ep_ret += rwd
            scores[ep] = ep_ret
    return scores
            

if __name__ == "__main__":
    args = Args()
    ppo_eps = args.ppo_eps
    c_val_loss = args.c_val_loss
    c_entr_loss = args.c_entr_loss
    n_env_steps = args.n_env_steps
    rollout_len = args.rollout_len
    batch_size = args.batch_size
    device = args.device

    wandb.init(
        project = 'PPO',
        name = args.gym_id + str(random.randint(1e3,1e4)),
        config = args,
        monitor_gym=True,
        mode = 'offline',
    )

    # setup env
    env = gym.make(args.gym_id)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=args.n_eval_episodes)
    eval_env = gym.make(args.gym_id, render_mode="rgb_array")
    eval_env = gym.wrappers.NormalizeObservation(eval_env)
    eval_env = gym.wrappers.RecordVideo(eval_env, video_folder='videos', episode_trigger=lambda k: k % args.n_eval_episodes == 0)

    # set seed for reproducibility
    # env.seed(args.seed)
    # eval_env.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = ActorCritic(
        n_hidden=args.n_hidden, 
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.n, 
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    n_iters = int(n_env_steps / rollout_len)
    init_obs, info = env.reset()
    for iter in range(n_iters):

        *rollout, init_obs = get_rollout(agent, env, init_obs, rollout_len, args.gamma, device)
        observations, action_probs, rewards_to_go, values, advantages = rollout

        for epoch in range(args.n_epochs):
            inds = torch.arange(0, len(observations))
            inds = inds[torch.randperm(len(inds))]
            for step in range(len(observations) // batch_size):
                b_inds = inds[step*batch_size:(step+1)*batch_size]
                b_obs = observations[b_inds]
                b_act_prob = action_probs[b_inds].view(-1)
                b_rwd_to_go = rewards_to_go[b_inds].view(-1)
                # b_val = values[b_inds].view(-1)
                b_adv = advantages[b_inds].view(-1)
                
                # TODO: normalize batch?
                # b_rwd_to_go = (b_rwd_to_go - b_rwd_to_go.mean()) / (b_rwd_to_go.std() + 1e-8)
                # b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

                b_pred_val = agent.get_value(b_obs).view(-1)
                # b_pred_val = b_val + (b_pred_val - b_val).clip(-ppo_eps, ppo_eps)  # TODO: clip values?
                b_pred_act, b_pred_act_prob, b_pred_act_entropy = agent.get_action(b_obs)
                b_prob_ratio = b_pred_act_prob / b_act_prob

                ppo_loss = -(b_pred_act_prob * b_adv).sum()
                # ppo_loss = -torch.min(b_prob_ratio * b_adv, b_prob_ratio.clamp(1-ppo_eps, 1+ppo_eps) * b_adv).sum()  # TODO use mean? 
                value_loss = ((b_pred_val - b_rwd_to_go)**2).sum()  # TODO use mean?
                # TODO clip value loss?
                entropy_loss = -b_pred_act_entropy.sum()  # TODO use mean?

                loss = ppo_loss + c_val_loss * value_loss + c_entr_loss * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.grad_clip)
                optimizer.step()

            # print(f"| epoch: {epoch} | loss: {loss.item()} |")

        env_steps_trained = (iter+1)*rollout_len
        if iter % args.log_every == 0:
            print(f"| iter: {iter} | env steps trained: {env_steps_trained} | loss: {loss.item()} |")
            print(f"| ppo loss: {ppo_loss.item()} | value loss: {value_loss.item()} | entropy loss: {entropy_loss.item()} |")
            print(f"| lr: {optimizer.param_groups[0]['lr']} | ppo_eps: {ppo_eps}")
            print(f"| episode length mean: {sum(env.length_queue)/(max(len(env.length_queue), 1))} |")
            # print(f"| probabilities mean: {b_pred_act_prob.detach().mean().item()} |")
            wandb.log({'loss': loss.item()}, step = env_steps_trained)
        if iter % args.eval_every == 0:
            scores = eval(agent, eval_env, args.n_eval_episodes, device)
            score_mean = scores.mean().item()
            score_std = scores.std().item()
            print(f"| iter: {iter} | env steps trained: {env_steps_trained} | episodic return mean: {score_mean} | episodic return std: {score_std} |")
            wandb.log({'return_mean': score_mean, 'return_std': score_std}, step = env_steps_trained)
        if args.lr_decay:
            optimizer.param_groups[0]['lr'] = args.lr * (1 - iter / n_iters)
        if args.ppo_eps_decay:
            ppo_eps = args.ppo_eps * (1 - iter / n_iters)

    # torch.save(agent.state_dict(), save_path)
    env.close()
    eval_env.close()
