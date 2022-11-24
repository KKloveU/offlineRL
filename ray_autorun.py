import sys
import gym
import torch
import argparse
import os
import utils
import d4rl
import time
import ray
import time
import random
import numpy as np
import torch.nn as nn
from algo import BCQ,IQL,TD3_BC
from logger import Logger, setup_logger
from data_utils import d4rl_trajectories
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_list=['hopper-medium-expert-v2', 'ant-expert-v2', 'halfcheetah-medium-expert-v2',
'halfcheetah-random-v2', 'ant-random-v2', 'hopper-random-v2',
'halfcheetah-medium-v2',  'ant-medium-expert-v2', 'walker2d-expert-v2', 'ant-medium-replay-v2',
'ant-medium-v2', 'hopper-expert-v2', 'walker2d-random-v2',
'hopper-medium-v2', 'walker2d-medium-v2', 'halfcheetah-medium-replay-v2',
'halfcheetah-expert-v2', 'hopper-medium-replay-v2',
'walker2d-medium-expert-v2']


worng=['halfcheetah-medium-v0','hopper-random-v0','halfcheetah-expert-v0','ant-expert-v0',
    'halfcheetah-medium-expert-v0','hopper-expert-v0','halfcheetah-random-v0','ant-medium-v0','walker2d-expert-v0','ant-medium-expert-v0','walker2d-medium-v0',
    'ant-random-v0','hopper-medium-v0','walker2d-medium-expert-v0','walker2d-random-v0','hopper-medium-expert-v0']


def evaluate_policy(env,policy, mean, std, eval_episodes=10):
    all_returns = []
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        episodic_len = 0
        episodic_reward = 0
        while not done:
            obs = (np.array(obs).reshape(1, -1) - mean) / std
            action = policy.select_action(obs)
            obs, rew, done, info = env.step(action)
            episodic_reward += rew
            episodic_len += 1
            if episodic_len+1 == env._max_episode_steps:
                done = True
        all_returns.append(episodic_reward)
    all_returns = np.array(all_returns)
    avg_return = np.mean(all_returns)
    std_return = np.std(all_returns)
    median_return = np.median(all_returns)
    min_return = np.min(all_returns)

    d4rl_score = env.get_normalized_score(avg_return)

    print ("---------------------------------------")
    print ("Evaluation over %d episodes: %f | normalized score :%f " % (eval_episodes, avg_return, d4rl_score))
    print ("---------------------------------------")
    return avg_return, std_return, median_return, min_return, d4rl_score

@ray.remote
def run_algo(args):
    import d4rl
    init_time = time.time()
    env = gym.make(args.env_name)
    env.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(f'State dim :{state_dim}, Action dim: {action_dim}')
    print('Max action: ', max_action)

    # Load buffer
    replay_buffer = utils.ReplayBuffer()
    dataset = d4rl.qlearning_dataset(env)
    # dataset = env.unwrapped.get_dataset()

    # num_trajectories = int(args.buffer_size / env._max_episode_steps)
    d4rl_trajectories(dataset, env, replay_buffer, buffer_size=args.buffer_size)
    mean, std = replay_buffer.normalize_states()

    hparam_str_dict = dict(algo=args.algo_name, seed=args.seed, env=args.env_name,
                           batch_size=args.batch_size, buffer_size=args.buffer_size)
    variant = hparam_str_dict
    file_name = ','.join(['%s=%s' % (k, str(hparam_str_dict[k])) for k in sorted(hparam_str_dict.keys())])

    print ("---------------------------------------")
    print ("Settings: " + file_name)
    print ("---------------------------------------")

    logger=Logger()
    setup_logger(logger,file_name, variant=variant, log_dir=os.path.join(args.log_dir, file_name))


    if args.algo_name == 'BCQ':
        policy = BCQ.BCQ(state_dim=state_dim,
                            action_dim=action_dim,
                            max_action=max_action,
                            discount=args.gamma,
                            logger=logger)
    elif args.algo_name == 'IQL':
        policy = IQL.IQL(state_dim=state_dim,
                            action_dim=action_dim,
                            max_action=max_action,
                            hidden_dim=args.hidden_dim,
                            discount=args.gamma,
                            logger=logger)
    elif args.algo_name == 'BCQ-v2':
        policy = BCQ.BCQ(state_dim=state_dim,
                            action_dim=action_dim,
                            max_action=max_action,
                            cloning=True,
                            discount=args.gamma,
                            logger=logger)
    elif args.algo_name == 'TD3_BC':
        policy = TD3_BC.TD3_BC(state_dim=state_dim,
                            action_dim=action_dim,
                            max_action=max_action,
                            hidden_dim=args.hidden_dim,
                            discount=args.gamma,
                            logger=logger)
    else:
        sys.exit(f'Choose the right algo name, {args.algo_name} not found')

    training_iters = 0
    while training_iters < args.max_timesteps: 
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
        avg_return, std_return, median_return, min_return, d4rl_score = evaluate_policy(env, policy, mean, std)
        training_iters += args.eval_freq
        print("Training iterations: " + str(training_iters))
        logger.record_tabular('Training Epochs', int(training_iters // int(args.eval_freq)))
        logger.record_tabular('Eval/AverageReturn', avg_return)
        logger.record_tabular('Eval/StdReturn', std_return)
        logger.record_tabular('Eval/MedianReturn', median_return)
        logger.record_tabular('Eval/MinReturn', min_return)
        logger.record_tabular('Eval/D4RL_score', d4rl_score)
        logger.record_tabular('training time', (time.time() - init_time) / (60 * 60))
        logger.dump_tabular()
    # return 0


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="")    # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=float)      # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--version", default='0', type=str)
    parser.add_argument('--algo_name', default="TD3_BC", type=str)    # Which algo to run (see the options below in the main function)
    parser.add_argument('--log_dir', default='./data_tmp/', type=str) # Logging directory
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--buffer_size', default=1000000, type=int)
    parser.add_argument('--num_worker', default=1, type=int)
    parser.add_argument('--cuda',action='store_true')
    args = parser.parse_args()

    ray.init()
    print("start auto run_algo!")
    worker_list=[]
    for dataset_name in dataset_list:
        if len(worker_list)<args.num_worker:
            print("================= ready to run",dataset_name," =================")
            args.env_name=dataset_name
            if not args.cuda:
                worker_list.append(run_algo.remote(args))
            else:
                worker_list.append(run_algo.options(num_gpus=0.2).remote(args))
        else:
            done,worker_list=ray.wait(worker_list)
            # ray.get(done)

    print("=============== all start wait to finish! =====================")
    while(len(worker_list)):
        done,worker_list=ray.wait(worker_list)
    ray.shutdown()
    print("end!")
