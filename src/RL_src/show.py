import argparse
import os
import sys
import math
import numpy as np
import torch
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import custom_gym
from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
from matplotlib import animation
sys.path.append('a2c_ppo_acktr')

# seed = 0

args = get_args()
# env_name = "QuadRotor-v0"

env = make_vec_envs(
    args.env_name, 
    args.seed,
    1,
    None,
    None,
    device='cpu',
    allow_early_resets=False)

render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
if args.loadfile:
    model_path = os.path.join('trained_models/', args.algo, args.env_name + args.loadfile + '.pt')
    actor_critic, ob_rms = torch.load(model_path)
else:
    print('No Pretrained model loaded') 



recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)
obs = env.reset()
for i in range(6000):    
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=True)

    obs, reward, done, info = env.step(action)

    masks.fill_(0.0 if done else 1.0)

    if render_func is not None:
        render_func('human')
    