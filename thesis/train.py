import os
import sys
import time
import random
import argparse
import warnings
import pathlib
import importlib
from functools import partial as bind

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import ruamel.yaml

# Add embodied to path (adjust if needed)
import embodied
from embodied import wrappers

# Import cleanrl PPO agent structure
from ppo_agent import Agent, layer_init

# --- Config loading ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument('--configs', type=str, nargs='+', default=[], help="Which named configs to apply.")
    parser.add_argument('--seed', type=int, default=None, help="Override seed manually.")
    return parser.parse_args()

def load_config(argv=None):
    configs = ruamel.yaml.YAML(typ='safe').load(
        (embodied.Path(__file__).parent / 'config.yaml').read())
    
    parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
    config = embodied.Config(configs['defaults'])

    for name in parsed.configs:
        config = config.update(configs[name])
    config = embodied.Flags(config).parse(other)
    # args = embodied.Config(
    #     **config.run, logdir=config.logdir,
    #     batch_steps=config.batch_size * config.batch_length, policy_rollout_every=config.policy_rollout_every, full_policy_rollout_every=config.full_policy_rollout_every)
    print(config)

    return config

# --- Environment creation ---

def make_envs(config):
    suite, task = config['task'].split('_', 1)
    ctors = []
    for index in range(config['num_envs']):
        ctor = lambda: make_env(config)
        if config.get('envs', {}).get('parallel', 'none') != 'none':
            ctor = bind(embodied.Parallel, ctor, config['envs']['parallel'])
        if config.get('envs', {}).get('restart', True):
            ctor = bind(wrappers.RestartOnException, ctor)
        ctors.append(ctor)
    envs = [ctor() for ctor in ctors]
    return embodied.BatchEnv(envs, parallel=(config.get('envs', {}).get('parallel', 'none') != 'none'))

def make_env(config, **overrides):
    # You can add custom environments by creating and returning the environment
    # instance here. Environments with different interfaces can be converted
    # using `embodied.envs.from_gym.FromGym` and `embodied.envs.from_dm.FromDM`.
    suite, task = config.task.split('_', 1)
    if "TrailEnv" in task or "GridBlindPick" or "LavaTrail" in task:
        import trailenv

    ctor = {
        'dummy': 'embodied.envs.dummy:Dummy',
        'gym': 'embodied.envs.from_gym:FromGym',
        'gymnasium': 'embodied.envs.from_gymnasium:FromGymnasium',
        'dm': 'embodied.envs.from_dmenv:FromDM',
        'crafter': 'embodied.envs.crafter:Crafter',
        'dmc': 'embodied.envs.dmc:DMC',
        'atari': 'embodied.envs.atari:Atari',
        'dmlab': 'embodied.envs.dmlab:DMLab',
        'minecraft': 'embodied.envs.minecraft:Minecraft',
        'loconav': 'embodied.envs.loconav:LocoNav',
        'pinpad': 'embodied.envs.pinpad:PinPad',
        'robopianist': 'embodied.envs.robopianist:RoboPianist'
    }[suite]
    if isinstance(ctor, str):
        module, cls = ctor.split(':')
        module = importlib.import_module(module)
        ctor = getattr(module, cls)
    kwargs = config.env.get(suite, {})
    kwargs.update(overrides)
    if suite == 'robopianist':
        kwargs.update({
        'record': config.run.script == 'eval_only'  # record in eval only for now (single environment)
        })
        render_image = False
        if 'Pixel' in task:
            task = task.replace('Pixel', '')
        render_image = True
        kwargs.update({'render_image': render_image})

    env = ctor(task, **kwargs)
    return wrap_env(env, config)

def wrap_env(env, config):
    args = config.get('wrapper', {})
    for name, space in env.act_space.items():
        if name == 'reset':
            continue
        elif space.discrete:
            env = wrappers.OneHotAction(env, name)
        elif args.get('discretize', 0):
            env = wrappers.DiscretizeAction(env, name, args['discretize'])
        else:
            env = wrappers.NormalizeAction(env, name)

    env = wrappers.ExpandScalars(env)

    if args.get('length', 0):
        env = wrappers.TimeLimit(env, args['length'], args.get('reset', True))
    if args.get('checks', False):
        env = wrappers.CheckSpaces(env)

    for name, space in env.act_space.items():
        if not space.discrete:
            env = wrappers.ClipAction(env, name)

    return env


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="thesis/config.yaml", help="Path to config file.")
    parser.add_argument('--configs', type=str, nargs='+', default=['defaults'], help="List of config names to apply.")
    parser.add_argument('--seed', type=int, default=None, help="Optional override seed")
    return parser.parse_args(argv)  # <<<<< VERY IMPORTANT: pass argv here


# --- PPO Training ---
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parsed_args = parse_args(argv)
    config = load_config(argv)

    # Seeding
    seed = parsed_args.seed if parsed_args.seed is not None else config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    # Create a dummy environment just to detect action space type
    env = make_env(config)

    # Decide whether discrete or continuous
    action_spaces = [space for name, space in env.act_space.items() if name != "reset"]
    if len(action_spaces) != 1:
        raise ValueError("Only single-action-space environments are supported for now.")
    action_space = action_spaces[0]

    if action_space.discrete:
        ppo_script = "ppo"
    else:
        ppo_script = "ppo_continuous_action"

    print(f"[train.py] Detected action space type: {'Discrete' if action_space.discrete else 'Continuous'}")
    print(f"[train.py] Running script: {ppo_script}.py")

    # Now execute the correct PPO script
    import subprocess
    run_command = [
        sys.executable,  # usually "python"
        f"thesis/{ppo_script}.py",
        "--seed", str(seed),
        "--config", parsed_args.config,
    ]
    print("[train.py] Launching subprocess:", " ".join(run_command))
    subprocess.run(run_command, check=True)

if __name__ == "__main__":
    main()