import os
import sys
import time
import random
import argparse
import warnings
import pathlib
import importlib
import re
from functools import partial as bind
from types import SimpleNamespace

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

# Add thesis directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import embodied package and its components
import embodied
from embodied import wrappers
from embodied.core import Path, Flags, Config

# Import teacher-student components
from thesis.teacher_student.teacher import TeacherPolicy
from thesis.teacher_student.student import StudentPolicy
from thesis.teacher_student.replay_buffer import ReplayBuffer
from thesis.teacher_student.bc import BehavioralCloning

def dict_to_namespace(d):
    """Convert a dictionary to a SimpleNamespace recursively."""
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

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
    config_dict = embodied.Config(configs['defaults'])

    for name in parsed.configs:
        config_dict = config_dict.update(configs[name])
    config_dict = embodied.Flags(config_dict).parse(other)
    
    # Convert to SimpleNamespace
    config = dict_to_namespace(config_dict)
    print(config)

    return config

def make_envs(config):
    suite, task = config.task.split('_', 1)
    ctors = []
    for index in range(config.num_envs):
        ctor = lambda: make_env(config)
        if hasattr(config, 'envs') and hasattr(config.envs, 'parallel') and config.envs.parallel != 'none':
            ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
        if hasattr(config, 'envs') and hasattr(config.envs, 'restart') and config.envs.restart:
            ctor = bind(wrappers.RestartOnException, ctor)
        ctors.append(ctor)
    envs = [ctor() for ctor in ctors]
    return embodied.BatchEnv(envs, parallel=(hasattr(config, 'envs') and hasattr(config.envs, 'parallel') and config.envs.parallel != 'none'))

def make_env(config, **overrides):
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
    kwargs = getattr(config.env, suite, {})
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
    args = getattr(config, 'wrapper', {})
    for name, space in env.act_space.items():
        if name == 'reset':
            continue
        elif space.discrete:
            env = wrappers.OneHotAction(env, name)
        elif hasattr(args, 'discretize') and args.discretize:
            env = wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = wrappers.NormalizeAction(env, name)

    env = wrappers.ExpandScalars(env)

    if hasattr(args, 'length') and args.length:
        env = wrappers.TimeLimit(env, args.length, getattr(args, 'reset', True))
    if hasattr(args, 'checks') and args.checks:
        env = wrappers.CheckSpaces(env)

    for name, space in env.act_space.items():
        if not space.discrete:
            env = wrappers.ClipAction(env, name)

    return env

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parsed_args = parse_args()
    config = load_config(argv)

    # Seeding
    seed = parsed_args.seed if parsed_args.seed is not None else config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    # Create environment
    env = make_env(config)
    envs = make_envs(config)

    # Initialize components
    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
    
    teacher = TeacherPolicy(envs, config).to(device)
    student = StudentPolicy(envs, config).to(device)
    
    replay_buffer = ReplayBuffer(
        config.replay_buffer.capacity,
        envs.obs_space,
        config.full_keys,
        config.keys
    )
    
    bc_trainer = BehavioralCloning(student, config)
    
    # Initialize optimizers
    teacher_optimizer = optim.Adam(teacher.parameters(), lr=config.learning_rate, eps=1e-5)
    student_optimizer = optim.Adam(student.parameters(), lr=config.bc.learning_rate, eps=1e-5)
    
    # Initialize logging
    exp_name = os.path.basename(__file__)[: -len(".py")]
    run_name = f"{config.task}__{exp_name}__{seed}__{int(time.time())}"
    
    if config.track:
        import wandb
        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            sync_tensorboard=True,
            config=vars(config),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )
    
    # Training loop
    global_step = 0
    start_time = time.time()
    
    # Initialize episode tracking buffers
    episode_returns = np.zeros(config.num_envs)
    episode_lengths = np.zeros(config.num_envs)
    
    # Initialize video logging buffers
    video_frames = {key: [] for key in config.log_keys_video}
    last_video_log = 0
    video_log_interval = 10000  # Log a video every 10k steps
    
    # Initialize actions with zeros and reset flags
    action_shape = envs.act_space['action'].shape
    num_envs = config.num_envs
    
    acts = {
        'action': np.zeros((num_envs,) + action_shape, dtype=np.float32),
        'reset': np.ones(num_envs, dtype=bool)  # Reset all environments initially
    }
    
    # Get initial observations
    obs_dict = envs.step(acts)
    next_obs = {}
    for key in teacher.mlp_keys + teacher.cnn_keys:
        next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(device)
    next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(device)
    
    # Main training loop
    for iteration in range(1, config.total_timesteps // (config.num_envs * config.num_steps) + 1):
        if config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / (config.total_timesteps // (config.num_envs * config.num_steps))
            lrnow = frac * config.learning_rate
            teacher_optimizer.param_groups[0]["lr"] = lrnow
            student_optimizer.param_groups[0]["lr"] = lrnow
        
        # 1. Teacher collects transitions
        teacher_transitions = teacher.collect_transitions(envs, config.num_steps)
        
        # 2. Store transitions in replay buffer
        for transition in teacher_transitions:
            replay_buffer.add(transition)
        
        # 3. Update teacher policy using PPO
        # TODO: Implement PPO update for teacher
        
        # 4. Student learns from teacher via BC
        if len(replay_buffer) >= config.bc.batch_size:
            bc_metrics = bc_trainer.train(
                replay_buffer,
                config.bc.num_steps,
                config.bc.batch_size,
                writer,
                global_step
            )
            
            # Log BC metrics
            if config.track:
                wandb.log({
                    "bc/loss": bc_metrics['bc_loss'],
                    "bc/action_diff": bc_metrics.get('action_diff', None),
                    "metrics/global_step": global_step,
                })
        
        # 5. Optionally: Student fine-tunes with PPO
        # TODO: Implement PPO update for student
        
        # Update global step
        global_step += config.num_envs * config.num_steps
        
        # Log metrics
        if config.track:
            wandb.log({
                "charts/learning_rate": teacher_optimizer.param_groups[0]["lr"],
                "charts/SPS": int(global_step / (time.time() - start_time)),
                "metrics/global_step": global_step,
            })
    
    if config.save_model:
        model_path = f"runs/{run_name}/{exp_name}.cleanrl_model"
        torch.save({
            'teacher_state_dict': teacher.state_dict(),
            'student_state_dict': student.state_dict(),
        }, model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()

if __name__ == "__main__":
    main()
