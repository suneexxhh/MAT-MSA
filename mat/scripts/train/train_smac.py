#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
sys.path.append("../../")
from mat.config import get_config
from mat.envs.starcraft2.StarCraft2_Env import StarCraft2Env
from mat.envs.starcraft2.Random_StarCraft2_Env import RandomStarCraft2Env
from mat.envs.starcraft2.smac_maps import get_map_params
from mat.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from mat.runner.shared.smac_runner import SMACRunner as Runner
from mat.algorithms.mat.algorithm.scale_config import get_scale_config

"""Train script for SMAC."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                if all_args.random_agent_order:
                    env = RandomStarCraft2Env(all_args)
                else:
                    env = StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                if all_args.random_agent_order:
                    env = RandomStarCraft2Env(all_args)
                else:
                    env = StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--map_name', type=str, default='3m', help="Which smac map to run on")
    parser.add_argument('--eval_map_name', type=str, default='3m', help="Which smac map to eval on")
    parser.add_argument('--run_dir', type=str, default='', help="Which smac map to eval on")
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_false', default=True)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_false', default=True)
    parser.add_argument("--random_agent_order", action='store_true', default=False)
    # CMS层级数
    parser.add_argument(
        '--n_cms_levels',
        type=int,
        default=3,
        help='CMS 层级数 (建议 2-4)'
    )

    # CMS dropout
    parser.add_argument(
        '--cms_dropout',
        type=float,
        default=0.0,
        help='CMS 内部 dropout (建议 0.0 或 0.1)')

    # CMS 中间层 chunk_size
    parser.add_argument(
        '--cms_mid_chunk_size',
        type=int,
        default=4,
        help='CMS 中间层的 chunk_size (默认 4，控制中等时间尺度的记忆范围)')

    # 是否使用地图自适应配置
    parser.add_argument(
        '--use_map_adaptive',
        action='store_true',
        default=True,
        help='是否使用地图自适应配置 (默认 True，设置 --no-use_map_adaptive 禁用)')
    parser.add_argument(
        '--no-use_map_adaptive',
        dest='use_map_adaptive',
        action='store_false',
        help='禁用地图自适应配置，使用手动指定的 n_cms_levels 和 cms_mid_chunk_size')

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.use_map_adaptive:
        all_args.n_cms_levels, all_args.cms_mid_chunk_size = get_scale_config(
            all_args.map_name,
            default_levels=all_args.n_cms_levels,
            default_chunk_size=all_args.cms_mid_chunk_size,
        )

    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    num_agents = get_map_params(all_args.map_name)["n_agents"]
    all_args.run_dir = run_dir

    # ========== 打印项目名称 ==========
    print("\n" + "="*80)
    print("╔" + "═"*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  MAT+MSA: Multi-Scale Attention Ablation".center(78) + "║")
    print("║" + "  (Table 1 map-adaptive scale configuration)".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "═"*78 + "╝")
    print("="*80)

    # ========== 打印所有配置参数 ==========
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)

    # 基本信息
    print("\n[Basic Info]")
    print(f"  Environment:        {all_args.env_name}")
    print(f"  Map Name:           {all_args.map_name}")
    print(f"  Algorithm:          {all_args.algorithm_name}")
    print(f"  Experiment Name:    {all_args.experiment_name}")
    print(f"  Seed:               {all_args.seed}")
    print(f"  Device:             {device}")
    print(f"  Num Agents:         {num_agents}")

    # 训练参数
    print("\n[Training Parameters]")
    print(f"  Num Env Steps:      {all_args.num_env_steps}")
    print(f"  Episode Length:     {all_args.episode_length}")
    print(f"  n_rollout_threads:  {all_args.n_rollout_threads}")
    print(f"  n_training_threads: {all_args.n_training_threads if hasattr(all_args, 'n_training_threads') else 'N/A'}")
    print(f"  num_mini_batch:     {all_args.num_mini_batch if hasattr(all_args, 'num_mini_batch') else 'N/A'}")
    print(f"  ppo_epoch:          {all_args.ppo_epoch if hasattr(all_args, 'ppo_epoch') else 'N/A'}")
    print(f"  Learning Rate:      {all_args.lr if hasattr(all_args, 'lr') else 'N/A'}")
    print(f"  clip_param:         {all_args.clip_param if hasattr(all_args, 'clip_param') else 'N/A'}")

    # 网络架构
    print("\n[Network Architecture]")
    print(f"  n_block:            {all_args.n_block if hasattr(all_args, 'n_block') else 'N/A'}")
    print(f"  n_embd:             {all_args.n_embd if hasattr(all_args, 'n_embd') else 'N/A'}")
    print(f"  n_head:             {all_args.n_head if hasattr(all_args, 'n_head') else 'N/A'}")
    print(f"  encode_state:       {all_args.encode_state if hasattr(all_args, 'encode_state') else 'N/A'}")

    # CMS配置
    print("\n[CMS Configuration]")
    print(f"  n_cms_levels:       {all_args.n_cms_levels if hasattr(all_args, 'n_cms_levels') else 'N/A'}")
    print(f"  cms_mid_chunk_size: {all_args.cms_mid_chunk_size if hasattr(all_args, 'cms_mid_chunk_size') else 'N/A'}")
    print(f"  cms_dropout:        {all_args.cms_dropout if hasattr(all_args, 'cms_dropout') else 'N/A'}")

    # 其他配置
    print("\n[Other Settings]")
    print(f"  Use Wandb:          {all_args.use_wandb}")
    print(f"  Use Eval:           {all_args.use_eval}")
    print(f"  Save Interval:      {all_args.save_interval if hasattr(all_args, 'save_interval') else 'N/A'}")
    print(f"  Log Interval:       {all_args.log_interval if hasattr(all_args, 'log_interval') else 'N/A'}")

    # 运行目录
    print("\n[Directories]")
    print(f"  Run Directory:      {run_dir}")

    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
