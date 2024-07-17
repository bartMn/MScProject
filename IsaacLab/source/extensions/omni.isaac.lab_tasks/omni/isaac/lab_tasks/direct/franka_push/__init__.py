# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Push environment.
"""

import gymnasium as gym

from . import agents
from .franka_push_env import FrankaPushEnv, FrankaPushEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Franka-Push-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.franka_push:FrankaPushEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaPushEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FrankaPushPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
