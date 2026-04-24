from functools import partial
from typing import Literal

# from smac.env import MultiAgentEnv, StarCraft2Env
from .multiagentenv import MultiAgentEnv
from .starcraft2 import StarCraft2Env

# from .gfootball import GoogleFootballEnv
from .shaping import LLEPotentialShaping
from lle import LLE, Direction
import marlenv

import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
# REGISTRY["gf"] = partial(env_fn, env=GoogleFootballEnv)


def lle_fn(**kwargs):
    match kwargs["map"]:
        case str(map_file):
            env = LLE.from_file(map_file)
        case int(level):
            env = LLE.level(level)
        case other:
            raise ValueError(f"Invalid map: {other}")

    obs_type = kwargs.get("obs_type", "layered")
    state_type = kwargs.get("state_type", "flattened")
    env = env.obs_type(obs_type).state_type(state_type).single_objective().build()
    seed = kwargs.get("seed", None)
    if seed is not None:
        seed = int(seed)
        env.seed(seed)
    time_limit = env.width * env.height // 2
    return marlenv.adapters.PymarlAdapter(env, time_limit)


def str_to_bool(value: str) -> bool:
    if value in {"true", "True", "1"}:
        return True
    if value in {"false", "False", "0"}:
        return False
    raise ValueError(f"Can not convert {value} to boolean")


def shaped_lle(
    *,
    gamma: float,
    map: str | int,
    enable_shaped_subgoals: bool | str,
    reward_value: float = 0.1,
    obs_type: Literal["layered", "flattened", "partial3x3", "partial5x5", "partial7x7", "state", "image", "perspective"] = "layered",
    state_type: Literal["layered", "flattened", "partial3x3", "partial5x5", "partial7x7", "state", "image", "perspective"] = "flattened",
    seed: int | None = None,
):
    match map:
        case str(map_file):
            env = LLE.from_file(map_file)
        case int(level):
            env = LLE.level(level)
        case other:
            raise ValueError(f"Invalid map: {other}")

    if isinstance(enable_shaped_subgoals, str):
        enable_shaped_subgoals = str_to_bool(enable_shaped_subgoals)

    env = env.obs_type(obs_type).state_type(state_type).single_objective().build()
    if seed is not None:
        seed = int(seed)
        env.seed(seed)
    time_limit = env.width * env.height // 2
    l1 = env.world.laser_sources[4, 0]
    l2 = env.world.laser_sources[6, 12]
    env = LLEPotentialShaping(
        env,
        {l1: Direction.SOUTH, l2: Direction.SOUTH},
        gamma,
        reward_value=reward_value,
        enable_extras=bool(enable_shaped_subgoals),
    )
    return marlenv.adapters.PymarlAdapter(env, time_limit)
