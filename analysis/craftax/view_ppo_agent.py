import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from craftax.environment_base.wrappers import AutoResetEnvWrapper
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from craftax.craftax.constants import Action, Achievement
from models.actor_critic import ActorCriticConv, ActorCritic
from craftax.craftax.play_craftax import CraftaxRenderer


def main(path):
    with open(os.path.join(path, "config.yaml")) as f:
        raw_config = yaml.load(f, Loader=yaml.Loader)

        config = {}
        for key, value in raw_config.items():
            if isinstance(value, dict) and "value" in value:
                config[key] = value["value"]

    config["NUM_ENVS"] = 1

    orbax_checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpoint_manager = CheckpointManager(
        os.path.join(path, "policies"), orbax_checkpointer, options
    )

    if config["ENV_NAME"] == "Craftax-Symbolic-v1":
        from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv

        env = CraftaxSymbolicEnv(CraftaxSymbolicEnv.default_static_params())
        network = ActorCritic(len(Action), config["LAYER_SIZE"])
    elif config["ENV_NAME"] == "Craftax-Pixels-v1":
        from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv

        env = CraftaxPixelsEnv(CraftaxPixelsEnv.default_static_params())
        network = ActorCriticConv(len(Action), config["LAYER_SIZE"])
    else:
        raise ValueError(f"Unknown env: {config['ENV_NAME']}")

    env = AutoResetEnvWrapper(env)
    env_params = env.default_params

    init_x = jnp.zeros((config["NUM_ENVS"], *env.observation_space(env_params).shape))

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng, __rng = jax.random.split(rng, 3)
    network_params = network.init(_rng, init_x)

    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    train_state = checkpoint_manager.restore(config["TOTAL_TIMESTEPS"], items=train_state)

    obs, env_state = env.reset(key=__rng)
    done = 0

    renderer = CraftaxRenderer(env, env_params, pixel_render_size=1)

    while not renderer.is_quit_requested():
        done = np.array([done], dtype=bool)
        obs = jnp.expand_dims(obs, axis=0)

        pi, value = network.apply(train_state.params, obs)
        rng, _rng = jax.random.split(rng)
        action = pi.sample(seed=_rng)[0]
        # action = jnp.argmax(pi.probs[0, 0])

        if action is not None:
            rng, _rng = jax.random.split(rng)
            old_achievements = env_state.achievements
            obs, env_state, reward, done, info = env.step(
                _rng, env_state, action, env_params
            )
            new_achievements = env_state.achievements
            print_new_achievements(old_achievements, new_achievements)
            if done:
                print("\n")
        renderer.render(env_state)


def print_new_achievements(old_achievements, new_achievements):
    for i in range(len(old_achievements)):
        if old_achievements[i] == 0 and new_achievements[i] == 1:
            print(f"{Achievement(i).name} ({new_achievements.sum()}/{22})")


if __name__ == "__main__":
    checkpoint = (
        "/home/mans4835/PycharmProjects/Craftax_Baselines/wandb/run-20240329_115225-dlk3gdfi/files"
    )

    debug = False

    if debug:
        with jax.disable_jit():
            main(checkpoint)
    else:
        main(checkpoint)
