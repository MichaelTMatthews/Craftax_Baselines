# uv run --python 3.12 --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
# "jax[cuda12]>=0.7.1",
# "flax>=0.11.1",
# "optax",
# "orbax-checkpoint",
# "flashbax",
# "chex",
# "gymnax",
# "craftax",
# "argparse",
# "mlflow",
# "pygame",
# "matplotlib",
# "imageio",
# ]
# ///

import argparse
import os
from pathlib import Path
from typing import Any, Callable, NamedTuple

import flashbax as fbx
import gymnax
import jax
import jax.numpy as jnp
import mlflow
import optax
import orbax.checkpoint
import yaml
from craftax.craftax_env import make_craftax_env_from_name
from flax import nnx
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper

from logz.batch_logging import batch_log, create_log_dict
from wrappers import (
    AutoResetEnvWrapper,
    BatchEnvWrapper,
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)


class QNetwork(nnx.Module):
    def __init__(
        self,
        din: int,
        dout: int,
        rngs: nnx.Rngs,
    ):
        self.n_actions = dout
        self.linear1 = nnx.Linear(
            din,
            120,
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            120,
            84,
            rngs=rngs,
        )
        self.linear3 = nnx.Linear(
            84,
            dout,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        x = nnx.relu(x)
        x = self.linear3(x)
        return x

    def decide(self, x: jax.Array) -> jax.Array:
        q_values = self(x)
        action = jnp.argmax(q_values, axis=-1)
        return action

    def epsilon_greedy(
        self,
        x: jax.Array,
        epsilon: float,
        rng: jax.random.PRNGKey,
    ) -> jax.Array:
        explore_key, exploit_key = jax.random.split(rng)
        action = nnx.cond(
            jax.random.uniform(explore_key, ()) < epsilon,
            lambda: jax.random.randint(exploit_key, (config["n_envs"]), 0, self.n_actions),
            lambda: self.decide(x),
        )
        return action


class TimeStep(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    # info: jnp.ndarray


def make_run(config: dict[str, Any]) -> Callable:
    def lr_schedule(update_idx: int) -> float:
        n_updates = config["training"]["n_steps"] // config["training"]["update_frequency"]
        return config["training"]["lr"] * (1.0 - (update_idx / n_updates))

    def epsilon_schedule(step: int) -> float:
        return config["training"]["epsilon_final"] + (
            config["training"]["epsilon_initial"] - config["training"]["epsilon_final"]
        ) * jnp.exp(-step / config["training"]["epsilon_decay"])

    def run(rng: jax.random.PRNGKey):
        def step_fn(run_state, _):
            def update_fn(batch, agent, target_network, optim):
                def loss_fn(
                    agent,
                    target_network,
                ):
                    q_values = agent(batch.first.obs)
                    q_value = jnp.take_along_axis(
                        q_values,
                        jnp.expand_dims(batch.first.action, axis=-1),
                        axis=-1,
                    ).squeeze(axis=-1)

                    target_next = jnp.max(target_network(batch.second.obs), axis=-1)
                    target_value = (
                        batch.first.reward
                        + jnp.logical_not(batch.first.done).astype(batch.first.reward.dtype)
                        * config["training"]["gamma"]
                        * target_next
                    )

                    return jnp.mean((q_value - target_value) ** 2)

                loss, grads = nnx.value_and_grad(loss_fn)(agent, target_network)
                optim.update(agent, grads)

                return loss

            obs, agent, target_network, optim, env_state, buffer_state, step, epsilon, key = run_state

            key, action_key, step_key, buffer_key = jax.random.split(key, 4)

            epsilon = epsilon_schedule(step)
            action = agent.epsilon_greedy(obs, epsilon=epsilon, rng=action_key)

            next_obs, env_state, reward, done, info = env.step(
                step_key,
                env_state,
                action,
                env_params,
            )

            time_step = TimeStep(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                # info=info,
            )

            buffer_state = buffer.add(buffer_state, time_step)

            nnx.cond(
                (step >= config["training"]["learning_start"])
                & buffer.can_sample(buffer_state)
                & (step % config["training"]["update_frequency"] == 0),
                lambda batch, agent, target_network, optim: update_fn(batch, agent, target_network, optim),
                lambda a, b, c, d: jnp.array(0.0),
                batch := buffer.sample(buffer_state, buffer_key).experience,
                agent,
                target_network,
                optim,
            )

            nnx.cond(
                step % config["training"]["target_update_frequency"] == 0,
                lambda q_network: nnx.update(target_network, nnx.split(q_network)[1]),
                lambda _: None,
                agent,
            )

            run_state = (
                next_obs,
                agent,
                target_network,
                optim,
                env_state,
                buffer_state,
                step + config["n_envs"],
                epsilon,
                key,
            )

            return run_state, _

        key, buffer_key, agent_key, env_key, run_key = jax.random.split(rng, 5)

        buffer = fbx.make_flat_buffer(
            max_length=config["training"]["buffer_size"],
            min_length=config["training"]["batch_size"],
            sample_batch_size=config["training"]["batch_size"],
            add_sequences=False,
            # add_batch_size=config["n_envs"],
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )
        dummy_action = jax.random.randint(buffer_key, (config["n_envs"],), 0, env.action_space(env_params).n)
        _, dummy_env_state = env.reset(buffer_key, env_params)
        dummy_obs, _, dummy_reward, dummy_done, dummy_info = env.step(
            buffer_key, dummy_env_state, dummy_action, env_params
        )
        buffer_state = buffer.init(
            TimeStep(
                obs=dummy_obs,
                action=dummy_action,
                reward=dummy_reward,
                done=dummy_done,
                # info=dummy_info,
            )
        )

        agent = QNetwork(
            din=env.observation_space(env_params).shape[0],
            dout=env.action_space(env_params).n,
            rngs=nnx.Rngs(agent_key),
        )
        target_network = nnx.clone(agent)

        tx = optax.chain(
            optax.adam(
                learning_rate=lr_schedule if config["training"]["anneal_lr"] else config["training"]["lr"],
            )
        )

        optim = nnx.Optimizer(agent, tx, wrt=nnx.Param)

        obs, env_state = env.reset(env_key, env_params)

        run_state = (
            obs,
            agent,
            target_network,
            optim,
            env_state,
            buffer_state,
            step := 0,
            epsilon := epsilon_schedule(step),
            run_key,
        )
        run_state, _ = nnx.scan(
            step_fn,
            length=config["training"]["n_steps"] // config["n_envs"],
        )(run_state, None)

        return run_state, _

    env, env_params = gymnax.make(config["env"]["id"])

    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    if config["env"].get("optimistic_resets", False):
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["n_envs"],
            reset_ratio=config["env"]["optimistic_resets"]["reset_ratio"],
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(
            env,
            num_envs=config["n_envs"],
        )

    return run


if __name__ == "__main__":
    # parse config args
    args = argparse.ArgumentParser()
    args.add_argument(
        "--config",
        type=str,
        required=True,
        help="Select from `configs/*.yaml`",
    )
    args = args.parse_args()

    with open(Path(args.config)) as file:
        config = yaml.safe_load(file)
    config["training"]["n_steps"] = int(float(config["training"]["n_steps"]))
    config["training"]["lr"] = float(config["training"]["lr"])

    # assert config conflicts

    deterministic = config.get("deterministic", True)
    if deterministic:
        os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

    # init experiment run
    config["experiment_name"] = config.get(
        "experiment_name",
        f"""CartPole DQN {config["training"]["n_steps"] // 1e6}M""",
    )
    mlflow.set_experiment(config["experiment_name"])
    mlflow.start_run()
    mlflow.log_params(config)

    # start

    key = jax.random.PRNGKey(config["seed"])
    runs_keys = jax.random.split(key, config["n_runs"])

    run = make_run(config)
    run = nnx.jit(run)
    run = nnx.vmap(run)

    run_state, _ = run(runs_keys)
