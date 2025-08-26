# uv run --python 3.12 --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
# "jax[cuda12]",
# "distrax",
# "optax",
# "flax",
# "numpy",
# "black",
# "pre-commit",
# "argparse",
# "wandb",
# "mlflow",
# "orbax-checkpoint",
# "pygame",
# "gymnax",
# "chex",
# "matplotlib",
# "imageio",
# "craftax"
# ]
# ///


# TODO: Dependencies, update jax>=0.7.1 flax>=0.11.1, rm distrax


import argparse
import os
from pathlib import Path
from typing import Any, Callable, NamedTuple

import distrax
import jax
import jax.numpy as jnp
import mlflow
import optax
import orbax.checkpoint
import yaml
from craftax.craftax_env import make_craftax_env_from_name
from flax import nnx

from logz.batch_logging import batch_log, create_log_dict
from wrappers import (
    AutoResetEnvWrapper,
    BatchEnvWrapper,
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)


class ActorCritic(nnx.Module):
    def __init__(
        self,
        din: int,
        layer_width: int,
        dout: int,
        rngs: nnx.Rngs,
        activation: str = "tanh",
    ):
        self.activation = activation
        self.linear1 = nnx.Linear(
            din,
            layer_width,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            layer_width,
            layer_width,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )
        self.linear3 = nnx.Linear(
            layer_width,
            layer_width,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )
        self.linear4 = nnx.Linear(
            layer_width,
            dout,
            kernel_init=nnx.initializers.orthogonal(0.01),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )

        self.linear1_critic = nnx.Linear(
            din,
            layer_width,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )
        self.linear2_critic = nnx.Linear(
            layer_width,
            layer_width,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )
        self.linear3_critic = nnx.Linear(
            layer_width,
            layer_width,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )
        self.linear4_critic = nnx.Linear(
            layer_width,
            1,
            kernel_init=nnx.initializers.orthogonal(1.0),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )

    def __call__(self, x):
        if self.activation == "relu":
            activation = nnx.relu
        else:
            activation = nnx.tanh

        actor_mean = self.linear1(x)
        actor_mean = activation(actor_mean)
        actor_mean = self.linear2(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = self.linear3(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = self.linear4(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = self.linear1_critic(x)
        critic = activation(critic)
        critic = self.linear2_critic(critic)
        critic = activation(critic)
        critic = self.linear3_critic(critic)
        critic = activation(critic)
        critic = self.linear4_critic(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    next_obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    info: jnp.ndarray
    #
    value: jnp.ndarray
    log_prob: jnp.ndarray


def make_run(config: dict[str, Any]) -> Callable:
    gamma = config["training"]["gamma"]
    gae_lambda = config["training"]["gae_lambda"]
    norm_advantage = config["training"].get("norm_advantage", False)
    clip_coef = config["training"]["clip_coef"]
    clip_vloss = config["training"].get("clip_vloss", False)
    vf_coef = config["training"]["vf_coef"]
    ent_coef = config["training"]["ent_coef"]

    n_batch_steps = config["training"]["n_batch_steps"]
    n_batches = config["training"]["n_steps"] // config["n_envs"] // n_batch_steps
    batch_size = config["n_envs"] * n_batch_steps

    n_epochs = config["training"]["n_epochs"]
    n_minibatches = config["training"]["n_minibatches"]

    def lr_schedule(_idx):
        return config["training"]["lr"] * (1 - (_idx // (n_minibatches * n_epochs)) / n_batches)

    def run(rng: jax.random.PRNGKey):
        def batch_step(run_state, _):
            def step(run_state, _):
                obs, model, optim, env_state, batch_idx, key = run_state

                key, action_key, step_key = jax.random.split(key, 3)

                distribution, value = model(obs)
                action = distribution.sample(seed=action_key)
                log_prob = distribution.log_prob(action)

                next_obs, env_state, reward, done, info = env.step(
                    step_key,
                    env_state,
                    action,
                    env_params,
                )

                transition = Transition(
                    obs=obs,
                    action=action,
                    next_obs=next_obs,
                    reward=reward,
                    done=done,
                    info=info,
                    #
                    value=value,
                    log_prob=log_prob,
                )

                run_state = (
                    next_obs,
                    model,
                    optim,
                    env_state,
                    batch_idx,
                    key,
                )

                return run_state, transition

            def rollout_step(carry, transition):
                next_value, next_done, prev_advantage = carry
                reward = transition.reward
                value = transition.value

                # gae advantage
                delta = reward + gamma * next_value * jnp.logical_not(next_done) - value
                advantage = delta + gamma * gae_lambda * jnp.logical_not(next_done) * prev_advantage

                return (value, transition.done, advantage), (advantage + value, advantage)

            def epoch_update(update_state, _):
                def minibatch_update(model_optim, minibatch):
                    def loss_fn(model, transition, advantages, returns):
                        distribution, new_value = model(transition.obs)
                        new_log_prob = distribution.log_prob(transition.action)
                        entropy = distribution.entropy()

                        ratio = jnp.exp(new_log_prob - transition.log_prob)

                        policy_loss = jnp.maximum(
                            -advantages * ratio,
                            -advantages * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef),
                        ).mean()

                        value_loss = jnp.where(
                            clip_vloss,
                            0.5
                            * jnp.maximum(
                                (new_value - returns) ** 2,
                                (
                                    transition.value
                                    + jnp.clip(new_value - transition.value, -clip_coef, clip_coef)
                                    - returns
                                )
                                ** 2,
                            ),
                            0.5 * ((new_value - returns) ** 2),
                        ).mean()

                        entropy_loss = entropy.mean()

                        loss = policy_loss + value_loss * vf_coef - entropy_loss * ent_coef
                        return loss, (policy_loss, value_loss, entropy_loss)

                    model, optim = model_optim
                    batch, advantages, returns = minibatch

                    loss, grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch, advantages, returns)
                    optim.update(grads)

                    return model_optim, loss

                model, optim, batch, advantages, returns, key = update_state

                key, permutation_key = jax.random.split(key, 2)

                joint = (batch, advantages, returns)  # shape: (n_steps, n_envs, ...)
                flat_joint = jax.tree.map(  # shape: (batch_size := n_steps * n_envs, ...)
                    lambda x: x.reshape((batch_size,) + x.shape[2:]),
                    joint,
                )
                permutation = jax.random.permutation(permutation_key, batch_size)
                shuffled_joint = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0),
                    flat_joint,
                )
                minibatches = jax.tree.map(  # shape: (n_minibatches, minibatch_size, ...)
                    lambda x: jnp.reshape(x, [n_minibatches, -1] + list(x.shape[1:])),
                    shuffled_joint,
                )

                _, losses = nnx.scan(
                    minibatch_update,
                    length=n_minibatches,
                )((model, optim), minibatches)

                update_state = (model, optim, batch, advantages, returns, key)
                return update_state, losses

            run_state, batch = nnx.scan(
                step,
                length=n_batch_steps,
            )(run_state, None)
            obs, model, optim, env_state, batch_idx, batch_key = run_state

            _, (returns, advantages) = nnx.scan(
                rollout_step,
                reverse=True,
                unroll=16,
            )(
                (
                    next_value := batch.value[-1],  # bootstrap the last value
                    next_done := batch.done[-1],  # bootstrap the last done
                    prev_advantage := jnp.zeros_like(batch.value[-1]),
                ),
                batch,
            )

            advantages = jnp.where(
                norm_advantage,
                (advantages - advantages.mean()) / (advantages.std() + 1e-8),
                advantages,
            )

            update_state = (
                model,
                optim,
                batch,
                advantages,
                returns,
                batch_key,
            )
            update_state, losses = nnx.scan(
                epoch_update,
                length=n_epochs,
            )(update_state, None)

            model, optim, _, _, _, _ = update_state

            run_state = (
                obs,
                model,
                optim,
                env_state,
                batch_idx + 1,
                batch_key,
            )

            # region logging

            def do_metrics():
                def metrics_callback(metric_info, batch_idx):
                    # Add NUM_REPEATS for batch logging compatibility
                    config["NUM_REPEATS"] = config["n_runs"]
                    config["DEBUG"] = True  # Add DEBUG flag for batch logging
                    config["NUM_STEPS"] = n_batch_steps  # Steps per batch, not total steps
                    config["NUM_ENVS"] = config["n_envs"]

                    to_log = create_log_dict(metric_info, config)
                    batch_log(batch_idx, to_log, config)

                metric_info = jax.tree.map(
                    lambda x: (x * batch.info["returned_episode"]).sum() / batch.info["returned_episode"].sum(),
                    batch.info,
                )
                jax.debug.callback(
                    metrics_callback,
                    metric_info,
                    batch_idx,
                )

            def do_checkpoint():
                def checkpoint_callback(batch_idx, model_state):
                    run_dir = Path(mlflow.get_artifact_uri().replace("file://", ""))
                    ckpt_dir = run_dir / "checkpoints" / f"{batch_idx:06d}"
                    checkpointer = orbax.checkpoint.StandardCheckpointer()
                    checkpointer.save(ckpt_dir, model_state)

                _, model_state = nnx.split(model)
                jax.debug.callback(
                    checkpoint_callback,
                    batch_idx,
                    model_state,
                )

            def do_snapshot():
                def snapshot_callback(batch_idx, snapshot):
                    run_dir = Path(mlflow.get_artifact_uri().replace("file://", ""))
                    snapshot_dir = run_dir / "run_state_snapshots" / f"{batch_idx:06d}"
                    checkpointer = orbax.checkpoint.StandardCheckpointer()
                    checkpointer.save(snapshot_dir, snapshot)

                _, model_state = nnx.split(model)
                optim_state = nnx.state(optim)
                snapshot = {
                    "obs": obs,
                    "model_state": model_state,
                    "optim_state": optim_state,
                    "env_state": env_state,
                    "batch_idx": batch_idx,
                    "batch_key": batch_key,
                }
                jax.debug.callback(
                    snapshot_callback,
                    batch_idx,
                    snapshot,
                )

            jax.lax.cond(
                jnp.logical_or(
                    batch_idx % config.get("metrics_every", 1) == 0,
                    batch_idx == n_batches - 1,
                ),
                do_metrics,
                lambda: None,
            )

            jax.lax.cond(
                jnp.logical_or(
                    batch_idx % config.get("checkpoint_every", 10) == 0,
                    batch_idx == n_batches - 1,
                ),
                do_checkpoint,
                lambda: None,
            )

            jax.lax.cond(
                jnp.logical_or(
                    batch_idx % config.get("snapshot_every", jnp.inf) == 0,
                    batch_idx == n_batches - 1,
                ),
                do_snapshot,
                lambda: None,
            )

            # endregion

            return run_state, _

        key, model_key, env_key, batch_key = jax.random.split(rng, 4)

        if "Symbolic" in config["env"]["id"]:
            model = ActorCritic(
                din=env.observation_space(env_params).shape[0],
                layer_width=config["agent"]["layer_size"],
                dout=env.action_space(env_params).n,
                rngs=nnx.Rngs(model_key),
            )
        else:
            raise NotImplementedError("NNX ActorCriticConv not implemented.")

        tx = optax.chain(
            optax.clip_by_global_norm(config["training"]["max_grad_norm"]),
            optax.adam(
                learning_rate=lr_schedule if config["training"]["anneal_lr"] else config["training"]["lr"],
                eps=1e-5,
            ),
        )

        optim = nnx.Optimizer(model, tx)

        obs, env_state = env.reset(env_key, env_params)

        run_state = (
            obs,
            model,
            optim,
            env_state,
            batch_idx := 0,
            batch_key,
        )
        run_state, _ = nnx.scan(
            batch_step,
            length=n_batches,
        )(run_state, None)

        return run_state, _

    env = make_craftax_env_from_name(
        config["env"]["id"],
        not config["env"]["optimistic_resets"],
    )
    env_params = env.default_params

    env = LogWrapper(env)
    if config["env"]["optimistic_resets"]:
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
        f"""Crafter PPO {config["training"]["n_steps"] // 1e6}M""",
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
