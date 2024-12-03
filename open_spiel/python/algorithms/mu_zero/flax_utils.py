from typing import Callable

import flax
import jax
import optax
import chex

Optimizer = Callable[[chex.ArrayTree, chex.ArrayTree], chex.ArrayTree]  # (params, grads) -> params


def optax_optimizer(
    params: chex.ArrayTree,
    init_and_update: optax.GradientTransformation) -> Optimizer:
  """Creates a parameterized function that represents an optimizer."""
  init_fn, update_fn = init_and_update

  @chex.dataclass
  class OptaxOptimizer:
    """A jax-friendly representation of an optimizer state with the update."""
    state: chex.Array

    def __call__(self, params: chex.ArrayTree, grads: chex.ArrayTree) -> chex.ArrayTree:
      updates, self.state = update_fn(grads, self.state, params)  # pytype: disable=annotation-type-mismatch  # numpy-scalars
      return optax.apply_updates(params, updates)

  return OptaxOptimizer(state=init_fn(params))


def init_params_optimizer(
  network,
  rng_key: chex.PRNGKey,
  init_input,
  optimizer_init: optax.GradientTransformation = optax.chain(optax.adamw(1e-3), optax.clip(100)),
):
  params = network.init(rng_key, init_input)
  optimizer = optax_optimizer(params, optimizer_init)
  return params, optimizer
  
def init_network_with_optimizer(
  network_class,
  rng_key: chex.PRNGKey,
  init_input,
  optimizer_init: optax.GradientTransformation = optax.chain(optax.adamw(1e-3), optax.clip(100)),
  network_args: tuple = (),
):
  network = network_class(*network_args)
  params, optimizer = init_params_optimizer(network, rng_key, init_input, optimizer_init) 
  return network, params, optimizer
