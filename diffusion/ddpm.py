from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from diffusion.unet import UNet


class DDPM(nnx.Module):
    def __init__(
        self,
        epsilon_model: UNet,
        noise_schedule: dict[str, Any],
        num_steps: int,
        device: jax.Device,
    ):
        self.epsilon_model = epsilon_model
        self.noise_schedule = noise_schedule
        self.num_steps = num_steps
        self.device = device

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> tuple[jax.Array, jax.Array]:
        batch_size = x.shape[0]

        times = jax.random.randint(
            rngs.times(), shape=(batch_size,), minval=1, maxval=self.num_steps + 1
        )
        times = jax.device_put(times, self.device)
        epsilon = jax.random.normal(rngs.noise(), shape=x.shape)

        # Note: sqrtab shape is (1, 1, 1, 1) only when times is array shape (batch_size,)
        x_t = (
            self.noise_schedule["sqrtab"][times, None, None, None] * x
            + self.noise_schedule["sqrtmab"][times, None, None, None] * epsilon
        )
        epsilon_theta = self.epsilon_model(x_t, times / self.num_steps)
        return epsilon_theta, epsilon

    def sample(
        self,
        num_images: int,
        image_dimensions: tuple[int, int],
        image_channels: int,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        image_shape = (num_images, *image_dimensions, image_channels)
        x_i = jax.random.normal(rngs.noise(), shape=image_shape).to_device(self.device)
        epsilon_model = nnx.jit(self.epsilon_model)

        for i in range(self.num_steps, 0, -1):
            epsilon = (
                jax.random.normal(rngs.noise(), shape=image_shape).to_device(self.device)
                if i > 1
                else 0
            )
            epsilon_theta = epsilon_model(
                x_i, jnp.tile(jnp.array(i / self.num_steps).to_device(self.device), (num_images, 1))
            )
            x_i = (
                self.noise_schedule["oneover_sqrta"][i]
                * (x_i - epsilon_theta * self.noise_schedule["mab_over_sqrtmab"][i])
                + self.noise_schedule["sqrt_beta_t"][i] * epsilon
            )
        return x_i


def ddpm_schedule(beta1: float, beta2: float, time_steps: int) -> dict[str, jax.Array]:
    assert beta1 < beta2 < 1.0

    beta_t = (beta2 - beta1) * jnp.arange(0, time_steps + 1, dtype=jnp.float32) / time_steps + beta1
    sqrt_beta_t = jnp.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = jnp.log(alpha_t)
    alphabar_t = jnp.exp(jnp.cumsum(log_alpha_t, axis=0))

    sqrtab = jnp.sqrt(alphabar_t)
    oneover_sqrta = 1 / jnp.sqrt(alpha_t)

    sqrtmab = jnp.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }
