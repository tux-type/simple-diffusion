# ruff: noqa: E402
import ray
import ray.data
import logging

ray.data.DataContext.get_current().enable_progress_bars = False
logging.getLogger("ray.data").setLevel(logging.WARNING)

ray.init(num_cpus=12, num_gpus=1)

import jax
import jax.numpy as jnp
import matplotlib.image
import numpy as np
import optax
from flax import nnx
from tqdm import tqdm

from diffusion.ddpm import DDPM, ddpm_schedule
from diffusion.unet import UNet


def normalise_batch(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    data = np.array(samples["image"]).astype(np.float32)
    # Pad from 28x28 to 32x32 for easier downsampling and upsampling
    data = np.pad(
        data, pad_width=((0, 0), (2, 2), (2, 2)), mode="constant", constant_values=-1
    )
    normalised_data = ((data / 255.0) - 0.5) / 0.5
    samples["image"] = normalised_data
    return samples


def loss_fn(model, x, rngs):
    predictions, targets = model(x, rngs)
    loss = jnp.mean(optax.l2_loss(predictions=predictions, targets=targets))
    return loss, predictions


@nnx.jit
def train_step(
    model: DDPM, optimiser: nnx.Optimizer, metrics: nnx.MultiMetric, rngs: nnx.Rngs, x: jax.Array
):
    grad_fn = nnx.value_and_grad(f=loss_fn, has_aux=True)  # type: ignore
    (loss, predictions), grads = grad_fn(model, x, rngs)
    metrics.update(loss=loss)
    optimiser.update(grads)


def train_mnist(num_epochs: int):
    init_rngs = nnx.Rngs(params=0)
    training_rngs = nnx.Rngs(times=1, noise=2)
    sampling_rngs = nnx.Rngs(noise=3)

    noise_schedule = ddpm_schedule(beta1=1e-4, beta2=0.02, time_steps=1000)
    ddpm = DDPM(
        epsilon_model=UNet(1, 1, num_features=128, rngs=init_rngs),
        noise_schedule=noise_schedule,
        num_steps=1000,
        device=jax.devices("gpu")[0],
    )

    dataset = (
        ray.data.read_images("data/mnist", override_num_blocks=12)
        .map_batches(normalise_batch)
        .materialize()
    )

    optimiser = nnx.Optimizer(ddpm, optax.adam(2e-4))

    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

    metrics_history = {"train_loss": []}
    for i in range(num_epochs):
        dataset = dataset.random_shuffle().materialize()
        progress_bar = tqdm(
            dataset.iter_batches(
                prefetch_batches=1, batch_size=128, batch_format="numpy", drop_last=True
            )
        )
        for batch in progress_bar:
            x = jnp.array(batch["image"])[:, :, :, None]
            train_step(ddpm, optimiser, metrics, training_rngs, x)

            for metric, value in metrics.compute().items():
                metrics_history[f"train_{metric}"].append(value)
            loss = metrics_history["train_loss"][-1]
            progress_bar.set_description(f"loss: {loss:.4f}")

        if i % 5 == 0 or i == (num_epochs - 1):
            print(f"Sampling Epoch: {i}")
            x_0 = ddpm.sample(
                num_images=16,
                image_dimensions=(32, 32),
                image_channels=1,
                rngs=sampling_rngs,
            )
            x_0 = ((x_0 + 1) * 127.5).clip(0, 255)
            # Squeeze since grayscale
            matplotlib.image.imsave(
                f"imgs/numbers/image_{i}.png",
                jnp.concatenate(x_0, axis=1).squeeze(axis=2).astype(jnp.uint8),
                cmap="binary",
            )


if __name__ == "__main__":
    train_mnist(num_epochs=10)
