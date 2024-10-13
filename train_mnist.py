import jax
import jax.numpy as jnp
import matplotlib.image
import numpy as np
import optax
from datasets import Dataset, load_dataset
from flax import nnx
from tqdm import tqdm

from diffusion.ddpm import DDPM, ddpm_schedule
from diffusion.unet import UNet


def normalise_batch(samples):
    samples = samples.copy()
    data = np.array(samples["image"]).astype(np.float32)

    mean_per_sample = np.mean(data, axis=(1, 2), keepdims=True)
    std_per_sample = np.std(data, axis=(1, 2), keepdims=True)

    normalised_data = (data - mean_per_sample) / std_per_sample

    desired_mean = 0.5
    desired_std = 1

    scaled_data = (normalised_data * desired_std) + desired_mean
    # Pad from 28x28 to 32x32 for easier downsampling and upsampling
    scaled_data = np.pad(
        scaled_data, pad_width=((0, 0), (2, 2), (2, 2)), mode="constant", constant_values=0
    )
    samples["image"] = scaled_data
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
    # Set seed?

    init_rngs = nnx.Rngs(params=0)
    training_rngs = nnx.Rngs(times=1, noise=2)
    sampling_rngs = nnx.Rngs(noise=3)

    noise_schedule = ddpm_schedule(beta1=1e-4, beta2=0.02, time_steps=1000)
    ddpm = DDPM(
        epsilon_model=UNet(1, 1, num_features=128, rngs=init_rngs),
        noise_schedule=noise_schedule,
        num_steps=1000,
    )

    dataset: Dataset = load_dataset("mnist", keep_in_memory=True, split="train").with_format("jax")  # type: ignore
    normalised_dataset = dataset.map(normalise_batch, batched=True)

    optimiser = nnx.Optimizer(ddpm, optax.adam(2e-4))

    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

    metrics_history = {"train_loss": []}
    for i in range(num_epochs):
        normalised_dataset.shuffle()
        progress_bar = tqdm(normalised_dataset.iter(batch_size=128, drop_last_batch=True))
        for batch in progress_bar:
            # for step, batch in enumerate(normalised_dataset.iter(batch_size=128, drop_last_batch=True)):
            x = batch["image"][:, :, :, None]  # type: ignore
            train_step(ddpm, optimiser, metrics, training_rngs, x)

            for metric, value in metrics.compute().items():
                metrics_history[f"train_{metric}"].append(value)
            # print(f"[train] step: {step}, " f"loss: {metrics_history['train_loss'][-1]}, ")
            loss = metrics_history["train_loss"][-1]
            progress_bar.set_description(f"loss: {loss:.4f}")

        x_0 = ddpm.sample(
            num_images=16,
            image_dimensions=(32, 32),
            image_channels=1,
            rngs=sampling_rngs,
            device=jax.devices("gpu")[0],
        )
        # Squeeze since grayscale
        matplotlib.image.imsave(
            f"imgs/image_{i}.png",
            jnp.concatenate(x_0, axis=1).squeeze(axis=2).astype(jnp.uint8),
            cmap="binary",
        )


if __name__ == "__main__":
    train_mnist(num_epochs=10)
