import jax
import jax.numpy as jnp
from flax import nnx


class SirenEmbedding(nnx.Module):
    def __init__(self, out_features: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(1, out_features, use_bias=False, rngs=rngs)
        self.linear2 = nnx.Linear(out_features, out_features, rngs=rngs)

    def __call__(self, times: jax.Array) -> jax.Array:
        out = times.reshape(-1, 1)
        out = jnp.sin(self.linear1(out))
        out = self.linear2(out)
        return out


class ConvBlock(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        is_residual: bool = False,
    ):
        self.is_residual = is_residual
        self.conv1 = nnx.Sequential(
            nnx.Conv(
                in_features,
                out_features,
                kernel_size=(3, 3),
                strides=1,
                padding=1,
                rngs=rngs,
            ),
            nnx.GroupNorm(num_features=out_features, num_groups=8, rngs=rngs),
        )
        self.conv2 = nnx.Sequential(
            nnx.Conv(
                out_features,
                out_features,
                kernel_size=(3, 3),
                strides=1,
                padding=1,
                rngs=rngs,
            ),
            nnx.GroupNorm(num_features=out_features, num_groups=8, rngs=rngs),
        )
        self.conv3 = nnx.Sequential(
            nnx.Conv(
                out_features,
                out_features,
                kernel_size=(3, 3),
                strides=1,
                padding=1,
                rngs=rngs,
            ),
            nnx.GroupNorm(num_features=out_features, num_groups=8, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # TODO: Check that this rearrange did not break anything
        residual = self.conv1(x)
        # Why not swish?
        residual = nnx.relu(residual)
        out = self.conv2(residual)
        out = nnx.relu(out)
        out = self.conv3(out)
        out = nnx.relu(out)
        if self.is_residual:
            out = residual + out
            return out / 1.414
        else:
            return out


class DownBlock(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        self.conv = ConvBlock(in_features, out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        out = self.conv(x)
        out = nnx.max_pool(out, window_shape=(2, 2), strides=(2, 2))
        return out


class UpBlock(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        self.conv = nnx.Sequential(
            # Due to differences with pytorch potentially replace with resize + regular conv
            nnx.ConvTranspose(
                in_features,
                out_features,
                kernel_size=(2, 2),
                strides=(2, 2),
                padding="SAME",
                rngs=rngs,
            ),
            ConvBlock(out_features, out_features, rngs=rngs),
            ConvBlock(out_features, out_features, rngs=rngs),
        )

    def __call__(self, x: jax.Array, skip: jax.Array) -> jax.Array:
        out = jnp.concatenate((x, skip), axis=3)
        out = self.conv(out)
        return out


class UNet(nnx.Module):
    def __init__(
        self, in_features: int, out_features: int, rngs: nnx.Rngs, num_features: int = 256
    ):
        self.in_features = in_features
        self.out_features = out_features

        self.num_features = num_features

        self.image_projection = ConvBlock(in_features, num_features, is_residual=True, rngs=rngs)
        self.time_embedding = SirenEmbedding(2 * num_features, rngs=rngs)

        self.down1 = DownBlock(num_features, num_features, rngs=rngs)
        self.down2 = DownBlock(num_features, 2 * num_features, rngs=rngs)
        self.down3 = DownBlock(2 * num_features, 2 * num_features, rngs=rngs)

        self.up0 = nnx.Sequential(
            nnx.ConvTranspose(
                2 * num_features, 2 * num_features, kernel_size=(4, 4), strides=4, rngs=rngs
            ),
            nnx.GroupNorm(num_features=2 * num_features, num_groups=8, rngs=rngs),
        )

        self.up1 = UpBlock(4 * num_features, 2 * num_features, rngs=rngs)
        self.up2 = UpBlock(4 * num_features, num_features, rngs=rngs)
        self.up3 = UpBlock(2 * num_features, num_features, rngs=rngs)

        self.feature_aggregation = nnx.Conv(
            2 * num_features, self.out_features, kernel_size=(3, 3), strides=1, padding=1, rngs=rngs
        )

    def __call__(self, x: jax.Array, times: jax.Array):
        x = self.image_projection(x)

        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        # TODO: Check if correct; figure out why the avg_pool here & thro full meaning
        # This reduces from (4 x 4) to vector (1 x 1)
        thro = nnx.relu(nnx.avg_pool(down3, window_shape=(4, 4), strides=(4, 4)))
        time_embedding = self.time_embedding(times).reshape(-1, 1, 1, self.num_features * 2)
        # From vector to (4 x 4)
        thro = nnx.relu(self.up0(thro + time_embedding))

        up1 = self.up1(thro, down3) + time_embedding
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)

        out = self.feature_aggregation(jnp.concatenate((up3, x), axis=3))

        return out
