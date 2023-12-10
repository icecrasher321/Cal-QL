import jax
import jax.numpy as jnp
from flax import linen as nn
from .jax_utils import batch_to_jax
from .replay_buffer import subsample_batch

class Autoencoder(nn.Module):
    input_dim: int
    encoding_dim: int = 16  # This can be adjusted
    noise_stddev: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic=True):
        encoded = nn.Dense(self.encoding_dim)(x)
        encoded = nn.relu(encoded)

        if not deterministic:
            noise = self.noise_stddev * jax.random.normal(jax.random.PRNGKey(0), encoded.shape)
            encoded += noise

        decoded = nn.Dense(self.input_dim)(encoded)
        return decoded

    def train_step(self, optimizer, batch, deterministic=False):
        def loss_fn(params):
            reconstructed = self.apply(params, batch, deterministic=deterministic)
            loss = jnp.mean(jnp.square(reconstructed - batch))  # Mean Squared Error
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, gradients = grad_fn(optimizer.target)
        optimizer = optimizer.apply_gradient(gradients)
        return optimizer, loss
    
    def batch_data(self, data, batch_size=512, shuffle=True):
        if shuffle:
            shuffled_indices = jax.random.permutation(jax.random.PRNGKey(0), data.shape[0])
            print(len(shuffled_indices))
            data = data[shuffled_indices]

        num_batches = jnp.ceil(data.shape[0] / batch_size).astype(int)

        batches = [data[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

        return batches

    def train_loop(self, optimizer, dataset, num_epochs):
        batches = self.batch_data(dataset)
        for epoch in range(num_epochs):
            print("EPOCH=", epoch)
            for curr_batch in batches:
                optimizer, loss = self.train_step(optimizer, curr_batch, deterministic=False)
                
        return optimizer, loss

