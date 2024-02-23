from jax import numpy as jnp
import numpy as np
import jax
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding

mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(4, 8)[:, None, None, None, :],
                         axis_names=("data", "seq", "expert", "fsdp", "model"),)

batch_size = 4
target_length = 1024
source_length = 1024
num_heads = 32
per_head_dim = 128

def init_cpu():  # Initing on Neuron causes compiler failures.
    probs = jnp.ones((batch_size, num_heads, target_length, source_length))
    v_proj = jnp.ones((batch_size, source_length, num_heads, per_head_dim))
    return probs, v_proj

def move_to_neuron(probs, v_proj):
    probs = jax.device_put(probs)
    v_proj = jax.device_put(v_proj)
    return probs, v_proj

cpu_device = jax.devices('cpu')[0]
with jax.default_device(cpu_device):
    probs, v_proj = init_cpu()

def apply_attention(probs, v_proj):
    context = jnp.einsum("bnts,bsnh->btnh", probs, v_proj).astype(v_proj.dtype)
    return context

move_to_neuron = jax.jit(move_to_neuron, in_shardings=(NamedSharding(mesh, PartitionSpec('data', 'model', None, None)),
                                                       NamedSharding(mesh, PartitionSpec('data', None, 'model', None))))
probs, v_proj = move_to_neuron(probs, v_proj)
apply_attention = jax.jit(apply_attention, in_shardings=(NamedSharding(mesh, PartitionSpec('data', 'model', None, None)),
                                                         NamedSharding(mesh, PartitionSpec('data', None, 'model', None))))
context = apply_attention(probs, v_proj)
print(context)
