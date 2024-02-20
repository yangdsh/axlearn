import contextlib

# pylint: disable=too-many-lines,duplicate-code,no-self-use

import jax
import numpy as np
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding
from axlearn.common import attention, test_utils, utils
from axlearn.common.attention import (
    ParallelTransformerLayer,
    TransformerLayer, scaled_hidden_dim, TransformerFeedForwardLayer, MultiheadAttention, FusedQKVLinear,
)
from axlearn.common.layers import RMSNorm, set_bias_recursively
from axlearn.common.module import functional as F
from axlearn.common.test_utils import NeuronTestCase, assert_allclose, dummy_segments_positions
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

def backup(self):
    self.assertEqual(
        {
            "feed_forward": {
                "dropout1": {},
                "dropout2": {},
                "linear1": {"weight": (16, 64)},
                "linear2": {"weight": (64, 16)},
                "stochastic_depth": {},
            },
            "norm": {"scale": (16,)},
            "self_attention": {
                "dropout": {},
                "i_proj": {
                    "k_proj": {"weight": (16, 4, 4)},
                    "q_proj": {"weight": (16, 4, 4)},
                    "v_proj": {"weight": (16, 4, 4)},
                },
                "o_proj": {"weight": (16, 4, 4)},
                "scale_key": {},
                "scale_query": {},
            },
        },
        #utils.shapes(layer_params),
        utils.shapes(''),
    )
class TransformerTest(NeuronTestCase):
    """Tests ParallelTransformerLayer."""

    def test_with_golden_value(self):
        """A test of TransformerLayer by comparing results to a golden value."""
        mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(4, 8)[:, None, None, None, :],
                                 axis_names=("data", "seq", "expert", "fsdp", "model"),)
        with mesh:
            model_dim = 256
            num_heads = 8
            cfg = TransformerLayer.default_config().set(name="test", input_dim=model_dim)
            #print(cfg)
            cfg.feed_forward.set(hidden_dim=scaled_hidden_dim(4))
            cfg.self_attention.attention.set(num_heads=num_heads)
            cfg.self_attention.attention.input_linear = FusedQKVLinear.default_config()
            cfg.self_attention.norm = RMSNorm.default_config()
            cfg.feed_forward.norm = RMSNorm.default_config()
            set_bias_recursively(cfg, bias=False)
            set_double_shard_weights_config(
                cfg,
                batch_axis_names='data',
                fsdp_axis_names='fsdp',
                tp_axis_names='model',
                seq_axis_names='model',
            )
            print(cfg)
            layer: TransformerLayer = cfg.instantiate(parent=None)
            print(layer)

            layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))

            batch_size, tgt_len = 4, 256
            rng = np.random.default_rng(seed=123)
            target = rng.random([batch_size, tgt_len, model_dim], dtype=np.float32)
            mask = attention.make_causal_mask(tgt_len)
            mask = jnp.tile(mask[None, None, :, :], (batch_size, num_heads, 1, 1))
            input_tensor = jnp.asarray(target)
            input_tensor = jax.device_put(input_tensor, NamedSharding(mesh, PartitionSpec('data', 'model', None)))
            @jax.jit
            def run():
                layer_outputs, _ = F(
                    layer,
                    inputs=dict(data=input_tensor, self_attention_logit_biases=mask),
                    state=layer_params,
                    is_training=True,
                    prng_key=jax.random.PRNGKey(0),
                )
                return layer_outputs
            layer_outputs = run()
            self.assertEqual(target.shape, layer_outputs.data.shape)



def set_double_shard_weights_config(
        cfg: Union[TransformerLayer.Config, Sequence[TransformerLayer.Config]],
        *,
        batch_axis_names: Union[str, Sequence[str]] = ("data", "fsdp"),
        fsdp_axis_names: Union[str, Sequence[str]] = "fsdp",
        tp_axis_names: Union[str, Sequence[str]] = "model",
        seq_axis_names: Union[str, Sequence[str]] = "seq",
):
    """Sets `cfg` to shard FFN and attention weights over both fsdp and tp axes.

    Args:
        cfg: (A sequence of) Transformer layer config to apply sharding spec to.
        batch_axis_names: Axis name(s) over which we shard the batch dimension of output tensors.
        fsdp_axis_names: Axis name(s) over which we shard fully-sharded-data-parallel tensors.
        tp_axis_names: Axis name(s) over which we shard tensor-parallel tensors.
        seq_axis_names: Axis name(s) over which we shard sequence-parallel tensors.
    """

    # pytype: disable=attribute-error
    def set_attn_partition_specs(attn_layer: MultiheadAttention.Config):
        # Shard weights.
        input_linear_cfg = attn_layer.input_linear
        if hasattr(input_linear_cfg, "input_linear"):
            input_linear_cfg = input_linear_cfg.input_linear
        input_linear_cfg.layer.param_partition_spec = (fsdp_axis_names, tp_axis_names, None)
        attn_layer.output_linear.param_partition_spec = (fsdp_axis_names, tp_axis_names, None)
        #attn_layer.output_linear.output_partition_spec = (batch_axis_names, seq_axis_names, None)

    def set_ffn_partition_specs(ff_layer: TransformerFeedForwardLayer.Config):
        # Shard weights.
        ff_layer.linear1.param_partition_spec = (fsdp_axis_names, tp_axis_names)
        ff_layer.linear2.param_partition_spec = (tp_axis_names, fsdp_axis_names)
        # Encourage the right activation sharding.
        ff_layer.linear1.output_partition_spec = (batch_axis_names, None, tp_axis_names)
        ff_layer.linear2.output_partition_spec = (batch_axis_names, seq_axis_names, None)

    if not isinstance(cfg, Sequence):
        cfg = [cfg]

    for layer_cfg in cfg:
        set_attn_partition_specs(layer_cfg.self_attention.attention)
        if layer_cfg.cross_attention is not None:
            set_attn_partition_specs(layer_cfg.cross_attention.attention)
        if isinstance(layer_cfg.feed_forward, TransformerFeedForwardLayer.Config):
            set_ffn_partition_specs(layer_cfg.feed_forward)