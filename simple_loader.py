""""""



""""""


class Args20b:
    vocab_size = 25216
    hidden_size = 6144
    num_attention_heads = 64
    rotary_pct = 0.25
    rotary_emb_base = 10000
    layernorm_epsilon = 1e-5
    num_layers = 44


class ArgsDummy:
    vocab_size = 50432
    hidden_size = 64
    num_attention_heads = 4


    rotary_pct = 0.25
    rotary_emb_base = 10000
    layernorm_epsilon = 1e-5
    num_layers = 2
import torch


class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000, device=None):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # [sx, 1 (b * np), hn]
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


# @torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = cos[offset:q.shape[0] + offset, ...], sin[offset:q.shape[0] + offset, ...]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

  
  import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
import math




class NeoX20BModel(nn.Module):
    def __init__(self, args, use_cache=False, device=None):
        super().__init__()
        self.use_cache = use_cache
        self.embed_in = nn.Embedding(args.vocab_size, args.hidden_size, device=device)
        self.layer_list = nn.ModuleList([])
        for layer_i in range(args.num_layers):
            self.layer_list.append(TransformerLayer(args, use_cache, device=device))
        self.final_layer_norm = nn.LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            device=device,
        )
        self.logits_out = nn.Linear(
            args.hidden_size,
            args.vocab_size,
            bias=False,
            device=device,
        )

    def forward(self, x, attention_mask, layer_past=None):
        if layer_past is None:
            layer_past = [None] * len(self.layer_list)
        kv_cache_list = []
        hidden_states = self.embed_in(x)
        hidden_states = self.pre_transformer_transpose(hidden_states)
        for layer_i, layer in enumerate(self.layer_list):
            hidden_states, kv_cache = layer(hidden_states, attention_mask, layer_past=layer_past[layer_i])
            kv_cache_list.append(kv_cache)
        hidden_states = self.post_transformer_transpose(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.logits_out(hidden_states)
        if self.use_cache:
            return logits, kv_cache_list
        else:
            return logits

    @classmethod
    def pre_transformer_transpose(cls, x):
        return x.transpose(0, 1).contiguous()

    @classmethod
    def post_transformer_transpose(cls, x):
        return x.transpose(0, 1).contiguous()


class TransformerLayer(nn.Module):
    def __init__(self, args, use_cache, device=None):
        super().__init__()
        self.use_cache = use_cache
        self.input_layernorm = nn.LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            device=device,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            device=device,
        )
        self.attention = SelfAttention(args, self.use_cache, device=device)
        self.mlp = MLP(args)

    def forward(self, x, attention_mask, layer_past=None):
        residual = x
        layernorm_output = self.input_layernorm(x)
        attention_output, attention_bias, kv_cache = self.attention(
            layernorm_output,
            attention_mask,
            layer_past=layer_past,
        )
        attention_output = attention_output + attention_bias.expand_as(attention_output)
        mlp_output, mlp_bias = self.mlp(self.post_attention_layernorm(x))
        output = mlp_output + mlp_bias.expand_as(mlp_output) + attention_output
        output = residual + output
        return output, kv_cache


class SelfAttention(nn.Module):
    def __init__(self, args, use_cache=False, device=None):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.use_cache = use_cache
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size_per_attention_head = args.hidden_size // args.num_attention_heads
        self.rotary_ndims = int(self.hidden_size_per_attention_head * args.rotary_pct)
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims,
            base=args.rotary_emb_base,
            device=device,
        )
        self.query_key_value = nn.Linear(
            args.hidden_size,
            (args.hidden_size//2)*3,
            device=device,
        )
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        self.dense = LinearSkipAddBias(
            args.hidden_size//2,
            args.hidden_size,
            device=device,
        )

    def forward(self, hidden_states, attention_mask, layer_past=None):
        has_layer_past = layer_past is not None and layer_past.numel() > 0

        # Compute QKV
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        qkv = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_qkv_shape = qkv.size()[:-1] + (
            self.num_attention_heads,
            3 * self.hidden_size_per_attention_head,
        )
        qkv = qkv.view(*new_qkv_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        query_layer = qkv[..., :self.hidden_size_per_attention_head]
        key_layer = qkv[..., self.hidden_size_per_attention_head: 2 * self.hidden_size_per_attention_head]
        value_layer = qkv[..., 2 * self.hidden_size_per_attention_head:]

        # Compute rotary embeddings
        query_rot, query_pass = (
            query_layer[..., : self.rotary_ndims],
            query_layer[..., self.rotary_ndims:],
        )
        key_rot, key_pass = (
            key_layer[..., : self.rotary_ndims],
            key_layer[..., self.rotary_ndims:],
        )
        seq_len = key_layer.shape[0]
        offset = 0
        if has_layer_past:
            offset = layer_past[0].shape[0]
            seq_len += offset
        cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
        query_layer, key_layer = rotary.apply_rotary_pos_emb(
            query_rot, key_rot, cos, sin, offset=offset,
        )
        query_layer = torch.cat((query_layer, query_pass), dim=-1)
        key_layer = torch.cat((key_layer, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=0)
        if self.use_cache:
            kv_cache = torch.stack((key_layer, value_layer))
        else:
            kv_cache = None

        # Compute attention
        # noinspection PyTypeChecker
        context_layer = self.attention(
            query_layer, key_layer, value_layer, attention_mask
        )

        # Reshape outputs
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.dense(context_layer)

        return output, bias, kv_cache

    def attention(self, query_layer, key_layer, value_layer, attention_mask):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=query_layer.device,
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if self.use_cache:
            attention_mask = attention_mask[
                             ...,
                             :attention_scores.size(3),
                             :attention_scores.size(3),
                             ]

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        masked_scores = attention_mask_func(attention_scores,
                                            attention_mask) if attention_mask is not None else attention_scores
        attention_probs = torch.nn.Softmax(dim=-1)(masked_scores)

        #         # This is actually dropping out entire tokens to attend to, which might
        #         # seem a bit unusual, but is taken from the original Transformer paper.
        #         attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer


class MLP(nn.Module):
    def __init__(self, args, device=None):
        super().__init__()
        ff_dim = 2 * args.hidden_size
        self.dense_h_to_4h = LinearSkipAddBias(
            args.hidden_size,
            ff_dim,
            device=device,
        )
        self.dense_4h_to_h = LinearSkipAddBias(
            ff_dim,
            args.hidden_size,
            device=device,
        )

    def forward(self, hidden_states):
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = bias_gelu_impl(
            intermediate_parallel,
            bias_parallel,
        )
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class LinearSkipAddBias(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device))
        self.bias = nn.Parameter(torch.empty(out_features, device=device))

    def forward(self, x: Tensor):
        return F.linear(x, self.weight), self.bias


# noinspection PyAbstractClass
class GeLUFunction(torch.autograd.Function):
    # noinspection PyMethodOverriding
    @staticmethod
    # bias is an optional argument
    def forward(ctx, inputs, bias):
        ctx.save_for_backward(inputs, bias)
        return bias_gelu(bias, inputs)

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx, grad_output):
        inputs, bias = ctx.saved_tensors
        tmp = bias_gelu_back(grad_output, bias, inputs)
        return tmp, tmp


bias_gelu_impl = GeLUFunction.apply


def generate_mask(seq_len):
    return torch.tril(torch.ones((1, 1, seq_len, seq_len))) < 0.5


def attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(ltor_mask, -10000.0)
    return attention_scores


# @torch.jit.script
def bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
# @torch.jit.script
def bias_gelu_back(g, bias, y):
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * (
            (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    return ff * g
  
  
  
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
import math




class NeoX20BModel(nn.Module):
    def __init__(self, args, use_cache=False, device=None):
        super().__init__()
        self.use_cache = use_cache
        self.embed_in = nn.Embedding(args.vocab_size, args.hidden_size, device=device)
        self.layer_list = nn.ModuleList([])
        for layer_i in range(args.num_layers):
            self.layer_list.append(TransformerLayer(args, use_cache, device=device))
        self.final_layer_norm = nn.LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            device=device,
        )
        self.logits_out = nn.Linear(
            args.hidden_size,
            args.vocab_size,
            bias=False,
            device=device,
        )

    def forward(self, x, attention_mask, layer_past=None):
        if layer_past is None:
            layer_past = [None] * len(self.layer_list)
        kv_cache_list = []
        hidden_states = self.embed_in(x)
        hidden_states = self.pre_transformer_transpose(hidden_states)
        for layer_i, layer in enumerate(self.layer_list):
            hidden_states, kv_cache = layer(hidden_states, attention_mask, layer_past=layer_past[layer_i])
            kv_cache_list.append(kv_cache)
        hidden_states = self.post_transformer_transpose(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.logits_out(hidden_states)
        if self.use_cache:
            return logits, kv_cache_list
        else:
            return logits

    @classmethod
    def pre_transformer_transpose(cls, x):
        return x.transpose(0, 1).contiguous()

    @classmethod
    def post_transformer_transpose(cls, x):
        return x.transpose(0, 1).contiguous()


class TransformerLayer(nn.Module):
    def __init__(self, args, use_cache, device=None):
        super().__init__()
        self.use_cache = use_cache
        self.input_layernorm = nn.LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            device=device,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            device=device,
        )
        self.attention = SelfAttention(args, self.use_cache, device=device)
        self.mlp = MLP(args)

    def forward(self, x, attention_mask, layer_past=None):
        residual = x
        layernorm_output = self.input_layernorm(x)
        attention_output, attention_bias, kv_cache = self.attention(
            layernorm_output,
            attention_mask,
            layer_past=layer_past,
        )
        attention_output = attention_output + attention_bias.expand_as(attention_output)
        mlp_output, mlp_bias = self.mlp(self.post_attention_layernorm(x))
        output = mlp_output + mlp_bias.expand_as(mlp_output) + attention_output
        output = residual + output
        return output, kv_cache


class SelfAttention(nn.Module):
    def __init__(self, args, use_cache=False, device=None):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.use_cache = use_cache
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size_per_attention_head = args.hidden_size // args.num_attention_heads
        self.rotary_ndims = int(self.hidden_size_per_attention_head * args.rotary_pct)
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims,
            base=args.rotary_emb_base,
            device=device,
        )
        self.query_key_value = nn.Linear(
            args.hidden_size,
            (args.hidden_size//2)*3,
            device=device,
        )
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        self.dense = LinearSkipAddBias(
            args.hidden_size//2,
            args.hidden_size,
            device=device,
        )

    def forward(self, hidden_states, attention_mask, layer_past=None):
        has_layer_past = layer_past is not None and layer_past.numel() > 0

        # Compute QKV
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        qkv = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_qkv_shape = qkv.size()[:-1] + (
            self.num_attention_heads,
            3 * self.hidden_size_per_attention_head,
        )
        qkv = qkv.view(*new_qkv_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        query_layer = qkv[..., :self.hidden_size_per_attention_head]
        key_layer = qkv[..., self.hidden_size_per_attention_head: 2 * self.hidden_size_per_attention_head]
        value_layer = qkv[..., 2 * self.hidden_size_per_attention_head:]

        # Compute rotary embeddings
        query_rot, query_pass = (
            query_layer[..., : self.rotary_ndims],
            query_layer[..., self.rotary_ndims:],
        )
        key_rot, key_pass = (
            key_layer[..., : self.rotary_ndims],
            key_layer[..., self.rotary_ndims:],
        )
        seq_len = key_layer.shape[0]
        offset = 0
        if has_layer_past:
            offset = layer_past[0].shape[0]
            seq_len += offset
        cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
        query_layer, key_layer = rotary.apply_rotary_pos_emb(
            query_rot, key_rot, cos, sin, offset=offset,
        )
        query_layer = torch.cat((query_layer, query_pass), dim=-1)
        key_layer = torch.cat((key_layer, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=0)
        if self.use_cache:
            kv_cache = torch.stack((key_layer, value_layer))
        else:
            kv_cache = None

        # Compute attention
        # noinspection PyTypeChecker
        context_layer = self.attention(
            query_layer, key_layer, value_layer, attention_mask
        )

        # Reshape outputs
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.dense(context_layer)

        return output, bias, kv_cache

    def attention(self, query_layer, key_layer, value_layer, attention_mask):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=query_layer.device,
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if self.use_cache:
            attention_mask = attention_mask[
                             ...,
                             :attention_scores.size(3),
                             :attention_scores.size(3),
                             ]

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        masked_scores = attention_mask_func(attention_scores,
                                            attention_mask) if attention_mask is not None else attention_scores
        attention_probs = torch.nn.Softmax(dim=-1)(masked_scores)

        #         # This is actually dropping out entire tokens to attend to, which might
        #         # seem a bit unusual, but is taken from the original Transformer paper.
        #         attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer


class MLP(nn.Module):
    def __init__(self, args, device=None):
        super().__init__()
        ff_dim = 2 * args.hidden_size
        self.dense_h_to_4h = LinearSkipAddBias(
            args.hidden_size,
            ff_dim,
            device=device,
        )
        self.dense_4h_to_h = LinearSkipAddBias(
            ff_dim,
            args.hidden_size,
            device=device,
        )

    def forward(self, hidden_states):
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = bias_gelu_impl(
            intermediate_parallel,
            bias_parallel,
        )
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class LinearSkipAddBias(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device))
        self.bias = nn.Parameter(torch.empty(out_features, device=device))

    def forward(self, x: Tensor):
        return F.linear(x, self.weight), self.bias


# noinspection PyAbstractClass
class GeLUFunction(torch.autograd.Function):
    # noinspection PyMethodOverriding
    @staticmethod
    # bias is an optional argument
    def forward(ctx, inputs, bias):
        ctx.save_for_backward(inputs, bias)
        return bias_gelu(bias, inputs)

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx, grad_output):
        inputs, bias = ctx.saved_tensors
        tmp = bias_gelu_back(grad_output, bias, inputs)
        return tmp, tmp


bias_gelu_impl = GeLUFunction.apply


def generate_mask(seq_len):
    return torch.tril(torch.ones((1, 1, seq_len, seq_len))) < 0.5


def attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(ltor_mask, -10000.0)
    return attention_scores


# @torch.jit.script
def bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
# @torch.jit.script
def bias_gelu_back(g, bias, y):
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * (
            (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    return ff * g
  
 

import os
from tqdm import auto as tqdm_lib

import torch



def create_model(checkpoint_path, use_cache=False, device=torch.device("cuda:1")):
    """
    To prevent allocation memory on CPU, we initialize on 'meta' and individually
    port each module over to 'device' as we load each state dict.
    :param checkpoint_path: Path to the checkpoint folder
    :param use_cache: whether to use cache (i.e. for efficient generation)
    :param device: device that you want the model to end up on
    :return: model
    """
    # Instantiate model
    pbar = tqdm_lib.tqdm(total=48)
    pbar.set_description("Instantiating model")
    model = NeoX20BModel(Args20b, use_cache=use_cache, device="meta")
    model = model.half().to_empty(device=device)
    pbar.update(1)

    # Load transformer layers
    for layer_i in range(Args20b.num_layers):
        pbar.set_description(f"Loading layer {layer_i}")
        filename = f"layer_{layer_i + 2:02d}-model_00-model_states.pt"
        loaded = torch.load(os.path.join(checkpoint_path, filename))
        model.layer_list[layer_i].load_state_dict(loaded)
        del loaded
        pbar.update(1)

    # Load input embedding
    pbar.set_description(f"Loading input embedding")
    loaded = torch.load(os.path.join(checkpoint_path, "layer_00-model_00-model_states.pt"))
    model.embed_in.load_state_dict({"weight": loaded["word_embeddings.weight"]})
    del loaded
    pbar.update(1)

    # Load final layer norm
    pbar.set_description(f"Loading final layer norm")
    loaded = torch.load(os.path.join(checkpoint_path, "layer_47-model_00-model_states.pt"))
    model.final_layer_norm.load_state_dict({
        "weight": loaded["norm.weight"],
        "bias": loaded["norm.bias"],
    })
    del loaded
    pbar.update(1)

    # Load output embedding
    pbar.set_description(f"Loading output embedding")
    loaded = torch.load(os.path.join(checkpoint_path, "layer_48-model_00-model_states.pt"))
    model.logits_out.load_state_dict({
        "weight": loaded["final_linear.weight"],
    })
    del loaded
    pbar.update(1)
    pbar.set_description("Done.")

    return model


def create_dummy_model(use_cache=False, device=torch.device("cpu")):
    model = NeoX20BModel(ArgsDummy, use_cache=use_cache).half().to(device)
    return model
