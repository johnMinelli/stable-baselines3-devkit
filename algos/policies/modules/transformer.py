import numpy as np
import torch
from torch import nn

from algos.policies.modules.gru import GRUGate


class SinusoidalPE(nn.Module):
    """Relative positional encoding."""

    def __init__(self, dim, min_timescale=2.0, max_timescale=1e4):
        """
        Initialize sinusoidal positional encoding.

        :param dim: Embedding dimension
        :param min_timescale: Minimum timescale for frequency generation
        :param max_timescale: Maximum timescale for frequency generation
        """
        super().__init__()
        freqs = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer("inv_freqs", inv_freqs)

    def forward(self, seq_len):
        """
        Compute positional embedding.

        :param seq_len: Sequence length
        :return: Positional embedding with shape (seq_len, embedding_dim)
        """
        seq = torch.arange(int(seq_len) - 1, -1, -1.0)
        sinusoidal_inp = seq.view(-1, 1) * self.inv_freqs.view(1, -1)
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim=-1)
        return pos_emb


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, embed_dim, num_heads):
        """
        Initialize multi-head attention.

        :param embed_dim: Dimensionality of the embeddings
        :param num_heads: Number of attention heads
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for multi-head attention.

        :param query: Query tensor of shape (batch_size, query_len, embed_dim)
        :param key: Key tensor of shape (batch_size, key_len, embed_dim)
        :param value: Value tensor of shape (batch_size, value_len, embed_dim)
        :param mask: Optional attention mask of shape (batch_size, query_len, key_len)
        :return: Output tensor of shape (batch_size, query_len, embed_dim)
        """
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]

        # Project Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape for multi-head attention: (batch, len, embed) -> (batch, num_heads, len, head_dim)
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            # Reshape mask to match attention scores
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, query_len, key_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back: (batch, num_heads, query_len, head_dim) -> (batch, query_len, embed)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, query_len, self.embed_dim)

        # Final linear projection
        output = self.out_proj(attn_output)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, gru_bias=2.0, **kwargs):
        """
        Initialize a Transformer Block.

        :param embed_dim: Dimensionality of the input embeddings
        :param num_heads: Number of attention heads
        :param gru_bias: Bias value for GRU gates (default: 2.0)
        :param kwargs: Additional keyword arguments
        """
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.gate1 = GRUGate(embed_dim, gru_bias)
        self.gate2 = GRUGate(embed_dim, gru_bias)

        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, query, key, mask=None):
        """
        Forward pass of the Transformer Block.

        :param query: Query tensor.
        :param key: Key tensor.
        :param mask: Mask tensor for attention, indicating which elements to attend to.
        :returns: Output tensor after the Transformer Block.
        """
        norm_key = self.layer_norm1(key)
        Y = self.attention(self.layer_norm1(query), norm_key, norm_key, mask)
        out = self.gate1(query, Y)
        E = self.ffn(self.layer_norm2(out))
        out = self.gate2(out, E)
        assert not torch.isnan(out).any(), "Transformer block returned NaN!"

        return out


class GatedTransformerXL(nn.Module):
    """Gated Transformer XL model with memory mechanism."""

    def __init__(
        self, input_size: int, embed_dim=256, num_blocks=6, num_heads=8, max_episode_steps=10000, gru_bias=2.0, **kwargs
    ) -> None:
        """
        Initialize a Gated Transformer XL model.

        :param input_size: Dimensionality of the input
        :param embed_dim: Embedding dimension (default: 256)
        :param num_blocks: Number of transformer blocks (default: 6)
        :param num_heads: Number of attention heads (default: 8)
        :param max_episode_steps: Maximum number of episode steps (default: 500)
        :param gru_bias: Bias value for GRU gates (default: 2.0)
        :param kwargs: Additional keyword arguments
        """
        super().__init__()
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.heads_dim = self.embed_dim // self.num_heads
        self.max_episode_steps = max_episode_steps
        self.activation = nn.GELU()
        # Input embedding layer
        self.linear_embedding = nn.Linear(input_size, self.embed_dim)
        nn.init.orthogonal_(self.linear_embedding.weight, np.sqrt(2))
        self.pos_embedding = SinusoidalPE(dim=self.embed_dim)(self.max_episode_steps)

        # Instantiate transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.embed_dim, self.num_heads, gru_bias=gru_bias, **kwargs) for _ in range(self.num_blocks)]
        )

    def forward(self, x, memories, mask, memory_indices):
        """
        Forward pass through the Gated Transformer XL.

        :param x: Input tensor (query) of shape (batch_size, input_size)
        :param memories: Whole episode memories of shape (batch_size, memory_length, num_blocks, embed_dim)
        :param mask: Attention mask (dtype: bool) of shape (batch_size, memory_length)
        :param memory_indices: Memory window indices (dtype: long) of shape (batch_size, memory_length)
        :return: Tuple of (output tensor, updated memories)
        """
        assert torch.all(memory_indices<self.max_episode_steps), "Memory indices exceeded the allocated positional embedding vector length."

        # Feed embedding layer and activate
        h = self.activation(self.linear_embedding(x))
        if self.pos_embedding.device != memory_indices.device:
            self.pos_embedding = self.pos_embedding.to(device=memory_indices.device)

        # Add positional encoding to every transformer block input
        pos_embedding = self.pos_embedding[memory_indices.long()]
        memories = memories + pos_embedding.unsqueeze(2)  # no pos_embedding on first element
        # h = h + self.pos_embedding[(memory_indices[:,0]+mask[:,0].sum(-1)).long()]  # pos embedding only on first element then memory is stored with pos embedding

        # add mask following the sequence adopted
        mask = torch.cat((mask[:, [-1]], torch.ones_like(mask[:, 0, 0, None, None])), dim=2)
        # _mask = torch.cat((mask, torch.zeros_like(mask[:, :, [0]])), dim=2)  # for a sequence >1
        # mask = torch.cat((_mask, torch.cat((mask[:, [-1]], torch.ones_like(mask[:,0,0,None,None])), dim=2)), dim=1)

        # Forward transformer blocks
        out_memories = []
        for i, block in enumerate(self.transformer_blocks):
            out_memories.append(h.detach())
            seq = torch.cat([memories[:, :, i], h.unsqueeze(1)], 1)
            seq = block(h.unsqueeze(1), seq, mask)  # args: query, key, mask
            h = seq[:, -1].squeeze()
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
        return h, torch.stack(out_memories, dim=1)
