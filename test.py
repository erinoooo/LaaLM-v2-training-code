"""
LaaLM-v2 Interactive Inference
Test your trained model with persistent filesystem state!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from dataclasses import dataclass

# ============================================================================
# MODEL ARCHITECTURE (copy from training script)
# ============================================================================

@dataclass
class ModelConfig:
    vocab_size: int = 8000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 512
    dropout: float = 0.1
    
    def __post_init__(self):
        self.n_params = self.calculate_params()
    
    def calculate_params(self):
        emb = self.vocab_size * self.d_model
        pos_emb = self.max_seq_len * self.d_model
        attn_qkv = 3 * self.d_model * self.d_model
        attn_out = self.d_model * self.d_model
        ffn = 2 * self.d_model * self.d_ff
        layer_norm = 2 * 2 * self.d_model
        per_layer = attn_qkv + attn_out + ffn + layer_norm
        final_ln = 2 * self.d_model
        lm_head = self.vocab_size * self.d_model
        total = emb + pos_emb + (per_layer * self.n_layers) + final_ln + lm_head
        return total

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x, seq_len):
        return (
            self.cos_cached[:seq_len].to(x.device),
            self.sin_cached[:seq_len].to(x.device),
        )

def apply_rotary_emb(q, k, cos, sin):
    q_rot = torch.cat((-q[..., q.shape[-1]//2:], q[..., :q.shape[-1]//2]), dim=-1)
    k_rot = torch.cat((-k[..., k.shape[-1]//2:], k[..., :k.shape[-1]//2]), dim=-1)
    q_out = q * cos + q_rot * sin
    k_out = k * cos + k_rot * sin
    return q_out, k_out

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rope(x, T)
        q, k = apply_rotary_emb(q, k, cos, sin)
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class LaaLMModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
    
    def forward(self, idx):
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class LaaLMTerminal:
    def __init__(self, checkpoint_path, tokenizer_path, device="cuda"):
        self.device = device
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Load model
        print("Loading model...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        self.model = LaaLMModel(config)
        
        # Clean state dict (remove _orig_mod prefix)
        state_dict = checkpoint['model_state_dict']
        cleaned_state = {}
        for key, value in state_dict.items():
            clean_key = key.replace('_orig_mod.', '')
            cleaned_state[clean_key] = value
        
        self.model.load_state_dict(cleaned_state)
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f"✓ Model loaded: {config.n_params/1e6:.1f}M parameters")
        
        # Initialize conversation
        self.conversation = []
        self.system_prompt = """INIT_STATE:
CWD=/home/user
FILES=[]
ENV=USER:user,HOME:/home/user"""
        
        self.conversation.append(("system", self.system_prompt))
    
    def run_command(self, command, max_tokens=256):
        """Run a bash command and get output"""
        
        # Add command to conversation
        self.conversation.append(("user", command))
        
        # Build prompt from full conversation history
        prompt = ""
        for role, content in self.conversation:
            if role == "system":
                prompt += f"{content}\n"
            elif role == "user":
                prompt += f"$ {content}\n"
            elif role == "assistant":
                prompt += f"{content}\n"
        
        # Tokenize
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(self.device)
        
        # Truncate if too long (keep last 400 tokens to fit context)
        if input_ids.shape[1] > 400:
            input_ids = input_ids[:, -400:]
        
        # Generate
        with torch.no_grad():
            generated = input_ids
            
            for _ in range(max_tokens):
                # Forward pass
                logits = self.model(generated)
                
                # Get next token (greedy)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                
                # Stop if EOS or newline after output
                next_token_id = next_token.item()
                if next_token_id == 3:  # EOS token
                    break
                
                # Append
                generated = torch.cat([generated, next_token], dim=1)
                
                # Decode to check for stopping
                decoded = self.tokenizer.decode(generated[0].tolist())
                
                # Stop if we see a new command prompt
                if decoded.strip().endswith("$ ") and len(decoded) > len(prompt) + 10:
                    break
        
        # Decode output
        full_output = self.tokenizer.decode(generated[0].tolist())
        
        # Extract just the response (after the prompt)
        response = full_output[len(prompt):].strip()
        
        # Clean up response (remove any trailing $ or prompts)
        if "$ " in response:
            response = response.split("$ ")[0].strip()
        
        # Add to conversation
        self.conversation.append(("assistant", response))
        
        return response
    
    def reset(self):
        """Reset conversation (fresh filesystem)"""
        self.conversation = [("system", self.system_prompt)]
        print("✓ Filesystem reset")

# ============================================================================
# INTERACTIVE SHELL
# ============================================================================

def main():
    print("="*60)
    print("LaaLM-v2 Interactive Terminal")
    print("="*60)
    print()
    
    # Initialize
    terminal = LaaLMTerminal(
        checkpoint_path="checkpoints/laalm_v2_final.pt",
        tokenizer_path="laalm_v2_tokenizer.json",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("\n" + "="*60)
    print("Ready! Type bash commands (or 'exit' to quit, 'reset' to clear)")
    print("="*60)
    print()
    
    # Interactive loop
    while True:
        try:
            command = input("\n$ ").strip()
            
            if not command:
                continue
            
            if command.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
            
            if command.lower() == 'reset':
                terminal.reset()
                continue
            
            # Run command
            output = terminal.run_command(command)
            
            # Print output
            if output:
                print(output)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
