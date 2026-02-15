"""
LaaLM-v2 Interactive Inference - FIXED VERSION
Better stopping logic to prevent hallucination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from dataclasses import dataclass

# [Model architecture code - same as before, keeping it short here]
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
        return (self.cos_cached[:seq_len].to(x.device), self.sin_cached[:seq_len].to(x.device))

def apply_rotary_emb(q, k, cos, sin):
    q_rot = torch.cat((-q[..., q.shape[-1]//2:], q[..., :q.shape[-1]//2]), dim=-1)
    k_rot = torch.cat((-k[..., k.shape[-1]//2:], k[..., :k.shape[-1]//2]), dim=-1)
    return q * cos + q_rot * sin, k * cos + k_rot * sin

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(x, T)
        q, k = apply_rotary_emb(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        return self.dropout(self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C)))

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
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
    def forward(self, idx):
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm_f(x))

# ============================================================================
# FIXED INFERENCE ENGINE
# ============================================================================

class LaaLMTerminal:
    def __init__(self, checkpoint_path, tokenizer_path, device="cuda"):
        self.device = device
        print("Loading tokenizer...")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        print("Loading model...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        self.model = LaaLMModel(config)
        state_dict = checkpoint['model_state_dict']
        cleaned_state = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(cleaned_state)
        self.model = self.model.to(device).eval()
        
        print(f"✓ Model loaded: {config.n_params/1e6:.1f}M parameters")
        
        self.conversation = []
        self.system_prompt = "INIT_STATE:\nCWD=/home/user\nFILES=[]\nENV=USER:user,HOME:/home/user"
    
    def run_command(self, command, max_tokens=150):
        """Run command with MUCH better stopping"""
        
        # Build conversation
        conv_text = f"{self.system_prompt}\n"
        for cmd, out in self.conversation:
            conv_text += f"$ {cmd}\n{out}\n"
        conv_text += f"$ {command}\n"
        
        # Tokenize
        encoded = self.tokenizer.encode(conv_text)
        input_ids = torch.tensor([encoded.ids[-400:]], dtype=torch.long).to(self.device)  # Keep last 400 tokens
        prompt_len = input_ids.shape[1]
        
        # Generate
        generated_tokens = []
        with torch.no_grad():
            for step in range(max_tokens):
                logits = self.model(input_ids)
                next_token = logits[:, -1, :].argmax(dim=-1)
                
                # Decode incrementally to check stopping
                generated_tokens.append(next_token.item())
                response_so_far = self.tokenizer.decode(generated_tokens)
                
                # STOP CONDITIONS:
                # 1. If we see a new command prompt
                if '\n$' in response_so_far or response_so_far.endswith('$ '):
                    break
                
                # 2. If we see double newline (end of output)
                if '\n\n' in response_so_far and len(response_so_far) > 10:
                    break
                
                # 3. If response is getting too long and we have content
                if step > 100 and len(response_so_far.strip()) > 50:
                    break
                
                # 4. EOS token
                if next_token.item() == 3:
                    break
                
                # Continue generation
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        # Clean response
        response = self.tokenizer.decode(generated_tokens).strip()
        
        # Remove any command prompts that leaked
        if '\n$' in response:
            response = response.split('\n$')[0]
        if '$ ' in response:
            response = response.split('$ ')[0]
        
        response = response.strip()
        
        # Save to conversation
        self.conversation.append((command, response))
        
        # Keep conversation from getting too long
        if len(self.conversation) > 20:
            self.conversation = self.conversation[-15:]
        
        return response
    
    def reset(self):
        self.conversation = []
        print("✓ Conversation reset")

def main():
    print("="*60)
    print("LaaLM-v2 Interactive Terminal (FIXED)")
    print("="*60)
    print()
    
    terminal = LaaLMTerminal(
        checkpoint_path="checkpoints/laalm_v2_final.pt",
        tokenizer_path="laalm_v2_tokenizer.json",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("\n" + "="*60)
    print("Ready! Type bash commands")
    print("Commands: 'reset' (clear), 'exit' (quit)")
    print("="*60)
    print()
    
    while True:
        try:
            command = input("$ ").strip()
            
            if not command:
                continue
            if command.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
            if command.lower() == 'reset':
                terminal.reset()
                continue
            
            output = terminal.run_command(command)
            if output:
                print(output)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
