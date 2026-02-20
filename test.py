"""
LaaLM-v2 Interactive Inference (TPU)
Uses v3 delimiter format matching training data
"""

import torch
from tokenizers import Tokenizer

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ImportError as e:
    _msg = str(e)
    if "undefined symbol" in _msg or "_XLAC" in _msg:
        raise SystemExit(
            "\n[ERROR] torch_xla version mismatch with torch.\n"
            "torch and torch_xla must be exactly the same version.\n\n"
            "Fix — run these commands in your venv:\n"
            "  pip show torch               # note the version (e.g. 2.5.1)\n"
            "  pip uninstall torch_xla -y\n"
            "  pip install torch_xla==<same-version-as-torch>\n\n"
            "Or reinstall both together:\n"
            "  pip install torch==2.5.1 torch_xla==2.5.0\n\n"
            f"Original error: {_msg}"
        ) from None
    raise

from model import LaaLMv2Config, LaaLMModel

# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class LaaLMTerminal:
    def __init__(self, checkpoint_path, tokenizer_path, device=None):
        self.device = device if device is not None else xm.xla_device()
        print("Loading tokenizer...")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        print("Loading model...")
        # Load checkpoint to CPU first, then move to TPU
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']

        self.model = LaaLMModel(config)
        state_dict = checkpoint['model_state_dict']
        cleaned_state = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(cleaned_state)
        self.model = self.model.to(torch.bfloat16).to(self.device).eval()

        print(f"Model loaded: {config.n_params/1e6:.1f}M parameters")

        self.max_seq_len = config.max_seq_len
        self.conversation = []
        # System prompt matches training data format exactly
        self.system_prompt = (
            "### SYSTEM ###\n"
            "CWD=/home/user\n"
            "FILES=[]\n"
            "ENV=USER:user,HOME:/home/user\n"
            "### END SYSTEM ###"
        )

        # Cache delimiter token sequences for stop detection
        self._end_output_str = "### END OUTPUT ###"
        self._end_command_str = "### END COMMAND ###"

    def run_command(self, command, max_tokens=200):
        """Run command using the v3 delimiter format matching training data"""

        # Build conversation in the exact format the model was trained on
        conv_text = self.system_prompt + "\n\n"
        for prev_cmd, prev_out in self.conversation:
            conv_text += f"### COMMAND ###\n{prev_cmd}\n### END COMMAND ###\n\n"
            conv_text += f"### OUTPUT ###\n{prev_out}\n### END OUTPUT ###\n\n"

        # Add the new command and prompt for output
        conv_text += f"### COMMAND ###\n{command}\n### END COMMAND ###\n\n"
        conv_text += "### OUTPUT ###\n"

        # Tokenize (keep last max_seq_len tokens for context window)
        encoded = self.tokenizer.encode(conv_text)
        input_ids = torch.tensor(
            [encoded.ids[-self.max_seq_len:]], dtype=torch.long
        ).to(self.device)

        # Generate
        generated_tokens = []
        with torch.no_grad():
            for step in range(max_tokens):
                logits, _ = self.model(input_ids)
                next_token = logits[:, -1, :].argmax(dim=-1)

                generated_tokens.append(next_token.item())
                response_so_far = self.tokenizer.decode(generated_tokens)

                # Stop if we see the end-of-output delimiter
                if self._end_output_str in response_so_far:
                    break

                # Stop if we see the start of a new command block
                if "### COMMAND ###" in response_so_far:
                    break

                # Stop on EOS token (id=3)
                if next_token.item() == 3:
                    break

                # Safety: stop if response is very long with content
                if step > 150 and len(response_so_far.strip()) > 100:
                    break

                # Continue generation
                input_ids = torch.cat(
                    [input_ids, next_token.unsqueeze(0)], dim=1
                )

        # Clean response: extract content before end delimiter
        response = self.tokenizer.decode(generated_tokens).strip()

        # Remove any trailing delimiter markers
        if self._end_output_str in response:
            response = response.split(self._end_output_str)[0]
        if "### COMMAND ###" in response:
            response = response.split("### COMMAND ###")[0]

        response = response.strip()

        # Save to conversation history
        self.conversation.append((command, response))

        # Keep conversation from getting too long
        if len(self.conversation) > 20:
            self.conversation = self.conversation[-15:]

        return response

    def reset(self):
        self.conversation = []
        print("Conversation reset")

def main():
    print("=" * 60)
    print("LaaLM-v2 Interactive Terminal")
    print("=" * 60)
    print()

    # Prefer best model (lowest val loss) if available, fall back to final
    import os
    best_path = "checkpoints_v2/laalm_v2_best.pt"
    final_path = "checkpoints_v2/laalm_v2_final.pt"
    checkpoint_path = best_path if os.path.exists(best_path) else final_path

    terminal = LaaLMTerminal(
        checkpoint_path=checkpoint_path,
        tokenizer_path="laalm_v2_tokenizer_v3.json",
        device=xm.xla_device(),
    )

    print()
    print("=" * 60)
    print("Ready! Type bash commands")
    print("Commands: 'reset' (clear), 'exit' (quit)")
    print("=" * 60)
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
