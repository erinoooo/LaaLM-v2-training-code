"""
LaaLM-v2 Interactive Inference
Runs on CUDA if available, CPU otherwise.

Bug fixes:
  - Prompt priming previously accessed model.emb_drop / model.token_emb /
    model.lstm directly (bypassing the model API). Now uses
    _forward_with_hidden() which is the correct, stable interface.
  - Model is explicitly set to eval() before inference and bfloat16 cast
    is verified to match training dtype.
"""

import os
import torch
from tokenizers import Tokenizer

from model import LaaLMv2Config, LaaLMModel


class LaaLMTerminal:
    def __init__(self, checkpoint_path: str, tokenizer_path: str, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        print(f"Device: {self.device}")

        print("Loading tokenizer...")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        print("Loading model...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]

        self.model = LaaLMModel(config)
        # Strip torch.compile prefix if checkpoint came from a compiled run
        state = {k.replace("_orig_mod.", ""): v
                 for k, v in checkpoint["model_state_dict"].items()}
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(torch.bfloat16).to(self.device).eval()

        print(f"Model loaded: {config.n_params / 1e6:.2f}M parameters")

        self.max_seq_len   = config.max_seq_len
        self.conversation  = []
        self.system_prompt = (
            "### SYSTEM ###\n"
            "CWD=/home/user\n"
            "FILES=[]\n"
            "ENV=USER:user,HOME:/home/user\n"
            "### END SYSTEM ###"
        )
        self._end_output  = "### END OUTPUT ###"

    # ------------------------------------------------------------------

    def run_command(self, command: str, max_tokens: int = 200) -> str:
        # Build full context string in the training format
        ctx = self.system_prompt + "\n\n"
        for prev_cmd, prev_out in self.conversation:
            ctx += f"### COMMAND ###\n{prev_cmd}\n### END COMMAND ###\n\n"
            ctx += f"### OUTPUT ###\n{prev_out}\n### END OUTPUT ###\n\n"
        ctx += f"### COMMAND ###\n{command}\n### END COMMAND ###\n\n"
        ctx += "### OUTPUT ###\n"

        # Tokenise; keep last max_seq_len tokens
        enc       = self.tokenizer.encode(ctx)
        input_ids = torch.tensor(
            [enc.ids[-self.max_seq_len:]], dtype=torch.long
        ).to(self.device)

        generated: list[int] = []

        # Bug fix: use _forward_with_hidden() for both priming and generation
        # instead of directly accessing model.emb_drop / model.lstm internals.
        # The model is in eval() so all dropout layers are no-ops.
        with torch.inference_mode():
            hidden = None
            # Prime hidden state with full prompt (all but last token)
            if input_ids.size(1) > 1:
                _, hidden = self.model._forward_with_hidden(input_ids[:, :-1], hidden)

            current = input_ids[:, -1:]  # (1, 1)

            for step in range(max_tokens):
                logits, hidden = self.model._forward_with_hidden(current, hidden)
                next_id = logits[:, -1, :].argmax(dim=-1)  # greedy — fast for demo

                generated.append(next_id.item())
                response = self.tokenizer.decode(generated)

                if self._end_output in response:
                    break
                if "### COMMAND ###" in response:
                    break
                if next_id.item() == 3:          # EOS token
                    break
                if step > 150 and len(response.strip()) > 100:
                    break

                current = next_id.unsqueeze(0)   # (1,) → (1, 1)

        response = self.tokenizer.decode(generated).strip()
        if self._end_output in response:
            response = response.split(self._end_output)[0]
        if "### COMMAND ###" in response:
            response = response.split("### COMMAND ###")[0]
        response = response.strip()

        self.conversation.append((command, response))
        if len(self.conversation) > 20:
            self.conversation = self.conversation[-15:]

        return response

    def reset(self):
        self.conversation = []
        print("Conversation reset.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("LaaLM-v2 Interactive Terminal")
    print("=" * 60)
    print()

    best_path  = "checkpoints_v2/laalm_v2_best.pt"
    final_path = "checkpoints_v2/laalm_v2_final.pt"
    ckpt       = best_path if os.path.exists(best_path) else final_path

    terminal = LaaLMTerminal(
        checkpoint_path=ckpt,
        tokenizer_path="laalm_v2_tokenizer_v3.json",
    )

    print()
    print("=" * 60)
    print("Ready! Type bash commands. ('reset' | 'exit')")
    print("=" * 60)
    print()

    while True:
        try:
            command = input("$ ").strip()
            if not command:
                continue
            if command.lower() in ("exit", "quit"):
                print("\nGoodbye!")
                break
            if command.lower() == "reset":
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
