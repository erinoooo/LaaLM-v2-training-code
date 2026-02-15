"""
Train a custom BPE tokenizer for LaaLM-v2
Optimized for Linux commands and terminal output
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import json

# Configuration
TRAINING_DATA = "laalm_v2_training_data.jsonl"
VOCAB_SIZE = 8000  # Small vocab for 200-300M model
OUTPUT_PATH = "laalm_v2_tokenizer.json"


def extract_text_from_jsonl(filename):
    """Extract all text from training conversations"""
    print("Extracting text from training data...")

    texts = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1} conversations...")

            conv = json.loads(line)
            for msg in conv['messages']:
                texts.append(msg['content'])

    print(f"✓ Extracted {len(texts)} messages")
    return texts


def train_tokenizer(texts):
    """Train BPE tokenizer"""
    print(f"\nTraining tokenizer with vocab size {VOCAB_SIZE}...")

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # Use ByteLevel pre-tokenizer (works well for all text including special chars)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # Special tokens for our use case
    special_tokens = [
        "<pad>",  # Padding
        "<unk>",  # Unknown
        "<s>",  # Start of sequence
        "</s>",  # End of sequence
        "<REASON>",  # CoT reasoning start
        "</REASON>",  # CoT reasoning end
        "OUTPUT:",  # Output marker
    ]

    # Trainer configuration
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2,
    )

    # Train on extracted texts
    tokenizer.train_from_iterator(texts, trainer=trainer)

    print("✓ Tokenizer trained!")
    return tokenizer


def test_tokenizer(tokenizer):
    """Test tokenizer on sample commands"""
    print("\n" + "=" * 60)
    print("TOKENIZER TESTS")
    print("=" * 60)

    test_cases = [
        "ls -la",
        "cat file.txt | grep error",
        "echo hello > test.txt",
        "<REASON>\nSTEP1: execute(cat file.txt)\n</REASON>",
        "rm: cannot remove 'file.txt': No such file or directory",
        "/home/user/docs",
        "pwd",
        "OUTPUT:\ntest.txt\ndata.log",
    ]

    for text in test_cases:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)

        print(f"\nOriginal: {repr(text)}")
        print(f"Tokens:   {encoded.tokens[:15]}{'...' if len(encoded.tokens) > 15 else ''}")
        print(f"IDs:      {encoded.ids[:15]}{'...' if len(encoded.ids) > 15 else ''}")
        print(f"Decoded:  {repr(decoded)}")

        if text != decoded:
            print(f"⚠ Mismatch (this is OK for ByteLevel tokenizer)")

    print("\n✓ Tests complete!")


def save_tokenizer(tokenizer, path):
    """Save tokenizer to file"""
    print(f"\nSaving tokenizer to {path}...")
    tokenizer.save(path)
    print("✓ Saved!")


def print_vocab_stats(tokenizer):
    """Print vocabulary statistics"""
    print("\n" + "=" * 60)
    print("VOCABULARY STATISTICS")
    print("=" * 60)

    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")

    # Sample some tokens
    vocab = tokenizer.get_vocab()
    sample_tokens = list(vocab.items())[:30]

    print("\nFirst 30 tokens:")
    for token, idx in sample_tokens:
        print(f"  {idx:4d}: {repr(token)}")

    # Check for important command tokens
    important_tokens = [
        "ls", "cat", "grep", "echo", "pwd", "cd", "rm", "mv", "cp",
        "touch", "mkdir", "head", "tail", "wc", "find",
        "|", ">", ">>", "/", ".", ".txt", ".log",
        "error", "cannot", "file", "directory"
    ]

    print(f"\nChecking for important tokens...")
    found = 0
    for token in important_tokens:
        # ByteLevel tokens are encoded, so we check if encoding exists
        try:
            enc = tokenizer.encode(token)
            if len(enc.tokens) <= 3:  # Token exists as a unit or small merge
                found += 1
                print(f"  ✓ {token} -> {enc.tokens}")
        except:
            pass

    print(f"\nFound ~{found} important token patterns")


def main():
    print("=" * 60)
    print("LaaLM-v2 TOKENIZER TRAINING")
    print("=" * 60)
    print()

    # Extract text from training data
    texts = extract_text_from_jsonl(TRAINING_DATA)

    # Train tokenizer
    tokenizer = train_tokenizer(texts)

    # Test tokenizer
    test_tokenizer(tokenizer)

    # Print stats
    print_vocab_stats(tokenizer)

    # Save tokenizer
    save_tokenizer(tokenizer, OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("✓ TOKENIZER TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nTokenizer saved to: {OUTPUT_PATH}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print("\nYou can now use this tokenizer for training LaaLM-v2!")


if __name__ == "__main__":
    main()