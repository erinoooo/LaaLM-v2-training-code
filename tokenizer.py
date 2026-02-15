"""
Train tokenizer v3 for LaaLM-v2
Includes special delimiter tokens
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import json

# Config
TRAINING_DATA = "laalm_v2_training_data_v3.jsonl"
VOCAB_SIZE = 8000
OUTPUT_PATH = "laalm_v2_tokenizer_v3.json"

def extract_text():
    """Extract all text from training conversations"""
    print("Extracting text from training data...")
    
    texts = []
    with open(TRAINING_DATA, 'r') as f:
        for i, line in enumerate(f):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1} conversations...")
            
            conv = json.loads(line)
            texts.append(conv['text'])
    
    print(f"✓ Extracted {len(texts)} conversations")
    return texts

def train_tokenizer(texts):
    """Train BPE tokenizer with special delimiters"""
    print(f"\nTraining tokenizer with vocab size {VOCAB_SIZE}...")
    
    # Initialize
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    
    # Special tokens - INCLUDING delimiters
    special_tokens = [
        "<pad>",
        "<unk>",
        "<s>",
        "</s>",
        "### SYSTEM ###",
        "### END SYSTEM ###",
        "### COMMAND ###",
        "### END COMMAND ###",
        "### OUTPUT ###",
        "### END OUTPUT ###",
        "<REASON>",
        "</REASON>",
    ]
    
    # Train
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2,
    )
    
    tokenizer.train_from_iterator(texts, trainer=trainer)
    print("✓ Tokenizer trained!")
    return tokenizer

def test_tokenizer(tokenizer):
    """Test tokenizer"""
    print("\n" + "="*60)
    print("TOKENIZER TESTS")
    print("="*60)
    
    test_cases = [
        "### COMMAND ###\nls\n### END COMMAND ###",
        "### OUTPUT ###\nfile.txt\n### END OUTPUT ###",
        "cat file.txt | grep error",
        "<REASON>\nSTEP1: execute(ls)\n</REASON>",
    ]
    
    for text in test_cases:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        
        print(f"\nOriginal: {repr(text[:50])}")
        print(f"Tokens:   {encoded.tokens[:10]}")
        print(f"Decoded:  {repr(decoded[:50])}")
    
    print("\n✓ Tests complete!")

def save_tokenizer(tokenizer, path):
    """Save tokenizer"""
    print(f"\nSaving tokenizer to {path}...")
    tokenizer.save(path)
    print("✓ Saved!")

def main():
    print("="*60)
    print("LaaLM-v2.1 TOKENIZER TRAINING")
    print("="*60)
    print()
    
    texts = extract_text()
    tokenizer = train_tokenizer(texts)
    test_tokenizer(tokenizer)
    save_tokenizer(tokenizer, OUTPUT_PATH)
    
    print("\n" + "="*60)
    print("✓ TOKENIZER TRAINING COMPLETE!")
    print("="*60)
    print(f"\nTokenizer saved to: {OUTPUT_PATH}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print("\nSpecial tokens learned:")
    print("  - ### COMMAND ###")
    print("  - ### END COMMAND ###")
    print("  - ### OUTPUT ###")
    print("  - ### END OUTPUT ###")
    print("\nReady for training LaaLM-v2.1!")

if __name__ == "__main__":
    main()
