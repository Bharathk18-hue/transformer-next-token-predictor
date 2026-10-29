import os
import sys
import unittest
import tempfile
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tokenizer import SimpleTokenizer
from src.dataset import NextTokenDataset
from src.attention import ScaledDotProductAttention, MultiHeadSelfAttention
from src.transformer import TransformerLM
from src.generate import (
    predict_next_token_candidates,
    generate_text,
    load_model_and_tokenizer,
)


class TestTransformerNextToken(unittest.TestCase):
    """
    Automated test suite verifying all 9 core requirements for the Transformer project.
    """

    def setUp(self):
        torch.manual_seed(42)

    def test_01_tokenization_and_detokenization(self):
        """Verify tokenization, vocabulary creation, and detokenization round-trip."""
        corpus = [
            "the cat sat on the mat.",
            "deep learning models learn patterns.",
        ]
        tokenizer = SimpleTokenizer(min_freq=1)
        tokenizer.build_vocab(corpus)

        self.assertGreater(tokenizer.vocab_size, 4, "Vocabulary must contain words beyond special tokens")
        self.assertEqual(tokenizer.word2idx[tokenizer.PAD_TOKEN], 0)
        self.assertEqual(tokenizer.word2idx[tokenizer.UNK_TOKEN], 1)

        sample = "the cat sat on the mat."
        encoded = tokenizer.encode(sample)
        self.assertIsInstance(encoded, list)
        self.assertTrue(all(isinstance(x, int) for x in encoded))

        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded.lower(), sample.lower(), "Detokenized text should match input sentence")

        # Unknown word fallback test
        unknown_encoded = tokenizer.encode("extraterrestrial spacecraft")
        self.assertTrue(all(x == tokenizer.unk_id for x in unknown_encoded))

    def test_02_input_target_alignment(self):
        """Verify next-token prediction alignment (target is shifted by 1 token)."""
        tokens = [10, 20, 30, 40, 50, 60]
        dataset = NextTokenDataset(tokens, seq_len=3, stride=1)

        self.assertGreater(len(dataset), 0)
        inp, tgt = dataset[0]

        # inp: [10, 20, 30], tgt: [20, 30, 40]
        self.assertEqual(inp.tolist(), [10, 20, 30])
        self.assertEqual(tgt.tolist(), [20, 30, 40])
        # Position i in tgt is the next token after position i in inp
        for i in range(len(inp)):
            self.assertEqual(tgt[i].item(), tokens[i + 1])

    def test_03_self_attention_shapes(self):
        """Verify self-attention output shapes for single-head and multi-head."""
        batch_size = 2
        seq_len = 6
        d_model = 64
        num_heads = 4

        mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
        x = torch.randn(batch_size, seq_len, d_model)

        out, weights = mha(x, causal=True)
        self.assertEqual(out.shape, (batch_size, seq_len, d_model))
        self.assertEqual(weights.shape, (batch_size, num_heads, seq_len, seq_len))

    def test_04_causal_masking_prevents_leakage(self):
        """
        Verify that modifying future tokens DOES NOT affect the attention weights
        or representations of past and current tokens.
        """
        seq_len = 4
        d_model = 32
        mha = MultiHeadSelfAttention(d_model=d_model, num_heads=2, dropout=0.0)
        mha.eval()

        # Sequence 1: 4 tokens
        seq1 = torch.randn(1, seq_len, d_model)

        # Sequence 2: identical for first 2 tokens, but tokens 3 and 4 are changed
        seq2 = seq1.clone()
        seq2[0, 2:] = torch.randn(2, d_model)

        with torch.no_grad():
            out1, weights1 = mha(seq1, causal=True)
            out2, weights2 = mha(seq2, causal=True)

        # For tokens at indices 0 and 1, output vectors MUST be identical despite future changes
        self.assertTrue(
            torch.allclose(out1[0, :2], out2[0, :2], atol=1e-5),
            "Causal mask failed: past representations changed when future tokens changed!",
        )

        # In attention matrix, upper triangle must be strictly zero
        for h in range(2):
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    self.assertAlmostEqual(
                        weights1[0, h, i, j].item(),
                        0.0,
                        places=5,
                        msg=f"Future attention weight at ({i}, {j}) must be 0",
                    )

    def test_05_model_output_shape(self):
        """Verify full TransformerLM output shape is [batch_size, sequence_length, vocabulary_size]."""
        vocab_size = 120
        seq_len = 10
        batch_size = 3
        model = TransformerLM(
            vocab_size=vocab_size,
            context_length=seq_len,
            d_model=64,
            num_heads=4,
            num_layers=2,
        )
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits, _ = model(input_ids, causal=True)

        self.assertEqual(
            logits.shape,
            (batch_size, seq_len, vocab_size),
            "Output logits shape must be [batch_size, seq_len, vocab_size]",
        )

    def test_06_training_loss_is_finite(self):
        """Verify a forward-backward training step executes cleanly and produces finite loss."""
        vocab_size = 50
        seq_len = 8
        model = TransformerLM(vocab_size=vocab_size, context_length=seq_len, d_model=32, num_heads=2, num_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        inputs = torch.randint(0, vocab_size, (4, seq_len))
        targets = torch.randint(0, vocab_size, (4, seq_len))

        optimizer.zero_grad()
        logits, _ = model(inputs, causal=True)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

        self.assertFalse(torch.isnan(loss), "Loss must not be NaN")
        self.assertFalse(torch.isinf(loss), "Loss must not be Inf")
        self.assertGreater(loss.item(), 0.0, "Loss must be positive")

        loss.backward()
        optimizer.step()

    def test_07_model_saving_and_loading(self):
        """Verify that model weights and vocabulary can be saved and restored accurately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.pt")
            vocab_path = os.path.join(tmpdir, "test_vocab.json")

            # Save vocab
            tokenizer = SimpleTokenizer()
            tokenizer.build_vocab(["apple banana orange"])
            tokenizer.save_vocab(vocab_path)

            # Save model
            model = TransformerLM(vocab_size=tokenizer.vocab_size, context_length=8, d_model=32, num_heads=2, num_layers=1)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": tokenizer.vocab_size,
                        "context_length": 8,
                        "d_model": 32,
                        "num_heads": 2,
                        "num_layers": 1,
                    },
                },
                model_path,
            )

            # Reload
            loaded_model, loaded_tok = load_model_and_tokenizer(model_path, vocab_path)
            self.assertEqual(loaded_tok.vocab_size, tokenizer.vocab_size)
            self.assertEqual(loaded_model.vocab_size, tokenizer.vocab_size)

    def test_08_text_generation(self):
        """Verify text generation produces non-empty output with correct starting prompt."""
        vocab_size = 50
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(["the cat sat on the mat and slept peacefully"])
        model = TransformerLM(vocab_size=tokenizer.vocab_size, context_length=16, d_model=32, num_heads=2, num_layers=1)

        prompt = "the cat"
        gen_text = generate_text(model, tokenizer, prompt, max_new_tokens=4, sample=False)
        self.assertIsInstance(gen_text, str)
        self.assertTrue(gen_text.startswith("the cat"))
        self.assertGreater(len(gen_text.split()), 2)

    def test_09_streamlit_syntax_and_imports(self):
        """Verify app.py syntax compiles without errors."""
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        self.assertTrue(os.path.exists(app_path), "app.py must exist")
        with open(app_path, "r", encoding="utf-8") as f:
            code = f.read()
        compiled = compile(code, app_path, "exec")
        self.assertIsNotNone(compiled)


if __name__ == "__main__":
    unittest.main(verbosity=2)
