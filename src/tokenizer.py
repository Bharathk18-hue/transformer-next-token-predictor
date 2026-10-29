import json
import re
from typing import List, Dict, Optional


class SimpleTokenizer:
    """
    A beginner-friendly word-level tokenizer with special token support.
    Maps words to integer IDs and reverses integer IDs back to text.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN,
        ]
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        self.word2idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.idx2word = {idx: token for idx, token in enumerate(self.special_tokens)}

    @property
    def pad_id(self) -> int:
        return self.word2idx[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.word2idx[self.UNK_TOKEN]

    @property
    def sos_id(self) -> int:
        return self.word2idx[self.SOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.word2idx[self.EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    def tokenize_text(self, text: str) -> List[str]:
        """
        Cleans and splits raw text into words and punctuation tokens.
        """
        text = text.lower().strip()
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens

    def build_vocab(self, texts: List[str]) -> None:
        """
        Builds vocabulary from a list of text strings.
        """
        self._init_special_tokens()
        freq_counter: Dict[str, int] = {}

        for text in texts:
            tokens = self.tokenize_text(text)
            for token in tokens:
                freq_counter[token] = freq_counter.get(token, 0) + 1

        next_idx = len(self.word2idx)
        for token, count in sorted(freq_counter.items(), key=lambda item: -item[1]):
            if count >= self.min_freq and token not in self.word2idx:
                self.word2idx[token] = next_idx
                self.idx2word[next_idx] = token
                next_idx += 1

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Converts text string into a list of integer token IDs.
        """
        tokens = self.tokenize_text(text)
        ids = [self.word2idx.get(token, self.unk_id) for token in tokens]

        if add_special_tokens:
            ids = [self.sos_id] + ids + [self.eos_id]

        return ids

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """
        Converts a list of integer token IDs back to human-readable text.
        """
        words = []
        for tid in token_ids:
            word = self.idx2word.get(tid, self.UNK_TOKEN)
            if skip_special and word in self.special_tokens:
                continue
            words.append(word)

        # Format spacing neatly around punctuation
        text = ""
        for i, word in enumerate(words):
            if word in {".", ",", "!", "?", ";", ":"}:
                text = text.rstrip() + word + " "
            else:
                text += word + " "

        return text.strip()

    def save_vocab(self, filepath: str) -> None:
        """
        Saves vocabulary mappings to a JSON file.
        """
        data = {
            "min_freq": self.min_freq,
            "word2idx": self.word2idx,
            "idx2word": {str(k): v for k, v in self.idx2word.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_vocab(self, filepath: str) -> None:
        """
        Loads vocabulary mappings from a JSON file.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.min_freq = data.get("min_freq", 1)
        self.word2idx = data["word2idx"]
        self.idx2word = {int(k): v for k, v in data["idx2word"].items()}


if __name__ == "__main__":
    sample_corpus = [
        "The cat sat on the mat.",
        "The dog barked in the quiet park.",
        "Machine learning is fascinating and powerful."
    ]
    tokenizer = SimpleTokenizer(min_freq=1)
    tokenizer.build_vocab(sample_corpus)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    test_sentence = "The cat is sitting on the mat."
    encoded = tokenizer.encode(test_sentence)
    decoded = tokenizer.decode(encoded)
    print(f"Raw:     {test_sentence}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
