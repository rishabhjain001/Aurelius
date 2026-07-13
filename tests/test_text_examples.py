"""
tests/test_text_examples.py — Unit tests for Tokenizer behavior.

Validates: Requirements 5.3, 5.4
"""

import pytest
from encoders.text import Tokenizer


class TestTokenizerTruncation:
    """Tests for Requirement 5.3: captions longer than max_seq_len are truncated."""

    def test_caption_truncation(self):
        """Encoding a string that produces >77 tokens returns exactly 77 tokens.

        Validates: Requirements 5.3
        """
        tokenizer = Tokenizer(max_seq_len=77, pad_id=0)

        # "apple " repeated 100 times yields ~200 GPT-2 BPE tokens — well above 77
        long_text = "apple " * 100
        result = tokenizer.encode(long_text)

        assert len(result) == 77, (
            f"Expected 77 tokens after truncation, got {len(result)}"
        )


class TestTokenizerPadding:
    """Tests for Requirement 5.4: captions shorter than max_seq_len are padded."""

    def test_caption_padding(self):
        """Encoding a short string pads the result to exactly 77 tokens with pad_id.

        Validates: Requirements 5.4
        """
        pad_id = 0
        tokenizer = Tokenizer(max_seq_len=77, pad_id=pad_id)

        # "hello" encodes to 1 GPT-2 BPE token — far below 77
        result = tokenizer.encode("hello")

        # Total length must be exactly 77
        assert len(result) == 77, (
            f"Expected 77 tokens after padding, got {len(result)}"
        )

        # Find where real tokens end: the first occurrence of pad_id marks the
        # start of the padding region (GPT-2 token 0 is "!", so we check the
        # tail is all pad_id values)
        actual_token_count = len(tokenizer._enc.encode("hello"))
        padding_region = result[actual_token_count:]

        assert all(t == pad_id for t in padding_region), (
            f"Expected all padding positions to be pad_id={pad_id}, "
            f"but got: {padding_region}"
        )
