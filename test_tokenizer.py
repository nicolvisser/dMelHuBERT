from dataclasses import dataclass
from typing import List, Tuple

import torch
from tokenizers import Tokenizer

tokenizer: Tokenizer = Tokenizer.from_file("tokenizer.json")

encodings = tokenizer.encode_batch(["HELLO WORLD", "CAT SAT"])


@dataclass
class Encoding:
    """
    A class that represents an encoding of a text.
    This is not the same as the tokenizers.Encoding class.
    But that one is only an interface to some C code, so we cannot write to an instance of it or instantiate a new one.
    """

    ids: List[int]
    tokens: List[str]
    attention_mask: List[int]
    special_tokens_mask: List[int]
    type_ids: List[int]
    word_ids: List[int]
    offsets: List[Tuple[int, int]]
    sequence_ids: List[int]


def post_process(encoding: Encoding, blank_id: int, blank_token: str):
    ids = []
    tokens = []
    attention_mask = []
    special_tokens_mask = []
    type_ids = []
    word_ids = []
    offsets = []
    sequence_ids = []

    prev_id = None
    for idx, (id, tk, am, stm, ti, wi, of, si) in enumerate(
        zip(
            encoding.ids,
            encoding.tokens,
            encoding.attention_mask,
            encoding.special_tokens_mask,
            encoding.type_ids,
            encoding.word_ids,
            encoding.offsets,
            encoding.sequence_ids,
        )
    ):
        if idx > 0 and id == prev_id:
            # make provisions for the blank token
            ids.append(blank_id)
            tokens.append(blank_token)
            attention_mask.append(1)
            special_tokens_mask.append(1)
            type_ids.append(0)
            word_ids.append(None)
            offsets.append((idx, idx))
            sequence_ids.append(si)

        # add the current token
        ids.append(id)
        tokens.append(tk)
        attention_mask.append(am)
        special_tokens_mask.append(stm)
        type_ids.append(ti)
        word_ids.append(wi)
        offsets.append(of)
        sequence_ids.append(si)

        # remember the current token id
        prev_id = id

    new_encoding = Encoding(
        ids=ids,
        tokens=tokens,
        attention_mask=attention_mask,
        special_tokens_mask=special_tokens_mask,
        type_ids=type_ids,
        word_ids=word_ids,
        offsets=offsets,
        sequence_ids=sequence_ids,
    )

    return new_encoding


for enc in encodings:
    print(enc.ids)
    print(enc.attention_mask)
    print(enc.type_ids)
    print(enc.tokens)
    print()

for enc in encodings:
    enc2 = post_process(enc, 0, "-")
    print(enc2.ids)
    print(enc2.attention_mask)
    print(enc2.type_ids)
    print(enc2.tokens)
    print()
