"""Encode full training/development datasets to uint16 numpy arrays (2.7 d).

uint16 is appropriate because both vocab sizes (10K and 32K) fit within the
0..65535 range; uint8 is too small and uint32 wastes half the storage.
"""

import os

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer

TOK_DIR = "output/tokenizer"
OUT_DIR = "output/encoded"

SPECIAL_TOKENS = ["<|endoftext|>"]

CHUNK = 1_000_000  # number of token ids to accumulate before converting to an array


def encode_to_npy(tokenizer: Tokenizer, input_path: str, output_path: str) -> int:
    chunks: list[np.ndarray] = []
    buf: list[int] = []
    pbar = tqdm(unit="tok", desc=os.path.basename(input_path))

    with open(input_path, encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            buf.append(token_id)
            if len(buf) >= CHUNK:
                chunks.append(np.array(buf, dtype=np.uint16))
                pbar.update(CHUNK)
                buf = []

    if buf:
        chunks.append(np.array(buf, dtype=np.uint16))
        pbar.update(len(buf))
    pbar.close()

    arr = np.concatenate(chunks)
    np.save(output_path, arr)
    print(f"  saved {output_path}: {arr.shape[0]:,} tokens, {arr.nbytes / 1e9:.3f} GB")
    return arr.shape[0]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    ts = Tokenizer.from_files(
        f"{TOK_DIR}/tinystories_vocab.pkl",
        f"{TOK_DIR}/tinystories_merges.pkl",
        SPECIAL_TOKENS,
    )
    owt = Tokenizer.from_files(
        f"{TOK_DIR}/owt_vocab.pkl",
        f"{TOK_DIR}/owt_merges.pkl",
        SPECIAL_TOKENS,
    )

    encode_to_npy(ts, "data/TinyStoriesV2-GPT4-train.txt", f"{OUT_DIR}/tinystories_train_ids.npy")
    encode_to_npy(ts, "data/TinyStoriesV2-GPT4-valid.txt", f"{OUT_DIR}/tinystories_valid_ids.npy")
    encode_to_npy(owt, "data/owt_train.txt", f"{OUT_DIR}/owt_train_ids.npy")
    encode_to_npy(owt, "data/owt_valid.txt", f"{OUT_DIR}/owt_valid_ids.npy")


if __name__ == "__main__":
    main()
