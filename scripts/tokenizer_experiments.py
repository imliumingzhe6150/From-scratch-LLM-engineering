"""Tokenizer experiments for assignment 2.7 (a), (b), (c).

Prints:
  (a) compression ratio (bytes/token) of TinyStories(10K) and OpenWebText(32K)
      on their own sampled documents;
  (b) compression ratio of OpenWebText documents tokenized by the TinyStories
      tokenizer (cross-tokenizer);
  (c) encoding throughput (bytes/sec) and the estimated time to tokenize the
      Pile dataset (825 GB).
"""

import time

from cs336_basics.tokenizer import Tokenizer

TOK_DIR = "output/tokenizer"
TS_VOCAB = f"{TOK_DIR}/tinystories_vocab.pkl"
TS_MERGES = f"{TOK_DIR}/tinystories_merges.pkl"
OWT_VOCAB = f"{TOK_DIR}/owt_vocab.pkl"
OWT_MERGES = f"{TOK_DIR}/owt_merges.pkl"

TS_TRAIN = "data/TinyStoriesV2-GPT4-train.txt"
OWT_TRAIN = "data/owt_train.txt"

SPECIAL_TOKENS = ["<|endoftext|>"]
N_DOCS = 10
THROUGHPUT_SAMPLE_CHARS = 10_000_000  # ~10 MB of text to time encoding on


def read_documents(path: str, n: int) -> list[str]:
    """Return the first `n` documents, split on the inline ``<|endoftext|>`` marker."""
    docs: list[str] = []
    cur: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split("<|endoftext|>")
            # All parts except the last are the end of complete documents.
            for part in parts[:-1]:
                cur.append(part)
                doc = "".join(cur)
                if doc.strip():
                    docs.append(doc)
                cur = []
                if len(docs) >= n:
                    return docs
            cur.append(parts[-1])
    if cur:
        doc = "".join(cur)
        if doc.strip() and len(docs) < n:
            docs.append(doc)
    return docs


def compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> tuple[int, int, float]:
    total_bytes = sum(len(doc.encode("utf-8")) for doc in docs)
    total_tokens = sum(len(tokenizer.encode(doc)) for doc in docs)
    return total_bytes, total_tokens, total_bytes / total_tokens


def measure_throughput(tokenizer: Tokenizer, text: str) -> float:
    # Use process_time (CPU time), not perf_counter: perf_counter includes
    # macOS suspend time, which inflates the result if the machine slept.
    start = time.process_time()
    tokenizer.encode(text)
    elapsed = time.process_time() - start
    return len(text.encode("utf-8")) / elapsed


def main() -> None:
    ts = Tokenizer.from_files(TS_VOCAB, TS_MERGES, SPECIAL_TOKENS)
    owt = Tokenizer.from_files(OWT_VOCAB, OWT_MERGES, SPECIAL_TOKENS)

    # (a) Each tokenizer encodes its own corpus's documents.
    ts_docs = read_documents(TS_TRAIN, N_DOCS)
    owt_docs = read_documents(OWT_TRAIN, N_DOCS)

    ts_b, ts_t, ts_r = compression_ratio(ts, ts_docs)
    owt_b, owt_t, owt_r = compression_ratio(owt, owt_docs)

    print("(a) compression ratio (bytes/token)")
    print(f"    TinyStories (10K): {ts_b} bytes / {ts_t} tokens = {ts_r:.3f} bytes/token")
    print(f"    OpenWebText (32K): {owt_b} bytes / {owt_t} tokens = {owt_r:.3f} bytes/token")

    # (b) OpenWebText documents tokenized by the TinyStories tokenizer.
    cross_b, cross_t, cross_r = compression_ratio(ts, owt_docs)
    print("(b) OpenWebText docs with TinyStories (10K) tokenizer")
    print(f"    {cross_b} bytes / {cross_t} tokens = {cross_r:.3f} bytes/token")

    # (c) Throughput and Pile estimate.
    with open(OWT_TRAIN, encoding="utf-8") as f:
        sample = f.read(THROUGHPUT_SAMPLE_CHARS)
    ts_tput = measure_throughput(ts, sample)
    owt_tput = measure_throughput(owt, sample)
    pile_bytes = 825 * 1e9
    pile_seconds = pile_bytes / owt_tput
    print("(c) throughput")
    print(f"    TinyStories tokenizer: {ts_tput:,.0f} bytes/sec")
    print(f"    OpenWebText tokenizer: {owt_tput:,.0f} bytes/sec")
    print(f"    Pile (825 GB) with OWT tokenizer: {pile_seconds/3600:,.1f} hours = {pile_seconds/86400:,.1f} days")


if __name__ == "__main__":
    main()
