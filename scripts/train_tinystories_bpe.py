import os
import pickle
import time

from cs336_basics.tokenizer import train_bpe


def main() -> None:
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10_000
    special_tokens = ["<|endoftext|>"]

    output_dir = "output/tokenizer"
    os.makedirs(output_dir, exist_ok=True)

    vocab_path = os.path.join(output_dir, "tinystories_vocab.pkl")
    merges_path = os.path.join(output_dir, "tinystories_merges.pkl")

    start_time = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=input_path,
        special_tokens=special_tokens,
        vocab_size=vocab_size,
    )
    end_time = time.perf_counter()

    longest_token = max(vocab.values(), key=len)

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    print(f"总时间 (train_bpe): {end_time - start_time:.4f} 秒")
    print(f"vocab 大小: {len(vocab)}")
    print(f"merges 数量: {len(merges)}")
    print(f"最长 token repr: {longest_token!r}")
    print(f"最长 token 字节长度: {len(longest_token)}")


if __name__ == "__main__":
    main()
