import pickle
from collections.abc import Iterable, Iterator

import regex as re
from tqdm import tqdm

from .pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def initialize_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {}

    for byte_value in range(256):
        vocab[byte_value] = bytes([byte_value])

    next_id = 256
    for special_token in special_tokens:
        vocab[next_id] = special_token.encode("utf-8")
        next_id += 1

    return vocab



def split_on_special_tokens(text: str, special_tokens: list[str]) ->list[str]:
    if not special_tokens:
        return [text]

    sorted_tokens = sorted(special_tokens, key=len, reverse=True)
    escaped_tokens = [re.escape(token) for token in sorted_tokens]

    pattern = "(?:" + "|".join(escaped_tokens) + ")"

    return re.split(pattern, text)



def count_pretokens(text:str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    pretoken_counts: dict[tuple[bytes, ...], int] = {}

    segments = split_on_special_tokens(text, special_tokens)

    for segment in segments:
        regex_matches = re.finditer(PAT, segment)
        for match in regex_matches:
            pretoken = match.group().encode("utf-8")
            units = tuple(bytes([byte]) for byte in pretoken)
            if units in pretoken_counts:
                pretoken_counts[units] += 1
            else:
                pretoken_counts[units] = 1

    return pretoken_counts


def count_pretokens_chunk(args: tuple[str, int, int, str, list[str]]) -> dict[tuple[bytes, ...], int]:
    input_path, start, end, encoding, special_tokens = args

    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode(encoding, errors="ignore")

    return count_pretokens(chunk, special_tokens)

def count_pretokens_parallel(input_path: str, special_tokens: list[str], num_processes: int) -> dict[tuple[bytes, ...], int]:
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(
            file, num_processes, b"<|endoftext|>"
        )

    args_list = [
        (input_path, boundaries[i], boundaries[i + 1], "utf-8", special_tokens)
        for i in range(len(boundaries) - 1)
    ]
    if len(args_list) == 1:
        return count_pretokens_chunk(args_list[0])

    actual_processes = min(num_processes, len(args_list))

    from multiprocessing import Pool

    with Pool(processes=actual_processes) as pool:
        results = pool.map(count_pretokens_chunk, args_list)

    combined_counts: dict[tuple[bytes, ...], int] = {}
    for result in results:
        for units, count in result.items():
            if units in combined_counts:
                combined_counts[units] += count
            else:
                combined_counts[units] = count

    return combined_counts


def count_pairs(pretoken_counts: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    pair_counts: dict[tuple[bytes, bytes], int] = {}

    for units, count in pretoken_counts.items():
        for i in range(len(units) - 1):
            pair = (units[i], units[i + 1])
            if pair in pair_counts:
                pair_counts[pair] += count
            else:
                pair_counts[pair] = count

    return pair_counts



def find_best_pair(pair_counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    best_pair = max(pair_counts, key=lambda pair:(pair_counts[pair], pair))
    return best_pair



def merge_pretoken(pretoken: tuple[bytes, ...], pair: tuple[bytes, bytes]) ->tuple[bytes, ...]:
    merged_units: list[bytes] = []
    i = 0
    while i < len(pretoken):
        if i < len(pretoken) - 1 and (pretoken[i], pretoken[i + 1]) == pair:
            merged_units.append(pretoken[i] + pretoken[i + 1])
            i += 2
        else:
            merged_units.append(pretoken[i])
            i += 1
    return tuple(merged_units)



def merge_all_pretokens(pretoken_counts: dict[tuple[bytes, ...], int], best_pair: tuple[bytes, bytes]) -> dict[tuple[bytes, ...], int]:
    new_pretoken_counts: dict[tuple[bytes, ...], int] = {}

    for pretoken, count in pretoken_counts.items():
        merged_pretoken = merge_pretoken(pretoken, best_pair)
        if merged_pretoken in new_pretoken_counts:
            new_pretoken_counts[merged_pretoken] += count
        else:
            new_pretoken_counts[merged_pretoken] = count

    return new_pretoken_counts


def build_pair_index(pretoken_counts: dict[tuple[bytes, ...], int]):
    words = []
    frequencies = []
    pair_counts = {}
    pair_to_words_ids = {}

    for word_id, (pretoken, freq) in enumerate(pretoken_counts.items()):
        word = list(pretoken)
        words.append(word)
        frequencies.append(freq)

        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
            pair_to_words_ids.setdefault(pair, set()).add(word_id)

    return words, frequencies, pair_counts, pair_to_words_ids


def update_pair_index(best_pair: tuple[bytes, bytes], words: list[list[bytes]], frequencies: list[int], pair_counts: dict[tuple[bytes, bytes], int], pair_to_words_ids: dict[tuple[bytes, bytes], set[int]]):
    affected_word_ids = list(pair_to_words_ids.get(best_pair, set()))

    for word_id in affected_word_ids:
        word = words[word_id]
        freq = frequencies[word_id]

        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] -= freq
            if pair_counts[pair] == 0:
                del pair_counts[pair]
            ids = pair_to_words_ids.get(pair)
            if ids:
                ids.discard(word_id)
                if not ids:
                    del pair_to_words_ids[pair]

        merged_word = merge_pretoken(tuple(word), best_pair)
        words[word_id] = list(merged_word)

        for i in range(len(merged_word) - 1):
            new_pair = (merged_word[i], merged_word[i + 1])
            pair_counts[new_pair] = pair_counts.get(new_pair, 0) + freq
            pair_to_words_ids.setdefault(new_pair, set()).add(word_id)



def train_bpe(input_path: str, special_tokens: list[str], vocab_size: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = initialize_vocab(special_tokens)

    merge: list[tuple[bytes, bytes]] = []

    print("开始预分词（并行读取文件）...")
    pretoken_counts = count_pretokens_parallel(input_path, special_tokens, num_processes=4)
    print(f"预分词完成：{len(pretoken_counts)} 个唯一 pretoken，开始 BPE 合并...")

    words, frequencies, pair_counts, pair_to_word_ids = build_pair_index(pretoken_counts)

    pbar = tqdm(total=vocab_size - len(vocab))
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = find_best_pair(pair_counts)
        merge.append(best_pair)

        update_pair_index(best_pair, words, frequencies, pair_counts, pair_to_word_ids)

        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        pbar.update(1)

    pbar.close()

    return vocab, merge


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens is not None else []

        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}

        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.token_to_id[token_bytes] = new_id

        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        special_token_set = set(self.special_tokens)
        for segment in self._split_on_special_tokens(text):
            if segment in special_token_set:
                ids.append(self.token_to_id[segment.encode("utf-8")])
            else:
                for match in re.finditer(PAT, segment):
                    piece = match.group().encode("utf-8")
                    ids.extend(self._encode_piece(piece))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join(self.vocab[i] for i in ids)
        return text_bytes.decode("utf-8", errors="replace")

    def _split_on_special_tokens(self, text: str) -> list[str]:
        if not self.special_tokens:
            return [text]
        sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(token) for token in sorted_tokens) + ")"
        return re.split(pattern, text)

    def _encode_piece(self, piece: bytes) -> list[int]:
        word = [bytes([byte]) for byte in piece]
        while len(word) >= 2:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank = float("inf")
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.merge_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break

            new_word: list[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

        return [self.token_to_id[unit] for unit in word]
