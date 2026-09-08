"""Generate text from a Transformer checkpoint and BPE tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_basics.generation import generate_tokens
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

END_OF_TEXT = "<|endoftext|>"


def default_device() -> str:
    """Choose the best accelerator available to this PyTorch installation."""

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a completion from a trained TransformerLM checkpoint."
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--merges-path", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=default_device())

    model = parser.add_argument_group("model architecture")
    model.add_argument("--context-length", type=int, default=256)
    model.add_argument("--d-model", type=int, default=512)
    model.add_argument("--num-layers", type=int, default=4)
    model.add_argument("--num-heads", type=int, default=16)
    model.add_argument("--d-ff", type=int, default=1344)
    model.add_argument("--rope-theta", type=float, default=10_000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_files(
        str(args.vocab_path),
        str(args.merges_path),
        special_tokens=[END_OF_TEXT],
    )
    end_token_id = tokenizer.token_to_id[END_OF_TEXT.encode("utf-8")]

    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )
    checkpoint = torch.load(
        args.checkpoint_path,
        weights_only=True,
        map_location=args.device,
    )
    model.load_state_dict(checkpoint["model"])

    torch.manual_seed(args.seed)
    prompt_tokens = tokenizer.encode(args.prompt)
    # GPT-style tokenizers use the end-of-text token to begin generation when
    # there is no textual prompt. It is context only and is not printed.
    model_prompt_tokens = prompt_tokens or [end_token_id]
    completion_tokens = generate_tokens(
        model,
        model_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        end_token_id=end_token_id,
    )
    visible_completion = completion_tokens
    if visible_completion and visible_completion[-1] == end_token_id:
        visible_completion = visible_completion[:-1]
    print(tokenizer.decode(prompt_tokens + visible_completion))


if __name__ == "__main__":
    main()
