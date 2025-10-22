import argparse
import ast
from typing import List, Sequence


def parse_list(s: str):
    try:
        return ast.literal_eval(s)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Cannot parse list from: {s}. Error: {e}")


def elemwise_ratio(a: Sequence[int], b: Sequence[int]) -> List[int]:
    if len(a) != len(b):
        raise ValueError(f"Mismatched lengths in elementwise ratio: {len(a)} vs {len(b)}")
    out = []
    for ai, bi in zip(a, b):
        if bi == 0:
            raise ValueError("Division by zero in stride ratio")
        if ai % bi != 0:
            # allow non-integer ratio? In nnU-Net we need integer kernel sizes.
            raise ValueError(f"Stride is not divisible per-axis: next={a} prev={b}")
        out.append(ai // bi)
    return out


def derive_pool_op_kernel_sizes(
    down_stride: List[Sequence[int]],
    embedding_patch_size: Sequence[int] = None,
    depths: Sequence[int] = None,
    num_heads: Sequence[int] = None,
    window_size: List[Sequence[int]] = None,
):
    stages = len(down_stride)
    if stages < 1:
        raise ValueError("down_stride must have at least one stage")

    # Sanity: down_stride must be cumulative and strictly increasing or equal per axis
    for i in range(1, stages):
        prev, curr = down_stride[i - 1], down_stride[i]
        if len(prev) != len(curr):
            raise ValueError(f"down_stride dims mismatch at stage {i-1}->{i}: {prev} vs {curr}")
        for d, (p, c) in enumerate(zip(prev, curr)):
            if c % p != 0:
                raise ValueError(
                    f"down_stride must be cumulative. Axis {d} stage {i}: {c} is not a multiple of previous {p}"
                )

    if embedding_patch_size is not None:
        if len(embedding_patch_size) != len(down_stride[0]):
            raise ValueError(
                f"embedding_patch_size dims {embedding_patch_size} must match down_stride[0] dims {down_stride[0]}"
            )
        # In this codebase, Encoder.input_resolution uses pretrain_img_size // down_stride[i].
        # So the first stage stride equals the patch embedding stride along each axis.
        if list(embedding_patch_size) != list(down_stride[0]):
            raise ValueError(
                f"Expected down_stride[0] == embedding_patch_size. Got {down_stride[0]} vs {embedding_patch_size}"
            )

    if depths is not None and len(depths) != stages:
        raise ValueError(
            f"depths length ({len(depths)}) must equal len(down_stride) ({stages}) (one per stage)."
        )
    if num_heads is not None and len(num_heads) != stages:
        raise ValueError(
            f"num_heads length ({len(num_heads)}) must equal len(down_stride) ({stages}) (one per stage)."
        )

    # pool_op_kernel_sizes are per transition between stages (stages-1 entries)
    pool = []
    for i in range(stages - 1):
        prev = down_stride[i]
        nxt = down_stride[i + 1]
        pool.append(elemwise_ratio(nxt, prev))

    # Optional checks with window sizes: input_resolution = crop_size // down_stride[i].
    # The model pads to multiples of window_size so we just warn instead of failing.
    if window_size is not None:
        if len(window_size) != stages:
            print(
                f"[warn] window_size length ({len(window_size)}) != stages ({stages}). Skipping window checks."
            )
        # else: cannot validate without crop_size; user can check separately.

    return pool


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Derive nnU-Net style pool_op_kernel_sizes from nnFormer params: "
            "down_stride, embedding_patch_size, depths, num_heads, window_size."
        )
    )
    ap.add_argument("--down-stride", type=parse_list, required=True,
                    help="JSON/Python list of per-stage cumulative strides, e.g. [[1,4,4],[1,8,8],[2,16,16],[4,32,32]]")
    ap.add_argument("--embedding-patch-size", type=parse_list, required=False,
                    help="JSON/Python list like [1,4,4]; should equal down_stride[0]")
    ap.add_argument("--depths", type=parse_list, required=False,
                    help="Per-stage depths (unused except for length check)")
    ap.add_argument("--num-heads", type=parse_list, required=False,
                    help="Per-stage num_heads (unused except for length check)")
    ap.add_argument("--window-size", type=parse_list, required=False,
                    help="Per-stage window sizes (unused, only for optional checks)")

    args = ap.parse_args()

    pool = derive_pool_op_kernel_sizes(
        down_stride=args.down_stride,
        embedding_patch_size=args.embedding_patch_size,
        depths=args.depths,
        num_heads=args.num_heads,
        window_size=args.window_size,
    )
    print(pool)


if __name__ == "__main__":
    main()

