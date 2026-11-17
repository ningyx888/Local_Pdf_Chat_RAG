"""
分块实验脚本（不依赖 LLM / 不建索引）

用法：
    uv run python tools/inspect_chunks.py "挖掘机维修案例(样例）.pdf"
    uv run python tools/inspect_chunks.py "某文件.pdf" --sizes 200 400 800
    uv run python tools/inspect_chunks.py "某文件.pdf" --show 3 --overlap 40

它只做四件事：
    1. 调 core.document_loader.extract_text 看能提出多少字
    2. 调 core.text_splitter.split_text 在不同 chunk_size 下切
    3. 打印长度分布 + 相邻块是否真的重叠
    4. 判断块尾是否被"切在半句话上"
"""

import argparse
import os
import sys

# 让脚本能从仓库根目录导入 core.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.document_loader import extract_text          # noqa: E402
from core.text_splitter import split_text             # noqa: E402


def describe_lengths(chunks):
    lens = sorted(len(c) for c in chunks)
    n = len(lens)
    total = sum(lens)
    short = sum(1 for x in lens if x < 50)
    return {
        "n": n,
        "min": lens[0] if lens else 0,
        "max": lens[-1] if lens else 0,
        "avg": (total / n) if n else 0.0,
        "median": lens[n // 2] if lens else 0,
        "short": short,
        "chars": total,
    }


def detect_overlap(a, b, expected):
    """相邻两块 a、b 的开头是否复用了 a 的结尾。返回 (实测重叠字符数, 是否达到 expected)。"""
    if not a or not b:
        return 0, False
    limit = min(len(a), len(b), max(expected * 2, 1))
    best = 0
    for k in range(limit, 0, -1):
        if a.endswith(b[:k]):
            best = k
            break
    return best, best >= expected


def cut_mid_sentence(chunk):
    """块尾不是句末标点，就认为被切断了。"""
    if not chunk:
        return False
    tail = chunk.rstrip()
    return not tail.endswith(("。", "！", "？", ".", "!", "?", ";", "；"))


def main():
    ap = argparse.ArgumentParser(description="看看 PDF 被切成什么样")
    ap.add_argument("path", help="待分析的文档路径")
    ap.add_argument("--sizes", nargs="+", type=int, default=[200, 400, 800],
                    help="要对比的 chunk_size，默认 200 400 800")
    ap.add_argument("--overlap", type=int, default=40, help="chunk_overlap，默认 40")
    ap.add_argument("--show", type=int, default=2, help="打印前 N 块全文，默认 2")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"[错误] 文件不存在: {args.path}")
        return 1

    text = extract_text(args.path)
    if not text:
        print("[错误] extract_text 返回空字符串。可能是扫描件（无文字层）或不支持的格式。")
        return 1

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    print(f"文件      : {args.path}")
    print(f"提取字符数: {len(text):,}")
    print(f"段落数    : {len(paragraphs):,}   (按空行分隔)")
    print()

    for size in args.sizes:
        chunks = split_text(text, chunk_size=size, chunk_overlap=args.overlap)
        s = describe_lengths(chunks)
        print(f"--- chunk_size={size}, overlap={args.overlap} ---")
        print(f"  块数 {s['n']:>4} | 最短 {s['min']:>4} | 中位 {s['median']:>4} | "
              f"最长 {s['max']:>4} | 平均 {s['avg']:.0f} | 短块(<50字) {s['short']}")

        if len(chunks) > 1:
            overlaps, cut = [], 0
            for a, b in zip(chunks, chunks[1:]):
                got, ok = detect_overlap(a, b, args.overlap)
                overlaps.append((got, ok))
            cut = sum(1 for c in chunks if cut_mid_sentence(c))
            hit = sum(1 for _, ok in overlaps if ok)
            print(f"  相邻重叠达标 {hit}/{len(overlaps)} 对 | 块尾半句截断 {cut}/{len(chunks)} 块")
        print()

    # 原文抽样
    base = split_text(text, chunk_size=args.sizes[0], chunk_overlap=args.overlap)
    show = min(args.show, len(base))
    for i in range(show):
        c = base[i]
        print(f"=== 块 {i}（{len(c)} 字符，来源于 chunk_size={args.sizes[0]}）===")
        print(c)
        print()

    # 重叠证据：第一对
    if len(base) > 1:
        got, ok = detect_overlap(base[0], base[1], args.overlap)
        print("=== 相邻块重叠检查（块0 与 块1）===")
        if got:
            print(f"实测复用 {got} 字符（{ok and '达到' or '未达到'} overlap={args.overlap}）")
            print(f"块0 结尾: ...{base[0][-got:]!r}")
            print(f"块1 开头: {base[1][:got]!r}...")
        else:
            print(f"未检测到重叠：说明块0 结尾已有天然段落边界，overlap={args.overlap} 没被触发。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())