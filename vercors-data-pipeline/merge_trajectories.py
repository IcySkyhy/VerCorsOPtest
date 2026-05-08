#!/usr/bin/env python3
"""
合并所有 trajectories_c_*.jsonl 和旧 trajectories_*.jsonl（C 语言，不含 cuda），
在每条记录上补全 "model" 字段。模型名从文件名提取。

用法：
    python merge_trajectories.py                          # 合并所有 C 轨迹
    python merge_trajectories.py --lang cuda               # 合并所有 CUDA 轨迹
    python merge_trajectories.py --lang c --out merged.jsonl   # 指定输出文件
    python merge_trajectories.py --dry-run                  # 预览不写入
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict


def find_trajectory_files(lang: str, output_dir: str) -> list[tuple[str, str]]:
    """
    扫描 output/ 目录，返回 [(文件路径, 模型名), ...]。

    支持新命名: trajectories_{lang}_{model}.jsonl
    兼容旧命名: trajectories_{model}.jsonl
    """
    results = []
    if not os.path.isdir(output_dir):
        return results

    for fname in os.listdir(output_dir):
        if not fname.endswith(".jsonl"):
            continue
        base = fname[:-6]  # strip .jsonl

        # 新格式: trajectories_c_deepseek.jsonl 或 trajectories_cuda_deepseek.jsonl
        m = re.match(r'trajectories_(' + re.escape(lang) + r')_(.+)', base)
        if m:
            model_name = m.group(2)
            results.append((os.path.join(output_dir, fname), model_name))
            continue

        # 旧格式: trajectories_deepseek.jsonl（无 lang 标记，假设都是 C）
        if lang == "c":
            m = re.match(r'trajectories_(.+)', base)
            if m:
                model_name = m.group(1)
                # 排除 stats 文件
                if model_name.startswith("stats"):
                    continue
                results.append((os.path.join(output_dir, fname), model_name))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="合并 VerCors 轨迹文件，补全 model 字段"
    )
    parser.add_argument("--lang", type=str, default="c", choices=["c", "cuda"],
                        help="语言: c | cuda")
    parser.add_argument("--out", type=str, default=None,
                        help="输出文件路径（默认 output/trajectories_{lang}_merged.jsonl）")
    parser.add_argument("--dry-run", action="store_true", help="预览不写入")
    args = parser.parse_args()

    # 推断 output 目录: 脚本所在目录下的 output/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")

    files = find_trajectory_files(args.lang, output_dir)

    if not files:
        print(f"未找到任何 {'C' if args.lang == 'c' else 'CUDA'} 轨迹文件")
        print(f"搜索目录: {output_dir}")
        sys.exit(1)

    print(f"找到 {len(files)} 个轨迹文件:")
    for path, model in sorted(files, key=lambda x: x[1]):
        line_count = sum(1 for _ in open(path, "r", encoding="utf-8"))
        print(f"  {os.path.basename(path):45s} → model={model:<15s} ({line_count} 条)")

    if args.dry_run:
        print("\n[预览模式] 不写入文件")
        return

    # 合并
    merged = []
    stats = defaultdict(int)  # model → count

    for path, model in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  警告: 跳过无效 JSON 行: {path}: {line[:80]}...")
                    continue
                # 优先用数据本身的 model/lang（新版 agent 已自动写入）
                # 兼容旧数据：文件名推断的 model 作为 fallback
                record_model = record.get("model", model)
                record_lang = record.get("lang", args.lang)
                record["model"] = record_model
                record["lang"] = record_lang
                merged.append(record)
                stats[record_model] += 1

    # 写入
    out_path = args.out or os.path.join(output_dir, f"trajectories_{args.lang}_merged.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in merged:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n合并完成 → {out_path}")
    print(f"总计: {len(merged)} 条")
    for model, count in sorted(stats.items()):
        print(f"  {model}: {count} 条")


if __name__ == "__main__":
    main()
