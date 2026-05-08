#!/usr/bin/env python3
"""
将旧 per-model 轨迹文件合并到统一 total 文件，补全 model/lang 字段。

用法：
    python merge_trajectories.py                          # 合并所有 C 轨迹到 trajectories_c_total.jsonl
    python merge_trajectories.py --lang cuda               # CUDA 轨迹
    python merge_trajectories.py --dry-run                  # 预览不写入
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict


def find_source_files(lang: str, output_dir: str, total_name: str) -> list[tuple[str, str]]:
    """
    扫描 output/ 中除 total 文件外的所有 JSONL。

    支持格式:
      trajectories_{lang}_{model}.jsonl  （新 per-model）
      trajectories_{model}.jsonl         （旧 per-model，lang=c 时）
    跳过: trajectories_{lang}_total.jsonl（自身）
    """
    results = []
    if not os.path.isdir(output_dir):
        return results

    for fname in os.listdir(output_dir):
        if not fname.endswith(".jsonl"):
            continue
        if fname == total_name:
            continue          # 跳过自身
        base = fname[:-6]

        # 新格式: trajectories_c_glm-openai.jsonl
        m = re.match(r'trajectories_(' + re.escape(lang) + r')_(.+)', base)
        if m:
            model_name = m.group(2)
            results.append((os.path.join(output_dir, fname), model_name))
            continue

        # 旧格式: trajectories_glm.jsonl（无 lang 标记，仅 C 时匹配）
        if lang == "c":
            m = re.match(r'trajectories_(.+)', base)
            if m:
                model_name = m.group(1)
                if model_name.startswith("stats"):
                    continue
                results.append((os.path.join(output_dir, fname), model_name))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="将 per-model 轨迹合并到统一 total 文件"
    )
    parser.add_argument("--lang", type=str, default="c", choices=["c", "cuda"],
                        help="语言: c | cuda")
    parser.add_argument("--out", type=str, default=None,
                        help="输出文件路径（默认 output/trajectories_{lang}_total.jsonl）")
    parser.add_argument("--dry-run", action="store_true", help="预览不写入")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    total_name = f"trajectories_{args.lang}_total.jsonl"
    out_path = args.out or os.path.join(output_dir, total_name)

    files = find_source_files(args.lang, output_dir, total_name)

    if not files:
        print(f"未找到需要合并的 {'C' if args.lang == 'c' else 'CUDA'} per-model 轨迹文件")
        print(f"搜索目录: {output_dir}")
        sys.exit(1)

    print(f"找到 {len(files)} 个源文件:")
    for path, model in sorted(files, key=lambda x: x[1]):
        line_count = sum(1 for _ in open(path, "r", encoding="utf-8"))
        print(f"  {os.path.basename(path):45s} → model={model:<15s} ({line_count} 条)")

    if args.dry_run:
        print("\n[预览模式] 不写入文件")
        return

    # 先加载 total 已有数据（去重用）
    seen_ids = set()
    existing = []
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    seen_ids.add(rec.get("id", ""))
                    existing.append(rec)
                except json.JSONDecodeError:
                    pass
        print(f"  total 文件已有 {len(existing)} 条记录")

    # 合并新数据
    stats = defaultdict(int)
    new_count = 0

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
                rid = record.get("id", "")
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)

                record_model = record.get("model", model)
                record_lang = record.get("lang", args.lang)
                record["model"] = record_model
                record["lang"] = record_lang

                existing.append(record)
                stats[record_model] += 1
                new_count += 1

    # 写入
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in existing:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n合并完成 → {out_path}")
    print(f"新增: {new_count} 条 | 总计: {len(existing)} 条")
    for model, count in sorted(stats.items()):
        print(f"  {model}: +{count} 条")


if __name__ == "__main__":
    main()
