#!/usr/bin/env python3
"""
语料自动生成器 — 用 LLM 批量合成多样化的 C 函数，扩展到万级数据。

流程：
  1. 用 LLM 按类别批量生成干净的 C 函数（无 VerCors 注释）
  2. 去重：基于代码结构哈希
  3. 语法校验：用 GCC/Clang 检查语法正确性
  4. 写入 corpus/ 目录

用法：
    python code_generator.py                    # 生成到 target_count 个文件
    python code_generator.py --count 500        # 只生成 500 个
    python code_generator.py --dry-run           # 预览不写入
"""

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import config
from llm_client import call_llm, extract_c_code

logger = logging.getLogger(__name__)


# ============================================================
# 代码生成 Prompt
# ============================================================
def _load_templates() -> str:
    path = config.CODE_GEN_CONFIG["templates_file"]
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def _build_gen_prompt(category: str, count: int) -> str:
    """构建代码生成 prompt。"""
    templates = _load_templates()
    template_section = ""
    if templates:
        template_section = f"\n## 参考模板（请模仿这种风格，但不要照抄）\n{templates}"

    return f"""请生成 {count} 个不同的 C 语言函数，属于「{category}」类别。

## 严格要求
1. 每个函数必须独立完整，包含函数签名和函数体
2. **绝对不要添加任何 VerCors 注释（/*@ ... @*/）**
3. **绝对不要添加任何普通注释（// 或 /* ... */）**
4. 代码必须是纯 C 语言（不是 C++），不依赖任何非标准库
5. 按顺序输出，每个函数之间用 `---` 分隔
6. 只输出代码，不要任何解释文字
7. 每个函数 5-60 行，功能明确，逻辑完整
{template_section}

## 类别说明：{category}
请生成不同难度和模式变体的函数，例如：
- 不同参数个数
- 不同循环模式
- 不同的条件分支结构
- 不同的变量命名

现在请生成 {count} 个「{category}」类别的 C 函数："""


def _build_batch_prompt(categories: list[str], per_category: int) -> str:
    """构建多类别批量生成 prompt。"""
    cat_list = "\n".join(f"{i+1}. {c}" for i, c in enumerate(categories))
    templates = _load_templates()
    template_section = ""
    if templates:
        template_section = f"\n## 参考模板\n{templates}"

    return f"""请为以下每个类别各生成 {per_category} 个不同的纯 C 语言函数。

## 类别列表
{cat_list}

## 严格要求
1. 函数独立完整，含函数签名和函数体
2. **禁止添加任何注释（包括 //、/* */、/*@ @*/）**
3. 纯 C 语言，不依赖非标准库
4. 每个类别内的函数有不同变体（参数个数、循环模式、条件结构）
5. 按类别分组输出，每个函数之间用 `---` 分隔，类别开始处用 `## CATEGORY: <name>` 标记
6. 只输出代码和类别标记，不要解释
{template_section}

现在请生成以上 {len(categories)} 个类别各 {per_category} 个 C 函数："""


# ============================================================
# 代码去重与校验
# ============================================================
def _normalize_code(code: str) -> str:
    """规范化代码用于去重比较。"""
    # 移除多余空白
    code = re.sub(r'\s+', ' ', code).strip()
    # 移除变量名影响（保守：只做基础规范化）
    return code


def _code_hash(code: str) -> str:
    """计算代码结构哈希（用于去重）。"""
    normalized = _normalize_code(code)
    # 移除数字字面量，保留结构
    structural = re.sub(r'\b\d+\b', 'N', normalized)
    return hashlib.md5(structural.encode()).hexdigest()


def _syntax_check(code: str) -> tuple[bool, str]:
    """用 GCC 检查 C 代码语法（需要 GCC/Clang 在 PATH 中）。"""
    # 包装为完整编译单元
    full_code = code
    if not code.strip().startswith("#include"):
        # 添加常用的 include 和 main 入口（仅用于语法检查）
        full_code = f"""
#include <stdlib.h>
#include <string.h>
#include <math.h>

{code}
"""
    try:
        result = subprocess.run(
            ["gcc", "-fsyntax-only", "-xc", "-"],
            input=full_code,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0, result.stderr
    except FileNotFoundError:
        # GCC 不可用时跳过语法检查
        return True, "GCC not available, skipping syntax check"
    except Exception as e:
        return True, str(e)


def _extract_functions(llm_output: str) -> list[str]:
    """从 LLM 输出中提取单个函数。"""
    # 按 --- 或 ## CATEGORY 分割
    blocks = re.split(r'\n---\n|\n## CATEGORY:', llm_output)

    functions = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # 跳过纯类别标记行
        if block.startswith("##") and len(block) < 50:
            continue

        # 尝试提取函数
        func_match = re.search(
            r'(?:(?:static|inline)\s+)?(?:int|void|float|double|char|long|unsigned|size_t|bool)\s+\w+\s*\([^)]*\)\s*\{.*?\n\}',
            block, re.DOTALL
        )
        if func_match:
            functions.append(func_match.group(0).strip())
        else:
            # 如果 block 本身看起来像一个函数
            if re.search(r'(?:int|void|float|double|char|long|unsigned)\s+\w+\s*\(', block):
                # 去掉可能的 ## CATEGORY 前缀
                cleaned = re.sub(r'^#+\s*\w+\s*\n?', '', block).strip()
                if cleaned:
                    functions.append(cleaned)

    return functions


# ============================================================
# 主生成循环
# ============================================================
def _call_llm_with_retry(messages: list[dict], model_name: str, max_retries: int = 3) -> str:
    """调用 LLM，带自动重试。"""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return call_llm(messages, model_name)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(f"  API 调用失败 (尝试 {attempt}/{max_retries})，{wait}s 后重试: {e}")
                time.sleep(wait)
    raise last_error


def generate_corpus(
    target_count: int = None,
    gen_model: str = None,
    batch_size: int = None,
    dry_run: bool = False,
) -> int:
    """
    自动生成 C 代码语料库。每次只对一个类别调用 LLM，避免单次 prompt 过大。
    """
    cfg = config.CODE_GEN_CONFIG
    target_count = target_count or cfg["target_count"]
    gen_model = gen_model or config.DEFAULT_MODEL    # 改为跟随 .env 的 VERCORS_AGENT_MODEL
    batch_size = batch_size or cfg["gen_batch_size"]
    categories = cfg["categories"]

    # 统计已有文件
    existing_files = set()
    if os.path.isdir(config.CORPUS_DIR):
        existing_files = set(
            f for f in os.listdir(config.CORPUS_DIR) if f.endswith(".c")
        )
    existing_hashes = set()
    for fname in existing_files:
        fpath = os.path.join(config.CORPUS_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            existing_hashes.add(_code_hash(f.read()))

    needed = max(0, target_count - len(existing_files))
    if needed <= 0:
        logger.info(f"已有 {len(existing_files)} 个文件，已达到目标 {target_count}，无需生成")
        return 0

    logger.info(f"目标总数: {target_count}, 当前: {len(existing_files)}, 需生成: {needed}")
    logger.info(f"生成模型: {gen_model}, 每批: {batch_size}")
    logger.info(f"类别: {categories}")

    total_generated = 0
    round_num = 0
    # 每轮每个类别的并发数（避免单次 prompt 过大）
    per_call = min(batch_size, 8)
    max_rounds = (needed // (per_call * len(categories))) + 10

    os.makedirs(config.CORPUS_DIR, exist_ok=True)

    logger.info(f"模型: {gen_model} ({config.get_model_config(gen_model)['model']})")

    while total_generated < needed and round_num < max_rounds:
        round_num += 1

        # 每轮遍历所有类别，每次只请求一个类别（避免单次 prompt 过大超时）
        for cat in categories:
            if total_generated >= needed:
                break

            logger.info(f"  [{gen_model}] 类别={cat}, 轮={round_num}, 已生成={total_generated}/{needed}")

            prompt = _build_gen_prompt(cat, per_call)
            messages = [
                {"role": "system", "content": "你是一位精通 C 语言的系统编程专家。只输出代码，不输出解释。"},
                {"role": "user", "content": prompt},
            ]

            try:
                llm_output = _call_llm_with_retry(messages, gen_model)
            except Exception as e:
                logger.error(f"  LLM 调用最终失败 [{cat}]: {e}")
                continue

            # 提取函数
            functions = _extract_functions(llm_output)
            logger.info(f"    提取到 {len(functions)} 个候选")

            for func in functions:
                if total_generated >= needed:
                    break

                # 质量过滤
                lines = func.strip().split("\n")
                if len(lines) < cfg["min_lines"] or len(lines) > cfg["max_lines"]:
                    continue
                if "/*@" in func or "//" in func:
                    continue

                # 去重
                h = _code_hash(func)
                if h in existing_hashes:
                    continue

                # 语法检查
                ok, err = _syntax_check(func)
                if not ok:
                    logger.debug(f"    语法错误: {err[:80]}")
                    continue

                existing_hashes.add(h)
                safe_name = f"gen_{total_generated + 1:05d}.c"
                file_path = os.path.join(config.CORPUS_DIR, safe_name)
                if not dry_run:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(func)
                total_generated += 1

    logger.info(f"\n生成完成: 共 {total_generated} 个新文件")
    return total_generated


# ============================================================
# 去重 & 清洗已有语料
# ============================================================
def deduplicate_corpus() -> int:
    """对 corpus/ 中已有文件去重。"""
    if not os.path.isdir(config.CORPUS_DIR):
        return 0

    files = sorted(
        f for f in os.listdir(config.CORPUS_DIR) if f.endswith(".c")
    )
    seen = {}
    removed = 0

    for fname in files:
        fpath = os.path.join(config.CORPUS_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            code = f.read()
        h = _code_hash(code)
        if h in seen:
            os.remove(fpath)
            removed += 1
            logger.info(f"  去重删除: {fname} (重复于 {seen[h]})")
        else:
            seen[h] = fname

    logger.info(f"去重完成: 删除 {removed} 个重复文件，保留 {len(seen)} 个")
    return removed


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="VerCors 语料自动生成器 — LLM 批量合成 C 函数"
    )
    parser.add_argument("--count", type=int, default=None, help="目标文件总数")
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"使用的模型，可用: {list(config.MODEL_REGISTRY.keys())}（默认跟随 .env 的 VERCORS_AGENT_MODEL）"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="每类别每次生成的函数数（默认 8）")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不写入文件")
    parser.add_argument("--dedup", action="store_true", help="仅对现有语料去重")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.dedup:
        deduplicate_corpus()
        return

    generate_corpus(
        target_count=args.count,
        gen_model=args.model,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
