#!/usr/bin/env python3
"""
语料自动生成器 — 用 LLM 批量合成多样化的 C / CUDA 函数，扩展到万级数据。

流程：
  1. 用 LLM 按类别批量生成干净的 C / CUDA 函数（无 VerCors 注释）
  2. 去重：基于代码结构哈希
  3. 语法校验：GCC（C）/ NVCC（CUDA）
  4. 写入 corpus/c/ 或 corpus/cuda/

用法：
    python code_generator.py                        # C，默认数量
    python code_generator.py --lang cuda --count 500
    python code_generator.py --lang c --count 10000
    python code_generator.py --dry-run
"""

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Optional

import config
from llm_client import call_llm

logger = logging.getLogger(__name__)

# ── 每个语言的扩展名 ──
LANG_EXT = {"c": ".c", "cuda": ".cu"}

# ── CUDA 专用关键词（用于提取函数）──
CUDA_KEYWORDS = r'__global__|__device__|__host__|threadIdx|blockIdx|blockDim|gridDim'


# ============================================================
# Prompt 构建
# ============================================================
def _load_templates(lang: str) -> str:
    key = "cuda_templates_file" if lang == "cuda" else "templates_file"
    path = config.CODE_GEN_CONFIG[key]
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def _build_gen_prompt(category: str, count: int, lang: str) -> str:
    """构建代码生成 prompt，区分 C / CUDA。"""
    templates = _load_templates(lang)
    template_section = ""
    if templates:
        template_section = f"\n## 参考模板（请模仿风格，不要照抄）\n{templates}"

    if lang == "cuda":
        return _cuda_gen_prompt(category, count, template_section)
    else:
        return _c_gen_prompt(category, count, template_section)


def _c_gen_prompt(category: str, count: int, template_section: str) -> str:
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
请生成不同难度和模式变体的函数：
- 不同参数个数
- 不同循环模式
- 不同的条件分支结构
- 不同的变量命名

现在请生成 {count} 个「{category}」类别的 C 函数："""


def _cuda_gen_prompt(category: str, count: int, template_section: str) -> str:
    return f"""请生成 {count} 个不同的 CUDA kernel 函数，属于「{category}」类别。

## CUDA 格式要求
1. 每个函数用 `__global__ void` 声明（GPU kernel）
2. 使用 `threadIdx.x`, `blockIdx.x`, `blockDim.x`, `gridDim.x` 计算线程 ID
3. **绝对不要添加任何 VerCors 注释（/*@ ... @*/）**
4. **绝对不要添加任何普通注释（// 或 /* ... */）**
5. 按顺序输出，每个 kernel 之间用 `---` 分隔
6. 只输出代码，不要任何解释文字
7. 每个 kernel 5-60 行，功能明确，逻辑完整
{template_section}

## 类别说明：{category}
请生成不同难度和模式变体的 CUDA kernel：
- 不同线程索引计算方式（1D / 2D grid）
- 不同的边界检查模式
- 不同的循环展开方式

现在请生成 {count} 个「{category}」类别的 CUDA kernel："""


# ============================================================
# 代码去重与校验
# ============================================================
def _normalize_code(code: str) -> str:
    return re.sub(r'\s+', ' ', code).strip()


def _code_hash(code: str) -> str:
    normalized = _normalize_code(code)
    structural = re.sub(r'\b\d+\b', 'N', normalized)
    return hashlib.md5(structural.encode()).hexdigest()


def _syntax_check(code: str, lang: str) -> tuple[bool, str]:
    """语法检查：C 用 GCC，CUDA 用 NVCC。"""
    if lang == "cuda":
        return _nvcc_check(code)
    return _gcc_check(code)


def _gcc_check(code: str) -> tuple[bool, str]:
    if not code.strip().startswith("#include"):
        code = f"#include <stdlib.h>\n#include <string.h>\n#include <math.h>\n\n{code}"
    try:
        result = subprocess.run(
            ["gcc", "-fsyntax-only", "-xc", "-"],
            input=code, capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0, result.stderr
    except FileNotFoundError:
        return True, "GCC not available, skipping syntax check"
    except Exception as e:
        return True, str(e)


def _nvcc_check(code: str) -> tuple[bool, str]:
    """NVCC 语法检查 — 如果 NVCC 不可用则跳过。"""
    full = f'extern "C" {{\n{code}\n}}'
    try:
        result = subprocess.run(
            ["nvcc", "--cuda", "-fsyntax-only", "-o", os.devnull, "-"],
            input=full, capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0, result.stderr
    except FileNotFoundError:
        return True, "NVCC not available, skipping syntax check"
    except Exception as e:
        return True, str(e)


# ============================================================
# 函数提取
# ============================================================
def _extract_functions(llm_output: str, lang: str) -> list[str]:
    """从 LLM 输出中提取单个函数 / kernel。"""
    blocks = re.split(r'\n---\n|\n## CATEGORY:', llm_output)
    functions = []

    for block in blocks:
        block = block.strip()
        if not block or (block.startswith("##") and len(block) < 50):
            continue

        if lang == "cuda":
            funcs = _extract_cuda_kernels(block)
        else:
            funcs = _extract_c_functions(block)

        functions.extend(funcs)

    return functions


def _extract_c_functions(block: str) -> list[str]:
    funcs = []
    func_match = re.search(
        r'(?:(?:static|inline)\s+)?(?:int|void|float|double|char|long|unsigned|size_t|bool)\s+\w+\s*\([^)]*\)\s*\{.*?\n\}',
        block, re.DOTALL,
    )
    if func_match:
        funcs.append(func_match.group(0).strip())
    elif re.search(r'(?:int|void|float|double|char|long|unsigned)\s+\w+\s*\(', block):
        cleaned = re.sub(r'^#+\s*\w+\s*\n?', '', block).strip()
        if cleaned:
            funcs.append(cleaned)
    return funcs


def _extract_cuda_kernels(block: str) -> list[str]:
    funcs = []
    func_match = re.search(
        r'__global__\s+void\s+\w+\s*\([^)]*\)\s*\{.*?\n\}',
        block, re.DOTALL,
    )
    if func_match:
        funcs.append(func_match.group(0).strip())
    elif re.search(r'__global__|threadIdx|blockIdx', block):
        cleaned = re.sub(r'^#+\s*\w+\s*\n?', '', block).strip()
        if cleaned:
            funcs.append(cleaned)
    return funcs


# ============================================================
# 主生成循环
# ============================================================
def _call_llm_with_retry(messages: list[dict], model_name: str, max_retries: int = 3) -> str:
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


def _scan_existing(corpus_dir: str, ext: str) -> tuple[set, int]:
    """扫描已有文件，返回 (哈希集合, 最大 gen 编号)。"""
    existing_hashes = set()
    counter = 0
    if os.path.isdir(corpus_dir):
        for fname in os.listdir(corpus_dir):
            if not fname.endswith(ext):
                continue
            fpath = os.path.join(corpus_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                existing_hashes.add(_code_hash(f.read()))
            m = re.match(r'gen_(\d+)' + re.escape(ext), fname)
            if m:
                counter = max(counter, int(m.group(1)))
    return existing_hashes, counter


def generate_corpus(
    target_count: int = None,
    gen_model: str = None,
    batch_size: int = None,
    lang: str = "c",
    dry_run: bool = False,
) -> int:
    """
    自动生成 C / CUDA 语料。每次只对一个类别调用 LLM。
    """
    cfg = config.CODE_GEN_CONFIG
    target_count = target_count or cfg["target_count"].get(lang, 5000)
    gen_model = gen_model or config.DEFAULT_MODEL
    batch_size = batch_size or cfg["gen_batch_size"]
    categories = cfg["categories"].get(lang, cfg["categories"]["c"])
    ext = LANG_EXT[lang]

    # 确定输出目录
    if lang == "cuda":
        corpus_dir = config.CORPUS_CUDA_DIR
    else:
        corpus_dir = config.CORPUS_C_DIR
    os.makedirs(corpus_dir, exist_ok=True)

    # 扫描已有（优先扫描目标目录，再 fallback 到 corpus/ 根目录的遗留文件）
    existing_hashes, gen_counter_offset = _scan_existing(corpus_dir, ext)
    existing_count = sum(1 for f in os.listdir(corpus_dir) if f.endswith(ext))

    # 向后兼容：如果 corpus/c/ 为空但 corpus/ 根有遗留 gen_*.c
    if existing_count == 0 and lang == "c":
        parent_dir = config.CORPUS_DIR
        if os.path.isdir(parent_dir):
            parent_hashes, parent_counter = _scan_existing(parent_dir, ext)
            if parent_counter > 0:
                logger.warning(f"corpus/c/ 为空，发现 {parent_counter} 个遗留 gen_*.c 在 corpus/ 根")
                logger.warning("新文件将写入 corpus/c/，建议: mv corpus/gen_*.c corpus/c/")
                existing_hashes.update(parent_hashes)
                gen_counter_offset = parent_counter
    needed = max(0, target_count - existing_count)

    if needed <= 0:
        logger.info(f"已有 {existing_count} 个 {lang} 文件，已达到目标 {target_count}")
        return 0

    logger.info(f"[{lang}] 目标: {target_count}, 当前: {existing_count}, 需生成: {needed}")
    logger.info(f"模型: {gen_model} ({config.get_model_config(gen_model)['model']})")
    logger.info(f"类别: {categories}")

    total_generated = 0
    round_num = 0
    per_call = min(batch_size, 8)
    max_rounds = (needed // (per_call * len(categories))) + 10

    os.makedirs(corpus_dir, exist_ok=True)

    lang_sys = "CUDA C/C++" if lang == "cuda" else "C"

    while total_generated < needed and round_num < max_rounds:
        round_num += 1

        for cat in categories:
            if total_generated >= needed:
                break

            logger.info(f"  [{gen_model}] [{lang}] 类别={cat}, 轮={round_num}, 已生成={total_generated}/{needed}")

            prompt = _build_gen_prompt(cat, per_call, lang)
            messages = [
                {"role": "system", "content": f"你是一位精通 {lang_sys} 的系统编程专家。只输出代码，不输出解释。"},
                {"role": "user", "content": prompt},
            ]

            try:
                llm_output = _call_llm_with_retry(messages, gen_model)
            except Exception as e:
                logger.error(f"  LLM 调用最终失败 [{cat}]: {e}")
                continue

            functions = _extract_functions(llm_output, lang)
            logger.info(f"    提取到 {len(functions)} 个候选")

            for func in functions:
                if total_generated >= needed:
                    break

                lines = func.strip().split("\n")
                if len(lines) < cfg["min_lines"] or len(lines) > cfg["max_lines"]:
                    continue
                if "/*@" in func or "//" in func:
                    continue

                h = _code_hash(func)
                if h in existing_hashes:
                    continue

                if lang == "cuda":
                    if not re.search(CUDA_KEYWORDS, func):
                        continue  # 不含 CUDA 关键字，跳过

                ok, err = _syntax_check(func, lang)
                if not ok:
                    logger.debug(f"    语法错误: {err[:80]}")
                    continue

                existing_hashes.add(h)
                counter = gen_counter_offset + total_generated + 1
                safe_name = f"gen_{counter:05d}{ext}"
                file_path = os.path.join(corpus_dir, safe_name)
                if not dry_run:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(func)
                total_generated += 1

        logger.info(f"  本轮结束，累计: {total_generated}/{needed}")

    logger.info(f"\n[{lang}] 生成完成: 共 {total_generated} 个新文件 → {corpus_dir}")
    return total_generated


# ============================================================
# 去重 & 清洗
# ============================================================
def deduplicate_corpus(lang: str = "c") -> int:
    corpus_dir = config.CORPUS_CUDA_DIR if lang == "cuda" else config.CORPUS_C_DIR
    os.makedirs(corpus_dir, exist_ok=True)
    ext = LANG_EXT[lang]

    files = sorted(f for f in os.listdir(corpus_dir) if f.endswith(ext))
    seen = {}
    removed = 0
    for fname in files:
        fpath = os.path.join(corpus_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            h = _code_hash(f.read())
        if h in seen:
            os.remove(fpath)
            removed += 1
            logger.info(f"  去重删除 [{lang}]: {fname} (重复于 {seen[h]})")
        else:
            seen[h] = fname
    logger.info(f"[{lang}] 去重: 删除 {removed}, 保留 {len(seen)}")
    return removed


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="VerCors 语料自动生成器 — C / CUDA 批量合成"
    )
    parser.add_argument("--count", type=int, default=None, help="目标文件总数")
    parser.add_argument("--lang", type=str, default="c", choices=["c", "cuda"], help="语言: c | cuda")
    parser.add_argument("--model", type=str, default=None,
                        help=f"模型，可用: {list(config.MODEL_REGISTRY.keys())}")
    parser.add_argument("--batch-size", type=int, default=None, help="每类别每次生成数（默认 8）")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不写入文件")
    parser.add_argument("--dedup", action="store_true", help="仅去重")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.dedup:
        deduplicate_corpus(args.lang)
        return

    generate_corpus(
        target_count=args.count,
        gen_model=args.model,
        batch_size=args.batch_size,
        lang=args.lang,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
