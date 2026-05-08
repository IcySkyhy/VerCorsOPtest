#!/usr/bin/env python3
"""
VerCors 数据采集 Agent — ProofWright 风格自动注释生成与验证闭环。

支持多模型（DeepSeek / GLM-5.1）对比，跨模型自动回退。

用法：
    python vercors_agent.py                        # corpus/c/ 下所有 .c，默认模型
    python vercors_agent.py --lang cuda             # corpus/cuda/ 下所有 .cu
    python vercors_agent.py --lang c --file foo.c  # 单个 C 文件
    python vercors_agent.py --model glm             # 使用 GLM-5.1
    python vercors_agent.py --model deepseek --fallback glm  # 主+回退
    python vercors_agent.py --resume                # 断点续传
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime
from typing import Optional

import config
from llm_client import call_llm, call_llm_with_fallback, extract_c_code

# ============================================================
# 日志设置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# Prompt 模板加载
# ============================================================
def load_prompt(name: str) -> str:
    """加载 prompts/ 目录下的模板文件。"""
    path = os.path.join(config.PROMPTS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt 模板不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# 启动时加载模板
SYSTEM_PROMPT = load_prompt("system_prompt.txt")
ANNOTATE_USER_TEMPLATE = load_prompt("annotate_user.txt")
FEEDBACK_USER_TEMPLATE = load_prompt("feedback_user.txt")


# ============================================================
# VerCors 调用与解析
# ============================================================
def run_vercors(file_path: str) -> tuple[bool, str]:
    """
    调用 test_op.sh 验证指定的 .c 文件。
    返回 (是否通过, 完整输出文本)。
    """
    # 确保文件存在
    if not os.path.exists(file_path):
        return False, f"文件不存在: {file_path}"

    # 构建命令：bash test_op.sh <file>
    cmd = ["bash", config.TEST_OP_SH, file_path]
    logger.info(f"  执行验证: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.VERCORS_TIMEOUT,
            cwd=os.path.dirname(config.TEST_OP_SH) or ".",
        )
        output = result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired:
        logger.warning(f"  VerCors 验证超时 ({config.VERCORS_TIMEOUT}s)")
        return False, f"验证超时（>{config.VERCORS_TIMEOUT} 秒），SMT 求解器可能卡死"
    except Exception as e:
        return False, f"VerCors 调用异常: {str(e)}"

    success = is_verification_successful(output)
    return success, output


def is_verification_successful(output: str) -> bool:
    """判断 VerCors 输出是否表示验证通过。"""
    success_markers = [
        "Verification completed successfully",
        "Pass",
        "0 errors",
    ]
    for marker in success_markers:
        if marker.lower() in output.lower():
            return True
    return False


def extract_errors(output: str) -> str:
    """从 VerCors 输出中提取关键错误信息用于反馈。"""
    lines = output.split("\n")
    error_lines = []
    for line in lines:
        lower = line.lower()
        if any(kw in lower for kw in ["error", "fail", "warn", "cannot", "invalid", "unsat"]):
            error_lines.append(line)

    if not error_lines:
        # 如果没有显式错误，返回最后 30 行
        error_lines = lines[-30:]

    return "\n".join(error_lines)


# ============================================================
# 核心 Agent 循环：单文件处理
# ============================================================
def process_file(
    clean_file_path: str,
    model_name: str = None,
    fallback_model: str = None,
    enable_cross_model: bool = True,
    lang: str = "c",
) -> Optional[dict]:
    """
    对单个干净 C 文件执行多轮注释生成 + 验证循环。

    Args:
        clean_file_path: 干净 C 文件路径
        model_name: 主模型名
        fallback_model: 备选模型名（跨模型回退时使用）
        enable_cross_model: 主模型失败后是否启用跨模型回退

    返回：
        成功 → 轨迹字典（直接可写入 JSONL）
        失败 → None
    """
    model_name = model_name or config.DEFAULT_MODEL
    file_id = os.path.splitext(os.path.basename(clean_file_path))[0]
    logger.info(f"{'='*60}")
    logger.info(f"处理文件: {file_id}  |  模型: {model_name}")
    if fallback_model:
        logger.info(f"跨模型回退: {fallback_model}")
    logger.info(f"{'='*60}")

    # 读取干净代码
    with open(clean_file_path, "r", encoding="utf-8") as f:
        clean_code = f.read().strip()

    trajectory_rounds = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # ── Round 1：首轮生成 ──
    user_prompt = ANNOTATE_USER_TEMPLATE.replace("{clean_code}", clean_code)
    messages.append({"role": "user", "content": user_prompt})

    current_model = model_name
    total_retries = config.MAX_RETRIES

    for round_num in range(1, total_retries + 1):
        logger.info(f"  ── Round {round_num}/{total_retries} [{current_model}] ──")

        # 调用 LLM（支持跨模型回退）
        used_model = current_model
        try:
            if config.CROSS_MODEL_FALLBACK and enable_cross_model and fallback_model:
                llm_response, used_model = call_llm_with_fallback(
                    messages, current_model, fallback_model
                )
            else:
                llm_response = call_llm(messages, current_model)
                used_model = current_model
        except Exception as e:
            logger.error(f"所有模型调用均失败: {e}")
            trajectory_rounds.append({
                "round": round_num,
                "model": current_model,
                "llm_raw_response": f"API_ERROR: {e}",
                "annotated_code": "",
                "vercors_output": str(e),
                "passed": False,
            })
            if config.CROSS_MODEL_FALLBACK and enable_cross_model and fallback_model:
                # 切换到备选模型继续
                current_model = fallback_model
            continue

        # 提取代码
        try:
            annotated_code = extract_c_code(llm_response)
        except ValueError:
            logger.warning("代码提取失败，使用原始响应")
            annotated_code = llm_response.strip()

        # 写入临时文件
        os.makedirs(config.TEMP_DIR, exist_ok=True)
        temp_file = os.path.join(
            config.TEMP_DIR, f"{file_id}_round{round_num}.c"
        )
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(annotated_code)

        # 调用 VerCors 验证
        passed, vercors_output = run_vercors(temp_file)

        # 记录本轮
        round_record = {
            "round": round_num,
            "model": used_model,
            "llm_raw_response": llm_response,
            "annotated_code": annotated_code,
            "vercors_output": vercors_output,
            "passed": passed,
        }
        trajectory_rounds.append(round_record)

        if passed:
            logger.info(f"验证通过！模型: {used_model}, 轮数: {round_num}")
            return {
                "id": file_id,
                "lang": lang,
                "model": model_name,           # 用户选的主模型 key（如 "glm-openai"）
                "actual_model": used_model,     # 实际成功的模型（可能与 model_name 不同）
                "original_code": clean_code,
                "final_annotated_code": annotated_code,
                "final_vercors_output": vercors_output,
                "total_rounds": round_num,
                "trajectory": trajectory_rounds,
                "timestamp": datetime.now().isoformat(),
            }

        # 验证失败：构造反馈
        logger.info(f"验证失败 [{used_model}]，准备反馈...")
        errors = extract_errors(vercors_output)
        feedback_prompt = (
            FEEDBACK_USER_TEMPLATE
            .replace("{vercors_error}", errors)
            .replace("{previous_code}", annotated_code)
        )
        messages.append({"role": "assistant", "content": llm_response})
        messages.append({"role": "user", "content": feedback_prompt})

        # 消息长度控制
        if len(messages) > 12:
            messages = [messages[0]] + messages[-11:]

    # 所有重试耗尽
    # 如果主模型失败且启用了跨模型回退，尝试用备选模型独立重试
    if (config.CROSS_MODEL_FALLBACK and enable_cross_model
            and fallback_model and current_model != fallback_model):
        logger.info(f"  🔄 主模型 {model_name} 全部失败，切换到备选 {fallback_model}")
        return process_file(
            clean_file_path,
            model_name=fallback_model,
            fallback_model=None,
            enable_cross_model=False,
            lang=lang,
        )

    logger.warning(f"超过最大重试次数，标记为失败")
    return None


# ============================================================
# 批量处理入口
# ============================================================
def save_trajectory(trajectory: dict, model_name: str = None, lang: str = "c") -> None:
    """将一条成功轨迹追加到 JSONL 文件（按模型+语言分文件）。"""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    model_key = model_name or trajectory.get("model", config.DEFAULT_MODEL)
    out_file = os.path.join(config.OUTPUT_DIR, f"trajectories_{lang}_{model_key}.jsonl")
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(trajectory, ensure_ascii=False) + "\n")


def save_failed(file_id: str, clean_code: str, trajectory: list) -> None:
    """保存失败案例供后续分析。"""
    os.makedirs(config.FAILED_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fail_file = os.path.join(config.FAILED_DIR, f"{file_id}_{timestamp}.json")
    with open(fail_file, "w", encoding="utf-8") as f:
        json.dump({
            "id": file_id,
            "original_code": clean_code,
            "trajectory": trajectory,
            "timestamp": timestamp,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"  失败案例已保存: {fail_file}")


def get_completed_ids(model_name: str = None, lang: str = "c") -> set:
    """读取已有轨迹文件，返回已完成的 file id 集合（兼容新旧命名）。"""
    model_key = model_name or config.DEFAULT_MODEL
    # 新命名: trajectories_{lang}_{model}.jsonl  旧命名: trajectories_{model}.jsonl
    candidates = [
        os.path.join(config.OUTPUT_DIR, f"trajectories_{lang}_{model_key}.jsonl"),
        os.path.join(config.OUTPUT_DIR, f"trajectories_{model_key}.jsonl"),
        # 也检查反向语言（如果旧文件按旧格式存了 CUDA 数据但没有 lang 标记）
    ]
    completed = set()
    for out_file in candidates:
        if not os.path.exists(out_file):
            continue
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    completed.add(record.get("id", ""))
                except json.JSONDecodeError:
                    continue
    if completed:
        logger.info(f"  从已有轨迹加载了 {len(completed)} 个已完成 id "
                     f"(模型={model_key}, 语言={lang})")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VerCors 数据采集 Agent — 多模型支持"
    )
    parser.add_argument("--file", type=str, default=None, help="只处理指定的文件")
    parser.add_argument("--lang", type=str, default="c", choices=["c", "cuda"], help="语言: c | cuda（决定扫描哪个 corpus 子目录）")
    parser.add_argument("--resume", action="store_true", help="跳过已有成功轨迹的文件")
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"使用的模型，可用: {list(config.MODEL_REGISTRY.keys())}"
    )
    parser.add_argument(
        "--fallback", type=str, default=None,
        help="备选模型（主模型失败时回退）"
    )
    parser.add_argument(
        "--no-fallback", action="store_true",
        help="禁用跨模型回退"
    )
    args = parser.parse_args()

    model_name = args.model or config.DEFAULT_MODEL
    lang = args.lang
    fallback_model = args.fallback
    # 自动确定备选模型
    if not args.no_fallback and fallback_model is None:
        available = list(config.MODEL_REGISTRY.keys())
        if len(available) > 1:
            fb = [m for m in available if m != model_name]
            if fb:
                fallback_model = fb[0]

    # 检查 API Key
    cfg = config.get_model_config(model_name)
    if cfg.get("api_key", "").startswith("sk-your-"):
        logger.error(f"请先设置 {model_name.upper()}_API_KEY 环境变量或修改 .env")
        sys.exit(1)

    # 确保所有输出目录存在（包括 corpus 子目录）
    for d in [config.OUTPUT_DIR, config.FAILED_DIR, config.TEMP_DIR,
              config.CORPUS_C_DIR, config.CORPUS_CUDA_DIR]:
        os.makedirs(d, exist_ok=True)

    # 收集待处理文件（按 --lang 只扫描对应子目录，有空则 fallback 到 corpus/ 根）
    if args.file:
        corpus_dir = config.CORPUS_CUDA_DIR if lang == "cuda" else config.CORPUS_C_DIR
        files = [args.file] if os.path.isabs(args.file) else [os.path.join(corpus_dir, args.file)]
    else:
        if lang == "cuda":
            corpus_dir = config.CORPUS_CUDA_DIR
            ext = ".cu"
        else:
            corpus_dir = config.CORPUS_C_DIR
            ext = ".c"

        files = sorted(
            os.path.join(corpus_dir, f)
            for f in os.listdir(corpus_dir)
            if f.endswith(ext)
        )

        # 向后兼容：如果 corpus/c/ 为空但 corpus/ 根有遗留 .c 文件，扫描根目录
        if not files and lang == "c":
            parent_dir = config.CORPUS_DIR
            if os.path.isdir(parent_dir):
                legacy = sorted(
                    os.path.join(parent_dir, f)
                    for f in os.listdir(parent_dir)
                    if f.endswith(".c")
                )
                if legacy:
                    logger.warning(f"corpus/c/ 为空，发现 {len(legacy)} 个遗留文件在 corpus/ 根目录")
                    logger.warning("建议: mv corpus/*.c corpus/c/")
                    files = legacy
                    corpus_dir = parent_dir

        if not files:
            logger.error(f"{corpus_dir} 中没有 {ext} 文件")
            logger.error(f"请先运行: python code_generator.py --lang {lang} --count 500")
            sys.exit(1)

    logger.info(f"语料来源: {corpus_dir}")

    # 断点续传
    if args.resume:
        completed = get_completed_ids(model_name, lang)
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] not in completed]
        logger.info(f"断点续传：跳过 {len(completed)} 个已完成，剩余 {len(files)} 个")
        if not files:
            logger.info("所有文件已处理完毕")
            return

    logger.info(f"准备处理 {len(files)} 个文件 [{lang}]")
    logger.info(f"主模型: {model_name} ({cfg['model']})")
    if fallback_model:
        fb_cfg = config.get_model_config(fallback_model)
        logger.info(f"回退模型: {fallback_model} ({fb_cfg['model']})")
    logger.info(f"最大重试: {config.MAX_RETRIES} 轮")
    logger.info(f"跨模型回退: {'启用' if config.CROSS_MODEL_FALLBACK and not args.no_fallback else '禁用'}")

    stats = {"total": len(files), "passed": 0, "failed": 0, "total_rounds": 0, "model": model_name, "lang": lang}

    for i, file_path in enumerate(files, 1):
        file_id = os.path.splitext(os.path.basename(file_path))[0]
        logger.info(f"\n[{i}/{len(files)}] {file_id}")

        try:
            trajectory = process_file(
                file_path,
                model_name=model_name,
                fallback_model=fallback_model,
                enable_cross_model=not args.no_fallback,
                lang=lang,
            )
        except Exception as e:
            logger.error(f"  处理异常: {e}")
            logger.error(traceback.format_exc())
            stats["failed"] += 1
            continue

        if trajectory:
            save_trajectory(trajectory, model_name, lang)
            stats["passed"] += 1
            stats["total_rounds"] += trajectory["total_rounds"]
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                clean_code = f.read()
            save_failed(file_id, clean_code, [])
            stats["failed"] += 1

        avg_rounds = stats["total_rounds"] / stats["passed"] if stats["passed"] > 0 else 0
        logger.info(f"当前统计: 通过 {stats['passed']}, 失败 {stats['failed']}, "
                     f"平均轮数 {avg_rounds:.1f}")

    # 最终报告
    logger.info(f"\n{'='*60}")
    logger.info(f"最终统计 [{lang}] [{model_name}]")
    logger.info(f"{'='*60}")
    logger.info(f"  总计:   {stats['total']}")
    logger.info(f"  通过:   {stats['passed']} ({stats['passed']/stats['total']*100:.1f}%)")
    logger.info(f"  失败:   {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    if stats["passed"] > 0:
        logger.info(f"  平均轮数: {stats['total_rounds']/stats['passed']:.1f}")
    logger.info(f"  轨迹文件: output/trajectories_{lang}_{model_name}.jsonl")

    stats_out = os.path.join(config.OUTPUT_DIR, f"stats_{lang}_{model_name}.json")
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
