#!/usr/bin/env python3
"""
VerCors 数据采集 Agent — ProofWright 风格自动注释生成与验证闭环。

用 DeepSeek v4 Pro 作为 Teacher 模型，对无注释的 C 代码自动添加 VerCors 契约注释，
通过「生成 → 验证 → 报错反馈 → 修正」的多轮循环，采集成功的训练轨迹。

用法：
    python vercors_agent.py                    # 处理 corpus/ 下所有 .c 文件
    python vercors_agent.py --file foo.c       # 只处理指定文件
    python vercors_agent.py --resume            # 跳过已有轨迹的文件
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

from openai import OpenAI

import config

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
# DeepSeek API 客户端
# ============================================================
_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """懒加载 DeepSeek API 客户端。"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            timeout=config.API_TIMEOUT,
        )
    return _client


def call_deepseek(
    messages: list[dict],
    model: str = config.DEEPSEEK_MODEL,
    temperature: float = config.API_TEMPERATURE,
    max_tokens: int = config.API_MAX_TOKENS,
) -> str:
    """调用 DeepSeek API，返回模型回复文本。"""
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def extract_c_code(llm_output: str) -> str:
    """从 LLM 输出中提取 C 代码（兼容多种格式）。"""
    # 策略 1：匹配 ```c ... ``` 代码块
    pattern = r"```c\s*\n(.*?)```"
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 策略 2：匹配 ``` ... ``` 无语言标记
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # 启发式判断：包含 /*@ 则认为是 C 代码
        if "/*@" in code or "int " in code or "void " in code:
            return code

    # 策略 3：直接返回整个输出（希望 LLM 只输出了代码）
    if "/*@" in llm_output:
        return llm_output.strip()

    raise ValueError("无法从 LLM 输出中提取 C 代码块")


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
def process_file(clean_file_path: str) -> Optional[dict]:
    """
    对单个干净 C 文件执行多轮注释生成 + 验证循环。

    返回：
        成功 → 轨迹字典（直接可写入 JSONL）
        失败 → None
    """
    file_id = os.path.splitext(os.path.basename(clean_file_path))[0]
    logger.info(f"\n{'='*60}")
    logger.info(f"处理文件: {file_id}")
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

    for round_num in range(1, config.MAX_RETRIES + 1):
        logger.info(f"  ── Round {round_num}/{config.MAX_RETRIES} ──")

        # 调用 DeepSeek
        try:
            llm_response = call_deepseek(messages)
        except Exception as e:
            logger.error(f"  DeepSeek API 调用失败: {e}")
            # API 错误视为本轮失败，但如果还有重试机会就继续
            trajectory_rounds.append({
                "round": round_num,
                "llm_raw_response": f"API_ERROR: {e}",
                "annotated_code": "",
                "vercors_output": str(e),
                "passed": False,
            })
            continue

        # 提取代码
        try:
            annotated_code = extract_c_code(llm_response)
        except ValueError as e:
            logger.warning(f"  代码提取失败: {e}")
            # 将整个响应当作代码试试
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

        # 记录本轮到轨迹
        round_record = {
            "round": round_num,
            "llm_raw_response": llm_response,
            "annotated_code": annotated_code,
            "vercors_output": vercors_output,
            "passed": passed,
        }
        trajectory_rounds.append(round_record)

        if passed:
            logger.info(f"  ✅ 验证通过！总轮数: {round_num}")
            return {
                "id": file_id,
                "original_code": clean_code,
                "final_annotated_code": annotated_code,
                "final_vercors_output": vercors_output,
                "total_rounds": round_num,
                "trajectory": trajectory_rounds,
                "timestamp": datetime.now().isoformat(),
            }

        # 验证失败：构造反馈消息
        logger.info(f"  ❌ 验证失败，准备反馈...")
        errors = extract_errors(vercors_output)
        feedback_prompt = (
            FEEDBACK_USER_TEMPLATE
            .replace("{vercors_error}", errors)
            .replace("{previous_code}", annotated_code)
        )
        messages.append({"role": "assistant", "content": llm_response})
        messages.append({"role": "user", "content": feedback_prompt})

        # 控制消息长度，防止超出上下文窗口（DeepSeek 上下文 128K，保守截断）
        if len(messages) > 12:
            # 保留 system + 最近 6 轮（12 条消息）
            messages = [messages[0]] + messages[-11:]

    # 所有重试耗尽
    logger.warning(f"  💀 超过最大重试次数 ({config.MAX_RETRIES})，标记为失败")
    return None


# ============================================================
# 批量处理入口
# ============================================================
def save_trajectory(trajectory: dict) -> None:
    """将一条成功轨迹追加到 JSONL 文件。"""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(config.TRAJECTORIES_FILE, "a", encoding="utf-8") as f:
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


def get_completed_ids() -> set:
    """读取已有轨迹文件，返回已完成的 file id 集合（支持断点续传）。"""
    if not os.path.exists(config.TRAJECTORIES_FILE):
        return set()
    completed = set()
    with open(config.TRAJECTORIES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                completed.add(record.get("id", ""))
            except json.JSONDecodeError:
                continue
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VerCors 数据采集 Agent — DeepSeek 驱动"
    )
    parser.add_argument(
        "--file", type=str, default=None, help="只处理指定的 .c 文件"
    )
    parser.add_argument(
        "--resume", action="store_true", help="跳过已有成功轨迹的文件"
    )
    args = parser.parse_args()

    # 检查 API Key
    if config.DEEPSEEK_API_KEY == "sk-your-api-key-here":
        logger.error("请先设置 DEEPSEEK_API_KEY 环境变量或修改 config.py")
        sys.exit(1)

    # 确保目录存在
    for d in [config.OUTPUT_DIR, config.FAILED_DIR, config.TEMP_DIR]:
        os.makedirs(d, exist_ok=True)

    # 收集待处理文件
    if args.file:
        files = [args.file] if os.path.isabs(args.file) else [os.path.join(config.CORPUS_DIR, args.file)]
    else:
        files = sorted([
            os.path.join(config.CORPUS_DIR, f)
            for f in os.listdir(config.CORPUS_DIR)
            if f.endswith(".c")
        ])
        if not files:
            logger.error(f"corpus/ 目录中没有 .c 文件: {config.CORPUS_DIR}")
            sys.exit(1)

    # 断点续传
    if args.resume:
        completed = get_completed_ids()
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] not in completed]
        logger.info(f"断点续传模式：跳过 {len(completed)} 个已完成，剩余 {len(files)} 个待处理")
        if not files:
            logger.info("所有文件已处理完毕，无需继续")
            return

    logger.info(f"准备处理 {len(files)} 个文件")
    logger.info(f"语料目录: {config.CORPUS_DIR}")
    logger.info(f"输出目录: {config.OUTPUT_DIR}")
    logger.info(f"模型: {config.DEEPSEEK_MODEL}")
    logger.info(f"最大重试轮数: {config.MAX_RETRIES}")

    # 统计
    stats = {"total": len(files), "passed": 0, "failed": 0, "total_rounds": 0}

    for i, file_path in enumerate(files, 1):
        file_id = os.path.splitext(os.path.basename(file_path))[0]
        logger.info(f"\n[{i}/{len(files)}] {file_id}")

        try:
            trajectory = process_file(file_path)
        except Exception as e:
            logger.error(f"  处理异常: {e}")
            logger.error(traceback.format_exc())
            stats["failed"] += 1
            continue

        if trajectory:
            save_trajectory(trajectory)
            stats["passed"] += 1
            stats["total_rounds"] += trajectory["total_rounds"]
        else:
            # 读取原始代码以保存失败案例
            with open(file_path, "r", encoding="utf-8") as f:
                clean_code = f.read()
            save_failed(file_id, clean_code, [])
            stats["failed"] += 1

        # 实时统计
        avg_rounds = stats["total_rounds"] / stats["passed"] if stats["passed"] > 0 else 0
        logger.info(f"  📊 当前统计: 通过 {stats['passed']}, 失败 {stats['failed']}, "
                     f"平均轮数 {avg_rounds:.1f}")

    # ── 最终报告 ──
    logger.info(f"\n{'='*60}")
    logger.info("最终统计")
    logger.info(f"{'='*60}")
    logger.info(f"  总计:   {stats['total']}")
    logger.info(f"  通过:   {stats['passed']} ({stats['passed']/stats['total']*100:.1f}%)")
    logger.info(f"  失败:   {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    if stats["passed"] > 0:
        logger.info(f"  平均轮数: {stats['total_rounds']/stats['passed']:.1f}")
    logger.info(f"  轨迹文件: {config.TRAJECTORIES_FILE}")

    # 保存统计
    with open(config.STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
