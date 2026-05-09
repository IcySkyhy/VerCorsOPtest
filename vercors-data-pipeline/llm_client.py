"""
统一 LLM 客户端 — 支持 DeepSeek (openai) 和 GLM-5.1 (zai-sdk)。

特性：
  - 统一 call_llm(messages, model_name) 接口
  - GLM-5.1 自动启用深度思考 (thinking={type: "enabled"})
  - 跨模型自动回退：主模型失败时切换到备选
  - 与 vercors_agent.py 无缝集成
"""

import logging
import re
from typing import Optional

from openai import OpenAI

import config

logger = logging.getLogger(__name__)

# ============================================================
# 客户端缓存
# ============================================================
_openai_clients: dict[str, OpenAI] = {}
_zai_client = None  # GLM 专用 zai-sdk 客户端，懒加载


def _get_openai_client(model_name: str) -> OpenAI:
    """获取或创建 openai 兼容客户端（DeepSeek / GLM-openai / gpt-5.5 responses）。"""
    if model_name not in _openai_clients:
        cfg = config.get_model_config(model_name)
        provider = cfg.get("provider", "openai")
        if provider not in ("openai", "responses"):
            raise ValueError(f"模型 {model_name} 的 provider={provider} 不支持 openai 客户端")
        _openai_clients[model_name] = OpenAI(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            timeout=cfg["timeout"],
        )
    return _openai_clients[model_name]


def _get_zai_client():
    """懒加载 zai-sdk 客户端（GLM-5.1），使用 coding 套餐端点。"""
    global _zai_client
    if _zai_client is None:
        cfg = config.get_model_config("glm")

        # zai-sdk 通过环境变量 ZHIPUAI_BASE_URL 读取自定义端点
        # 必须在 import 前设置，否则 SDK 会用默认的 /api/paas/v4
        import os as _os
        _os.environ.setdefault("ZHIPUAI_BASE_URL", cfg["base_url"])

        try:
            from zai import ZhipuAiClient
        except ImportError:
            raise ImportError(
                "GLM 模型需要 zai-sdk，请执行: pip install zai-sdk"
            )

        # 尝试显式传 base_url（SDK >= 0.2.0 支持）
        try:
            _zai_client = ZhipuAiClient(
                api_key=cfg["api_key"],
                base_url=cfg["base_url"],
            )
        except TypeError:
            # 旧版 SDK 不支持 base_url 参数，回退到纯 api_key
            _zai_client = ZhipuAiClient(api_key=cfg["api_key"])

    return _zai_client


# ============================================================
# 统一调用接口
# ============================================================
def call_llm(
    messages: list[dict],
    model_name: str = None,
) -> str:
    """
    调用指定模型，返回回复文本。

    Args:
        messages: 标准 messages 列表
        model_name: "deepseek" | "glm"，默认使用 config.DEFAULT_MODEL

    Returns:
        模型回复文本
    """
    model_name = model_name or config.DEFAULT_MODEL
    cfg = config.get_model_config(model_name)
    provider = cfg.get("provider", "openai")

    if provider == "zai":
        return _call_glm(messages, cfg)
    elif provider == "responses":
        return _call_responses(messages, model_name, cfg)
    else:
        return _call_openai(messages, model_name, cfg)


def _call_responses(messages: list[dict], model_name: str, cfg: dict) -> str:
    """通过 OpenAI Responses API 调用（如 gpt-5.5 代理），失败时回退到 Chat Completions。"""
    client = _get_openai_client(model_name)
    # 尝试 Responses API
    try:
        response = client.responses.create(
            model=cfg["model"],
            input=_messages_to_text(messages),
            temperature=cfg["temperature"],
            max_output_tokens=cfg["max_tokens"],
        )
        return response.output_text
    except Exception as e:
        logger.warning(f"Responses API 调用失败 ({e})，回退到 Chat Completions")
        # 许多代理实际只支持 /v1/chat/completions
        return _call_openai(messages, model_name, cfg)


def _messages_to_text(messages: list[dict]) -> str:
    """将 messages 列表转为纯文本（Responses API 输入格式）。"""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"[System Instruction]\n{content}")
        elif role == "user":
            parts.append(content)
        elif role == "assistant":
            parts.append(content)
    return "\n\n".join(parts)


def _call_openai(messages: list[dict], model_name: str, cfg: dict) -> str:
    """通过 openai 兼容接口调用（DeepSeek / GLM-openai）。"""
    client = _get_openai_client(model_name)
    kwargs = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": cfg["temperature"],
        "max_tokens": cfg["max_tokens"],
    }
    # 透传 extra_body（如 thinking、reasoning_effort）
    extra = cfg.get("extra_body")
    if extra:
        kwargs["extra_body"] = extra
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def _call_glm(messages: list[dict], cfg: dict) -> str:
    """通过 zai-sdk 调用 GLM-5.1，自动启用深度思考。"""
    client = _get_zai_client()

    # GLM-5.1 推理/编码任务建议开启深度思考
    kwargs = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": cfg["temperature"],
        "max_tokens": cfg["max_tokens"],
    }

    # 深度思考（仅首轮 system+user 时最有效）
    thinking = cfg.get("thinking")
    if thinking is not None:
        kwargs["thinking"] = thinking

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


# ============================================================
# 跨模型回退调用
# ============================================================
def call_llm_with_fallback(
    messages: list[dict],
    primary_model: str = None,
    fallback_model: str = None,
) -> tuple[str, str]:
    """
    先尝试主模型，失败时自动切换到备选模型。

    Returns:
        (回复文本, 实际使用的模型名)
    """
    primary_model = primary_model or config.DEFAULT_MODEL

    # 确定备选模型
    if fallback_model is None:
        available = list(config.MODEL_REGISTRY.keys())
        fallback_model = available[1] if len(available) > 1 and available[0] == primary_model else available[0]

    # 尝试主模型
    try:
        result = call_llm(messages, primary_model)
        return result, primary_model
    except Exception as e:
        logger.warning(f"主模型 {primary_model} 调用失败: {e}")
        logger.info(f"回退到备选模型: {fallback_model}")

    # 尝试备选模型
    try:
        result = call_llm(messages, fallback_model)
        return result, fallback_model
    except Exception as e:
        raise RuntimeError(
            f"主模型 {primary_model} 和备选 {fallback_model} 均调用失败: {e}"
        )


# ============================================================
# C 代码提取（从 LLM 输出中）
# ============================================================
def extract_c_code(llm_output: str) -> str:
    """从 LLM 输出中提取 C / CUDA 代码（兼容多种格式）。"""
    # 策略 1：匹配 ```c / ```cu / ```cuda ... ``` 代码块
    m = re.search(r"```(?:c|cu|cuda)\s*\n(.*?)```", llm_output, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 策略 2：匹配 ``` ... ``` 无语言标记
    m = re.search(r"```\s*\n(.*?)```", llm_output, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if "/*@" in code or "//@" in code or "int " in code or "void " in code or "__global__" in code:
            return code

    # 策略 3：CUDA //@ 格式（无代码块包裹）
    if "//@" in llm_output:
        return llm_output.strip()

    # 策略 4：C 格式 /*@（无代码块包裹）
    if "/*@" in llm_output:
        return llm_output.strip()

    raise ValueError("无法从 LLM 输出中提取 C/CUDA 代码块")
