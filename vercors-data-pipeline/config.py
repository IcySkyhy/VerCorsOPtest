"""
VerCors 数据采集 Agent — 配置文件
支持 DeepSeek / GLM 多模型对比，以及大规模语料自动合成。
"""

import os

# ── 自动加载 .env 文件（如果 python-dotenv 已安装）──
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env_path)
except ImportError:
    pass

# ============================================================
# 模型注册表
# ============================================================
#   - DeepSeek:  openai 兼容接口，pip install openai
#   - GLM:      zai-sdk 专用接口，pip install zai-sdk
# ============================================================
MODEL_REGISTRY = {
    "deepseek": {
        "provider": "openai",                # 使用 openai 兼容客户端
        "api_key": os.environ.get("DEEPSEEK_API_KEY", "sk-your-api-key-here"),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-pro",
        "temperature": 0.3,
        "max_tokens": 65536,
        "timeout": 180,
    },
    "glm": {
        "provider": "zai",                   # 使用 zai-sdk 客户端
        "api_key": os.environ.get("GLM_API_KEY", "sk-your-glm-key-here"),
        "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
        "model": "glm-5.1",
        "temperature": 0.3,
        "max_tokens": 65536,                 # GLM-5.1 最大输出 128K
        "timeout": 120,
        # GLM-5.1 深度思考（编码任务建议开启）
        "thinking": {"type": "enabled"},
    },
    # 备选：如果 zai-sdk URL 有问题，用 openai 兼容路径直连 coding 端点
    "glm-openai": {
        "provider": "openai",
        "api_key": os.environ.get("GLM_API_KEY", "sk-your-glm-key-here"),
        "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
        "model": "glm-5.1",
        "temperature": 0.3,
        "max_tokens": 65536,
        "timeout": 120,
    },
}

# ── 默认使用的模型（可通过 --model 命令行覆盖）──
DEFAULT_MODEL = os.environ.get("VERCORS_AGENT_MODEL", "deepseek")

# ── 跨模型回退：主模型失败后，尝试用备选模型 ──
CROSS_MODEL_FALLBACK = False          # 是否启用跨模型回退
FALLBACK_RETRIES = 5                 # 回退到备选模型后的最大重试轮数

# ── 兼容旧版调用（单模型模式）──
def get_model_config(name: str = None) -> dict:
    """获取指定模型的配置字典。"""
    key = name or DEFAULT_MODEL
    if key not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {key}，可用: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key]

# 向后兼容别名
_ds = MODEL_REGISTRY["deepseek"]
DEEPSEEK_API_KEY = _ds["api_key"]
DEEPSEEK_BASE_URL = _ds["base_url"]
DEEPSEEK_MODEL = _ds["model"]
API_TEMPERATURE = _ds["temperature"]
API_MAX_TOKENS = _ds["max_tokens"]
API_TIMEOUT = _ds["timeout"]

# ============================================================
# VerCors 工具路径（服务器上的绝对路径）
# ============================================================
TEST_OP_SH = "/workspace/I/qimeng5/huyan/VerCorsOPtest/test_op.sh"
VERCORS_BIN = "/workspace/I/qimeng5/huyan/download/usr/share/vercors/vercors"
VERCORS_TIMEOUT = 120

# ============================================================
# Agent 循环参数
# ============================================================

MAX_RETRIES = 10            # 最大重试轮数（首轮 + 4 次反馈修正）
TEMP_DIR = "temp"    
     
# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "corpus")       # 干净 C 代码库
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")     # Prompt 模板
OUTPUT_DIR = os.path.join(BASE_DIR, "output")       # 成功轨迹
FAILED_DIR = os.path.join(BASE_DIR, "failed")       # 失败案例
TEMP_DIR = os.path.join(BASE_DIR, "temp")           # 临时文件

# ============================================================
# 输出文件（模型隔离，便于对比）
# ============================================================
def _trajectory_file(model_name: str) -> str:
    return os.path.join(OUTPUT_DIR, f"trajectories_{model_name}.jsonl")

def _stats_file(model_name: str) -> str:
    return os.path.join(OUTPUT_DIR, f"stats_{model_name}.json")

# 向后兼容
TRAJECTORIES_FILE = os.path.join(OUTPUT_DIR, "trajectories.jsonl")
STATS_FILE = os.path.join(OUTPUT_DIR, "stats.json")

# ============================================================
# 语料自动生成参数（扩展到万级数据）
# ============================================================
CODE_GEN_CONFIG = {
    "target_count": 10000,                  # 目标语料总数
    "gen_batch_size": 8,                    # 每类别每次生成的函数数（过大易超时）
    "max_concurrent": 1,                    # 并发请求数（单线程顺序，避免限流）
    "templates_file": os.path.join(BASE_DIR, "prompts", "code_gen_templates.txt"),
    # 代码生成使用的模型（--model 命令行覆盖，或跟随 VERCORS_AGENT_MODEL）
    "gen_model": DEFAULT_MODEL,             # 默认跟随 .env
    "verify_model": DEFAULT_MODEL,          # 驱动 VerCors 验证的模型
    "min_lines": 5,
    "max_lines": 60,
    "categories": [
        "arithmetic",
        "array_readonly",
        "array_readwrite",
        "conditional",
        "nested_loop",
        "prefix_cumulative",
        "multi_array",
    ],
}

# ============================================================
# 日志
# ============================================================
LOG_FILE = os.path.join(BASE_DIR, "agent.log")
