"""
VerCors 数据采集 Agent — 配置文件
部署到服务器后，修改 DEEPSEEK_API_KEY 和路径即可运行。
"""

import os

# ── 自动加载 .env 文件（如果 python-dotenv 已安装）──
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env_path)
except ImportError:
    # python-dotenv 未安装时静默跳过，依赖系统环境变量
    pass

# ============================================================
# DeepSeek API 配置（兼容 openai 库）
# ============================================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"          # DeepSeek v4 Pro

# API 调用参数
API_TEMPERATURE = 0.3                     # 代码生成建议低温度
API_MAX_TOKENS = 4096
API_TIMEOUT = 120                          # 单次 API 调用超时（秒）

# ============================================================
# VerCors 工具路径（服务器上的绝对路径）
# ============================================================
TEST_OP_SH = "/workspace/I/qimeng5/huyan/VerCorsOPtest/test_op.sh"
VERCORS_BIN = "/workspace/I/qimeng5/huyan/download/usr/share/vercors/vercors"

# VerCors 验证超时（秒），防止 SMT 求解器卡死
VERCORS_TIMEOUT = 120

# ============================================================
# Agent 循环参数
# ============================================================
MAX_RETRIES = 5            # 最大重试轮数（首轮 + 4 次反馈修正）
TEMP_DIR = "temp"          # 临时文件目录（存放每轮生成的 .c 文件）

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "corpus")       # 干净 C 代码库存放处
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")     # Prompt 模板
OUTPUT_DIR = os.path.join(BASE_DIR, "output")       # 成功轨迹输出
FAILED_DIR = os.path.join(BASE_DIR, "failed")        # 失败案例
TEMP_DIR = os.path.join(BASE_DIR, TEMP_DIR)

# ============================================================
# 输出文件
# ============================================================
TRAJECTORIES_FILE = os.path.join(OUTPUT_DIR, "trajectories.jsonl")   # 训练数据
STATS_FILE = os.path.join(OUTPUT_DIR, "stats.json")                  # 统计信息

# ============================================================
# 日志
# ============================================================
LOG_FILE = os.path.join(BASE_DIR, "agent.log")
