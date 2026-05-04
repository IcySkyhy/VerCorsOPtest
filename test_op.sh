#!/bin/sh

export PATH="/workspace/I/qimeng5/huyan/download/my_clang/bin:$PATH"

TARGET_FILE=${1:-"test.c"}

# 检查文件是否存在
if [ ! -f "$TARGET_FILE" ]; then
    echo "错误: 找不到文件 '$TARGET_FILE'"
    exit 1
fi

BASE_NAME=$(basename "$TARGET_FILE" .c)
TEST_TIME=$(date +"%Y%m%d-%H%M%S")
OUTPUT_FILE="${BASE_NAME}-${TEST_TIME}.out"

echo "正在启动 $TARGET_FILE 的 VerCors 验证"
echo "验证日志将保存至 $OUTPUT_FILE"
echo "---------------------------------------------------"

# 5. 执行官方引擎，并将输出和报错重定向到日志文件中
# 使用 2>&1 | tee 可以让你在屏幕上实时看到进度的同时，把日志存进文件
/workspace/I/qimeng5/huyan/download/usr/share/vercors/vercors "$TARGET_FILE" 2>&1 | tee "$OUTPUT_FILE"

echo "---------------------------------------------------"
echo "验证执行完毕"