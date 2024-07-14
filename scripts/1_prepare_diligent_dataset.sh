#!/bin/bash

# 设置基础路径
BASE_DIR="/home/qingpowuwu/Project_15_illumination/5_SDPS-Net-2019"
SCRIPT_DIR="$BASE_DIR/scripts"
ORIGINAL_DATASET="/home/qingpowuwu/Project_15_illumination/0_Dataset_Original/DiLiGenT/pmsData"
PROJECT_DATA_DIR="$BASE_DIR/data/datasets/DiLiGenT/pmsData"

# 创建必要的目录
mkdir -p "$PROJECT_DATA_DIR"

# 复制 filenames.txt 到 names.txt（如果需要的话）
if [ ! -f "$PROJECT_DATA_DIR/names.txt" ]; then
    cp "$ORIGINAL_DATASET/ballPNG/filenames.txt" "$PROJECT_DATA_DIR/names.txt"
fi

# 复制 objects.txt
cp "$ORIGINAL_DATASET/objects.txt" "$PROJECT_DATA_DIR/objects.txt"

# 运行 Python 脚本来裁剪数据
python "$SCRIPT_DIR/cropDiLiGenTData.py" --input_dir "$ORIGINAL_DATASET"

echo "数据准备完成。"