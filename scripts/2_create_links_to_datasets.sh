#!/bin/bash

# 设置工作目录
cd /home/qingpowuwu/Project_15_illumination/5_SDPS-Net-2019

# 创建必要的目录结构
mkdir -p data/datasets
cd data/datasets

# 定义源目录和目标名称
declare -A datasets=(
    ["/home/qingpowuwu/Project_15_illumination/0_Downloaded_Original/PS-FCN/PS_Blobby_Dataset"]="PS_Blobby_Dataset"
    ["/home/qingpowuwu/Project_15_illumination/0_Downloaded_Original/PS-FCN/PS_Sculpture_Dataset"]="PS_Sculpture_Dataset"
    ["/home/qingpowuwu/Project_15_illumination/0_Downloaded_Original/PS-FCN/PS_Test_Sphere_Bunny"]="PS_Test_Sphere_Bunny"
    ["/home/qingpowuwu/Project_15_illumination/0_Dataset_Original/DiLiGenT"]="DiLiGenT"
)

# 创建符号链接
for src in "${!datasets[@]}"; do
    dst="${datasets[$src]}"
    if [ ! -L "$dst" ]; then
        ln -s "$src" "$dst"
        echo "创建符号链接: $dst -> $src"
    else
        echo "符号链接已存在: $dst"
    fi
done

# 处理 DiLiGenT 数据集
if [ -d "DiLiGenT/pmsData" ]; then
    cd DiLiGenT/pmsData/

    # 生成 objects.txt 文件（排除 objects.txt 本身）
    ls | sed '/objects.txt/d' > objects.txt
    echo "更新了 objects.txt 文件"

    # 复制 filenames.txt
    if [ -f "ballPNG/filenames.txt" ]; then
        cp ballPNG/filenames.txt .
        echo "复制了 filenames.txt 文件"
    else
        echo "警告: ballPNG/filenames.txt 不存在"
    fi

    cd ../../
else
    echo "警告: DiLiGenT/pmsData 目录不存在"
fi

# 返回到根目录
cd ../../

echo "操作完成。符号链接已创建，DiLiGenT 数据集的 objects.txt 已更新，filenames.txt 已复制（如果存在）。"