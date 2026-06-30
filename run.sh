
#!/bin/bash
# ==========================================================
# UIGEN-T2 自动化数据清洗与 SFT 训练流水线脚本
# ==========================================================
# 若任何一个步骤报错，立刻停止脚本执行
set -e

echo "=================================================="
echo " 开始执行 UIGEN-T2 自动化流水线"
echo "=================================================="

echo -e "\n>>> [步骤 1/5]: 执行数据画像 (Data Profiling) ..."
python 01_data_profiling.py

echo -e "\n>>> [步骤 2/5]: 执行数据清洗与重组 (Data Cleaning) ..."
python 02_data_cleaning.py

echo -e "\n>>> [步骤 3/5]: SFT 训练 - 原始数据控制组 (RAW) ..."
python 03_train.py --mode raw --max_steps 1000

echo -e "\n>>> [步骤 4/5]: SFT 训练 - 清洗后实验组 (CLEAN) ..."
python 03_train.py --mode clean --max_steps 1000

echo -e "\n>>> [步骤 5/5]: 生成 Loss 对比曲线图 (Plotting Loss) ..."
python 04_plot_loss.py

echo -e "\n=================================================="
echo " 恭喜！流水线执行完毕，所有结果均已生成。 "
echo " 图表保存在 figures/ 目录，日志保存在 logs/ 目录。"
echo "=================================================="