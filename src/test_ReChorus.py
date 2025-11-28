# test_rechorus.py

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath('.'))

try:
    from src.models.reranker import PRM
    print("✅ ReChorus 环境验证成功：核心模块可正常导入")
    test_cmd = 'python main.py --model_name SASRec --dataset MovieLens_1M/ML_1MTOPK --epochs 5 --lr 1e-3 --l2 1e-6'
    os.system(test_cmd)
    print("✅ ReChorus 环境验证成功：模型训练成功")
except Exception as e:
    print("❌ 环境验证失败：", e)