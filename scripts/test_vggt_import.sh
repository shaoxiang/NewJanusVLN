#!/bin/bash
#
# 快速测试：验证 VGGT 导入是否修复
#

echo "测试 VGGT 导入..."

export PYTHONPATH=$PWD/src

python3 -c "
try:
    from qwen_vl.model.vggt.models.vggt import VGGT
    print('✅ VGGT 导入成功！')
    print('   可以运行预计算脚本了')
except Exception as e:
    print('❌ VGGT 导入失败：', e)
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "  测试通过！现在可以运行："
    echo ""
    echo "  单 GPU 预计算："
    echo "    bash scripts/run_precompute.sh"
    echo ""
    echo "  8 GPU 并行预计算（推荐）："
    echo "    bash scripts/precompute_8gpu.sh"
    echo "========================================="
else
    echo ""
    echo "测试失败，请检查环境配置"
    exit 1
fi
