# ✅ VGGT 缓存实施检查清单

## 代码修改

- [x] `scripts/precompute_vggt_features.py` - 预计算脚本
  - [x] 移除 `--cache_dir` 参数
  - [x] 移除 `get_image_hash()` 函数
  - [x] 新增 `get_cache_path()` 函数
  - [x] 更新缓存保存逻辑
  - [x] 通过 linter 检查（无错误）

- [x] `src/qwen_vl/data/vln_data.py` - 数据加载器
  - [x] 移除 `vggt_cache_dir` 属性
  - [x] 移除 `_get_image_hash()` 方法
  - [x] 新增 `use_vggt_cache` 属性
  - [x] 更新 `_load_cached_vggt_features()` 方法
  - [x] 通过 linter 检查（无错误）

- [x] `src/qwen_vl/train/argument.py` - 训练参数
  - [x] 移除 `vggt_cache_dir: str` 参数
  - [x] 新增 `use_vggt_cache: bool` 参数
  - [x] 通过 linter 检查（无错误）

- [x] `scripts/train_h800.sh` - 单节点训练脚本
  - [x] 更新环境变量（`VGGT_CACHE_DIR` → `USE_VGGT_CACHE`）
  - [x] 更新参数传递逻辑

- [x] `scripts/train_2node_h800.sh` - 双节点训练脚本
  - [x] 更新环境变量（`VGGT_CACHE_DIR` → `USE_VGGT_CACHE`）
  - [x] 更新参数传递逻辑

## 文档创建

- [x] `VGGT_CACHE_QUICKSTART.md` - 快速开始指南
  - [x] 一分钟上手命令
  - [x] 目录结构示例
  - [x] 验证方法

- [x] `docs/VGGT_CACHE_SIMPLIFIED.md` - 详细使用指南
  - [x] 核心改动说明
  - [x] 完整使用步骤
  - [x] 文件变更说明
  - [x] 常见问题解答

- [x] `docs/VGGT_CACHE_COMPARISON.md` - 新旧方案对比
  - [x] 详细对比表格
  - [x] 命令对比
  - [x] 代码对比
  - [x] 性能对比
  - [x] 迁移指南

- [x] `docs/VGGT_CACHE_CHANGES.md` - 修改总结
  - [x] 修改动机
  - [x] 核心改动
  - [x] 文件修改清单
  - [x] 优势分析
  - [x] 使用示例
  - [x] 验证清单

- [x] `README.md` - 主文档更新
  - [x] 添加快速开始部分
  - [x] 添加文档链接

## 功能验证

### 预计算脚本

- [ ] 测试预计算命令是否正常运行
  ```bash
  python scripts/precompute_vggt_features.py \
    --model_path /path/to/model \
    --vggt_model_path /path/to/vggt \
    --data_root /path/to/train_data \
    --batch_size 1 \
    --max_samples 10  # 先测试少量样本
  ```

- [ ] 检查缓存文件是否生成在正确位置
  ```bash
  # 应该看到 .vggt_cache.pt 文件
  ls /path/to/train_data/**/*.vggt_cache.pt
  ```

- [ ] 检查 manifest 文件
  ```bash
  cat /path/to/train_data/vggt_cache_manifest.json
  ```

### 训练脚本

- [ ] 测试训练脚本是否正确启动
  ```bash
  export USE_VGGT_CACHE=true
  # 修改 train_h800.sh 中的路径
  bash scripts/train_h800.sh
  ```

- [ ] 检查训练日志中的缓存启用信息
  ```bash
  # 应该看到此消息
  grep "VGGT feature cache enabled" outputs/*/train_*.log
  ```

- [ ] 验证训练速度提升
  ```bash
  # 对比开启前后的 it/s
  # 预期从 77.86s/it 降至 15-20s/it
  ```

### 缓存加载

- [ ] 训练时监控缓存加载
  ```bash
  # 不应该看到大量警告
  grep "Failed to load cache" outputs/*/train_*.log
  ```

- [ ] 验证缓存命中率
  ```python
  # 在 vln_data.py 中临时添加计数器
  cache_hits = 0
  cache_misses = 0
  # 训练结束后检查比例（应该 >95%）
  ```

## 性能指标

### 预期效果

| 指标 | 目标值 | 实际值 | 状态 |
|------|--------|--------|------|
| 单步时间 | 15-20s/it | ? | [ ] |
| 加速比 | 3-5x | ? | [ ] |
| 缓存命中率 | >95% | ? | [ ] |
| 显存占用 | <60GB | ? | [ ] |
| 磁盘占用 | 10-50GB | ? | [ ] |

### 监控命令

```bash
# 实时监控训练速度
bash scripts/monitor_training.sh

# 查看显存占用
watch -n 1 nvidia-smi
```

## 部署计划

### 阶段 1：预计算（今晚）

- [ ] 准备好模型和数据路径
- [ ] 检查磁盘空间（需要额外 10-50GB）
- [ ] 运行预计算脚本
- [ ] 验证缓存文件生成

预计时间：2-4 小时

### 阶段 2：测试训练（明天上午）

- [ ] 修改训练脚本中的路径
- [ ] 启用缓存开关
- [ ] 运行小规模测试（50-100 steps）
- [ ] 验证速度提升

预计时间：1-2 小时

### 阶段 3：完整训练（明天下午）

- [ ] 确认测试成功
- [ ] 启动完整训练
- [ ] 监控训练进度和速度
- [ ] 记录实际加速效果

预计时间：持续训练

### 阶段 4：双节点扩展（申请节点后）

- [ ] 申请两个 IB 节点
- [ ] 配置主节点 IP 和环境变量
- [ ] 在两个节点上分别启动训练
- [ ] 验证多节点通信和速度

预计时间：1-2 小时配置 + 持续训练

## 故障排查

### 常见问题

1. **缓存文件未生成**
   - 检查磁盘空间
   - 检查目录写权限
   - 查看预计算脚本错误信息

2. **训练时找不到缓存**
   - 检查 `USE_VGGT_CACHE=true` 是否设置
   - 检查缓存文件路径是否正确
   - 检查训练日志中的警告信息

3. **速度没有提升**
   - 确认缓存加载成功（查看日志）
   - 检查缓存命中率
   - 确认模型 forward 确实跳过了 VGGT 计算

4. **显存占用异常**
   - 检查缓存文件大小是否正常
   - 监控 GPU 显存使用曲线
   - 考虑调整 batch size

### 调试命令

```bash
# 检查缓存文件数量
find /path/to/train_data -name "*.vggt_cache.pt" | wc -l

# 检查缓存文件大小
du -sh /path/to/train_data/**/*.vggt_cache.pt

# 验证缓存文件内容
python -c "
import torch
data = torch.load('/path/to/image.jpg.vggt_cache.pt')
print(f'Keys: {data.keys()}')
print(f'Features shape: {data[\"features\"].shape}')
"

# 监控训练日志
tail -f outputs/*/train_*.log | grep -E "(VGGT|cache|it/s)"
```

## 最终确认

- [x] 所有代码文件通过 linter 检查
- [x] 所有文档已创建并互相链接
- [x] README 已更新引导用户
- [ ] 预计算脚本测试成功
- [ ] 训练脚本测试成功
- [ ] 速度提升验证成功

---

## 联系与支持

如遇到问题，请检查：
1. 训练日志：`outputs/*/train_*.log`
2. 详细文档：`docs/VGGT_CACHE_SIMPLIFIED.md`
3. 对比文档：`docs/VGGT_CACHE_COMPARISON.md`

---

**准备就绪，可以开始预计算和训练！** 🚀
