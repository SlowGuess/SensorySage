# PH-LLM 复现指南 - 混合训练版本

本项目复现论文《A personal health large language model for sleep and fitness coaching》中的SFT训练部分。

## 📚 论文核心方法

根据论文，训练采用**混合训练（Mixture Training）**而非分阶段训练：
- 将Insights、Etiology、Recommendations三个任务的数据混合在一起
- 使用Teacher Forcing：在训练Etiology和Recommendations时，prompt中包含专家的标准答案
- 一次性训练，模型同时学习三个任务

## 🚀 快速开始（服务器部署）

### 1. 同步代码到服务器

```bash
# 在本地执行
rsync -avz --exclude='*.pyc' --exclude='__pycache__' \
    --exclude='.git' --exclude='checkpoints' \
    /Users/xucai/Project/verl/ user@server:/path/to/verl/
```

### 2. 在服务器上安装环境

```bash
cd /path/to/verl

# 运行自动安装脚本
bash setup_server.sh

# 或手动安装
pip install -e .
pip install -e ".[gpu]"
```

### 3. 下载模型

```bash
# 登录HuggingFace（需要token）
huggingface-cli login

# 下载Llama-3-8B-Instruct
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir verl/models/Llama-3-8B-Instruct/LLM-Research/Meta-Llama-3-8B-Instruct
```

**注意**: 需要在HuggingFace上申请Llama-3的访问权限

### 4. 转换数据

```bash
# 将JSONL转换为Parquet格式
python data_processing/convert_mixed_sleep_to_parquet.py \
    --input_dir dataset/raw \
    --output_dir dataset/parquet/sleep_mixed
```

### 5. 验证环境

```bash
# 运行验证脚本
python verify_setup.py
```

### 6. 开始训练

```bash
# 使用nohup后台运行
nohup bash scripts/run_sleep_mixed_sft.sh > training.log 2>&1 &

# 或使用screen
screen -S sleep_sft
bash scripts/run_sleep_mixed_sft.sh
# Ctrl+A+D 分离screen

# 查看日志
tail -f training.log
```

## 📁 项目结构

```
verl/
├── data_processing/
│   ├── convert_mixed_sleep_to_parquet.py  # 数据转换脚本（新增）
│   └── convert_sleep_data.py              # 原始数据转换脚本
├── dataset/
│   ├── raw/
│   │   ├── sleep_train_with_ids.jsonl     # 训练数据（已生成）
│   │   ├── sleep_validation_with_ids.jsonl # 验证数据（已生成）
│   │   └── sleep_test_with_ids.jsonl      # 测试数据（已生成）
│   └── parquet/
│       └── sleep_mixed/
│           ├── train.parquet              # 转换后的训练数据
│           ├── val.parquet                # 转换后的验证数据
│           └── test.parquet               # 转换后的测试数据
├── scripts/
│   ├── run_sleep_mixed_sft.sh             # 混合训练脚本（新增）
│   └── run_llama3_8b_sft.sh               # 原始训练脚本
├── verl/
│   ├── trainer/
│   │   └── fsdp_sft_trainer.py            # FSDP SFT训练器
│   └── models/
│       └── Llama-3-8B-Instruct/           # 模型目录
├── checkpoints/                            # 训练checkpoint保存目录
├── setup_server.sh                         # 服务器环境安装脚本（新增）
├── verify_setup.py                         # 环境验证脚本（新增）
├── SERVER_DEPLOYMENT_CHECKLIST.md          # 部署检查清单（新增）
└── README_REPRODUCTION.md                  # 本文件
```

## 🔧 训练配置

### 硬件配置
- **GPU**: 4×A100 (80GB)
- **内存**: 建议256GB+
- **磁盘**: 建议500GB+

### 训练超参数
```yaml
# 数据配置
micro_batch_size_per_gpu: 2
train_batch_size: 32
max_length: 8192
truncation: right

# 模型配置
model: Llama-3-8B-Instruct
strategy: fsdp2
dtype: bfloat16
gradient_checkpointing: True

# 优化器配置
learning_rate: 2e-5
weight_decay: 0.01
clip_grad: 1.0

# 训练配置
total_epochs: 3
save_freq: 1
test_freq: 1
```

### 预期训练时间
- 每个epoch: 约30-60分钟
- 总训练时间（3 epochs）: 约1.5-3小时
- 内存占用: 每张GPU约40-60GB

## 📊 数据格式

### 输入数据（JSONL）
```json
{
    "case_study_id": "SC16543",
    "task_type": "insights",
    "prompt": "You are a sleep medicine expert...",
    "completion": "**Sleep Routine:**..."
}
```

### 训练数据（Parquet）
```python
{
    "prompt": "You are a sleep medicine expert...",
    "response": "**Sleep Routine:**...",
    "case_study_id": "SC16543",
    "task_type": "insights"
}
```

## 🎯 训练流程

### 数据准备流程
1. **原始数据**: `sleep_case_studies.all.jsonl`
2. **拆分数据**: 使用你的处理脚本生成三个任务的混合数据
3. **转换格式**: 使用`convert_mixed_sleep_to_parquet.py`转换为Parquet

### 训练流程
1. **加载模型**: Llama-3-8B-Instruct
2. **加载数据**: 混合的Parquet数据
3. **训练**: 使用FSDP2进行分布式训练
4. **保存**: 每个epoch保存checkpoint

### 推理流程（训练后）
1. 输入数据 → 生成Insights
2. 数据 + Insights → 生成Etiology
3. 数据 + Insights + Etiology → 生成Recommendations

## 🔍 常见问题

### Q1: CUDA Out of Memory
**解决方案**:
- 减小`micro_batch_size_per_gpu`（当前为2）
- 减小`max_length`（当前为8192）
- 确保gradient checkpointing已启用

### Q2: 模型下载失败
**解决方案**:
- 检查HuggingFace token
- 确认有Llama-3访问权限
- 使用镜像站点（如hf-mirror.com）

### Q3: 数据格式错误
**解决方案**:
- 确保JSONL文件包含`prompt`和`completion`字段
- 运行`verify_setup.py`检查数据格式
- 重新运行数据转换脚本

### Q4: 训练速度慢
**解决方案**:
- 检查是否使用了FSDP2
- 确认flash-attention已安装
- 调整batch size和序列长度

## 📈 监控训练

### 使用WandB
```bash
# 登录wandb
wandb login

# 训练会自动上传到wandb
# 项目名: PH-LLM-Reproduction
# 实验名: sleep_mixed_sft_llama3_8b
```

### 查看日志
```bash
# 实时查看训练日志
tail -f training.log

# 查看GPU使用情况
watch -n 1 nvidia-smi

# 查看checkpoint
ls -lh checkpoints/sleep_mixed_sft_llama3_8b/
```

## 📝 重要文件说明

### 新增文件
1. **convert_mixed_sleep_to_parquet.py**: 将混合训练的JSONL数据转换为Parquet格式
2. **run_sleep_mixed_sft.sh**: 混合训练启动脚本
3. **setup_server.sh**: 服务器环境自动安装脚本
4. **verify_setup.py**: 训练前环境验证脚本
5. **SERVER_DEPLOYMENT_CHECKLIST.md**: 详细的部署检查清单

### 关键配置
- 训练脚本: `scripts/run_sleep_mixed_sft.sh`
- 训练器: `verl/trainer/fsdp_sft_trainer.py`
- 数据集: `verl/utils/dataset/sft_dataset.py`

## 🎓 参考资料

- 论文: [A personal health large language model for sleep and fitness coaching](https://www.nature.com/articles/s41591-025-03888-0)
- verl文档: https://verl.readthedocs.io/
- verl GitHub: https://github.com/volcengine/verl

## 📞 获取帮助

如果遇到问题:
1. 查看`training.log`中的错误信息
2. 运行`verify_setup.py`检查环境
3. 查看`SERVER_DEPLOYMENT_CHECKLIST.md`
4. 查看verl GitHub Issues

## ✅ 检查清单

部署前请确认:
- [ ] 代码已同步到服务器
- [ ] Python环境已配置（Python 3.8+）
- [ ] PyTorch和CUDA正常工作
- [ ] verl已安装
- [ ] Llama-3-8B-Instruct模型已下载
- [ ] 数据已转换为Parquet格式
- [ ] 训练脚本路径已确认
- [ ] GPU可见且可用（4×A100）
- [ ] WandB已配置（可选）
- [ ] 磁盘空间充足（至少500GB）

## 🎉 开始训练

一切准备就绪后，运行:
```bash
bash scripts/run_sleep_mixed_sft.sh
```

祝训练顺利！🚀
