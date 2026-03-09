# 服务器部署检查清单

## 📋 部署前准备

### 1. 硬件环境确认
- ✅ 4张 A100 (80GB) GPU
- ✅ 足够的CPU内存（建议至少256GB）
- ✅ 足够的磁盘空间（至少500GB用于模型和数据）

### 2. 软件环境要求
- Python 3.10+
- CUDA 11.8+ 或 12.1+
- PyTorch 2.0+（建议2.6.0）
- 网络连接（用于下载模型和依赖）

---

## 🚀 部署步骤

### Step 1: 同步代码到服务器
```bash
# 在本地执行
rsync -avz --exclude='*.pyc' --exclude='__pycache__' \
    --exclude='.git' --exclude='checkpoints' \
    /Users/xucai/Project/verl/ user@server:/path/to/verl/
```

### Step 2: 安装依赖

#### 2.1 创建虚拟环境（推荐）
```bash
cd /path/to/verl
python3 -m venv venv
source venv/bin/activate
```

#### 2.2 安装PyTorch（根据CUDA版本选择）
```bash
# CUDA 11.8
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 2.3 安装verl和依赖
```bash
# 安装verl核心依赖
pip install -e .

# 安装GPU相关依赖（flash-attention等）
pip install -e ".[gpu]"

# 如果需要vLLM支持（可选，用于推理加速）
# pip install -e ".[vllm]"
```

#### 2.4 验证安装
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import verl; print('verl installed successfully')"
```

### Step 3: 下载模型

#### 3.1 下载Llama-3-8B-Instruct
```bash
# 创建模型目录
mkdir -p verl/models/Llama-3-8B-Instruct/LLM-Research

# 使用huggingface-cli下载（需要先安装）
pip install huggingface-hub

# 下载模型（需要HuggingFace token）
huggingface-cli login
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir verl/models/Llama-3-8B-Instruct/LLM-Research/Meta-Llama-3-8B-Instruct

# 或者使用git lfs
# git lfs install
# cd verl/models/Llama-3-8B-Instruct/LLM-Research
# git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
```

**注意**: 需要在HuggingFace上申请Llama-3的访问权限

### Step 4: 准备数据

#### 4.1 转换JSONL到Parquet格式
```bash
# 确保已经有处理好的JSONL文件在dataset/raw/目录下
python data_processing/convert_mixed_sleep_to_parquet.py \
    --input_dir dataset/raw \
    --output_dir dataset/parquet/sleep_mixed
```

#### 4.2 验证数据
```bash
# 检查生成的parquet文件
ls -lh dataset/parquet/sleep_mixed/

# 应该看到:
# train.parquet
# val.parquet
# test.parquet

# 验证数据格式
python -c "
import pandas as pd
df = pd.read_parquet('dataset/parquet/sleep_mixed/train.parquet')
print(f'训练样本数: {len(df)}')
print(f'列名: {df.columns.tolist()}')
print(f'第一个样本的prompt长度: {len(df.iloc[0][\"prompt\"])}')
"
```

### Step 5: 配置训练脚本

检查并修改 `scripts/run_sleep_mixed_sft.sh`:

```bash
# 确认以下路径正确:
# 1. TRAIN_DATA 路径
# 2. VAL_DATA 路径
# 3. MODEL_PATH 路径
# 4. SAVE_PATH 路径

# 根据GPU数量调整:
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用4张GPU
NPROC_PER_NODE=4
```

### Step 6: 测试运行（可选但推荐）

#### 6.1 小规模测试
```bash
# 修改训练脚本，使用小数据集测试
# 在run_sleep_mixed_sft.sh中临时修改:
# trainer.total_epochs=1
# data.train_batch_size=8

# 运行测试
bash scripts/run_sleep_mixed_sft.sh
```

### Step 7: 正式训练

```bash
# 使用nohup或screen在后台运行
nohup bash scripts/run_sleep_mixed_sft.sh > training.log 2>&1 &

# 或使用screen
screen -S sleep_sft
bash scripts/run_sleep_mixed_sft.sh
# Ctrl+A+D 分离screen

# 查看日志
tail -f training.log
```

---

## 🔍 常见问题排查

### 问题1: CUDA Out of Memory
**解决方案**:
- 减小 `data.micro_batch_size_per_gpu` (当前为2)
- 减小 `data.max_length` (当前为8192)
- 启用gradient checkpointing (已启用)
- 使用混合精度训练 (已使用bfloat16)

### 问题2: 模型下载失败
**解决方案**:
- 检查HuggingFace token是否有效
- 检查是否有Llama-3访问权限
- 使用镜像站点（如hf-mirror.com）

### 问题3: 数据加载错误
**解决方案**:
- 检查parquet文件是否正确生成
- 验证prompt和response字段是否存在
- 检查数据路径是否正确

### 问题4: 多GPU训练失败
**解决方案**:
- 检查NCCL环境变量
- 验证所有GPU可见: `nvidia-smi`
- 检查网络配置（多节点训练时）

---

## 📊 监控训练进度

### 使用WandB（推荐）
```bash
# 登录wandb
wandb login

# 训练会自动上传到wandb
# 项目名: PH-LLM-Reproduction
# 实验名: sleep_mixed_sft_llama3_8b
```

### 查看本地日志
```bash
# 查看训练日志
tail -f training.log

# 查看checkpoint
ls -lh checkpoints/sleep_mixed_sft_llama3_8b/
```

---

## 📈 预期训练时间和资源

### 数据规模估算
根据你的数据:
- 训练集: ~20MB JSONL → 约1500-2000个样本 × 3任务 = 4500-6000个训练样本
- 验证集: ~4.5MB JSONL → 约300-400个样本 × 3任务 = 900-1200个验证样本

### 训练时间估算（4×A100 80GB）
- 每个epoch: 约30-60分钟（取决于序列长度）
- 总训练时间（3 epochs）: 约1.5-3小时
- 内存占用: 每张GPU约40-60GB

### Checkpoint大小
- 每个checkpoint: 约16GB（Llama-3-8B）
- 建议保留空间: 至少100GB

---

## ✅ 部署完成检查

- [ ] 代码已同步到服务器
- [ ] Python环境已配置
- [ ] PyTorch和CUDA正常工作
- [ ] verl已安装
- [ ] Llama-3-8B-Instruct模型已下载
- [ ] 数据已转换为Parquet格式
- [ ] 训练脚本路径已确认
- [ ] GPU可见且可用
- [ ] WandB已配置（可选）
- [ ] 磁盘空间充足

---

## 🎯 下一步

训练完成后:
1. 评估模型性能
2. 使用测试集进行推理
3. 对比baseline模型
4. 根据需要调整超参数重新训练

---

## 📞 需要帮助？

如果遇到问题，请检查:
1. `training.log` 中的错误信息
2. GPU内存使用情况: `nvidia-smi`
3. 磁盘空间: `df -h`
4. verl GitHub Issues: https://github.com/volcengine/verl/issues
