#!/usr/bin/env python3
"""
训练前环境验证脚本
检查所有必要的文件、路径和配置是否正确
"""

import os
import sys
from pathlib import Path

# 颜色定义
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_status(status, message):
    """打印状态信息"""
    if status == "ok":
        print(f"{Colors.GREEN}✓{Colors.END} {message}")
        return True
    elif status == "error":
        print(f"{Colors.RED}✗{Colors.END} {message}")
        return False
    elif status == "warning":
        print(f"{Colors.YELLOW}⚠{Colors.END} {message}")
        return True
    else:
        print(f"{Colors.BLUE}ℹ{Colors.END} {message}")
        return True

def check_python_packages():
    """检查Python包"""
    print("\n" + "="*60)
    print("检查Python包...")
    print("="*60)

    packages = {
        'torch': 'PyTorch',
        'verl': 'verl',
        'transformers': 'Transformers',
        'pandas': 'Pandas',
        'pyarrow': 'PyArrow',
        'hydra': 'Hydra-core',
        'wandb': 'WandB',
        'peft': 'PEFT',
        'tensordict': 'TensorDict',
    }

    all_ok = True
    for pkg, name in packages.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print_status("ok", f"{name} (版本: {version})")
        except ImportError:
            print_status("error", f"{name} 未安装")
            all_ok = False

    return all_ok

def check_cuda():
    """检查CUDA"""
    print("\n" + "="*60)
    print("检查CUDA和GPU...")
    print("="*60)

    try:
        import torch

        if torch.cuda.is_available():
            print_status("ok", f"CUDA可用 (版本: {torch.version.cuda})")
            gpu_count = torch.cuda.device_count()
            print_status("ok", f"检测到 {gpu_count} 张GPU")

            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print_status("info", f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

            return gpu_count >= 4
        else:
            print_status("error", "CUDA不可用")
            return False
    except Exception as e:
        print_status("error", f"检查CUDA时出错: {e}")
        return False

def check_model_path():
    """检查模型路径"""
    print("\n" + "="*60)
    print("检查模型路径...")
    print("="*60)

    model_path = Path("verl/models/Llama-3-8B-Instruct/LLM-Research/Meta-Llama-3-8B-Instruct")

    if model_path.exists():
        # 检查关键文件
        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]

        all_files_exist = True
        for file in required_files:
            file_path = model_path / file
            if file_path.exists():
                print_status("ok", f"找到 {file}")
            else:
                print_status("warning", f"缺少 {file}")
                all_files_exist = False

        # 检查模型权重文件
        safetensors_files = list(model_path.glob("*.safetensors"))
        bin_files = list(model_path.glob("*.bin"))

        if safetensors_files:
            print_status("ok", f"找到 {len(safetensors_files)} 个safetensors权重文件")
            return True
        elif bin_files:
            print_status("ok", f"找到 {len(bin_files)} 个bin权重文件")
            return True
        else:
            print_status("error", "未找到模型权重文件")
            return False
    else:
        print_status("error", f"模型路径不存在: {model_path}")
        print_status("info", "请运行以下命令下载模型:")
        print_status("info", "  huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \\")
        print_status("info", "    --local-dir verl/models/Llama-3-8B-Instruct/LLM-Research/Meta-Llama-3-8B-Instruct")
        return False

def check_data_files():
    """检查数据文件"""
    print("\n" + "="*60)
    print("检查数据文件...")
    print("="*60)

    # 检查原始JSONL文件
    raw_files = {
        'train': 'dataset/raw/sleep_train_with_ids.jsonl',
        'val': 'dataset/raw/sleep_validation_with_ids.jsonl',
        'test': 'dataset/raw/sleep_test_with_ids.jsonl',
    }

    jsonl_ok = True
    for split, path in raw_files.items():
        if Path(path).exists():
            size = Path(path).stat().st_size / 1024 / 1024  # MB
            print_status("ok", f"找到 {split} JSONL文件 ({size:.1f} MB)")
        else:
            print_status("error", f"缺少 {split} JSONL文件: {path}")
            jsonl_ok = False

    # 检查Parquet文件
    parquet_dir = Path("dataset/parquet/sleep_mixed")
    parquet_files = {
        'train': parquet_dir / 'train.parquet',
        'val': parquet_dir / 'val.parquet',
        'test': parquet_dir / 'test.parquet',
    }

    parquet_ok = True
    for split, path in parquet_files.items():
        if path.exists():
            size = path.stat().st_size / 1024 / 1024  # MB

            # 读取并验证格式
            try:
                import pandas as pd
                df = pd.read_parquet(path)

                if 'prompt' in df.columns and 'response' in df.columns:
                    print_status("ok", f"找到 {split} Parquet文件 ({size:.1f} MB, {len(df)} 样本)")
                else:
                    print_status("error", f"{split} Parquet文件缺少必要字段")
                    parquet_ok = False
            except Exception as e:
                print_status("error", f"读取 {split} Parquet文件失败: {e}")
                parquet_ok = False
        else:
            print_status("warning", f"缺少 {split} Parquet文件: {path}")
            parquet_ok = False

    if not parquet_ok and jsonl_ok:
        print_status("info", "请运行以下命令转换数据:")
        print_status("info", "  python data_processing/convert_mixed_sleep_to_parquet.py")

    return jsonl_ok and parquet_ok

def check_training_script():
    """检查训练脚本"""
    print("\n" + "="*60)
    print("检查训练脚本...")
    print("="*60)

    script_path = Path("scripts/run_sleep_mixed_sft.sh")

    if script_path.exists():
        print_status("ok", f"找到训练脚本: {script_path}")

        # 检查脚本是否可执行
        if os.access(script_path, os.X_OK):
            print_status("ok", "脚本具有执行权限")
        else:
            print_status("warning", "脚本没有执行权限")
            print_status("info", f"运行: chmod +x {script_path}")

        return True
    else:
        print_status("error", f"训练脚本不存在: {script_path}")
        return False

def check_directories():
    """检查必要目录"""
    print("\n" + "="*60)
    print("检查目录结构...")
    print("="*60)

    required_dirs = [
        "checkpoints",
        "dataset/raw",
        "dataset/parquet/sleep_mixed",
        "verl/models",
        "scripts",
        "data_processing",
    ]

    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            print_status("ok", f"目录存在: {dir_path}")
        else:
            print_status("warning", f"目录不存在: {dir_path}")
            print_status("info", f"创建目录: mkdir -p {dir_path}")
            all_ok = False

    return all_ok

def estimate_training_resources():
    """估算训练资源"""
    print("\n" + "="*60)
    print("训练资源估算...")
    print("="*60)

    try:
        import pandas as pd
        train_path = Path("dataset/parquet/sleep_mixed/train.parquet")

        if train_path.exists():
            df = pd.read_parquet(train_path)
            num_samples = len(df)

            # 估算训练时间（基于经验值）
            batch_size = 32
            num_gpus = 4
            epochs = 3

            steps_per_epoch = num_samples // batch_size
            total_steps = steps_per_epoch * epochs

            # 假设每步0.5秒（A100）
            estimated_time_seconds = total_steps * 0.5
            estimated_time_hours = estimated_time_seconds / 3600

            print_status("info", f"训练样本数: {num_samples}")
            print_status("info", f"批次大小: {batch_size}")
            print_status("info", f"每epoch步数: {steps_per_epoch}")
            print_status("info", f"总训练步数: {total_steps}")
            print_status("info", f"预计训练时间: {estimated_time_hours:.1f} 小时")

            # 估算内存需求
            print_status("info", f"预计每GPU内存: 40-60 GB")
            print_status("info", f"预计checkpoint大小: ~16 GB")

    except Exception as e:
        print_status("warning", f"无法估算训练资源: {e}")

def main():
    """主函数"""
    print("\n" + "="*60)
    print("verl 训练环境验证")
    print("="*60)

    checks = [
        ("Python包", check_python_packages),
        ("CUDA和GPU", check_cuda),
        ("模型路径", check_model_path),
        ("数据文件", check_data_files),
        ("训练脚本", check_training_script),
        ("目录结构", check_directories),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print_status("error", f"{name} 检查失败: {e}")
            results[name] = False

    # 估算资源
    estimate_training_resources()

    # 总结
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)

    all_passed = all(results.values())

    for name, passed in results.items():
        status = "ok" if passed else "error"
        print_status(status, name)

    print("\n" + "="*60)
    if all_passed:
        print(f"{Colors.GREEN}✓ 所有检查通过！可以开始训练{Colors.END}")
        print("="*60)
        print("\n运行以下命令开始训练:")
        print("  bash scripts/run_sleep_mixed_sft.sh")
        return 0
    else:
        print(f"{Colors.RED}✗ 部分检查未通过，请修复后再训练{Colors.END}")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
