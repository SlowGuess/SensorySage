#!/usr/bin/env python3
"""
将VERL的FSDP checkpoint转换为标准HuggingFace格式 (通用版)
支持 SFT (根目录权重) 和 RL (actor子目录权重) 两种结构

使用方法:
python convert_fsdp_to_hf.py <checkpoint_input_dir> <hf_output_dir>
"""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pathlib import Path

def dtensor_to_tensor(tensor):
    """将DTensor转换为普通tensor"""
    if hasattr(tensor, '_local_tensor'):
        # DTensor对象，提取本地tensor
        return tensor._local_tensor
    elif hasattr(tensor, 'to_local'):
        # 另一种DTensor API
        return tensor.to_local()
    else:
        # 已经是普通tensor
        return tensor

def convert_fsdp_checkpoint_to_hf(checkpoint_dir, output_dir):
    base_path = Path(checkpoint_dir)
    output_path = Path(output_dir)

    print(f"[1/6] 分析 Checkpoint 结构: {base_path}")

    # --- 核心修改：自动探测权重路径 (Method B) ---
    actor_dir = base_path / "actor"
    
    # 检查是否有 actor 子目录且里面包含权重文件
    if actor_dir.exists() and list(actor_dir.glob("model_world_size_*.pt")):
        print(f"      -> 检测到 RL (PPO/GRPO) 格式")
        print(f"      -> 权重路径指向: {actor_dir}")
        weight_path = actor_dir
    else:
        print(f"      -> 检测到 SFT / 标准 FSDP 格式")
        print(f"      -> 权重路径指向: {base_path}")
        weight_path = base_path

    # --- 自动探测 Config 路径 ---
    # 优先在根目录找 huggingface，如果找不到去 actor 目录找
    hf_dir = base_path / "huggingface"
    if not hf_dir.exists():
        print(f"      -> 根目录未找到 huggingface，尝试在权重目录查找...")
        hf_dir = weight_path / "huggingface"
    
    if not hf_dir.exists():
        raise ValueError(f"❌ 错误: 在 {base_path} 或 {weight_path} 中均未找到 'huggingface' 配置目录。")
    
    print(f"      -> Config/Tokenizer 路径: {hf_dir}")

    # 2. 从huggingface子目录加载tokenizer和config
    print(f"[2/6] 加载 Tokenizer 和 Config...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(hf_dir))
        config = AutoConfig.from_pretrained(str(hf_dir))
    except Exception as e:
        raise RuntimeError(f"加载 Config 失败: {e}")

    # 3. 从config初始化模型架构（不加载权重）
    print("[3/6] 初始化模型架构 (Empty Model)...")
    # 强制使用 bfloat16 以节省内存并匹配常见训练设置
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16 
    )
    model = model.cpu()

    # 4. 加载FSDP分片权重
    print(f"[4/6] 从 {weight_path.name} 加载 FSDP 分片权重...")
    rank_files = sorted(weight_path.glob("model_world_size_*_rank_*.pt"))

    if not rank_files:
        raise ValueError(f"❌ 错误: 在 {weight_path} 未找到任何 model_world_size_*_rank_*.pt 文件")

    print(f"      找到 {len(rank_files)} 个分片文件")

    # 加载所有rank的state dict
    state_dicts = []
    for rank_file in rank_files:
        print(f"      正在读取: {rank_file.name} ...")
        # weights_only=False 是因为 checkpoint 可能包含其他元数据
        state_dict = torch.load(rank_file, map_location="cpu", weights_only=False)
        state_dicts.append(state_dict)

    # 5. 合并FSDP分片并转换DTensor
    print("[5/6] 合并 FSDP 分片并处理 DTensor...")

    merged_state_dict = {}
    # 使用第一个rank的keys作为基准
    all_keys = list(state_dicts[0].keys())

    for key in all_keys:
        # 收集所有rank中这个key的tensor
        tensors = []
        for sd in state_dicts:
            if key in sd:
                # 转换DTensor为普通tensor
                tensor = dtensor_to_tensor(sd[key])
                tensors.append(tensor)

        if len(tensors) == 0:
            continue
        elif len(tensors) == 1:
            merged_state_dict[key] = tensors[0]
        else:
            # 尝试合并
            first_tensor = tensors[0]
            # 标量或单元素tensor通常是相同的元数据
            if first_tensor.dim() == 0 or first_tensor.numel() == 1:
                merged_state_dict[key] = first_tensor
            else:
                # 检查所有tensor是否完全相同（例如LayerNorm层在FSDP中通常是复制的）
                all_same = all(torch.equal(t, first_tensor) for t in tensors[1:])
                if all_same:
                    # 所有rank的值相同，直接使用
                    merged_state_dict[key] = first_tensor
                else:
                    # FSDP默认沿第0维切分参数，进行拼接
                    try:
                        merged_state_dict[key] = torch.cat(tensors, dim=0)
                    except Exception as e:
                        print(f"      警告: key '{key}' 合并失败，默认使用 Rank 0 的数据. Error: {e}")
                        merged_state_dict[key] = first_tensor

    print(f"      成功合并 {len(merged_state_dict)} 个参数")

    # 6. 加载合并后的权重到模型
    print("      正在将权重加载到 HF 模型架构中...")
    missing_keys, unexpected_keys = model.load_state_dict(merged_state_dict, strict=False)

    if missing_keys:
        print(f"      ⚠️ 警告: 缺失 Keys: {len(missing_keys)} 个 (如果是 _extra_state 等可忽略)")
    if unexpected_keys:
        print(f"      ⚠️ 警告: 多余 Keys: {len(unexpected_keys)} 个")

    # 7. 保存为标准HuggingFace格式
    print(f"[6/6] 保存标准 HF 模型到: {output_path} ...")
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(
        str(output_path),
        safe_serialization=True, # 保存为 safetensors
        max_shard_size="5GB"
    )
    tokenizer.save_pretrained(str(output_path))

    print("\n✅ 转换成功!")
    print(f"   输出目录: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Verl Checkpoint 转换工具 (SFT/RL 通用版)")
        print("使用方法:")
        print("  python convert_fsdp2hf.py <输入checkpoint路径> <输出hf路径>")
        print("\n示例:")
        print("  python data_processing/convert_fsdp2hf.py checkpoints/sleep_coach_rl/global_step_3 checkpoints/sleep_coach_rl_hf")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    output_dir = sys.argv[2]

    convert_fsdp_checkpoint_to_hf(checkpoint_dir, output_dir)