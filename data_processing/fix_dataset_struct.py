import pandas as pd

def fix_dataset(parquet_path):
    df = pd.read_parquet(parquet_path)

    def extract_user_message(prompt_array):
        """从prompt数组提取用户睡眠数据"""
        for msg in prompt_array:
            if msg.get('role') == 'user':
                return msg.get('content', '')
        return ''

    # 1. 添加data_source字段（数据集标识符）
    df['data_source'] = 'sleep_health_coaching'

    # 2. 更新extra_info，添加睡眠数据
    def update_extra_info(row):
        extra_info = row.get('extra_info', {})
        if not isinstance(extra_info, dict):
            extra_info = {}
        # 添加睡眠数据到extra_info
        extra_info['user_sleep_data'] = extract_user_message(row['prompt'])
        return extra_info

    df['extra_info'] = df.apply(update_extra_info, axis=1)

    # 保存
    df.to_parquet(parquet_path, index=False)
    print(f"✓ 已更新: {parquet_path}")
    print(f"  - 数据条数: {len(df)}")
    print(f"  - data_source: {df.iloc[0]['data_source']}")
    print(f"  - extra_info keys: {list(df.iloc[0]['extra_info'].keys())}")
    print(f"  - user_sleep_data长度: {len(df.iloc[0]['extra_info']['user_sleep_data'])} 字符\n")

# 处理训练集和验证集
fix_dataset('dataset/parquet/train_with_reward.parquet')
fix_dataset('dataset/parquet/val_with_reward.parquet')