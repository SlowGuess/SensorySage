"""
睡眠健康Coaching的自定义Reward函数
调用LLM-as-Judge API进行评分
"""
import requests
import logging

logger = logging.getLogger(__name__)

def sleep_coaching_reward(data_source, solution_str, ground_truth, extra_info=None):
    """
    调用Reward API进行评分
    
    Args:
        data_source: 数据集标识符 (如 'sleep_health_coaching')
        solution_str: 模型生成的response
        ground_truth: 参考答案（可能不用）
        extra_info: 额外信息，包含user_sleep_data
    
    Returns:
        dict: 包含 'score' 字段的字典
    """
    try:
        # 构建API请求
        api_url = "http://0.0.0.0:6009/get_reward2"

        payload = {
            "data_source": data_source,
            "response_str": solution_str,
            "ground_truth": ground_truth or "",
            "extra_info": extra_info or {}
        }

        # 调用API
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()

        # 确保返回包含 'score' 字段
        if 'score' not in result:
            logger.warning(f"API返回缺少'score'字段: {result}")
            result['score'] = 0.0

        logger.info(f"✓ Reward API返回: score={result['score']:.3f}")
        return result

    except Exception as e:
        logger.error(f"❌ Reward API调用失败: {e}")
        # 返回默认低分
        return {
            "score": 0.0,
            "error": str(e)
        }