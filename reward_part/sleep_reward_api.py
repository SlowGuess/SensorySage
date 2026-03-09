# reward_part/sleep_reward_api.py
"""
睡眠健康Coaching的LLM-as-Judge Reward API
使用阿里云百炼通义千问API根据rubric进行打分
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Union
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ 阿里云百炼API配置 ============
DASHSCOPE_API_KEY = "sk-fce73fb564d14c01b3cabb7221deec85"
DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
MODEL_NAME = "qwen-flash"  # 可选: qwen-turbo, qwen-plus, qwen-max

# 睡眠健康Coaching评分Rubric
SLEEP_COACHING_RUBRIC = """
你是一个睡眠健康专家评估员。请根据以下标准对睡眠健康coaching回答进行评分（0-10分）：

## 评分维度（总分10分）

### 1. 完整性 (0-3分)
- 3分：全面覆盖睡眠时长、质量、规律性等所有关键方面
- 2分：覆盖大部分关键方面，但有遗漏
- 1分：只涉及部分方面
- 0分：内容不完整或偏离主题

### 2. 数据驱动 (0-3分)
- 3分：充分引用用户的具体睡眠数据（时长、深睡比例、入睡时间等）
- 2分：引用了部分数据
- 1分：很少引用数据
- 0分：没有引用任何数据

### 3. 专业性 (0-2分)
- 2分：使用准确的睡眠医学术语，解释科学合理
- 1分：基本专业，但有小错误
- 0分：不专业或有明显错误

### 4. 可操作性 (0-2分)
- 2分：提供清晰、具体、可执行的建议
- 1分：建议较笼统
- 0分：没有实际建议

## 评分格式
请严格按照以下JSON格式输出，不要有任何其他文字：
{
    "completeness": <0-3>,
    "data_driven": <0-3>,
    "professional": <0-2>,
    "actionable": <0-2>,
    "total_score": <0-10>,
    "reasoning": "<简短说明>"
}
"""


def call_qwen_api(prompt: str, system_prompt: str = "") -> str:
    """
    调用阿里云百炼通义千问API
    
    Args:
        prompt: 用户输入
        system_prompt: 系统提示词
    
    Returns:
        模型响应文本
    """
    try:
        headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json"
        }

        # 阿里云百炼API格式
        data = {
            "model": MODEL_NAME,
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            "parameters": {
                "temperature": 0.1,  # 低温度保证评分稳定
                "max_tokens": 500,
                "result_format": "message"
            }
        }

        logger.info(f"调用通义千问API: {MODEL_NAME}")

        response = requests.post(
            DASHSCOPE_API_URL,
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()

        result = response.json()

        # 提取响应文本
        if "output" in result and "choices" in result["output"]:
            content = result["output"]["choices"][0]["message"]["content"]
            logger.info(f"API响应成功，长度: {len(content)}")
            return content
        else:
            logger.error(f"API响应格式异常: {result}")
            raise ValueError("API响应格式错误")

    except Exception as e:
        logger.error(f"通义千问API调用失败: {e}")
        # 返回默认低分JSON
        return json.dumps({
            "completeness": 0,
            "data_driven": 0,
            "professional": 0,
            "actionable": 0,
            "total_score": 0,
            "reasoning": f"API调用失败: {str(e)}"
        })


def evaluate_sleep_coaching(
    data_source: str,
    response_str: str,
    ground_truth: str,
    extra_info: Dict = None  # 新增参数
) -> Dict[str, float]:
    """
    使用LLM-as-Judge评估睡眠coaching回答
    
    Args:
        data_source: 用户睡眠数据（prompt中的用户信息）
        response_str: 模型生成的coaching回答
        ground_truth: 专家标注的参考答案
    
    Returns:
        评分字典
    """
    user_sleep_data = ""
    if extra_info:
          user_sleep_data = extra_info.get("user_sleep_data", "")
    # 构建评估prompt
    eval_prompt = f"""
请评估以下睡眠健康coaching回答的质量。

## 用户睡眠数据
{data_source[:500]}...

## 模型生成的回答
{response_str}

请根据评分标准给出分数，只输出JSON，不要有其他文字。
"""

    # 调用通义千问API
    llm_response = call_qwen_api(eval_prompt, SLEEP_COACHING_RUBRIC)

    try:
        # 解析JSON响应
        # 清理可能的markdown包裹
        json_text = llm_response.strip()
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0]

        json_text = json_text.strip()
        scores = json.loads(json_text)

        # 归一化到0-1范围（VERL期望的格式）
        normalized_score = scores["total_score"] / 10.0

        result = {
            "judge_score": float(normalized_score),
            "score": float(normalized_score),  # VERL需要的主要字段
            "completeness": float(scores.get("completeness", 0)),
            "data_driven": float(scores.get("data_driven", 0)),
            "professional": float(scores.get("professional", 0)),
            "actionable": float(scores.get("actionable", 0)),
            "reasoning": scores.get("reasoning", "")
        }

        logger.info(f"✅ 评分结果: score={result['score']:.2f}, details={scores}")
        return result

    except Exception as e:
        logger.error(f"❌ 解析LLM响应失败: {e}")
        logger.error(f"原始响应: {llm_response[:200]}...")
        # 返回默认低分
        return {
            "judge_score": 0.0,
            "score": 0.0,
            "error": str(e),
            "raw_response": llm_response[:200]
        }


@app.post("/get_reward2")
async def get_reward2(request: Request):
    """
    VERL调用的reward接口
    """
    json_data = await request.json()
    logger.info("=" * 50)
    logger.info("🎯 Reward API Called")

    # 提取数据
    data_source = json_data.get("data_source", "sleep_health_coaching")  # 数据集标识符
    extra_info = json_data.get("extra_info", {})
    user_sleep_data = extra_info.get("user_sleep_data", "")  # 实际的睡眠数据
    response_str = json_data.get("response_str", "")
    if isinstance(response_str, list):
      response_str = response_str[0]
    ground_truth = json_data.get("ground_truth", "")

    
    
    # 如果response_str是列表，取第一个
    if isinstance(response_str, list):
        response_str = response_str[0]

    logger.info(f"📝 Response length: {len(response_str)} chars")
    logger.info(f"📊 Data source length: {len(data_source)} chars")

    # 异步调用评估函数
    result = await asyncio.to_thread(
        evaluate_sleep_coaching,
        data_source,
        response_str,
        ground_truth,
        extra_info  # 新增
    )

    logger.info(f"📤 返回分数: {result.get('score', 0):.3f}")
    logger.info("=" * 50)

    return result


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "service": "sleep_coaching_reward_api",
        "model": MODEL_NAME,
        "api": "阿里云百炼通义千问"
    }


@app.get("/test")
async def test_api():
    """测试API连接"""
    try:
        response = call_qwen_api("你好", "你是一个助手")
        return {
            "status": "success",
            "response": response[:100],
            "model": MODEL_NAME
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 60)
    logger.info("🚀 启动Sleep Coaching Reward API服务")
    logger.info(f"📡 API: 阿里云百炼通义千问")
    logger.info(f"🤖 Model: {MODEL_NAME}")
    logger.info(f"🔑 API Key: {DASHSCOPE_API_KEY[:20]}...")
    logger.info(f"🌐 服务地址: http://0.0.0.0:6009")
    logger.info("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=6009)