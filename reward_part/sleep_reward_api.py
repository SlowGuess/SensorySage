#!/usr/bin/env python3
"""
睡眠健康 Coaching 的 LLM-as-Judge Reward API（Qwen3-8B 单卡版）
用于 GRPO 在线 RL 训练的本地 Reward 服务
"""

import asyncio
import json
import logging
import os
import re
from typing import Dict

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer

from rubric_config import (
    DATA_GROUNDING_RUBRICS,
    CAUSAL_COHERENCE_RUBRICS,
    REASONING_DEPTH_RUBRICS,
    QUALITY_RUBRICS,
    REWARD_WEIGHTS,
)

# =========================
# 日志配置
# =========================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# =========================
# FastAPI
# =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 模型配置
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "JUDGE_MODEL_PATH",
    os.path.join(BASE_DIR, "..", "verl", "models", "Qwen3-8B"),
)



judge_model = None
judge_tokenizer = None


def get_input_device(model) -> str:
    """多卡/单卡统一获取输入设备。单卡时通常就是 cuda:0。"""
    if hasattr(model, "hf_device_map"):
        embed_device = model.hf_device_map.get("model.embed_tokens", 0)
        if isinstance(embed_device, int):
            return f"cuda:{embed_device}"
        return str(embed_device)

    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def load_judge_model():
    """加载本地 Judge 模型（启动时调用一次）"""
    global judge_model, judge_tokenizer

    if judge_model is not None:
        return

    logger.info("=" * 60)
    logger.info("📦 Loading local Judge model...")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Path exists: {os.path.isdir(MODEL_PATH)}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    logger.info(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    try:
        judge_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )

        judge_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        judge_model.eval()

        logger.info("✅ Judge model loaded successfully!")
        logger.info(f"Input device: {get_input_device(judge_model)}")
        if hasattr(judge_model, "hf_device_map"):
            logger.info(f"Device map: {judge_model.hf_device_map}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Failed to load judge model: {e}")
        raise


def call_local_judge(prompt: str, max_new_tokens: int = 256) -> str:
    """
    调用本地 Judge 模型生成评分结果
    注意：Qwen3 显式关闭 thinking，提升结构化输出稳定性和速度。
    """
    global judge_model, judge_tokenizer

    if judge_model is None:
        load_judge_model()

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator for AI-generated sleep coaching advice. "
                "Return valid JSON only. No markdown, no explanations outside JSON."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    text = judge_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    input_device = get_input_device(judge_model)

    inputs = judge_tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
    ).to(input_device)

    with torch.no_grad():
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=judge_tokenizer.eos_token_id,
            pad_token_id=judge_tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = judge_tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def build_rubric_prompt(layer_name: str, rubrics: Dict, user_data: str, response: str) -> str:
    """
    构建某一层 Rubric 的评估 prompt
    """
    insights = ""
    etiology = ""
    recommendations = ""

    if "<INSIGHTS>" in response:
        insights_match = re.search(r"<INSIGHTS>(.*?)</INSIGHTS>", response, re.DOTALL)
        etiology_match = re.search(r"<ETIOLOGY>(.*?)</ETIOLOGY>", response, re.DOTALL)
        recommendations_match = re.search(r"<RECOMMENDATIONS>(.*?)</RECOMMENDATIONS>", response, re.DOTALL)

        if insights_match:
            insights = insights_match.group(1).strip()
        if etiology_match:
            etiology = etiology_match.group(1).strip()
        if recommendations_match:
            recommendations = recommendations_match.group(1).strip()
    else:
        recommendations = response

    rubric_descriptions = []
    for rubric_id, rubric_info in rubrics.items():
        rubric_desc = f"""## Rubric {rubric_id}: {rubric_info['name']}
{rubric_info['description']}
- Pass if: {rubric_info['pass_criteria']}
- Fail if: {rubric_info['fail_criteria']}
"""
        rubric_descriptions.append(rubric_desc)

    rubric_text = "\n".join(rubric_descriptions)

    output_format = "{\n"
    for rubric_id in rubrics.keys():
        output_format += f'  "{rubric_id}": {{"pass": true/false, "reasoning": "..."}},\n'
    output_format += '  "summary": "..."\n}'

    prompt = f"""You are evaluating a sleep coaching response. Evaluate the following {layer_name} rubrics.

# User Sleep Data
{user_data[:1200]}

# Model Response

## Insights
{insights if insights else "[Not available]"}

## Etiology
{etiology if etiology else "[Not available]"}

## Recommendations
{recommendations if recommendations else "[Not available]"}

# Evaluation Task
Evaluate the response against the rubrics below.

{rubric_text}

# Output Format
Return ONLY a JSON object:
{output_format}

Be strict, accurate, and concise.
"""
    return prompt


def evaluate_layer(layer_name: str, rubrics: Dict, user_data: str, response: str) -> Dict:
    """
    评估单层 Rubric
    """
    prompt = build_rubric_prompt(layer_name, rubrics, user_data, response)

    try:
        llm_response = call_local_judge(prompt, max_new_tokens=512)

        cleaned = llm_response.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group()

        result = json.loads(cleaned)

        passed_count = sum(
            1 for rubric_id in rubrics.keys()
            if result.get(rubric_id, {}).get("pass", False)
        )
        score = passed_count / len(rubrics)

        result["layer_score"] = score
        result["passed_count"] = passed_count
        result["total_count"] = len(rubrics)

        return result

    except Exception as e:
        logger.error(f"❌ Failed to evaluate {layer_name}: {e}")
        logger.error(f"Raw response: {llm_response[:300] if 'llm_response' in locals() else 'N/A'}")

        default_result = {
            rubric_id: {"pass": False, "reasoning": "Evaluation failed"}
            for rubric_id in rubrics.keys()
        }
        default_result["layer_score"] = 0.0
        default_result["passed_count"] = 0
        default_result["total_count"] = len(rubrics)
        default_result["error"] = str(e)

        return default_result


def compute_final_reward(
    dg_result: Dict,
    cc_result: Dict,
    rd_result: Dict,
    od_result: Dict,
) -> Dict:
    """
    计算最终 reward（无门控）
    """
    dg_score = dg_result.get("layer_score", 0.0)
    cc_score = cc_result.get("layer_score", 0.0)
    rd_score = rd_result.get("layer_score", 0.0)
    od_score = od_result.get("layer_score", 0.0)

    final_reward = (
        REWARD_WEIGHTS["data_grounding"] * dg_score
        + REWARD_WEIGHTS["causal_coherence"] * cc_score
        + REWARD_WEIGHTS["reasoning_depth"] * rd_score
        + REWARD_WEIGHTS["quality"] * od_score
    )

    return {
        "score": float(final_reward),
        "judge_score": float(final_reward),
        "components": {
            "data_grounding": float(dg_score),
            "causal_coherence": float(cc_score),
            "reasoning_depth": float(rd_score),
            "quality": float(od_score),
        },
        "weights": REWARD_WEIGHTS,
        "details": {
            "data_grounding": dg_result,
            "causal_coherence": cc_result,
            "reasoning_depth": rd_result,
            "quality": od_result,
        },
    }


def evaluate_sleep_coaching(
    data_source: str,
    response_str: str,
    ground_truth: str = "",
    extra_info: Dict = None,
) -> Dict:
    """
    完整评估睡眠 coaching 回答（4 层 Rubric）
    """
    user_sleep_data = ""
    if extra_info:
        user_sleep_data = extra_info.get("user_sleep_data", "")

    if not user_sleep_data:
        user_sleep_data = data_source

    logger.info("🔍 Evaluating with 4-layer Rubric system...")

    logger.info("  [1/4] Evaluating Data Grounding...")
    dg_result = evaluate_layer("Data Grounding", DATA_GROUNDING_RUBRICS, user_sleep_data, response_str)
    logger.info(f"      Score: {dg_result.get('layer_score', 0):.2f}")

    logger.info("  [2/4] Evaluating Causal Coherence...")
    cc_result = evaluate_layer("Causal Coherence", CAUSAL_COHERENCE_RUBRICS, user_sleep_data, response_str)
    logger.info(f"      Score: {cc_result.get('layer_score', 0):.2f}")

    logger.info("  [3/4] Evaluating Reasoning Depth...")
    rd_result = evaluate_layer("Reasoning Depth", REASONING_DEPTH_RUBRICS, user_sleep_data, response_str)
    logger.info(f"      Score: {rd_result.get('layer_score', 0):.2f}")

    logger.info("  [4/4] Evaluating Quality...")
    od_result = evaluate_layer("Quality", QUALITY_RUBRICS, user_sleep_data, response_str)
    logger.info(f"      Score: {od_result.get('layer_score', 0):.2f}")

    result = compute_final_reward(dg_result, cc_result, rd_result, od_result)

    logger.info(f"✅ Final Reward: {result['score']:.3f}")
    logger.info(
        f"   Components: DG={result['components']['data_grounding']:.2f}, "
        f"CC={result['components']['causal_coherence']:.2f}, "
        f"RD={result['components']['reasoning_depth']:.2f}, "
        f"OD={result['components']['quality']:.2f}"
    )

    return result


@app.post("/get_reward2")
async def get_reward2(request: Request):
    """
    VERL 调用的 reward 接口（在线 RL 训练）
    """
    json_data = await request.json()

    logger.info("=" * 60)
    logger.info("🎯 Reward API Called (Online RL)")

    data_source = json_data.get("data_source", "")
    extra_info = json_data.get("extra_info", {})
    user_sleep_data = extra_info.get("user_sleep_data", "")
    response_str = json_data.get("response_str", "")
    ground_truth = json_data.get("ground_truth", "")

    if isinstance(response_str, list):
        response_str = response_str[0]

    logger.info(f"📝 Response length: {len(response_str)} chars")
    logger.info(f"📊 User data length: {len(user_sleep_data)} chars")

    result = await asyncio.to_thread(
        evaluate_sleep_coaching,
        data_source,
        response_str,
        ground_truth,
        extra_info,
    )

    logger.info(f"📤 Returning reward: {result.get('score', 0):.3f}")
    logger.info("=" * 60)

    return result


@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    return {
        "status": "healthy",
        "service": "sleep_coaching_reward_api",
        "model": "Qwen3-8B (Local)",
        "rubrics": "19 binary rubrics (4 layers)",
        "weights": REWARD_WEIGHTS,
        "model_loaded": judge_model is not None,
    }


@app.get("/test")
async def test_api():
    """
    轻量测试接口：只检查模型是否加载，不做真实生成
    """
    try:
        if judge_model is None:
            load_judge_model()

        return {
            "status": "success",
            "model_loaded": True,
            "input_device": get_input_device(judge_model),
            "device_map": getattr(judge_model, "hf_device_map", {}),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 60)
    logger.info("🚀 Starting Sleep Coaching Reward API (Qwen3-8B)")
    logger.info("🤖 Model: Qwen3-8B (Local)")
    logger.info("📊 Rubrics: 19 binary (4 layers)")
    logger.info(f"⚖️  Weights: {REWARD_WEIGHTS}")
    logger.info("🌐 Server: http://0.0.0.0:6009")
    logger.info("=" * 60)

    load_judge_model()
    uvicorn.run(app, host="0.0.0.0", port=6009)