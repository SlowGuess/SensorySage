import requests
import logging
import os

logger = logging.getLogger(__name__)

API_URL = os.environ.get("REWARD_API_URL", "http://127.0.0.1:6009/get_reward2")

def sleep_coaching_reward(data_source, solution_str, ground_truth, extra_info=None):
    try:
        payload = {
            "data_source": data_source,
            "response_str": solution_str,
            "ground_truth": ground_truth or "",
            "extra_info": extra_info or {},
        }

        response = requests.post(API_URL, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()

        if "score" not in result:
            logger.warning(f"Reward API missing 'score': {result}")
            result["score"] = 0.0

        logger.info(f"Reward score={result['score']:.3f}")
        return result

    except Exception as e:
        logger.error(f"Reward API call failed: {e}")
        return {
            "score": 0.0,
            "error": str(e),
        }
