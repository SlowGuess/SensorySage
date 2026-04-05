#!/usr/bin/env python3
"""
测试 Reward API 服务
验证本地模型和 Rubric 评估系统是否正常工作
"""

import json
import time
import traceback
import requests

BASE_URL = "http://127.0.0.1:6009"


def test_health_check():
    """测试健康检查接口"""
    print("=" * 60)
    print("Test 1: Health Check")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        response.raise_for_status()
        result = response.json()

        print("✅ Health check passed!")
        print(f"   Status: {result.get('status')}")
        print(f"   Model: {result.get('model')}")
        print(f"   Rubrics: {result.get('rubrics')}")
        print(f"   Weights: {result.get('weights')}")
        print(f"   Model loaded: {result.get('model_loaded')}")
        return True

    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("\n💡 Hint: Make sure the API server is running:")
        print("   cd /home/lsy/Projects/verl/reward_part")
        print("   export CUDA_VISIBLE_DEVICES=3")
        print("   python3 sleep_reward_api.py")
        return False


def test_model_loading():
    """测试模型加载状态"""
    print("\n" + "=" * 60)
    print("Test 2: Model Loading")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/test", timeout=10)
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "success":
            print("✅ Model loaded successfully!")
            print(f"   Model loaded: {result.get('model_loaded')}")
            print(f"   Input device: {result.get('input_device')}")
            print(f"   Device map: {result.get('device_map')}")
            return True
        else:
            print(f"❌ Model loading failed: {result.get('error')}")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_reward_calculation():
    """测试完整的 reward 计算"""
    print("\n" + "=" * 60)
    print("Test 3: Reward Calculation")
    print("=" * 60)

    test_data = {
        "data_source": "sleep_health_coaching",
        "response_str": """<INSIGHTS>
The user shows a significant sleep onset latency of 60-90 minutes, well above the normal 10-20 minutes.
The total sleep time of 6-6.5 hours is below the recommended 7-9 hours for adults, indicating chronic sleep deprivation.
The afternoon nap at 1 PM may be reducing nighttime sleep pressure.
</INSIGHTS>

<ETIOLOGY>
The late caffeine consumption (last cup at 3 PM) means caffeine remains active at bedtime (11 PM), interfering with sleep drive.
The afternoon nap reduces homeostatic sleep pressure needed for nighttime sleep.
Smartphone use in bed creates dual problems: blue light suppresses melatonin, and engaging content keeps the mind active.
</ETIOLOGY>

<RECOMMENDATIONS>
1. Eliminate the afternoon nap to rebuild sleep pressure for nighttime.
2. Move caffeine cutoff to no later than 12 PM (noon).
3. Stop smartphone use 60 minutes before bedtime (by 10 PM).
4. Practice stimulus control: if unable to sleep after 20 minutes, get up and do a quiet activity.
</RECOMMENDATIONS>""",
        "ground_truth": "",
        "extra_info": {
            "user_sleep_data": """The user is a 45-year-old male office worker.
He reports difficulty falling asleep at night, usually taking 60-90 minutes to fall asleep.
He goes to bed around 11 PM but often lies awake until 12:30 AM or later.
He wakes up at 7 AM for work, getting approximately 6-6.5 hours of sleep per night.
He reports feeling tired during the day and sometimes takes a 30-minute nap after lunch around 1 PM.
He drinks 2-3 cups of coffee per day, with the last cup around 3 PM.
He uses his smartphone in bed while trying to fall asleep."""
        }
    }

    try:
        print("📤 Sending request...")
        print(f"   User data length: {len(test_data['extra_info']['user_sleep_data'])} chars")
        print(f"   Response length: {len(test_data['response_str'])} chars")

        start_time = time.time()

        response = requests.post(
            f"{BASE_URL}/get_reward2",
            json=test_data,
            timeout=300
        )
        response.raise_for_status()

        elapsed_time = time.time() - start_time
        result = response.json()

        print(f"\n✅ Reward calculation completed! (took {elapsed_time:.1f}s)")
        print("\n📊 Results:")
        print(f"   Final Reward: {result.get('score', 0):.3f}")

        if "components" in result:
            components = result["components"]
            print("\n   Component Scores:")
            print(f"      Data Grounding:    {components.get('data_grounding', 0):.2f}")
            print(f"      Causal Coherence:  {components.get('causal_coherence', 0):.2f}")
            print(f"      Reasoning Depth:   {components.get('reasoning_depth', 0):.2f}")
            print(f"      Quality:           {components.get('quality', 0):.2f}")

        if "weights" in result:
            print(f"\n   Weights: {result['weights']}")

        with open("test_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("\n💾 Full result saved to: test_result.json")
        return True

    except requests.exceptions.Timeout:
        print("❌ Request timeout (>5 minutes)")
        print("   This may happen if the 4-layer evaluation is still running.")
        return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("REWARD API TEST SUITE")
    print("=" * 60)
    print("\nThis will test the local Reward API service.")
    print("Make sure the API server is running first.\n")

    if not test_health_check():
        print("\n⚠️  API server not running. Please start it first:")
        print("   cd /home/lsy/Projects/verl/reward_part")
        print("   export CUDA_VISIBLE_DEVICES=3")
        print("   python3 sleep_reward_api.py")
        return

    if not test_model_loading():
        print("\n⚠️  Model loading failed. Check the API logs.")
        return

    print("\n⚠️  The next test may take around 20-90 seconds depending on prompt length.")
    input("Press Enter to continue...")

    if not test_reward_calculation():
        print("\n⚠️  Reward calculation failed. Check the API logs.")
        return

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe Reward API is ready for RL training.")
    print("\nNext steps:")
    print("1. Keep the API server running")
    print("2. Start GRPO RL training on GPUs 0,1,2")
    print("3. Keep reward judge on GPU 3")
    print("4. Monitor reward latency and score distribution")
    print("=" * 60)


if __name__ == "__main__":
    main()