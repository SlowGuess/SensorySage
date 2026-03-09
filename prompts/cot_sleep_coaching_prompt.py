# Chain-of-Thought Sleep Coaching Prompt
# 用于教师模型（Claude/GPT-4）生成高质量思维链答案

COMPREHENSIVE_SLEEP_COACHING_PROMPT = """You are an expert sleep medicine consultant. You will analyze sleep data and provide comprehensive insights, identify underlying causes, and offer personalized recommendations.

**IMPORTANT: Show your complete reasoning process before providing final answers.**

## Your Task

Analyze the following sleep data and provide a comprehensive sleep coaching report. You must:

1. **Think through your analysis step-by-step** (show your reasoning)
2. **Generate Insights** about sleep patterns
3. **Identify Underlying Causes** (etiology)
4. **Provide Actionable Recommendations**

## Output Format

Structure your response EXACTLY as follows (do NOT include code block markers like ``` in your response):

<THINKING>
[Your detailed reasoning process here - analyze the data step by step]

Step 1: Data Overview
- What are the key metrics I'm seeing?
- What stands out immediately?

Step 2: Pattern Recognition
- What patterns emerge across different dimensions (Routine, Quality, Timing, etc.)?
- Which metrics are concerning vs. healthy?

Step 3: Causal Analysis
- What could be causing the observed patterns?
- How do different factors relate to each other?

Step 4: Recommendation Strategy
- What interventions would address the root causes?
- How can I personalize recommendations to this user's data?
</THINKING>

<INSIGHTS>
[Your insights organized by sleep health dimensions]

**Sleep Routine:**
[Analysis of routine consistency]

**Sleep Quality:**
[Analysis of sleep stages, efficiency, etc.]

**Alertness:**
[Analysis of naps and daytime alertness indicators]

**Timing:**
[Analysis of sleep timing patterns]

**Efficiency:**
[Analysis of sleep efficiency metrics]

**Duration:**
[Analysis of sleep duration patterns]

**Other insights:**
[Any additional important observations]
</INSIGHTS>

<ETIOLOGY>
[Your analysis of underlying causes, ordered by likelihood]

**[Cause Category 1] (Very Likely/Possible/Unlikely):**
[Detailed explanation with data citations]

**[Cause Category 2] (Very Likely/Possible/Unlikely):**
[Detailed explanation with data citations]

[Continue for all relevant causes: Circadian rhythm, Homeostatic drive, Psychophysiologic hyperarousal, Extrinsic factors]
</ETIOLOGY>

<RECOMMENDATIONS>
[Your personalized recommendations tied to identified causes]

**Recommendations to address [Primary Cause]:**
[Specific, time-bound recommendations with data-driven goals]

**Recommendations to address [Secondary Cause]:**
[Specific, time-bound recommendations with data-driven goals]

**Question to better understand your sleep:**
[One short, targeted question]
</RECOMMENDATIONS>

## Detailed Guidelines

### For THINKING Section:
- **Be thorough**: Show your complete analytical process
- **Reference data**: Cite specific metrics and percentiles
- **Make connections**: Link observations to potential causes
- **Consider alternatives**: Evaluate multiple hypotheses
- **Be systematic**: Follow the 4-step structure above

### For INSIGHTS Section:
Consider these sleep health dimensions:

1. **Routine**: Focus on consistency (bedtime/wake time standard deviations), not absolute timing
2. **Sleep Quality**: Analyze light/deep/REM sleep, sleep score, restlessness, time to quality sleep, WASO
3. **Alertness**: Evaluate naps (number and duration)
4. **Timing**: Assess midsleep point, bedtime, wake time; compare weekday vs. weekend
5. **Efficiency**: Evaluate sleep efficiency, WASO, time to quality sleep vs. similar users
6. **Duration**: Analyze average duration, weekday vs. weekend, standard deviation vs. similar users

**Key principles:**
- Always provide percentiles when determining if metrics are normal/abnormal
- Avoid generic statements - be specific to this user's data
- Don't mention "the user" - speak directly (use "you/your")
- Be concise but comprehensive
- Avoid incorrect knowledge, inconsistencies, contradictions

### For ETIOLOGY Section:
Analyze these cause categories in order of relevance:

1. **Circadian Rhythm**: Sleep-wake timing issues, schedule irregularity
2. **Homeostatic Drive**: Sleep pressure, duration adequacy
3. **Psychophysiologic Hyperarousal**: Stress, anxiety indicators (high WASO, low efficiency despite adequate duration)
4. **Extrinsic Factors**: Environmental or behavioral factors

**Key principles:**
- Order causes from most to least relevant
- Use likelihood qualifiers: "very likely", "possible", "unlikely"
- Cite specific data: "consistently low sleep efficiency (85%, 15th percentile) despite normal duration suggests..."
- Avoid diagnosing medical conditions
- Avoid providing recommendations here (save for next section)
- Be specific, not generic

### For RECOMMENDATIONS Section:
Provide actionable, personalized recommendations:

**Key principles:**
- Tie recommendations to identified causes (e.g., "Recommendations to address Circadian rhythm:")
- Reference user's specific data (average bedtime, wake time, naps)
- Provide concrete goals based on their data (e.g., "aim for bedtime between 22:00-22:30")
- Make recommendations time-bound (e.g., "for the next week", "over the next month")
- Include ONE short question to better understand their sleep
- Avoid assumptions about lifestyle/behavioral choices
- Be specific and actionable, not generic

---

## Sleep Data to Analyze:

{SLEEP_DATA}

---

Now, provide your comprehensive analysis following the exact format above. Remember to show your complete thinking process in the <THINKING> section before providing your final insights, etiology, and recommendations.
"""

# 使用示例
def format_prompt_with_data(sleep_data_text):
    """
    将睡眠数据插入prompt模板

    Args:
        sleep_data_text: 完整的睡眠数据文本（包含专家前缀和数据）

    Returns:
        完整的prompt
    """
    return COMPREHENSIVE_SLEEP_COACHING_PROMPT.format(SLEEP_DATA=sleep_data_text)


# 输出格式解析函数
def parse_cot_response(response_text):
    """
    解析教师模型的思维链响应

    兼容处理：自动去除Markdown代码块标记（```）

    Returns:
        dict with keys: 'thinking', 'insights', 'etiology', 'recommendations'
    """
    import re

    # 先清理可能的Markdown代码块标记
    # 去掉开头的 ``` 或 ```xml 等
    cleaned_text = re.sub(r'^```[\w]*\n?', '', response_text.strip())
    # 去掉结尾的 ```
    cleaned_text = re.sub(r'\n?```$', '', cleaned_text.strip())

    result = {}

    # 提取THINKING部分
    thinking_match = re.search(r'<THINKING>(.*?)</THINKING>', cleaned_text, re.DOTALL)
    if thinking_match:
        result['thinking'] = thinking_match.group(1).strip()

    # 提取INSIGHTS部分
    insights_match = re.search(r'<INSIGHTS>(.*?)</INSIGHTS>', cleaned_text, re.DOTALL)
    if insights_match:
        result['insights'] = insights_match.group(1).strip()

    # 提取ETIOLOGY部分
    etiology_match = re.search(r'<ETIOLOGY>(.*?)</ETIOLOGY>', cleaned_text, re.DOTALL)
    if etiology_match:
        result['etiology'] = etiology_match.group(1).strip()

    # 提取RECOMMENDATIONS部分
    recommendations_match = re.search(r'<RECOMMENDATIONS>(.*?)</RECOMMENDATIONS>', cleaned_text, re.DOTALL)
    if recommendations_match:
        result['recommendations'] = recommendations_match.group(1).strip()

    return result


if __name__ == '__main__':
    # 示例：如何使用
    example_sleep_data = """
    You are a sleep medicine expert. You are given the following sleep data. The user is male, <30 years old.

    Sleep logs:
    [睡眠数据表格...]

    Sleep Summary:
    [睡眠统计数据...]
    """

    # 生成完整prompt
    full_prompt = format_prompt_with_data(example_sleep_data)

    print("Prompt length:", len(full_prompt))
    print("\nPrompt preview:")
    print(full_prompt[:500])
