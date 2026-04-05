"""
Rubric配置 - 用于RL训练的reward计算
19个二元Rubrics，分为4层，无门控机制
"""

# Layer 1: Data Grounding (权重提高到 0.35)
DATA_GROUNDING_RUBRICS = {
    "DG1": {
        "name": "Data Citation Accuracy",
        "description": "引用用户数据时，是否准确无误（数字、时间、事件等）",
        "pass_criteria": "所有引用的用户数据（如睡眠时长、入睡时间等）与原始输入完全一致",
        "fail_criteria": "存在任何数据引用错误、数字偏差或时间点错误"
    },
    "DG2": {
        "name": "Data Interpretation Validity",
        "description": "对用户数据的解读是否合理、符合常识",
        "pass_criteria": "所有基于数据的推断都有明确的逻辑链条，符合睡眠科学常识",
        "fail_criteria": "存在基于数据的不合理推断或违反常识的解释"
    },
    "DG3": {
        "name": "No Hallucination",
        "description": "是否避免编造用户数据中不存在的信息",
        "pass_criteria": "没有添加用户未提供的症状、习惯或背景信息",
        "fail_criteria": "编造了用户未提及的信息（如假设用户有某种习惯、症状等）"
    },
    "DG4": {
        "name": "Sufficient Data Support",
        "description": "所有结论是否都有用户数据支撑",
        "pass_criteria": "每个主要结论（insights/etiology/recommendations）都明确引用了支持它的用户数据",
        "fail_criteria": "存在缺乏数据支撑的空泛结论或猜测性陈述"
    }
}

# Layer 2: Causal Coherence (权重 0.35)
CAUSAL_COHERENCE_RUBRICS = {
    "CC1": {
        "name": "Insights → Etiology Coherence",
        "description": "Insights中识别的模式是否自然过渡到Etiology的根因分析",
        "pass_criteria": "Etiology明确引用了Insights中的关键发现，并基于这些发现展开根因分析",
        "fail_criteria": "Etiology与Insights脱节，或忽略了Insights中的重要发现"
    },
    "CC2": {
        "name": "Etiology → Recommendations Coherence",
        "description": "Recommendations是否针对性地解决Etiology中识别的根因",
        "pass_criteria": "每条建议都明确对应Etiology中的某个根因，形成清晰的'问题-解决方案'映射",
        "fail_criteria": "建议与根因脱节，或存在根因未被建议覆盖的情况"
    },
    "CC3": {
        "name": "Complete Reasoning Chain",
        "description": "从数据观察到根因再到建议，推理链条是否完整",
        "pass_criteria": "能够追溯每条建议的完整因果链：数据→Insights→Etiology→Recommendations",
        "fail_criteria": "推理链条存在断裂，无法追溯某些结论的来源"
    },
    "CC4": {
        "name": "No Causal Leaps",
        "description": "是否避免了因果推理中的逻辑跳跃",
        "pass_criteria": "每个因果推断都有明确的中间步骤，没有突然的结论",
        "fail_criteria": "存在未经解释的因果跳跃或缺失的推理步骤"
    },
    "CC5": {
        "name": "Internal Logical Consistency",
        "description": "三个section之间是否存在逻辑矛盾",
        "pass_criteria": "三个section的陈述相互支持，没有互相矛盾的信息",
        "fail_criteria": "存在前后矛盾的陈述或相互冲突的结论"
    }
}

# Layer 3: Reasoning Depth (权重 0.15)
REASONING_DEPTH_RUBRICS = {
    "RD1": {
        "name": "Visible Reasoning Process",
        "description": "是否展示了清晰的推理过程",
        "pass_criteria": "包含明确的推理步骤、证据权衡、假设检验等过程",
        "fail_criteria": "推理过程缺失、过于简单或仅是结论的堆砌"
    },
    "RD2": {
        "name": "Multi-Perspective Analysis",
        "description": "是否从多个角度分析问题（生理、心理、环境、习惯等）",
        "pass_criteria": "考虑了至少3个不同维度的因素，并分析了它们的相互作用",
        "fail_criteria": "分析单一、片面，只关注某一个维度"
    },
    "RD3": {
        "name": "Self-Correction Presence",
        "description": "是否存在自我质疑、修正或权衡不同可能性的过程",
        "pass_criteria": "明确展示了'但是...'、'另一方面...'、'更可能的是...'等自我修正",
        "fail_criteria": "没有任何自我质疑或修正，推理过程过于线性"
    },
    "RD4": {
        "name": "Deep vs Shallow Reasoning",
        "description": "推理是否深入到根本原因，而非停留在表面",
        "pass_criteria": "至少深入到'为什么会这样'的第二层或第三层原因",
        "fail_criteria": "仅描述现象或给出一层浅显的解释"
    }
}

# Layer 4: Quality (权重 0.15)
QUALITY_RUBRICS = {
    "OD1": {
        "name": "Relevance & Personalization",
        "description": "建议是否与用户的具体情况高度相关且个性化",
        "pass_criteria": "建议明确针对用户的具体情况（年龄、性别、作息、症状等），而非泛泛而谈",
        "fail_criteria": "建议过于通用，缺乏个性化，适用于任何人"
    },
    "OD2": {
        "name": "Accuracy & Evidence",
        "description": "建议是否准确且有科学依据",
        "pass_criteria": "建议符合睡眠科学常识，没有错误或误导性信息",
        "fail_criteria": "包含错误信息、过时建议或缺乏科学依据的说法"
    },
    "OD3": {
        "name": "Specificity & Actionability",
        "description": "建议是否具体且可执行",
        "pass_criteria": "建议包含具体的时间、方法、步骤等，用户可以立即采取行动",
        "fail_criteria": "建议模糊、抽象，用户不知道如何执行"
    },
    "OD4": {
        "name": "Comprehensiveness",
        "description": "是否全面覆盖了用户问题的各个方面",
        "pass_criteria": "识别了用户的主要和次要问题，给出了多角度的建议",
        "fail_criteria": "遗漏了明显的问题或只关注部分问题"
    },
    "OD5": {
        "name": "Clarity & Presentation",
        "description": "表达是否清晰、易懂、结构良好",
        "pass_criteria": "语言清晰、逻辑清楚、结构合理，易于理解",
        "fail_criteria": "表达混乱、逻辑不清或结构混乱"
    },
    "OD6": {
        "name": "Safety & Appropriateness",
        "description": "建议是否安全且适当（考虑用户的年龄、健康状况等）",
        "pass_criteria": "建议安全、适度，对特殊人群（如老年人）有适当的注意事项",
        "fail_criteria": "建议可能不安全或不适合用户的具体情况"
    }
}

# Reward权重配置（去掉门控机制）
REWARD_WEIGHTS = {
    'data_grounding': 0.35,    # 提高Data Grounding权重
    'causal_coherence': 0.35,  # Causal Coherence（核心创新）
    'reasoning_depth': 0.15,   # Reasoning Depth
    'quality': 0.15            # Quality
}

# 所有Rubrics汇总
ALL_RUBRICS = {
    **DATA_GROUNDING_RUBRICS,
    **CAUSAL_COHERENCE_RUBRICS,
    **REASONING_DEPTH_RUBRICS,
    **QUALITY_RUBRICS
}
