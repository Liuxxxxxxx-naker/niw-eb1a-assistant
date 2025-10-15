import os
import re
import json
import requests
import streamlit as st
from collections import defaultdict

# -----------------------------
# 0) 全局配置
# -----------------------------
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL_ID = "GLM-4.6"  # 统一使用这个大小写

st.set_page_config(page_title="NIW/EB-1A 智能评估助手", layout="wide")
st.title("🧑‍⚖️ NIW / EB-1A 智能评估助手")
st.markdown("基于 **GLM-4.6**，模拟 USCIS 审查逻辑，输出：Prong 分析、成功率与 Future Plan。")

# -----------------------------
# 1) 提示词
# -----------------------------
SYSTEM_PROMPT = r"""
你是一位顶尖的美国移民律师顾问，同时精通学术数据分析和USCIS（美国公民及移民服务局）的NIW（国家利益豁免）和EB-1A（杰出人才）申请审查逻辑。
你的任务是分析用户提供的科研背景资料，并模拟USCIS的审查标准，给出一份全面、专业的评估报告。
你必须严格遵循以下输出格式要求：
1.  你的全部回答必须是一个单一、有效的JSON对象。
2.  不要在JSON对象前后添加任何解释性文字、代码块标记（如```json）或任何其他内容。
3.  如果无法分析，请返回一个包含 "error": "Unable to process the input." 的JSON对象。
JSON对象的结构必须如下：
{
  "analysis_summary": {
    "field_of_expertise": "申请人的专业领域",
    "key_achievements": "主要成就的简要总结"
  },
  "prong_analysis": {
    "prong_1": {
      "score": 8,
      "reasoning": "论证过程，解释为什么满足或不满足Substantial Merit and National Importance。",
      "suggestions": "针对Prong 1的补强建议。"
    },
    "prong_2": {
      "score": 7,
      "reasoning": "论证过程，解释为什么申请人Well-Positioned to Advance the Field。",
      "suggestions": "针对Prong 2的补强建议。"
    },
    "prong_3": {
      "score": 9,
      "reasoning": "论证过程，解释为什么Benefit to the U.S. outweighs Labor Certification。",
      "suggestions": "针对Prong 3的补强建议。"
    }
  },
  "overall_assessment": {
    "total_score": 24,
    "success_probability_niw": "高",
    "success_probability_eb1a": "中",
    "overall_suggestions": "综合补强建议，包括证据类型、寻找推荐信的人选等。"
  },
  "future_plan_draft": [
    "第一条未来计划，与申请人研究方向衔接，体现美国国家意义。",
    "第二条未来计划，语言积极、可信、有行动路线。",
    "第三条未来计划，可以涉及教学、合作或技术转化。"
  ]
}
"""

USER_PROMPT_TEMPLATE = """
请根据以下申请人提供的科研资料，进行NIW/EB-1A智能评估。
申请人科研资料：
---
{user_input}
---
请严格按照系统提示词中定义的JSON格式输出你的评估报告。
"""

# -----------------------------
# 2) 工具函数
# -----------------------------
def get_api_key():
    # 优先 secrets → 环境变量 → 页面输入
    key = None
    try:
        key = st.secrets["ZHIPU_API_KEY"]
    except Exception:
        key = os.getenv("ZHIPU_API_KEY")

    if not key:
        st.info("未检测到 API Key，请在下方输入（仅保存在本页会话中）。")
        key = st.text_input("ZHIPU_API_KEY", type="password")
        if key:
            st.session_state["_ZHIPU_API_KEY_USER"] = key
        else:
            key = st.session_state.get("_ZHIPU_API_KEY_USER")

    return key

def strip_code_fences(text: str) -> str:
    # 去掉 ```json ... ``` 或 ``` ... ``` 围栏
    text = re.sub(r"^```.*?\n", "", text.strip(), flags=re.S)
    text = re.sub(r"\n```$", "", text.strip(), flags=re.S)
    return text.strip()

def extract_first_json(text: str) -> str:
    # 保底：提取第一个“{ ... }”大括号包裹的对象
    s = strip_code_fences(text)
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return s  # 就按原文返回，交给 json.loads 去尝试

def call_glm(user_input_text: str, api_key: str) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(user_input=user_input_text)}
        ],
        "temperature": 0.2
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}", "raw_response": getattr(resp, "text", "")}
    except (KeyError, ValueError) as e:
        return {"error": f"Unexpected API response structure: {e}", "raw_response": getattr(resp, "text", "")}

    # 尝试解析严格 JSON
    raw = extract_first_json(content)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON from model response.", "raw_content": content}

# -----------------------------
# 3) 页面与流程
# -----------------------------
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

api_key = get_api_key()
with st.form("assessment_form"):
    st.subheader("第一步：请输入您的科研资料")
    user_input = st.text_area(
        "请详细描述研究方向、代表论文（期刊/影响因子/引用）、审稿经历、奖项、合作等。",
        height=300,
        value=st.session_state.user_input
    )
    submitted = st.form_submit_button("开始智能评估")

if submitted:
    if not api_key:
        st.error("缺少 ZHIPU_API_KEY：请配置 Streamlit Secrets 或在页面输入框里填入。")
        st.stop()
    if not user_input.strip():
        st.warning("请输入科研资料后再评估。")
        st.stop()

    st.session_state.user_input = user_input
    with st.spinner("🤖 GLM-4.6 正在分析中..."):
        report = call_glm(user_input, api_key)

    if "error" in report:
        st.error(f"分析失败：{report['error']}")
        if "raw_content" in report:
            with st.expander("查看模型原始输出"):
                st.code(report["raw_content"])
        if "raw_response" in report:
            with st.expander("查看API原始响应"):
                st.code(report["raw_response"])
        st.stop()

    # -------------------------
    # 4) 结果展示
    # -------------------------
    st.success("✅ 评估完成！")
    st.header("评估报告")

    # 分析摘要
    st.subheader("📄 分析摘要")
    col1, col2 = st.columns(2)
    col1.write(f"**专业领域：** {report['analysis_summary'].get('field_of_expertise','-')}")
    col2.write(f"**关键成就：** {report['analysis_summary'].get('key_achievements','-')}")

    # Prong 分析
    st.subheader("⚖️ USCIS Prong 详细分析")
    for prong_key in ("prong_1", "prong_2", "prong_3"):
        pr = report.get("prong_analysis", {}).get(prong_key, {})
        score = pr.get("score", "-")
        with st.expander(f"{prong_key.replace('_',' ').title()}  |  Score: {score}/10"):
            st.write(f"**论证逻辑：** {pr.get('reasoning','-')}")
            st.info(f"**补强建议：** {pr.get('suggestions','-')}")

    # 综合评估
    st.subheader("📊 综合评估")
    oa = report.get("overall_assessment", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("NIW 成功概率", oa.get("success_probability_niw", "-"))
    c2.metric("EB-1A 成功概率", oa.get("success_probability_eb1a", "-"))
    c3.metric("综合得分", oa.get("total_score", "-"))
    st.write(f"**综合建议：** {oa.get('overall_suggestions','-')}")

    # Future Plan
    st.subheader("🧭 Future Plan 写作建议")
    for i, plan in enumerate(report.get("future_plan_draft", []), start=1):
        st.write(f"{i}. {plan}")

