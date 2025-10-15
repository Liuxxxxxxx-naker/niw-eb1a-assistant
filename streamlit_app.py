import os, re, json, requests, streamlit as st

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL_ID = "GLM-4.6"

st.set_page_config(page_title="NIW/EB-1A 智能评估助手", layout="wide")
st.title("🧑‍⚖️ NIW / EB-1A 智能评估助手")
st.caption("基于 GLM-4.6。输入可为中/英，输出统一英文 JSON 报告。")

SYSTEM_PROMPT = r"""
You are a senior U.S. immigration petition advisor and academic evaluator.
Regardless of input language, respond in English only with a single valid JSON object in the following schema and nothing else.
If you cannot process, return {"error": "Unable to process the input."}
{
  "analysis_summary": {"field_of_expertise": "string","key_achievements": "string"},
  "prong_analysis": {
    "prong_1": {"score": 0, "reasoning": "string", "suggestions": "string"},
    "prong_2": {"score": 0, "reasoning": "string", "suggestions": "string"},
    "prong_3": {"score": 0, "reasoning": "string", "suggestions": "string"}
  },
  "overall_assessment": {
    "total_score": 0,
    "success_probability_niw": "string",
    "success_probability_eb1a": "string",
    "overall_suggestions": "string"
  },
  "future_plan_draft": ["string","string","string"]
}
"""

USER_PROMPT_TEMPLATE = """
Please analyze the following NIW/EB-1A applicant. Input can be Chinese or English, but your output must be English JSON exactly per the schema.

Applicant profile:
---
{user_input}
---
Strictly output the JSON only, no extra text.
"""

def strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```.*?\n", "", s, flags=re.S)
    s = re.sub(r"\n```$", "", s, flags=re.S)
    return s.strip()

def extract_first_json(s: str) -> str:
    s = strip_code_fences(s)
    i, j = s.find("{"), s.rfind("}")
    return s[i:j+1] if i != -1 and j != -1 and j > i else s

def get_api_key() -> str:
    key = None
    try:
        key = st.secrets["ZHIPU_API_KEY"]
    except Exception:
        key = os.getenv("ZHIPU_API_KEY")
    if not key:
        st.info("未检测到 API Key，请在下方输入（仅本会话保存）。")
        key = st.text_input("ZHIPU_API_KEY", type="password")
        if key:
            st.session_state["_ZHIPU_API_KEY_USER"] = key
        else:
            key = st.session_state.get("_ZHIPU_API_KEY_USER")
    return key

def call_glm(user_input_text: str, api_key: str, temperature: float = 0.2) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(user_input=user_input_text)}
        ],
        "temperature": float(temperature),
    }
    resp = None
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}", "raw_response": resp.text if resp else ""}
    except (KeyError, ValueError) as e:
        return {"error": f"Unexpected API response structure: {e}", "raw_response": resp.text if resp else ""}

    raw = extract_first_json(content)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON from model response.", "raw_content": content}

with st.sidebar:
    st.header("⚙️ Settings")
    API_KEY = get_api_key()
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    show_raw = st.toggle("Show raw response (debug)", value=False)
    st.markdown("---")
    st.caption("建议：先在左侧填入 API Key。")

st.subheader("① 基本信息")
c1, c2 = st.columns(2)
name = c1.text_input("Name / Applicant")
field = c2.text_input("Field of Study")
institutions = st.text_input("Affiliations / Collaborations (comma separated)")
awards = st.text_area("Awards / Grants (optional)", height=70)
reviewer = st.text_area("Peer-review Experience (optional)", height=70)

st.subheader("② Publications（可直接编辑）")
if "pubs" not in st.session_state:
    st.session_state.pubs = [{"title":"", "journal":"", "year":"", "citations":0, "countries":"US;CN"}]

pubs_edited = st.data_editor(
    st.session_state.pubs,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "title": st.column_config.TextColumn("Title", width="medium"),
        "journal": st.column_config.TextColumn("Journal", width="medium"),
        "year": st.column_config.TextColumn("Year", width="small"),
        "citations": st.column_config.NumberColumn("Citations", min_value=0, step=1),
        "countries": st.column_config.TextColumn("Cited by Countries (US;CN;DE)")
    },
)
# 统一为 list[dict]
if hasattr(pubs_edited, "to_dict"):
    st.session_state.pubs = pubs_edited.to_dict("records")
else:
    st.session_state.pubs = pubs_edited

st.subheader("③ Additional Notes (optional)")
extra = st.text_area("Any context the model should consider", height=100)

def build_user_input() -> str:
    lines = []
    if name: lines.append(f"Name: {name}")
    if field: lines.append(f"Field: {field}")
    if institutions: lines.append(f"Collaborations: {institutions}")
    if awards: lines.append(f"Awards: {awards}")
    if reviewer: lines.append(f"Reviewer: {reviewer}")
    pubs_lines = []
    for p in st.session_state.pubs:
        t = (p.get("title") or "").strip()
        if not t: continue
        pubs_lines.append(f'- "{t}" ({p.get("journal","")}, {p.get("year","")}), citations={p.get("citations",0)}, cited_countries="{p.get("countries","")}"')
    if pubs_lines:
        lines.append("Publications:\n" + "\n".join(pubs_lines))
    if extra:
        lines.append("Notes:\n" + extra)
    return "\n".join(lines).strip()

run = st.button("🚀 开始智能评估", use_container_width=True)

if run:
    if not API_KEY:
        st.error("缺少 ZHIPU_API_KEY。请在侧边栏输入或配置 Secrets/环境变量。")
        st.stop()
    user_input_text = build_user_input()
    if not user_input_text:
        st.warning("请至少填写基本信息或一篇论文。")
        st.stop()

    with st.spinner("GLM-4.6 正在评估…"):
        report = call_glm(user_input_text, API_KEY, temperature=temperature)

    if "error" in report:
        st.error(f"失败：{report['error']}")
        if show_raw and "raw_content" in report:
            with st.expander("Raw Model Output"):
                st.code(report["raw_content"])
        if show_raw and "raw_response" in report:
            with st.expander("Raw API Response"):
                st.code(report["raw_response"])
        st.stop()

    st.success("✅ Completed")
    st.header("Evaluation Report")

    st.subheader("📄 Analysis Summary")
    a = report.get("analysis_summary", {})
    x1, x2 = st.columns(2)
    x1.write(f"**Field of Expertise:** {a.get('field_of_expertise','-')}")
    x2.write(f"**Key Achievements:** {a.get('key_achievements','-')}")

    st.subheader("⚖️ USCIS Prongs")
    for k, label in [("prong_1","Prong 1"), ("prong_2","Prong 2"), ("prong_3","Prong 3")]:
        pr = report.get("prong_analysis", {}).get(k, {})
        with st.expander(f"{label} | Score: {pr.get('score','-')}/10", expanded=False):
            st.write(f"**Reasoning:** {pr.get('reasoning','-')}")
            st.info(f"**Suggestions:** {pr.get('suggestions','-')}")

    st.subheader("📊 Overall Assessment")
    oa = report.get("overall_assessment", {})
    y1, y2, y3 = st.columns(3)
    y1.metric("NIW Probability", oa.get("success_probability_niw","-"))
    y2.metric("EB-1A Probability", oa.get("success_probability_eb1a","-"))
    y3.metric("Total Score", oa.get("total_score","-"))
    st.write(f"**Overall Suggestions:** {oa.get('overall_suggestions','-')}")

    st.subheader("🧭 Future Plan (Draft)")
    for i, plan in enumerate(report.get("future_plan_draft", []), start=1):
        st.write(f"{i}. {plan}")

    md = []
    md += ["# NIW/EB-1A Evaluation Report", "## Analysis Summary"]
    md += [f"- Field of Expertise: {a.get('field_of_expertise','-')}",
           f"- Key Achievements: {a.get('key_achievements','-')}",
           "## USCIS Prongs"]
    for k, label in [("prong_1","Prong 1"), ("prong_2","Prong 2"), ("prong_3","Prong 3")]:
        pr = report.get("prong_analysis", {}).get(k, {})
        md += [f"### {label} (Score: {pr.get('score','-')}/10)",
               f"- Reasoning: {pr.get('reasoning','-')}",
               f"- Suggestions: {pr.get('suggestions','-')}"]
    md += ["## Overall Assessment",
           f"- NIW Probability: {oa.get('success_probability_niw','-')}",
           f"- EB-1A Probability: {oa.get('success_probability_eb1a','-')}",
           f"- Total Score: {oa.get('total_score','-')}",
           f"- Overall Suggestions: {oa.get('overall_suggestions','-')}",
           "## Future Plan (Draft)"]
    for p in report.get("future_plan_draft", []):
        md.append(f"- {p}")
    st.download_button("📥 Download Markdown", "\n".join(md), file_name="niw_eb1a_report.md", mime="text/markdown")

st.markdown("""
<style>
    .stButton>button { border-radius: 10px; height: 3rem; font-weight: 600; }
    .stTextInput input, .stTextArea textarea { border-radius: 10px; }
    .stDataFrame, .stDataEditor { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)
