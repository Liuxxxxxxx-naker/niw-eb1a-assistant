import streamlit as st
import pandas as pd
import plotly.express as px
import requests, json, time, math
from math import log10
from typing import Dict, Any

# -----------------------
# 初始化配置
# -----------------------
st.set_page_config(page_title="NIW / EB-1A 智能评估助手", layout="wide")
st.title("🇺🇸 NIW / EB-1A 智能评估助手 (GLM-4.6)")

st.markdown("基于 **GLM-4.6**，模拟 USCIS 审查逻辑，结合论文、引用、以及 Petition Letter/Future Plan 综合评估。")

# -----------------------
# Sidebar 参数设置
# -----------------------
with st.sidebar:
    st.header("⚙️ 参数设置")
    API_KEY = st.text_input("ZHIPU_API_KEY", type="password")
    model_id = st.text_input("模型 ID", value="glm-4-6")
    eval_mode = st.radio("评估模式", ["NIW", "EB-1A", "Both"], horizontal=True)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    timeout_s = st.slider("超时时间 (秒)", 30, 180, 90, 10)

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# -----------------------
# Petition Letter / Future Plan 上传
# -----------------------
st.header("④ Petition Letter / Future Plan")
uploaded_petition = st.file_uploader("📄 上传文件 (.txt / .docx)", type=["txt", "docx"])
petition_text = ""
if uploaded_petition is not None:
    if uploaded_petition.name.endswith(".txt"):
        petition_text = uploaded_petition.read().decode("utf-8")
    else:
        import docx
        doc = docx.Document(uploaded_petition)
        petition_text = "\n".join([p.text for p in doc.paragraphs])
else:
    petition_text = st.text_area("或直接粘贴 Petition Letter / Future Plan 内容", height=250)

# -----------------------
# 用户输入（科研信息）
# -----------------------
st.header("① 基本科研信息")
col1, col2 = st.columns(2)
with col1:
    name = st.text_input("Name / Applicant")
    field = st.text_input("Field of Study")
with col2:
    aff = st.text_input("Affiliations / Collaborations (comma separated)")
awards = st.text_area("Awards / Grants (optional)")
peer = st.text_area("Peer-review Experience (optional)")

# Publications
st.header("② Publications（可直接编辑）")
if "pubs" not in st.session_state:
    st.session_state.pubs = pd.DataFrame([{"Title": "", "Journal": "", "Year": "", "Citations": 0, "Cited by Countries": ""}])
pubs_df = st.data_editor(st.session_state.pubs, num_rows="dynamic")
st.session_state.pubs = pubs_df

# Additional notes
st.header("③ Additional Notes (optional)")
notes = st.text_area("其他说明（例如项目、合作、专利、媒体报道等）")

# -----------------------
# 系统 Prompt 生成
# -----------------------
def build_system_prompt(mode="Both"):
    return f"""
You are a USCIS-style evaluator specializing in {mode} petitions.
Analyze the applicant’s research background, impact, citation data, and attached petition/future plan.
Output structured JSON including:
- USCIS Prongs (1–3) reasoning and suggestions
- Overall NIW/EB1A probability
- Petition review (argument strength, structure, improvement)
- Future plan recommendations
Ensure professional, concise English.
"""

# -----------------------
# 调用 GLM
# -----------------------
def call_glm(user_input_text: str, api_key: str,
             temperature: float = 0.2, mode: str = "Both",
             model_id: str = "glm-4-6", timeout_s: int = 90) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": build_system_prompt(mode)},
            {"role": "user", "content": user_input_text}
        ],
        "max_tokens": 2000
    }
    for attempt in range(3):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout_s)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return {"success": True, "content": text}
            else:
                time.sleep(2)
        except Exception as e:
            if attempt == 2:
                return {"error": f"API 请求失败: {e}"}
    return {"error": "请求超时或失败"}

# -----------------------
# 拼接用户输入
# -----------------------
def build_user_input():
    lines = [
        f"Applicant: {name}",
        f"Field: {field}",
        f"Affiliations: {aff}",
        f"Awards: {awards}",
        f"Peer-review: {peer}",
        f"Publications:\n{pubs_df.to_markdown(index=False)}",
        f"Notes: {notes}"
    ]
    if petition_text:
        lines.append(f"Petition/Future Plan:\n{petition_text}")
    return "\n".join(lines)

# -----------------------
# 二级传播模块
# -----------------------
TOP_VENUES = {"nature","science","cell","pnas","jacs","advanced materials","angewandte"}
def work_notability_score(work):
    c = work.get("cited_by_count",0)
    venue = ((work.get("host_venue") or {}).get("display_name") or "").lower()
    return log10(c+1) + (1 if any(k in venue for k in TOP_VENUES) else 0)

def get_second_order(openalex_id, per_l1=50, per_l2=30):
    url = f"https://api.openalex.org/works/{openalex_id}/citations"
    try:
        l1 = requests.get(url).json().get("results", [])
        l2_map = {}
        for w in l1[:per_l1]:
            wid = w.get("id")
            if not wid: continue
            sub = requests.get(f"https://api.openalex.org/works/{wid}/citations").json().get("results", [])
            for s in sub[:per_l2]:
                sid = s.get("id")
                if sid not in l2_map:
                    l2_map[sid] = s
        rows = []
        for s in l2_map.values():
            rows.append({
                "title": s.get("display_name",""),
                "venue": (s.get("host_venue") or {}).get("display_name",""),
                "year": s.get("publication_year",""),
                "cited_by": s.get("cited_by_count",0),
                "score": work_notability_score(s)
            })
        return pd.DataFrame(rows).sort_values(["score","cited_by"],ascending=False)
    except Exception as e:
        st.warning(f"二级传播抓取失败：{e}")
        return pd.DataFrame()

# -----------------------
# 运行评估
# -----------------------
if st.button("🚀 开始智能评估"):
    if not API_KEY:
        st.warning("请先输入 ZHIPU_API_KEY")
    else:
        ui_text = build_user_input()
        with st.spinner("GLM-4.6 正在分析中..."):
            report = call_glm(ui_text, API_KEY, temperature, eval_mode, model_id, timeout_s)
        if "error" in report:
            st.error(report["error"])
        else:
            st.subheader("🧠 AI 评估报告")
            st.markdown(report["content"])

# -----------------------
# 示例地图（引用国家）
# -----------------------
st.header("🌍 Citing Countries Overview (示例)")
sample = pd.DataFrame({"country":["CN","US","JP"],"count":[5,3,1]})
fig = px.choropleth(sample, locations="country", locationmode="ISO-3",
                    color="count", color_continuous_scale="Blues",
                    title="Citing Countries")
st.plotly_chart(fig, use_container_width=True)
