import os
import re
import json
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL_ID = "GLM-4.6"
OPENALEX = "https://api.openalex.org"

ISO2_TO_ISO3 = {
    "US":"USA","CN":"CHN","DE":"DEU","JP":"JPN","GB":"GBR","UK":"GBR","FR":"FRA","CA":"CAN","AU":"AUS",
    "IT":"ITA","ES":"ESP","KR":"KOR","IN":"IND","NL":"NLD","SE":"SWE","NO":"NOR","FI":"FIN","DK":"DNK",
    "CH":"CHE","AT":"AUT","BE":"BEL","SG":"SGP","HK":"HKG","TW":"TWN","IL":"ISR","SA":"SAU","AE":"ARE",
    "RU":"RUS","BR":"BRA","MX":"MEX","ZA":"ZAF","PL":"POL","TR":"TUR"
}

st.set_page_config(page_title="NIW/EB-1A 智能评估助手", layout="wide")
st.title("🧑‍⚖️ NIW / EB-1A 智能评估助手")
st.caption("GLM-4.6 + OpenAlex • 中英皆可输入，统一英文输出 • 地图+二级影响 • 可下载报告")

def strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```.*?\n", "", s, flags=re.S)
    s = re.sub(r"\n```$", "", s, flags=re.S)
    return s.strip()

def extract_first_json(s: str) -> str:
    s = strip_code_fences(s)
    i, j = s.find("{"), s.rfind("}")
    return s[i:j+1] if i != -1 and j != -1 and j > i else s

def build_system_prompt(mode: str) -> str:
    base = "You are a senior U.S. immigration petition advisor and academic evaluator. Respond in English only with a single JSON object."
    if mode == "NIW":
        schema = r'''
Output JSON:
{
  "analysis_summary":{"field_of_expertise":"string","key_achievements":"string"},
  "niw_prongs":{
    "prong_1":{"score":0,"reasoning":"string","suggestions":"string"},
    "prong_2":{"score":0,"reasoning":"string","suggestions":"string"},
    "prong_3":{"score":0,"reasoning":"string","suggestions":"string"}
  },
  "overall_assessment":{"niw_probability":"string","overall_suggestions":"string","total_score":0},
  "future_plan_draft":["string","string","string"]
}
Definitions:
- Prong 1: Substantial Merit & National Importance
- Prong 2: Well Positioned to Advance the Endeavor
- Prong 3: On balance, it benefits the U.S. to waive the job offer and labor certification
If you cannot process, return {"error":"Unable to process the input."}.
'''
    elif mode == "EB-1A":
        schema = r'''
Output JSON:
{
  "analysis_summary":{"field_of_expertise":"string","key_achievements":"string"},
  "eb1a_criteria":{
    "awards":"met|partial|not_met","membership":"met|partial|not_met","media":"met|partial|not_met",
    "judge_of_others":"met|partial|not_met","original_contribution":"met|partial|not_met",
    "authorship":"met|partial|not_met","exhibitions":"met|partial|not_met","leading_role":"met|partial|not_met",
    "high_salary":"met|partial|not_met","commercial_success":"met|partial|not_met"
  },
  "overall_assessment":{"eb1a_probability":"string","criteria_met_count":0,"overall_suggestions":"string"},
  "future_plan_draft":["string","string","string"]
}
Judge each criterion per USCIS; explain borderline cases in suggestions. If you cannot process, return {"error":"Unable to process the input."}.
'''
    else:
        schema = r'''
Output JSON:
{
  "analysis_summary":{"field_of_expertise":"string","key_achievements":"string"},
  "niw_prongs":{
    "prong_1":{"score":0,"reasoning":"string","suggestions":"string"},
    "prong_2":{"score":0,"reasoning":"string","suggestions":"string"},
    "prong_3":{"score":0,"reasoning":"string","suggestions":"string"}
  },
  "eb1a_criteria":{
    "awards":"met|partial|not_met","membership":"met|partial|not_met","media":"met|partial|not_met",
    "judge_of_others":"met|partial|not_met","original_contribution":"met|partial|not_met",
    "authorship":"met|partial|not_met","exhibitions":"met|partial|not_met","leading_role":"met|partial|not_met",
    "high_salary":"met|partial|not_met","commercial_success":"met|partial|not_met"
  },
  "overall_assessment":{"niw_probability":"string","eb1a_probability":"string","criteria_met_count":0,"total_score":0,"overall_suggestions":"string"},
  "future_plan_draft":["string","string","string"]
}
If you cannot process, return {"error":"Unable to process the input."}.
'''
    return base + "\n" + schema

USER_PROMPT_TEMPLATE = """
Analyze the following NIW/EB-1A applicant. Input can be Chinese or English, but your output must be English JSON exactly per the schema.

Applicant profile:
---
{user_input}
---
Return JSON only, no extra text.
"""

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

def call_glm(user_input_text: str, api_key: str, temperature: float = 0.2, mode: str = "Both") -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": build_system_prompt(mode)},
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

def ox_get(url, params=None):
    r = requests.get(url, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()

def resolve_work_by_title(title):
    j = ox_get(f"{OPENALEX}/works", {"search": title, "per-page": 1})
    items = j.get("results", [])
    return items[0] if items else None

def resolve_work_by_doi(doi):
    doi = doi.lower().strip().replace("https://doi.org/", "")
    try:
        return ox_get(f"{OPENALEX}/works/doi:{doi}")
    except Exception:
        return None

def get_citing_works(openalex_id, per_page=200, max_pages=10):
    all_items, cursor = [], "*"
    for _ in range(max_pages):
        j = ox_get(f"{OPENALEX}/works", {"filter": f"cites:{openalex_id}", "per-page": per_page, "cursor": cursor})
        items = j.get("results", [])
        all_items.extend(items)
        cursor = j.get("meta", {}).get("next_cursor")
        if not cursor or not items: break
    return all_items

def countries_from_authorships(item):
    out = []
    for a in item.get("authorships", []):
        for inst in a.get("institutions", []):
            cc = inst.get("country_code")
            if cc: out.append(cc.upper())
    return out

def aggregate_citing_countries(openalex_id):
    citing = get_citing_works(openalex_id)
    counts = {}
    for w in citing:
        for c in countries_from_authorships(w):
            counts[c] = counts.get(c, 0) + 1
    return counts, citing

def second_order_reach(citing_list, top_k=10):
    ranked = sorted(
        [(w.get("display_name",""), w.get("cited_by_count",0), w.get("id","")) for w in citing_list],
        key=lambda x: x[1], reverse=True
    )
    return ranked[:top_k]

def draw_country_map(country_counts: dict, title="Citing Countries (OpenAlex)"):
    if not country_counts:
        st.info("No citing countries found.")
        return
    rows = []
    for k, v in country_counts.items():
        cc3 = ISO2_TO_ISO3.get(k.upper(), k.upper())
        rows.append({"ISO3": cc3, "country": k.upper(), "count": v})
    df = pd.DataFrame(rows).sort_values("count", ascending=False)
    top_n = min(15, len(df))
    st.write(f"**Top {top_n} citing countries**")
    st.dataframe(df.head(top_n)[["country", "count"]], use_container_width=True)
    st.download_button("⬇️ Download country counts (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="citing_countries.csv", mime="text/csv")
    fig = px.choropleth(df, locations="ISO3", color="count",
                        hover_name="country",
                        hover_data={"ISO3": False, "count": True, "country": True},
                        color_continuous_scale="Blues", locationmode="ISO-3", title=title)
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>Count: %{z}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.bar(df.head(top_n), x="country", y="count", title="Top Citing Countries (bar)")
    st.plotly_chart(fig2, use_container_width=True)

with st.sidebar:
    st.header("⚙️ Settings")
    API_KEY = get_api_key()
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    eval_mode = st.radio("Evaluation target", ["NIW", "EB-1A", "Both"])
    show_raw = st.toggle("Show raw response (debug)", value=False)
    st.markdown("---")
    st.caption("左侧先填 API Key；下方可自动解析论文到表格。")

st.subheader("① 基本信息")
c1, c2 = st.columns(2)
name = c1.text_input("Name / Applicant")
field = c2.text_input("Field of Study")
institutions = st.text_input("Affiliations / Collaborations (comma separated)")
awards = st.text_area("Awards / Grants (optional)", height=70)
reviewer = st.text_area("Peer-review Experience (optional)", height=70)

st.subheader("② 数据来源")
source = st.radio("选择论文来源", ["手动输入/表格", "按标题快速解析（OpenAlex）", "按 DOI 快速解析（OpenAlex）"], horizontal=True)
title_input = doi_input = ""
if source == "按标题快速解析（OpenAlex）":
    title_input = st.text_input("输入论文标题（逐篇解析后加入）")
elif source == "按 DOI 快速解析（OpenAlex）":
    doi_input = st.text_input("输入 DOI（如 10.1021/acsami.xxxxxxx）")

colF1, colF2 = st.columns([1,1])
fetch_btn = colF1.button("🔎 解析并加入到 Publications")
clear_btn = colF2.button("🧹 清空 Publications")

st.subheader("② Publications（可直接编辑）")
if "pubs" not in st.session_state:
    st.session_state.pubs = [{"title":"", "journal":"", "year":"", "citations":0, "countries":"US;CN"}]

if fetch_btn:
    item = None
    if source.startswith("按标题") and title_input.strip():
        item = resolve_work_by_title(title_input.strip())
    elif source.startswith("按 DOI") and doi_input.strip():
        item = resolve_work_by_doi(doi_input.strip())
    if not item:
        st.warning("未解析到论文。")
    else:
        title = item.get("display_name","")
        venue = (item.get("host_venue",{}) or {}).get("display_name","")
        year  = (item.get("publication_year") or "")
        cites = item.get("cited_by_count", 0)
        cc, citing_list = aggregate_citing_countries(item["id"])
        cc_str = ";".join(sorted(cc.keys()))
        st.session_state.pubs.append({"title": title, "journal": venue, "year": year,
                                      "citations": cites, "countries": cc_str})
        st.success("已加入 Publications。")
        with st.expander("本篇引用国家分布地图"):
            draw_country_map(cc, title=f"Countries citing: {title[:50]}...")
        with st.expander("二级影响 Top10（引用你的论文的论文中，被引最多）"):
            for t, c, wid in second_order_reach(citing_list, top_k=10):
                st.write(f"- {t}  | cited_by={c}  | {wid}")

if clear_btn:
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
st.session_state.pubs = pubs_edited.to_dict("records") if hasattr(pubs_edited, "to_dict") else pubs_edited

st.subheader("③ Additional Notes (optional)")
extra = st.text_area("Any context the model should consider", height=100)

def build_user_input() -> str:
    lines = []
    if name: lines.append(f"Name: {name}")
    if field: lines.append(f"Field: {field}")
    if institutions: lines.append(f"Collaborations: {institutions}")
    if awards: lines.append(f"Awards: {awards}")
    if reviewer: lines.append(f"Reviewer: {reviewer}")
    pubs_lines, total_cites, all_cc = [], 0, {}
    for p in st.session_state.pubs:
        t = (p.get("title") or "").strip()
        if not t: continue
        cnum = int(p.get("citations") or 0)
        total_cites += cnum
        for cc in (p.get("countries","") or "").split(";"):
            cc = cc.strip().upper()
            if not cc: continue
            all_cc[cc] = all_cc.get(cc, 0) + 1
        pubs_lines.append(f'- "{t}" ({p.get("journal","")}, {p.get("year","")}), citations={cnum}, cited_countries="{p.get("countries","")}"')
    if pubs_lines: lines.append("Publications:\n" + "\n".join(pubs_lines))
    if all_cc:
        top_cc = sorted(all_cc.items(), key=lambda x: x[1], reverse=True)[:10]
        cc_str = ", ".join([f"{k}:{v}" for k,v in top_cc])
        lines.append("Bibliometrics Summary:\n" +
                     f"- Total citations (sum of listed): {total_cites}\n" +
                     f"- Top citing countries (count of papers that cite you): {cc_str}\n" +
                     "- Note: Country stats come from OpenAlex citing-works institutions.")
    if extra: lines.append("Notes:\n" + extra)
    return "\n".join(lines).strip()

run = st.button("🚀 开始智能评估", use_container_width=True)

if run:
    if not API_KEY:
        st.error("缺少 ZHIPU_API_KEY。请在侧边栏输入或配置 Secrets/环境变量。")
        st.stop()
    ui_text = build_user_input()
    if not ui_text:
        st.warning("请至少填写基本信息或一篇论文。")
        st.stop()

    st.subheader("🌍 Citing Countries Overview")
    all_cc = {}
    for p in st.session_state.pubs:
        for cc in (p.get("countries","") or "").split(";"):
            cc = cc.strip().upper()
            if not cc: continue
            all_cc[cc] = all_cc.get(cc, 0) + 1
    draw_country_map(all_cc, title="Citing Countries (all listed publications)")

    with st.spinner("GLM-4.6 正在评估…"):
        report = call_glm(ui_text, API_KEY, temperature=temperature, mode=eval_mode)

    if "error" in report:
        st.error(f"失败：{report['error']}")
        if show_raw and "raw_content" in report:
            with st.expander("Raw Model Output"): st.code(report["raw_content"])
        if show_raw and "raw_response" in report:
            with st.expander("Raw API Response"): st.code(report["raw_response"])
        st.stop()

    st.success("✅ Completed")
    st.header("Evaluation Report")

    st.subheader("📄 Analysis Summary")
    a = report.get("analysis_summary", {})
    x1, x2 = st.columns(2)
    x1.write(f"**Field of Expertise:** {a.get('field_of_expertise','-')}")
    x2.write(f"**Key Achievements:** {a.get('key_achievements','-')}")

    if "niw_prongs" in report:
        st.subheader("⚖️ NIW Prongs")
        for k, label in [("prong_1","Prong 1"), ("prong_2","Prong 2"), ("prong_3","Prong 3")]:
            pr = report["niw_prongs"].get(k, {})
            with st.expander(f"{label} | Score: {pr.get('score','-')}/10", expanded=False):
                st.write(f"**Reasoning:** {pr.get('reasoning','-')}")
                st.info(f"**Suggestions:** {pr.get('suggestions','-')}")

    if "eb1a_criteria" in report:
        st.subheader("🏅 EB-1A Criteria (10)")
        crit = report["eb1a_criteria"]
        crit_map = {
            "awards":"Lesser nationally/internationally recognized prizes",
            "membership":"Membership requiring outstanding achievements",
            "media":"Published material about the applicant",
            "judge_of_others":"Judge/review of others’ work",
            "original_contribution":"Original contributions of major significance",
            "authorship":"Authorship of scholarly articles",
            "exhibitions":"Exhibitions/showcases (arts/creative fields)",
            "leading_role":"Leading/critical role for distinguished orgs",
            "high_salary":"High salary/remuneration",
            "commercial_success":"Commercial success (performing arts)"
        }
        table = pd.DataFrame([(crit_map[k], crit.get(k,"-")) for k in crit_map], columns=["Criterion","Status"])
        st.dataframe(table, use_container_width=True)

    st.subheader("📊 Overall Assessment")
    oa = report.get("overall_assessment", {})
    cols = st.columns(3)
    cols[0].metric("NIW Probability", oa.get("niw_probability","-"))
    cols[1].metric("EB-1A Probability", oa.get("eb1a_probability", oa.get("eb1a_probability","-")))
    cols[2].metric("Total Score", oa.get("total_score", oa.get("criteria_met_count","-")))
    st.write(f"**Overall Suggestions:** {oa.get('overall_suggestions','-')}")

    st.subheader("🧭 Future Plan (Draft)")
    for i, plan in enumerate(report.get("future_plan_draft", []), start=1):
        st.write(f"{i}. {plan}")

    with st.expander("📝 NIW Petition Letter Outline (copy-ready)"):
        st.markdown("""
**1. Introduction** — One-sentence thesis of national importance and your role.  
**2. Prong 1 — Substantial Merit & National Importance** — Problem scale, U.S. policy anchors (DOE/EPA/NSF/IRA); evidence (high-citation papers, downstream adoption, cross-country citations).  
**3. Prong 2 — Well Positioned** — Unique skills, placements, collaborations, reviewer roles; letters/grants.  
**4. Prong 3 — Waiver Benefits** — Why labor cert slows public-interest progress; independence; planned U.S. partners.  
**5. Conclusion** — Summarize merits; request favorable adjudication.
""")

    with st.expander("📄 EB-1A Petition Letter Outline (copy-ready)"):
        st.markdown("""
**1. Introduction** — Field, impact theme, summary of criteria met.  
**2. Criteria Sections (choose ≥3)** — Original contributions; Authorship; Judge of others; Leading role; Media; Awards; Membership; High salary; Commercial success; Exhibitions.  
**3. Sustained Acclaim** — Trajectory, continuity, momentum.  
**4. Conclusion** — Restate “top of the field”.
""")

    with st.expander("🧭 Future Plan – 5-Step Structure"):
        st.markdown(f"""
**1. Goal** — 3–5 year objective tied to U.S. priorities.  
**2. Approach** — Methods/roadmap（e.g., membrane scale-up, pilots with national labs）.  
**3. Impact** — KPIs（TRL, CO₂ tons, patents, startups）.  
**4. Collaboration** — U.S. partners（{institutions or 'target orgs'}）.  
**5. Risk & Mitigation** — Key risks + plan B.  
*Example:* Deliver scalable CO₂-selective membranes with DOE labs, target TRL-6 and 30% cost reduction vs. amine scrubbing.
""")

    md = []
    md += ["# NIW/EB-1A Evaluation Report", "## Analysis Summary"]
    md += [f"- Field of Expertise: {a.get('field_of_expertise','-')}",
           f"- Key Achievements: {a.get('key_achievements','-')}"]
    if "niw_prongs" in report:
        md += ["## NIW Prongs"]
        for k, label in [("prong_1","Prong 1"), ("prong_2","Prong 2"), ("prong_3","Prong 3")]:
            pr = report["niw_prongs"].get(k, {})
            md += [f"### {label} (Score: {pr.get('score','-')}/10)",
                   f"- Reasoning: {pr.get('reasoning','-')}",
                   f"- Suggestions: {pr.get('suggestions','-')}"]
    if "eb1a_criteria" in report:
        md += ["## EB-1A Criteria (10)"]
        for k, v in report["eb1a_criteria"].items():
            md.append(f"- {k}: {v}")
    md += ["## Overall Assessment",
           f"- NIW Probability: {oa.get('niw_probability','-')}",
           f"- EB-1A Probability: {oa.get('eb1a_probability', oa.get('eb1a_probability','-'))}",
           f"- Total Score: {oa.get('total_score', oa.get('criteria_met_count','-'))}",
           f"- Overall Suggestions: {oa.get('overall_suggestions','-')}",
           "## Future Plan (Draft)"]
    for p in report.get("future_plan_draft", []): md.append(f"- {p}")
    st.download_button("📥 Download Markdown", "\n".join(md), file_name="niw_eb1a_report.md", mime="text/markdown")

st.markdown("""
<style>
    .stButton>button { border-radius: 10px; height: 3rem; font-weight: 600; }
    .stTextInput input, .stTextArea textarea { border-radius: 10px; }
    .stDataFrame, .stDataEditor { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

