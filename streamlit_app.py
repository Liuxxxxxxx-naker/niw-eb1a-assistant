import os, re, json, time, requests, pandas as pd, plotly.express as px, streamlit as st
from math import log10
from typing import Dict, Any, List

# ---------- Page ----------
st.set_page_config(page_title="US NIW / EB-1A 智能评估助手", layout="wide", page_icon="🇺🇸")

# ---------- Apple-like minimal CSS ----------
st.markdown("""
<style>
:root{
  --bg:#0B0C0F;
  --panel:#111317;
  --sub:#151922;
  --line:#222733;
  --text:#E8EBF2;
  --muted:#9AA4B2;
  --brand:#8EA8FF; /* subtle indigo */
  --ok:#9EE37D;
  --warn:#FFD266;
  --danger:#FF7A7A;
}
*{font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, Roboto, "Helvetica Neue", Arial, "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Source Han Sans", sans-serif;}
.block-container{padding-top:1.2rem; max-width:1200px;}
body{background:var(--bg); color:var(--text);}
h1,h2,h3{letter-spacing:.2px}
hr{border-color:var(--line)}
a{color:#AFC2FF}
.card{background:linear-gradient(180deg,var(--panel),#0f1116 70%); border:1px solid var(--line);
      border-radius:18px; padding:20px; margin:10px 0 16px; box-shadow: 0 8px 30px rgba(0,0,0,.35);}
.subtle{color:var(--muted); font-size:.95rem}
.kpi{background:linear-gradient(180deg, #0f1218, #0b0d12); border:1px solid var(--line); border-radius:16px; padding:14px 16px;}
.kpi .v{font-size:2.2rem; font-weight:800; letter-spacing:.2px}
.badge{display:inline-block; padding:.28rem .6rem; border:1px solid #2a3140; border-radius:999px; color:#CAD3E6; margin:.25rem .35rem 0 0;}
.hero{background:radial-gradient(1200px 600px at 20% -10%,rgba(142,168,255,.16),transparent),
                   radial-gradient(1000px 500px at 110% -20%,rgba(138,255,233,.12),transparent),
                   linear-gradient(180deg,var(--panel),#0e1116 80%);
      border:1px solid var(--line); border-radius:22px; padding:22px 22px 18px; margin:-6px 0 10px; box-shadow: 0 12px 40px rgba(0,0,0,.5);}
.hero h1{font-size:1.6rem; margin:0 0 6px}
.stepper{display:flex; gap:8px; margin-top:8px}
.stepper .s{flex:1; text-align:center; padding:.5rem 0; border-radius:10px; border:1px solid #2a3140; color:#D7DEEF;
            background:#121621}
.stepper .s.active{border-color:#38486E; background:#162033}
.stTabs [data-baseweb="tab-list"]{gap:8px}
.stTabs [data-baseweb="tab"]{border:1px solid #2a3140; background:#141821; padding-top:10px; padding-bottom:10px; border-radius:12px;}
.stTabs [aria-selected="true"]{border:1px solid #3b4b72!important; background:#162033!important; color:#E8EDFF!important}
.stButton>button{border-radius:12px; padding:.55rem 1rem; font-weight:600; border:1px solid #3a4560; background:#182037; color:#E6EAFF;}
.stButton>button:hover{border-color:#5569A3}
button[kind="primary"]{background:linear-gradient(180deg,#6E8BFF,#5671F5); border:0!important; color:white!important}
input, textarea{border-radius:12px!important; border:1px solid #2a3140!important}
.stDataFrame, .stDataEditor{border-radius:12px; border:1px solid var(--line)}
.empty{color:var(--muted); border:1px dashed #334055; border-radius:12px; padding:18px; text-align:center}
.fab{position:fixed; right:22px; bottom:22px; z-index:99; }
.pill{display:inline-block; padding:.2rem .5rem; border-radius:999px; border:1px solid #2a3140; color:#B9C7E8}
</style>
""", unsafe_allow_html=True)

# ---------- Hero ----------
st.markdown(f"""
<div class="hero">
  <h1>US NIW / EB-1A 智能评估助手（GLM-4.6）</h1>
  <div class="subtle">分步输入 · 自动抓取 OpenAlex · 二级传播（含影响因子） · Petition/Future Plan/推荐信联合评审 · JSON 卡片化输出</div>
  <div class="badge">📚 Publications 智能解析</div>
  <div class="badge">🗺️ 引用国家地图</div>
  <div class="badge">🔁 二级传播排行</div>
  <div class="badge">🧾 文本评审</div>
  <div class="badge">⚖️ NIW / EB-1A 双模式</div>
  <div class="stepper">
    <div class="s active">① Profile</div>
    <div class="s">② Publications</div>
    <div class="s">③ Impact</div>
    <div class="s">④ Documents</div>
    <div class="s">⑤ Evaluate</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Constants ----------
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
OPENALEX = "https://api.openalex.org"
ISO2_TO_ISO3 = {"US":"USA","CN":"CHN","DE":"DEU","JP":"JPN","GB":"GBR","UK":"GBR","FR":"FRA","CA":"CAN","AU":"AUS",
                "IT":"ITA","ES":"ESP","KR":"KOR","IN":"IND","NL":"NLD","SE":"SWE","NO":"NOR","FI":"FIN","DK":"DNK",
                "CH":"CHE","AT":"AUT","BE":"BEL","SG":"SGP","HK":"HKG","TW":"TWN","IL":"ISR","SA":"SAU","AE":"ARE",
                "RU":"RUS","BR":"BRA","MX":"MEX","ZA":"ZAF","PL":"POL","TR":"TUR"}
TOP_VENUES = {"nature","science","cell","pnas","jacs","advanced materials","angewandte",
              "energy & environmental science","neurips","icml","iclr","cvpr","aaai","kdd","www","acl"}
TOP_INSTITUTIONS = {"mit","stanford","harvard","berkeley","caltech","oxford","cambridge",
                    "google","deepmind","openai","meta ai","microsoft","ibm","tsinghua","pku","ustc","zhejiang"}

# ---------- Helpers ----------
def strip_fences(s:str)->str:
    s=s.strip()
    s=re.sub(r"^```.*?\n","",s,flags=re.S)
    s=re.sub(r"\n```$","",s,flags=re.S)
    return s.strip()

def extract_first_json(s:str)->str:
    s=strip_fences(s)
    i,j=s.find("{"),s.rfind("}")
    return s[i:j+1] if i!=-1 and j!=-1 and j>i else s

def ox_get(url, params=None):
    r = requests.get(url, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()

def resolve_work_by_title(title:str):
    j=ox_get(f"{OPENALEX}/works", {"search": title, "per-page": 1})
    return j.get("results",[None])[0]

def resolve_work_by_doi(doi:str):
    doi=doi.lower().strip().replace("https://doi.org/","")
    try: return ox_get(f"{OPENALEX}/works/doi:{doi}")
    except: return None

def get_citing_works(openalex_id:str, per_page=200, max_pages=8)->List[dict]:
    out, cursor = [], "*"
    for _ in range(max_pages):
        j = ox_get(f"{OPENALEX}/works", {"filter": f"cites:{openalex_id}", "per-page": per_page, "cursor": cursor})
        out += j.get("results",[])
        cursor = j.get("meta",{}).get("next_cursor")
        if not cursor: break
    return out

def citing_countries(openalex_id:str):
    citing=get_citing_works(openalex_id)
    counts={}
    for w in citing:
        for a in w.get("authorships",[]):
            for ins in a.get("institutions",[]):
                cc=ins.get("country_code")
                if cc:
                    cc=cc.upper()
                    counts[cc]=counts.get(cc,0)+1
    return counts, citing

class ImpactFactorDB:
    def __init__(self, df: pd.DataFrame|None):
        self.lookup={}
        if df is not None and not df.empty and set(df.columns)>={"venue","if"}:
            for _,r in df.iterrows():
                v=str(r["venue"]).strip().lower()
                try:self.lookup[v]=float(r["if"])
                except:pass
    def get(self, venue:str)->float:
        if not venue: return 0.0
        v=venue.lower().strip()
        if v in self.lookup: return self.lookup[v]
        if any(k in v for k in TOP_VENUES): return 25.0
        return 0.0

def notability_score(work:dict, ifdb:ImpactFactorDB)->float:
    c = work.get("cited_by_count",0)
    venue = ((work.get("host_venue") or {}).get("display_name") or "")
    insts = []
    for a in work.get("authorships", []):
        for ins in a.get("institutions", []):
            nm = (ins.get("display_name") or "").lower()
            if nm: insts.append(nm)
    bonus=0.0
    if any(k in venue.lower() for k in TOP_VENUES): bonus+=1.0
    if any(any(k in x for k in TOP_INSTITUTIONS) for x in insts): bonus+=1.0
    ifv = ifdb.get(venue)
    if_term = log10(ifv+1)
    return log10(c+1) + bonus + if_term

def second_order(openalex_id:str, ifdb:ImpactFactorDB,
                 per_l1=80, pages_l1=5, per_l2=80, pages_l2=2)->pd.DataFrame:
    l1=get_citing_works(openalex_id, per_page=per_l1, max_pages=pages_l1)
    l2_map, prog={}, st.progress(0.0, text="Fetching second-order impact…")
    total=max(1,len(l1))
    for i,w in enumerate(l1,1):
        try:
            wid=w.get("id"); prog.progress(i/total)
            if not wid: continue
            sub=get_citing_works(wid, per_page=per_l2, max_pages=pages_l2)
            for s in sub:
                sid=s.get("id")
                if not sid: continue
                if sid not in l2_map or s.get("cited_by_count",0) > l2_map[sid].get("cited_by_count",0):
                    l2_map[sid]=s
        except: pass
    prog.empty()
    rows=[]
    for s in l2_map.values():
        venue=(s.get("host_venue") or {}).get("display_name","")
        ifv=ifdb.get(venue)
        rows.append({
            "title": s.get("display_name",""),
            "venue": venue,
            "IF": ifv,
            "year": s.get("publication_year",""),
            "cited_by": s.get("cited_by_count",0),
            "openalex_id": s.get("id",""),
            "score": notability_score(s, ifdb)
        })
    return pd.DataFrame(rows).sort_values(["score","cited_by","IF"], ascending=False)

def draw_country_block(cc_map:dict, title="Citing Countries"):
    if not cc_map:
        st.markdown('<div class="empty">No citing countries yet. Add publications first.</div>', unsafe_allow_html=True); return
    rows=[]
    for k,v in cc_map.items():
        rows.append({"ISO3": ISO2_TO_ISO3.get(k,k), "country": k, "count": v})
    df = pd.DataFrame(rows).sort_values("count", ascending=False)
    st.dataframe(df[["country","count"]], use_container_width=True, height=260)
    fig = px.choropleth(df, locations="ISO3", color="count",
                        hover_name="country", color_continuous_scale="Blues",
                        title=title, locationmode="ISO-3")
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>Count: %{z}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("ZHIPU_API_KEY", type="password", value=os.getenv("ZHIPU_API_KEY",""))
    model_id = st.text_input("Model", value="GLM-4.6")
    eval_mode = st.radio("Evaluation Target", ["NIW","EB-1A","Both"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    timeout_s = st.slider("Request timeout (sec)", 30, 180, 90, 5)
    show_raw = st.toggle("Debug: show raw model output", False)

# ---------- Tabs ----------
tabs = st.tabs(["① Profile", "② Publications", "③ Impact", "④ Documents", "⑤ Evaluate"])

# ----- Tab 1: Profile -----
with tabs[0]:
    st.markdown('<div class="card"><div class="subtle">Tell us who you are. This page is light and quick.</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    name = c1.text_input("Name / Applicant")
    field = c2.text_input("Field of Study")
    aff  = st.text_input("Affiliations / Collaborations (comma separated)")
    c3,c4 = st.columns(2)
    awards = c3.text_area("Awards / Grants (optional)", height=90)
    peer   = c4.text_area("Peer-review Experience (optional)", height=90)
    st.markdown("</div>", unsafe_allow_html=True)

# ----- Tab 2: Publications -----
with tabs[1]:
    if "pubs" not in st.session_state:
        st.session_state.pubs = [{"title":"","journal":"","year":"","citations":0,"countries":"US;CN"}]
    st.markdown('<div class="card"><b>Publications</b> · 用标题或 DOI 解析加入，或直接编辑表格。</div>', unsafe_allow_html=True)
    src = st.radio("数据来源", ["手动编辑表格","按标题解析（OpenAlex）","按 DOI 解析（OpenAlex）"], horizontal=True)
    ti = di = ""
    cA,cB = st.columns([1,1])
    if src=="按标题解析（OpenAlex）":
        ti = cA.text_input("论文标题")
        if cB.button("🔎 按标题解析并加入", use_container_width=True):
            item = resolve_work_by_title(ti.strip()) if ti.strip() else None
            if not item: st.warning("未解析到论文。")
            else:
                title=item.get("display_name","")
                venue=(item.get("host_venue") or {}).get("display_name","")
                year=item.get("publication_year",""); cites=item.get("cited_by_count",0)
                cc,_ = citing_countries(item["id"])
                st.session_state.pubs.append({"title":title,"journal":venue,"year":year,"citations":cites,"countries":";".join(sorted(cc.keys()))})
                st.success("已加入 Publications。")
    elif src=="按 DOI 解析（OpenAlex）":
        di = cA.text_input("DOI，例如 10.1021/acsami.xxxxxxx")
        if cB.button("🔎 按 DOI 解析并加入", use_container_width=True):
            item = resolve_work_by_doi(di.strip()) if di.strip() else None
            if not item: st.warning("未解析到论文。")
            else:
                title=item.get("display_name","")
                venue=(item.get("host_venue") or {}).get("display_name","")
                year=item.get("publication_year",""); cites=item.get("cited_by_count",0)
                cc,_ = citing_countries(item["id"])
                st.session_state.pubs.append({"title":title,"journal":venue,"year":year,"citations":cites,"countries":";".join(sorted(cc.keys()))})
                st.success("已加入 Publications。")

    pubs_edited = st.data_editor(
        st.session_state.pubs, num_rows="dynamic", use_container_width=True,
        column_config={
            "title": st.column_config.TextColumn("Title", width="medium"),
            "journal": st.column_config.TextColumn("Journal", width="medium"),
            "year": st.column_config.TextColumn("Year", width="small"),
            "citations": st.column_config.NumberColumn("Citations", min_value=0, step=1),
            "countries": st.column_config.TextColumn("Cited by Countries (US;CN;DE)")
        }
    )
    st.session_state.pubs = pubs_edited.to_dict("records") if hasattr(pubs_edited,"to_dict") else pubs_edited
    st.caption("Tip: 先把代表作加入，后面 Impact 页就能看到地图和二级传播。")

# ----- Tab 3: Impact -----
with tabs[2]:
    st.markdown('<div class="card"><b>Citing Countries</b> · Map + table · Derived from OpenAlex citing-works institutions.</div>', unsafe_allow_html=True)
    all_cc={}
    for p in st.session_state.pubs:
        for cc in (p.get("countries","") or "").split(";"):
            cc=cc.strip().upper()
            if cc: all_cc[cc]=all_cc.get(cc,0)+1
    draw_country_block(all_cc, "Citing Countries (All listed publications)")

    st.markdown('<div class="card"><b>Second-order Impact</b> · Who cites the papers that cite you. Upload IF CSV (venue,if) to weigh ranking.</div>', unsafe_allow_html=True)
    if_csv = st.file_uploader("上传 IF CSV（可选，两列：venue,if）", type=["csv"])
    if_db = ImpactFactorDB(pd.read_csv(if_csv)) if if_csv else ImpactFactorDB(None)

    if any((p.get("title") or "").strip() for p in st.session_state.pubs):
        for p in st.session_state.pubs:
            t=(p.get("title") or "").strip()
            if not t: continue
            with st.expander(f"🔁 {t[:80]}"):
                w = resolve_work_by_title(t)
                if not w: st.write("未解析到 OpenAlex 条目。"); continue
                df2 = second_order(w["id"], if_db, per_l1=50, pages_l1=3, per_l2=60, pages_l2=1)
                if df2.empty: st.write("暂无二级传播数据"); continue
                def linkify(row):
                    oid=row["openalex_id"]
                    url = f"https://openalex.org/{oid.split('/')[-1]}" if oid else ""
                    return f"[{row['title']}]({url})"
                show = df2.head(20).copy()
                show["work"]=show.apply(linkify, axis=1)
                st.dataframe(show[["work","venue","IF","year","cited_by","score"]], use_container_width=True)
                st.download_button("⬇️ 下载 CSV", df2.to_csv(index=False).encode("utf-8"),
                                   file_name=f"second_order_{t[:32]}.csv", mime="text/csv")
    else:
        st.markdown('<div class="empty">请先在 Publications 添加至少一篇论文。</div>', unsafe_allow_html=True)

# ----- Tab 4: Documents -----
with tabs[3]:
    st.markdown('<div class="card"><b>Petition / Future Plan / Recommendation Letters</b> · 可选但很重要：显著提升评估质量与建议的针对性。</div>', unsafe_allow_html=True)
    colA, colB = st.columns(2)
    up_pet = colA.file_uploader("Upload Petition (.txt/.docx)", type=["txt","docx"], key="pet")
    up_plan = colA.file_uploader("Upload Future Plan (.txt/.docx)", type=["txt","docx"], key="plan")
    recos = colB.file_uploader("Upload Recommendation Letters (multi)", type=["txt","docx"], accept_multiple_files=True)

    def read_text(upload):
        if upload is None: return ""
        if upload.name.endswith(".txt"): return upload.read().decode("utf-8","ignore")
        else:
            import docx; return "\n".join(p.text for p in docx.Document(upload).paragraphs)

    petition_text_uploaded  = read_text(up_pet)
    futureplan_text_uploaded= read_text(up_plan)
    recos_texts_uploaded    = [read_text(f) for f in (recos or []) if f]

    # 互斥：上传了就禁用粘贴框；并显示字符数/份数
    st.caption(f"已载入：Petition {'✓ ('+str(len(petition_text_uploaded))+' chars)' if petition_text_uploaded else '—'} · Future Plan {'✓ ('+str(len(futureplan_text_uploaded))+' chars)' if futureplan_text_uploaded else '—'} · Recos {len(recos_texts_uploaded)} 封")

    petition_text = st.text_area("Or paste Petition", value="" if petition_text_uploaded else "", height=150, disabled=bool(petition_text_uploaded))
    futureplan_text = st.text_area("Or paste Future Plan", value="" if futureplan_text_uploaded else "", height=140, disabled=bool(futureplan_text_uploaded))
    recos_concat = st.text_area("Or paste Recommendation Letters (concat allowed)", value="" if recos_texts_uploaded else "", height=140, disabled=bool(recos_texts_uploaded))

# ----- Build prompt -----
def build_user_input()->str:
    lines=[]
    if name: lines.append(f"Name: {name}")
    if field: lines.append(f"Field: {field}")
    if aff: lines.append(f"Affiliations: {aff}")
    if awards: lines.append(f"Awards: {awards}")
    if peer: lines.append(f"Peer-review: {peer}")

    pubs_lines,total_cites,all_cc=[],0,{}
    for p in st.session_state.pubs:
        t=(p.get("title") or "").strip()
        if not t: continue
        cnum=int(p.get("citations") or 0); total_cites+=cnum
        for cc in (p.get("countries","") or "").split(";"):
            cc=cc.strip().upper()
            if cc: all_cc[cc]=all_cc.get(cc,0)+1
        pubs_lines.append(f'- "{t}" ({p.get("journal","")}, {p.get("year","")}), citations={cnum}, cited_countries="{p.get("countries","")}"')
    if pubs_lines: lines.append("Publications:\n"+"\n".join(pubs_lines))
    if all_cc:
        top_cc=sorted(all_cc.items(), key=lambda x:x[1], reverse=True)[:10]
        lines.append("Bibliometrics Summary:\n"+
                     f"- Total citations (sum of listed): {total_cites}\n"+
                     f"- Top citing countries: "+", ".join([f"{k}:{v}" for k,v in top_cc])+
                     "\n- Country stats derived from OpenAlex citing-works institutions.")

    # documents
    pet = (petition_text_uploaded or petition_text or "").strip()
    plan = (futureplan_text_uploaded or futureplan_text or "").strip()
    recs = ("\n\n".join(recos_texts_uploaded) or recos_concat or "").strip()
    if pet: lines.append("Petition Letter:\n"+pet)
    if plan: lines.append("Future Plan:\n"+plan)
    if recs: lines.append("Recommendation Letters Combined:\n"+recs)
    return "\n".join(lines).strip()

def build_system_prompt(mode:str)->str:
    base = f"You are a senior USCIS-style evaluator for {mode} cases. Respond in English with a single JSON object only."
    if mode=="NIW":
        schema = r'''
{
  "analysis_summary":{"field_of_expertise":"string","key_achievements":"string"},
  "niw_prongs":{
    "prong_1":{"score":0,"reasoning":"string","suggestions":"string"},
    "prong_2":{"score":0,"reasoning":"string","suggestions":"string"},
    "prong_3":{"score":0,"reasoning":"string","suggestions":"string"}
  },
  "overall_assessment":{"niw_probability":"string","overall_suggestions":"string","total_score":0},
  "petition_review":{"strength":0,"structure":"string","suggestions":"string"},
  "future_plan_review":{"strength":0,"structure":"string","suggestions":"string"},
  "recommendation_review":{"coverage":"string","gaps":"string","actions":"string"},
  "future_plan_draft":["string","string","string"]
}
Definitions: Prong1=Substantial merit & national importance; Prong2=Well positioned; Prong3=Balance test.
If you cannot process, return {"error":"Unable to process the input."}.
'''
    elif mode=="EB-1A":
        schema = r'''
{
  "analysis_summary":{"field_of_expertise":"string","key_achievements":"string"},
  "eb1a_criteria":{
    "awards":"met|partial|not_met","membership":"met|partial|not_met","media":"met|partial|not_met",
    "judge_of_others":"met|partial|not_met","original_contribution":"met|partial|not_met",
    "authorship":"met|partial|not_met","exhibitions":"met|partial|not_met","leading_role":"met|partial|not_met",
    "high_salary":"met|partial|not_met","commercial_success":"met|partial|not_met"
  },
  "overall_assessment":{"eb1a_probability":"string","criteria_met_count":0,"overall_suggestions":"string"},
  "petition_review":{"strength":0,"structure":"string","suggestions":"string"},
  "future_plan_review":{"strength":0,"structure":"string","suggestions":"string"},
  "recommendation_review":{"coverage":"string","gaps":"string","actions":"string"},
  "future_plan_draft":["string","string","string"]
}
If you cannot process, return {"error":"Unable to process the input."}.
'''
    else:
        schema = r'''
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
  "petition_review":{"strength":0,"structure":"string","suggestions":"string"},
  "future_plan_review":{"strength":0,"structure":"string","suggestions":"string"},
  "recommendation_review":{"coverage":"string","gaps":"string","actions":"string"},
  "future_plan_draft":["string","string","string"]
}
If you cannot process, return {"error":"Unable to process the input."}.
'''
    return base + "\n" + schema + "\nConsider petition/future plan/recommendation letters; evaluate argumentation quality and integrate into probabilities. Output one JSON only."

def call_glm(user_input_text:str, api_key:str, temperature:float=0.2,
             mode:str="Both", model_id:str="GLM-4.6", timeout_s:int=90)->Dict[str,Any]:
    headers={"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    payload={
        "model": model_id,
        "messages":[
            {"role":"system","content": build_system_prompt(mode)},
            {"role":"user","content": user_input_text}
        ],
        "temperature": float(temperature),
        "max_tokens": 2000
    }
    backoffs=[0,2,5]
    last=""
    for i,delay in enumerate(backoffs,1):
        if delay: time.sleep(delay)
        resp=None
        try:
            resp=requests.post(API_URL, headers=headers, json=payload, timeout=timeout_s)
            last=resp.text
            if resp.status_code in (429,500,502,503,504): 
                if i<len(backoffs): continue
                return {"error": f"API busy (status {resp.status_code})", "raw": last}
            resp.raise_for_status()
            data=resp.json()
            return {"ok":True,"content":data["choices"][0]["message"]["content"]}
        except requests.exceptions.ReadTimeout:
            if i<len(backoffs): continue
            return {"error": f"Timeout after {timeout_s}s", "raw": last}
        except Exception as e:
            if i<len(backoffs): continue
            return {"error": f"API request failed: {e}", "raw": last}

def parse_json_safe(text:str):
    raw=extract_first_json(text)
    try: return json.loads(raw)
    except: return None

# ----- Tab 5: Evaluate -----
with tabs[4]:
    st.markdown('<div class="card"><b>Run summary</b><div class="subtle">将输出：通过概率、NIW Prongs / EB-1A Criteria、文本评审、Future Plan 草稿、引用国家地图、二级传播 Top 等。</div></div>', unsafe_allow_html=True)
    # KPI preview
    c1,c2,c3 = st.columns(3)
    try:
        total_cites = sum(int(p.get("citations") or 0) for p in st.session_state.pubs)
        pub_count   = sum(1 for p in st.session_state.pubs if (p.get("title") or "").strip())
    except:
        total_cites, pub_count = 0, 0
    c1.markdown(f'<div class="kpi subtle">Publications<div class="v">{pub_count}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi subtle">Total Citations (listed)<div class="v">{total_cites}</div></div>', unsafe_allow_html=True)
    docs_ready = any([petition_text_uploaded, futureplan_text_uploaded, recos_texts_uploaded])
    c3.markdown(f'<div class="kpi subtle">Docs Loaded<div class="v">{"Yes" if docs_ready else "No"}</div></div>', unsafe_allow_html=True)

    disabled = (pub_count==0 and not docs_ready)
    if disabled:
        st.markdown('<div class="empty">请至少添加 1 篇论文或上传任意一个文档（Petition / Future Plan / Recos）以开始评估。</div>', unsafe_allow_html=True)

    if st.button("🚀 开始智能评估", use_container_width=True, disabled=disabled):
        if not api_key:
            st.error("缺少 ZHIPU_API_KEY"); st.stop()
        ui_text = build_user_input()
        with st.spinner("GLM-4.6 正在评估…"):
            res = call_glm(ui_text, api_key, temperature, eval_mode, model_id, timeout_s)

        if "error" in res:
            st.error(res["error"])
            if show_raw: st.code(res.get("raw",""))
            st.stop()

        text = res["content"]
        data = parse_json_safe(text)
        if not data:
            st.warning("模型未返回有效 JSON，以下为原文："); st.code(text); st.stop()

        st.success("✅ 结构化结果")
        # Summary
        a = data.get("analysis_summary", {})
        g1,g2 = st.columns(2)
        g1.markdown("**Field of Expertise**"); g1.markdown(f"<div class='kpi'><div class='v'>{a.get('field_of_expertise','-')}</div></div>", unsafe_allow_html=True)
        g2.markdown("**Key Achievements**"); g2.markdown(f"<div class='kpi'>{a.get('key_achievements','-')}</div>", unsafe_allow_html=True)
        st.markdown("---")

        # NIW / EB-1A
        if "niw_prongs" in data:
            st.subheader("⚖️ NIW Prongs")
            for k, label in [("prong_1","Prong 1"),("prong_2","Prong 2"),("prong_3","Prong 3")]:
                pr=data["niw_prongs"].get(k,{})
                with st.expander(f"{label} | Score {pr.get('score','-')}/10"):
                    st.write("**Reasoning:**", pr.get("reasoning","-"))
                    st.info(pr.get("suggestions","-"))
        if "eb1a_criteria" in data:
            st.subheader("🏅 EB-1A Criteria (10)")
            table=pd.DataFrame([(k,v) for k,v in data["eb1a_criteria"].items()], columns=["Criterion","Status"])
            st.dataframe(table, use_container_width=True)

        # Overall
        if "overall_assessment" in data:
            st.subheader("📊 Overall Assessment")
            oa=data["overall_assessment"]; c=st.columns(3)
            c[0].metric("NIW Probability", oa.get("niw_probability","-"))
            c[1].metric("EB-1A Probability", oa.get("eb1a_probability", oa.get("eb1a_probability","-")))
            c[2].metric("Score/Count", oa.get("total_score", oa.get("criteria_met_count","-")))
            st.write("**Overall Suggestions:**", oa.get("overall_suggestions","-"))

        # Text reviews
        if any(k in data for k in ["petition_review","future_plan_review","recommendation_review"]):
            st.subheader("🧾 Text Reviews")
            if "petition_review" in data:
                pr=data["petition_review"]; st.write("**Petition** — Strength:", pr.get("strength","-")); st.info(pr.get("suggestions","-"))
            if "future_plan_review" in data:
                fr=data["future_plan_review"]; st.write("**Future Plan** — Strength:", fr.get("strength","-")); st.info(fr.get("suggestions","-"))
            if "recommendation_review" in data:
                rr=data["recommendation_review"]; st.write("**Recommendation Letters** — Coverage:", rr.get("coverage","-")); st.info(rr.get("actions","-"))

        # Future Plan draft
        if "future_plan_draft" in data:
            st.subheader("🧭 Future Plan (Draft)")
            for i,p in enumerate(data["future_plan_draft"],1): st.write(f"{i}. {p}")

        st.download_button("📥 Download JSON", json.dumps(data, indent=2).encode("utf-8"),
                           "niw_eb1a_report.json","application/json")

# ---------- Floating CTA ----------
with st.container():
    st.markdown('<div class="fab">', unsafe_allow_html=True)
    st.button("🚀 开始评估", key="fab_run", use_container_width=False, disabled=False)
    st.markdown('</div>', unsafe_allow_html=True)
