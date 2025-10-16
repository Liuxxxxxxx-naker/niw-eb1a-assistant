import os, re, json, time, requests, pandas as pd, plotly.express as px, streamlit as st
from math import log10
from typing import Dict, Any, List

st.set_page_config(page_title="NIW / EB-1A Assistant", layout="wide", page_icon="🇺🇸")
st.markdown("""
<style>
:root { --card:#151a22; --muted:#9aa4b2; --ring:#7c9cff; }
.block-container { padding-top: 1.2rem; }
h1,h2,h3 { letter-spacing:.2px }
.small { color:var(--muted); font-size:.9rem }
hr { border-color:#232836 }
.card { background:#151a22; border:1px solid #232836; border-radius:16px; padding:18px; margin-bottom:14px; }
.metric { font-size:2rem; font-weight:700 }
.badge { display:inline-block; padding:.2rem .5rem; border:1px solid #2a3140; border-radius:.5rem; color:#cbd3df; margin-right:.35rem }
.stButton>button { border-radius:12px; height:2.8rem; font-weight:600; border:1px solid #2a3140 }
.stTextInput input, .stTextArea textarea { border-radius:10px; border:1px solid #2a3140 }
.stDataFrame, .stDataEditor { border-radius:12px; }
</style>
""", unsafe_allow_html=True)

st.title("🇺🇸 NIW / EB-1A 智能评估助手 (GLM-4.6)")
st.caption("中/英皆可输入 · 统一英文输出 · 地图 + 二级传播（含影响因子）· Petition/Future Plan/推荐信 文字级评审 · JSON 卡片渲染")

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
OPENALEX = "https://api.openalex.org"
ISO2_TO_ISO3 = {"US":"USA","CN":"CHN","DE":"DEU","JP":"JPN","GB":"GBR","UK":"GBR","FR":"FRA","CA":"CAN","AU":"AUS",
                "IT":"ITA","ES":"ESP","KR":"KOR","IN":"IND","NL":"NLD","SE":"SWE","NO":"NOR","FI":"FIN","DK":"DNK",
                "CH":"CHE","AT":"AUT","BE":"BEL","SG":"SGP","HK":"HKG","TW":"TWN","IL":"ISR","SA":"SAU","AE":"ARE",
                "RU":"RUS","BR":"BRA","MX":"MEX","ZA":"ZAF","PL":"POL","TR":"TUR"}

TOP_VENUES = {"nature","science","cell","pnas","lancet","jama","nejm","jacs","advanced materials","angewandte",
              "energy & environmental science","neurips","icml","iclr","cvpr","aaai","kdd","www","acl"}
TOP_INSTITUTIONS = {"mit","stanford","harvard","berkeley","caltech","oxford","cambridge",
                    "google","deepmind","openai","meta ai","microsoft","ibm","tsinghua","pku","ustc","zhejiang"}

# -------- utils
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
    return j.get("results", [None])[0]

def resolve_work_by_doi(doi:str):
    doi=doi.lower().strip().replace("https://doi.org/","")
    try: return ox_get(f"{OPENALEX}/works/doi:{doi}")
    except: return None

def get_citing_works(openalex_id:str, per_page=200, max_pages=8)->List[dict]:
    out, cursor = [], "*"
    for _ in range(max_pages):
        j = ox_get(f"{OPENALEX}/works", {"filter": f"cites:{openalex_id}", "per-page": per_page, "cursor": cursor})
        items=j.get("results",[])
        out.extend(items)
        cursor=j.get("meta",{}).get("next_cursor")
        if not cursor or not items: break
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

# -------- impact factor provider
class ImpactFactorDB:
    def __init__(self, df: pd.DataFrame|None):
        self.lookup={}
        if df is not None and not df.empty and set(df.columns)>={"venue","if"}:
            for _,r in df.iterrows():
                v=str(r["venue"]).strip().lower()
                try:
                    self.lookup[v]=float(r["if"])
                except:
                    pass
    def get(self, venue:str)->float:
        if not venue: return 0.0
        v=venue.lower().strip()
        if v in self.lookup: return self.lookup[v]
        # 粗略 fallback：顶刊关键词给一个大致 IF
        if any(k in v for k in TOP_VENUES): return 25.0
        return 0.0

def notability_score(work:dict, ifdb: ImpactFactorDB)->float:
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
    if_term = log10(ifv+1)  # IF 越大增益越多，但对数压缩
    return log10(c+1) + bonus + if_term

def second_order(openalex_id:str, ifdb: ImpactFactorDB,
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
    df=pd.DataFrame(rows).sort_values(["score","cited_by","IF"], ascending=False)
    return df

def draw_country_block(cc_map:dict, title="Citing Countries"):
    if not cc_map:
        st.info("No citing countries found."); return
    rows=[]
    for k,v in cc_map.items():
        rows.append({"ISO3": ISO2_TO_ISO3.get(k,k), "country": k, "count": v})
    df = pd.DataFrame(rows).sort_values("count", ascending=False)
    top_n = min(15, len(df))
    st.markdown(f"**Top {top_n} citing countries**")
    st.dataframe(df.head(top_n)[["country","count"]], use_container_width=True)
    st.download_button("⬇️ Download country counts (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="citing_countries.csv", mime="text/csv")
    fig = px.choropleth(df, locations="ISO3", color="count",
                        hover_name="country",
                        hover_data={"ISO3":False,"count":True,"country":True},
                        color_continuous_scale="Blues", locationmode="ISO-3", title=title)
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>Count: %{z}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.bar(df.head(top_n), x="country", y="count", title="Top Citing Countries (bar)")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("ZHIPU_API_KEY", type="password", value=os.getenv("ZHIPU_API_KEY",""))
    model_id = st.text_input("Model ID", value="GLM-4.6")
    eval_mode = st.radio("Evaluation Target", ["NIW","EB-1A","Both"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    timeout_s = st.slider("Request timeout (sec)", 30, 180, 90, 5)
    show_raw = st.toggle("Debug: show raw model output", False)

# ---------------- Petition / Plan / Recos ----------------
st.subheader("🧾 Petition & Future Plan & Recommendation Letters")
colA, colB = st.columns(2)
with colA:
    up_pet = st.file_uploader("Upload Petition Letter (.txt/.docx)", type=["txt","docx"], key="pet")
    up_plan = st.file_uploader("Upload Future Plan (.txt/.docx)", type=["txt","docx"], key="plan")
with colB:
    recos = st.file_uploader("Upload Recommendation Letters (multi, .txt/.docx)", type=["txt","docx"], accept_multiple_files=True)

def read_text_from_upload(upload):
    if upload is None: return ""
    if upload.name.endswith(".txt"):
        return upload.read().decode("utf-8","ignore")
    else:
        import docx
        return "\n".join(p.text for p in docx.Document(upload).paragraphs)

petition_text = read_text_from_upload(up_pet)
futureplan_text = read_text_from_upload(up_plan)
recos_texts = [read_text_from_upload(f) for f in (recos or [])]
petition_text = st.text_area("Or paste Petition Letter", value=petition_text, height=180)
futureplan_text = st.text_area("Or paste Future Plan", value=futureplan_text, height=160)
recos_concat = st.text_area("Or paste Recommendation Letters (one per block or concat)", value="\n\n".join([t for t in recos_texts if t]), height=160)

# ---------------- Applicant ----------------
st.subheader("🧑‍🔬 Applicant")
c1,c2 = st.columns(2)
name = c1.text_input("Name / Applicant")
field = c2.text_input("Field of Study")
aff  = st.text_input("Affiliations / Collaborations (comma separated)")
awards = st.text_area("Awards / Grants (optional)", height=70)
peer   = st.text_area("Peer-review Experience (optional)", height=70)

# ---------------- Publications ----------------
st.subheader("📚 Publications")
if "pubs" not in st.session_state:
    st.session_state.pubs = [{"title":"","journal":"","year":"","citations":0,"countries":"US;CN"}]

src = st.radio("数据来源", ["手动输入/表格","按标题解析（OpenAlex）","按DOI解析（OpenAlex）"], horizontal=True)
ti = di = ""
if src=="按标题解析（OpenAlex）":
    ti = st.text_input("论文标题")
elif src=="按DOI解析（OpenAlex）":
    di = st.text_input("DOI，例如 10.1021/acsami.xxxxxxx")

c3,c4 = st.columns([1,1])
if c3.button("🔎 解析并加入"):
    item=None
    if ti.strip(): item=resolve_work_by_title(ti.strip())
    if di.strip(): item=resolve_work_by_doi(di.strip())
    if not item: st.warning("未解析到论文。")
    else:
        title=item.get("display_name","")
        venue=(item.get("host_venue") or {}).get("display_name","")
        year=item.get("publication_year","")
        cites=item.get("cited_by_count",0)
        cc, citing_list = citing_countries(item["id"])
        cc_str=";".join(sorted(cc.keys()))
        st.session_state.pubs.append({"title":title,"journal":venue,"year":year,"citations":cites,"countries":cc_str})
        st.success("已加入 Publications。")
        with st.expander("本篇引用国家地图"):
            draw_country_block(cc, title=f"Citing Countries – {title[:60]}")
        with st.expander("🔁 二级传播（引用你的引用，含影响因子）"):
            ifdb = ImpactFactorDB(None)
            df2 = second_order(item["id"], ifdb)
            if not df2.empty:
                # OpenAlex链接
                def linkify(row):
                    oid=row["openalex_id"]
                    url = f"https://openalex.org/{oid.split('/')[-1]}" if oid else ""
                    return f"[{row['title']}]({url})"
                dfshow = df2.head(30).copy()
                dfshow["work"] = dfshow.apply(linkify, axis=1)
                st.dataframe(dfshow[["work","venue","IF","year","cited_by","score"]], use_container_width=True)
                st.download_button("⬇️ 下载 CSV", df2.to_csv(index=False).encode("utf-8"), "second_order.csv","text/csv")
            else:
                st.info("未获取到二级传播数据。")

if c4.button("🧹 清空表格"):
    st.session_state.pubs = [{"title":"","journal":"","year":"","citations":0,"countries":"US;CN"}]

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
st.session_state.pubs = pubs_edited.to_dict("records") if hasattr(pubs_edited, "to_dict") else pubs_edited

# -------- IF CSV 上传（用于二级传播评分权重）
st.subheader("📈 期刊影响因子 CSV（可选）")
if_csv = st.file_uploader("上传 IF CSV（两列：venue,if）", type=["csv"])
if_db = ImpactFactorDB(pd.read_csv(if_csv)) if if_csv else ImpactFactorDB(None)
st.caption("未上传时使用顶刊关键词作为近似权重。")

# ---------------- Notes ----------------
st.subheader("📝 Additional Notes")
notes = st.text_area("Any context to include (projects/patents/media/collabs etc.)", height=120)

# ---------------- Build prompt & call model ----------------
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

    if notes: lines.append("Notes:\n"+notes)

    if petition_text: lines.append("Petition Letter:\n"+petition_text)
    if futureplan_text: lines.append("Future Plan:\n"+futureplan_text)
    if recos_concat:
        lines.append("Recommendation Letters Combined:\n"+recos_concat)

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
    return base + "\n" + schema + "\nConsider the attached petition, future plan, and recommendation letters; evaluate argumentation quality and integrate into probabilities."

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

# ---------------- Run ----------------
st.markdown("<hr/>", unsafe_allow_html=True)
run = st.button("🚀 开始智能评估", use_container_width=True)

if run:
    if not api_key:
        st.error("缺少 ZHIPU_API_KEY"); st.stop()

    ui = build_user_input()

    st.subheader("🌍 Citing Countries Overview")
    all_cc={}
    for p in st.session_state.pubs:
        for cc in (p.get("countries","") or "").split(";"):
            cc=cc.strip().upper()
            if cc: all_cc[cc]=all_cc.get(cc,0)+1
    draw_country_block(all_cc, "Citing Countries (all listed publications)")

    with st.spinner("GLM-4.6 正在评估…"):
        res = call_glm(ui, api_key, temperature, eval_mode, model_id, timeout_s)

    if "error" in res:
        st.error(res["error"])
        if show_raw: st.code(res.get("raw",""))
        st.stop()

    text = res["content"]
    data = parse_json_safe(text)

    if data:
        st.success("✅ 结构化结果")
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            a=data.get("analysis_summary",{})
            cols=st.columns(2)
            cols[0].markdown("**Field of Expertise**"); cols[0].markdown(f"<div class='metric'>{a.get('field_of_expertise','-')}</div>", unsafe_allow_html=True)
            cols[1].markdown("**Key Achievements**"); cols[1].write(a.get("key_achievements","-"))
            st.markdown("</div>", unsafe_allow_html=True)

        if "niw_prongs" in data:
            st.markdown("### ⚖️ NIW Prongs")
            for k, label in [("prong_1","Prong 1"),("prong_2","Prong 2"),("prong_3","Prong 3")]:
                pr=data["niw_prongs"].get(k,{})
                with st.expander(f"{label} | Score {pr.get('score','-')}/10"):
                    st.write("**Reasoning:**", pr.get("reasoning","-"))
                    st.info(pr.get("suggestions","-"))

        if "eb1a_criteria" in data:
            st.markdown("### 🏅 EB-1A Criteria (10)")
            table=pd.DataFrame([(k,v) for k,v in data["eb1a_criteria"].items()], columns=["Criterion","Status"])
            st.dataframe(table, use_container_width=True)

        if "overall_assessment" in data:
            st.markdown("### 📊 Overall Assessment")
            oa=data["overall_assessment"]
            c=st.columns(3)
            c[0].markdown("**NIW Probability**"); c[0].markdown(f"<div class='metric'>{oa.get('niw_probability','-')}</div>", unsafe_allow_html=True)
            c[1].markdown("**EB-1A Probability**"); c[1].markdown(f"<div class='metric'>{oa.get('eb1a_probability', oa.get('eb1a_probability','-'))}</div>", unsafe_allow_html=True)
            c[2].markdown("**Score/Count**"); c[2].markdown(f"<div class='metric'>{oa.get('total_score', oa.get('criteria_met_count','-'))}</div>", unsafe_allow_html=True)
            st.write("**Overall Suggestions:**", oa.get("overall_suggestions","-"))

        if "petition_review" in data or "future_plan_review" in data or "recommendation_review" in data:
            st.markdown("### 🧾 Text Reviews")
            if "petition_review" in data:
                pr=data["petition_review"]; st.subheader("Petition")
                st.write("Strength:", pr.get("strength","-"))
                st.write("Structure:", pr.get("structure","-"))
                st.info(pr.get("suggestions","-"))
            if "future_plan_review" in data:
                fr=data["future_plan_review"]; st.subheader("Future Plan")
                st.write("Strength:", fr.get("strength","-"))
                st.write("Structure:", fr.get("structure","-"))
                st.info(fr.get("suggestions","-"))
            if "recommendation_review" in data:
                rr=data["recommendation_review"]; st.subheader("Recommendation Letters")
                st.write("Coverage:", rr.get("coverage","-"))
                st.write("Gaps:", rr.get("gaps","-"))
                st.info(rr.get("actions","-"))

        if "future_plan_draft" in data:
            st.markdown("### 🧭 Future Plan (Draft)")
            for i,p in enumerate(data["future_plan_draft"],1):
                st.write(f"{i}. {p}")

        st.download_button("📥 Download JSON", json.dumps(data, indent=2).encode("utf-8"),
                           "niw_eb1a_report.json", "application/json")
    else:
        st.warning("模型未返回有效 JSON，以下为原始文本：")
        st.code(text)

# -------- Combined second-order with IF --------
with st.expander("🔎 Amplified Impact – Combine all publications (with IF)"):
    combine=[]
    for p in st.session_state.pubs:
        t=(p.get("title") or "").strip()
        if not t: continue
        w=resolve_work_by_title(t)
        if not w: continue
        df2 = second_order(w["id"], if_db, per_l1=50, pages_l1=3, per_l2=60, pages_l2=1)
        if not df2.empty:
            df2["source_title"]=t
            combine.append(df2)
    if combine:
        all_df=pd.concat(combine, ignore_index=True).sort_values(["score","cited_by","IF"], ascending=False)
        def linkify(row):
            oid=row["openalex_id"]
            url = f"https://openalex.org/{oid.split('/')[-1]}" if oid else ""
            return f"[{row['title']}]({url})"
        dfshow=all_df.head(50).copy()
        dfshow["work"]=dfshow.apply(linkify, axis=1)
        st.dataframe(dfshow[["work","venue","IF","year","cited_by","score","source_title"]], use_container_width=True)
        st.download_button("⬇️ Download combined second-order (CSV)",
                           all_df.to_csv(index=False).encode("utf-8"),
                           "second_order_all.csv","text/csv")
        top3=all_df.head(3).to_dict("records")
        pts="; ".join([f"{r['title']} ({r['venue']}, IF={r['IF']}, {r['year']}, cited_by={r['cited_by']})" for r in top3])
        st.code(f"""The applicant's work shows second-order influence: {pts}. 
This evidences propagation in high-impact venues and leading institutions, supporting NIW Prong 1 and EB-1A major-significance contributions.""")
    else:
        st.info("请先通过“按标题/DOI解析”把代表作加入表格，再展开本卡片。")
