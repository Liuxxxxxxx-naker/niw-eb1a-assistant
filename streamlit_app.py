import os, re, json, time, requests, pandas as pd, plotly.express as px, streamlit as st
from math import log10

# ============ 页面配置 ============
st.set_page_config(page_title="NIW / EB-1A 智能评估助手", layout="wide", page_icon="🇺🇸")

# ============ 样式美化 ============
st.markdown("""
<style>
.hero { padding:18px 20px; border:1px solid #232836; background:#151a22; border-radius:16px; margin:-8px 0 8px 0; }
.hero h1 { margin:0 0 6px 0; font-size:1.4rem }
.chips span{ display:inline-block; padding:.28rem .55rem; margin:.2rem .35rem .2rem 0; border:1px solid #2a3140; border-radius:999px; color:#cbd3df; font-size:.86rem }
.card { background:#151a22; border:1px solid #232836; border-radius:16px; padding:18px; margin:8px 0 12px 0; }
.section-title { font-weight:700; margin-bottom:.4rem }
.empty { color:#9aa4b2; border:1px dashed #334055; border-radius:12px; padding:16px; text-align:center }
.step { color:#9aa4b2; font-size:.9rem }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { border:1px solid #2a3140; background:#141923; padding-top:10px; padding-bottom:10px; border-radius:10px; }
.stTabs [aria-selected="true"] { border:1px solid #3a4560 !important; background:#192131 !important; }
.stButton>button { border-radius:12px; height:2.7rem; font-weight:600; }
.fab { position:fixed; right:22px; bottom:22px; z-index:99; }
.kpi { background:#11151d; border:1px solid #273047; border-radius:14px; padding:12px 14px; }
.kpi .v { font-size:1.8rem; font-weight:800; margin-top:.2rem }
.small { color:#9aa4b2; font-size:.9rem }
hr { border-color:#232836; }
</style>
""", unsafe_allow_html=True)

# ============ 顶部 Hero ============
st.markdown("""
<div class="hero">
  <h1>NIW / EB-1A 智能评估助手（GLM-4.6）</h1>
  <div class="small">分步输入 · 自动抓取 OpenAlex 数据 · 二级传播（含影响因子） · Petition/Future Plan/推荐信联合评审 · 结构化 JSON 输出</div>
  <div class="chips" style="margin-top:6px;">
    <span>📚 Publications 智能解析</span>
    <span>🗺️ 引用国家地图</span>
    <span>🔁 二级传播排行</span>
    <span>🧾 Petition / Plan / Recos 评审</span>
    <span>⚖️ NIW / EB-1A 双模式</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ============ 常量和工具函数 ============
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
OPENALEX = "https://api.openalex.org"
ISO2_TO_ISO3 = {"US":"USA","CN":"CHN","DE":"DEU","JP":"JPN","GB":"GBR","FR":"FRA","CA":"CAN","AU":"AUS","KR":"KOR","IN":"IND"}

TOP_VENUES = {"nature","science","cell","pnas","advanced materials","energy & environmental science","jacs"}
TOP_INSTITUTIONS = {"mit","stanford","harvard","berkeley","caltech","oxford","cambridge","google","deepmind","openai","tsinghua","pku"}

def ox_get(url, params=None):
    r = requests.get(url, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()

def resolve_work_by_title(title:str):
    j = ox_get(f"{OPENALEX}/works", {"search": title, "per-page": 1})
    return j.get("results", [None])[0]

def resolve_work_by_doi(doi:str):
    doi=doi.lower().strip().replace("https://doi.org/","")
    try: return ox_get(f"{OPENALEX}/works/doi:{doi}")
    except: return None

def get_citing_works(openalex_id:str, per_page=100, max_pages=5):
    out, cursor = [], "*"
    for _ in range(max_pages):
        j = ox_get(f"{OPENALEX}/works", {"filter": f"cites:{openalex_id}", "per-page": per_page, "cursor": cursor})
        out += j.get("results", [])
        cursor = j.get("meta", {}).get("next_cursor")
        if not cursor: break
    return out

def citing_countries(openalex_id:str):
    citing = get_citing_works(openalex_id)
    counts={}
    for w in citing:
        for a in w.get("authorships",[]):
            for ins in a.get("institutions",[]):
                cc=ins.get("country_code")
                if cc:
                    counts[cc]=counts.get(cc,0)+1
    return counts, citing

class ImpactFactorDB:
    def __init__(self, df=None):
        self.lookup={}
        if df is not None and not df.empty:
            for _,r in df.iterrows():
                self.lookup[str(r["venue"]).lower()] = float(r["if"])
    def get(self, venue:str):
        if not venue: return 0
        v=venue.lower()
        if v in self.lookup: return self.lookup[v]
        if any(k in v for k in TOP_VENUES): return 20
        return 0

def notability_score(work, ifdb):
    c = work.get("cited_by_count",0)
    venue = (work.get("host_venue") or {}).get("display_name","")
    bonus = 0
    if any(k in venue.lower() for k in TOP_VENUES): bonus+=1
    if any(k in venue.lower() for k in TOP_INSTITUTIONS): bonus+=1
    return log10(c+1)+log10(ifdb.get(venue)+1)+bonus

def second_order(openalex_id:str, ifdb:ImpactFactorDB):
    l1 = get_citing_works(openalex_id)
    l2_map = {}
    for w in l1:
        try:
            sub = get_citing_works(w.get("id",""), per_page=40, max_pages=2)
            for s in sub:
                sid = s.get("id")
                if sid and sid not in l2_map:
                    l2_map[sid] = s
        except: pass
    rows=[]
    for s in l2_map.values():
        venue = (s.get("host_venue") or {}).get("display_name","")
        rows.append({
            "title": s.get("display_name",""),
            "venue": venue,
            "IF": ifdb.get(venue),
            "year": s.get("publication_year",""),
            "cited_by": s.get("cited_by_count",0),
            "score": notability_score(s, ifdb)
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False)

def draw_country_block(cc_map, title="Citing Countries"):
    if not cc_map:
        st.info("暂无引用国家数据"); return
    df=pd.DataFrame([{"country":k,"count":v,"ISO3":ISO2_TO_ISO3.get(k,k)} for k,v in cc_map.items()])
    fig=px.choropleth(df, locations="ISO3", color="count", hover_name="country", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

# ============ 侧边栏设置 ============
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("ZHIPU_API_KEY", type="password")
    model_id = st.text_input("Model ID", "GLM-4.6")
    eval_mode = st.radio("Evaluation Target", ["NIW","EB-1A","Both"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    timeout_s = st.slider("Timeout (sec)", 30, 180, 90)

# ============ 主体导航 ============
tabs = st.tabs(["① Profile", "② Publications", "③ Impact", "④ Documents", "⑤ Evaluate"])

# ---------- Tab 1 ----------
with tabs[0]:
    st.markdown('<div class="card"><div class="section-title">基本信息</div>', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    name=c1.text_input("Applicant Name")
    field=c2.text_input("Field of Study")
    aff=st.text_input("Affiliations")
    awards=st.text_area("Awards / Grants (optional)")
    peer=st.text_area("Peer Review (optional)")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Tab 2 ----------
with tabs[1]:
    if "pubs" not in st.session_state:
        st.session_state.pubs=[]
    st.markdown('<div class="card"><div class="section-title">Publications</div>', unsafe_allow_html=True)
    mode=st.radio("数据来源",["手动输入","按标题解析","按DOI解析"],horizontal=True)
    if mode=="按标题解析":
        title_in=st.text_input("输入论文标题")
        if st.button("解析标题"):
            r=resolve_work_by_title(title_in)
            if r:
                cc,_=citing_countries(r["id"])
                st.session_state.pubs.append({
                    "title":r.get("display_name",""),
                    "journal":(r.get("host_venue") or {}).get("display_name",""),
                    "year":r.get("publication_year",""),
                    "citations":r.get("cited_by_count",0),
                    "countries":";".join(cc.keys())
                })
    elif mode=="按DOI解析":
        doi_in=st.text_input("输入 DOI")
        if st.button("解析DOI"):
            r=resolve_work_by_doi(doi_in)
            if r:
                cc,_=citing_countries(r["id"])
                st.session_state.pubs.append({
                    "title":r.get("display_name",""),
                    "journal":(r.get("host_venue") or {}).get("display_name",""),
                    "year":r.get("publication_year",""),
                    "citations":r.get("cited_by_count",0),
                    "countries":";".join(cc.keys())
                })
    df = pd.DataFrame(st.session_state.pubs)
    st.dataframe(df,use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Tab 3 ----------
with tabs[2]:
    st.markdown('<div class="card"><div class="section-title">引用国家地图</div>', unsafe_allow_html=True)
    cc_all={}
    for p in st.session_state.pubs:
        for c in (p.get("countries") or "").split(";"):
            c=c.strip().upper()
            if c: cc_all[c]=cc_all.get(c,0)+1
    draw_country_block(cc_all)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="section-title">二级传播</div>', unsafe_allow_html=True)
    csv_file=st.file_uploader("上传影响因子CSV (venue,if)",type="csv")
    ifdb=ImpactFactorDB(pd.read_csv(csv_file)) if csv_file else ImpactFactorDB(None)
    for p in st.session_state.pubs:
        t=(p.get("title") or "").strip()
        if not t: continue
        with st.expander(f"🔁 {t[:80]}"):
            w=resolve_work_by_title(t)
            if not w: continue
            df2=second_order(w["id"],ifdb)
            if not df2.empty:
                st.dataframe(df2.head(10),use_container_width=True)
                st.download_button("下载CSV",df2.to_csv(index=False).encode("utf-8"),file_name=f"{t[:30]}.csv")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Tab 4 ----------
with tabs[3]:
    st.markdown('<div class="card"><div class="section-title">文档上传</div>', unsafe_allow_html=True)
    col1,col2=st.columns(2)
    pet=col1.file_uploader("Petition (.txt/.docx)",type=["txt","docx"])
    plan=col1.file_uploader("Future Plan (.txt/.docx)",type=["txt","docx"])
    recos=col2.file_uploader("Recommendation Letters (multi)",type=["txt","docx"],accept_multiple_files=True)
    def read_text(up):
        if not up: return ""
        if up.name.endswith(".txt"): return up.read().decode("utf-8","ignore")
        else:
            import docx
            return "\n".join(p.text for p in docx.Document(up).paragraphs)
    petition_text=read_text(pet)
    futureplan_text=read_text(plan)
    recos_text="\n\n".join([read_text(f) for f in recos or []])
    st.text_area("Petition 内容",petition_text,height=150)
    st.text_area("Future Plan 内容",futureplan_text,height=150)
    st.text_area("Recommendation Letters 内容",recos_text,height=150)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Tab 5 ----------
with tabs[4]:
    st.markdown('<div class="card"><div class="section-title">智能评估</div>', unsafe_allow_html=True)
    st.write("（这里可接入你的 GLM 评估逻辑，例如 call_glm + JSON 渲染）")
    st.button("🚀 开始评估", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 悬浮按钮
st.markdown('<div class="fab">', unsafe_allow_html=True)
st.button("🚀 开始评估", key="fab_run", use_container_width=False)
st.markdown('</div>', unsafe_allow_html=True)
