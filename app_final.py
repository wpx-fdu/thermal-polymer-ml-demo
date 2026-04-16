
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="热响应聚合物相变温区预测与工程筛选平台（最终版）",
    layout="wide",
)

# ========== Theme ==========
st.markdown(
    """
<style>
:root {
    --bg-soft: linear-gradient(180deg, #f5fbff 0%, #eef7ff 100%);
    --card-bg: rgba(255,255,255,0.92);
    --line: #dbeafe;
    --text-main: #0f172a;
    --text-sub: #5b6472;
    --blue-1: #eaf4ff;
    --blue-2: #cfe7ff;
    --blue-3: #93c5fd;
    --blue-4: #60a5fa;
    --blue-5: #2563eb;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-soft);
}
.block-container {
    padding-top: 1.0rem;
    padding-bottom: 2rem;
    max-width: 1220px;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(239,246,255,0.96));
    border: 1px solid rgba(147,197,253,0.35);
    border-radius: 18px;
    padding: 14px;
    box-shadow: 0 8px 24px rgba(37,99,235,0.08);
}
.app-hero {
    background: radial-gradient(circle at top right, rgba(147,197,253,0.22), transparent 28%),
                linear-gradient(135deg, rgba(255,255,255,0.95), rgba(239,246,255,0.98));
    border: 1px solid rgba(147,197,253,0.40);
    border-radius: 22px;
    padding: 22px 24px;
    box-shadow: 0 10px 28px rgba(37,99,235,0.08);
    margin-bottom: 14px;
}
.app-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,252,255,0.98));
    border: 1px solid rgba(147,197,253,0.28);
    border-radius: 20px;
    padding: 18px 20px;
    box-shadow: 0 8px 20px rgba(37,99,235,0.07);
    margin-bottom: 14px;
    backdrop-filter: blur(10px);
}
.small-note {
    color: var(--text-sub);
    font-size: 0.93rem;
}
.section-title {
    font-weight: 750;
    font-size: 1.05rem;
    color: var(--text-main);
    margin-bottom: 0.55rem;
}
.result-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(219,234,254,1), rgba(191,219,254,0.9));
    color: #1d4ed8;
    font-size: 0.87rem;
    font-weight: 700;
}
.info-pill {
    display: inline-block;
    margin-right: 8px;
    margin-top: 4px;
    padding: 5px 10px;
    border-radius: 999px;
    background: rgba(239,246,255,0.95);
    border: 1px solid rgba(147,197,253,0.35);
    color: #1e40af;
    font-size: 0.82rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ========== Model ==========
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "g2_rf_model.pkl"
META_PATH = APP_DIR / "g2_model_metadata.json"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, meta

model, meta = load_model()

@dataclass
class PredictionResult:
    predicted_lcst: Optional[float]
    model_group: str
    confidence_note: str
    engineering_score_10: Optional[float]
    thermal_window_label: str
    engineering_comment: str
    next_step: str

def infer_group(polymer_main: str) -> str:
    if polymer_main in ["PBA", "PnBMA"]:
        return "G2"
    if polymer_main in ["PEO", "PEGE", "PGME", "PEEGE", "PPO"]:
        return "G1"
    if polymer_main in ["PBzMA", "PBnMA", "PBMA"]:
        return "G3"
    return "Unknown"

def thermal_window_score(t: float) -> int:
    if 90 <= t <= 120:
        return 2
    if 60 <= t < 90 or 120 < t <= 160:
        return 1
    return 0

def application_proximity_score(system_form: str, has_salt: bool) -> int:
    if has_salt or system_form == "Electrolyte":
        return 2
    if system_form == "Ionogel":
        return 1
    return 0

def predict_lcst_g2_model(inputs: Dict[str, Any]) -> float:
    x = pd.DataFrame([{
        "Polymer_Main": inputs["polymer_main"],
        "Molecular_Weight_gmol": float(inputs["molecular_weight"]),
        "Polymer_Concentration_wt_pct": float(inputs["polymer_concentration"]),
        "IL_Cation": inputs["il_cation"],
        "IL_Anion": inputs["il_anion"],
        "Has_Salt": int(inputs["has_salt"]),
    }])
    pred = model.predict(x)[0]
    return round(float(pred), 1)

def run_inference(inputs: Dict[str, Any]) -> PredictionResult:
    group = infer_group(inputs["polymer_main"])

    if group == "G2":
        pred = predict_lcst_g2_model(inputs)
        tws = thermal_window_score(pred)
        aps = application_proximity_score(inputs["system_form"], inputs["has_salt"])
        score_10 = round(((tws + aps) / 4) * 10, 2)

        if 90 <= pred <= 120:
            thermal_label = "接近动力电池热保护窗口"
        elif pred < 90:
            thermal_label = "触发温区偏低，需警惕常温/中温误触发"
        else:
            thermal_label = "触发温区偏高，更适合进一步设计优化"

        return PredictionResult(
            predicted_lcst=pred,
            model_group="G2（PBA + PnBMA）真实随机森林模型已接入",
            confidence_note=f"当前调用的是基于 {meta['train_samples']} 条 G2 样本训练的随机森林模型。结果适合前端筛选，不替代实验验证。",
            engineering_score_10=score_10,
            thermal_window_label=thermal_label,
            engineering_comment="当前工程评分综合热窗口匹配与应用接近度，为原型版筛选指标。",
            next_step="建议结合分子量与聚合物浓度继续优化，并优先验证是否接近 90–120 ℃ 热保护窗口。",
        )

    if group == "G1":
        return PredictionResult(
            predicted_lcst=None,
            model_group="G1（Polyether）暂未接入正式预测模型",
            confidence_note="当前 G1 样本量偏小，更适合作为后续独立子模型建设对象。",
            engineering_score_10=None,
            thermal_window_label="仅提供分组提示，不建议直接输出数值预测",
            engineering_comment="建议先补充 PEO/polyether 体系样本，再开展单独建模。",
            next_step="优先收集不同分子量、不同端基、不同阳离子结构下的 polyether 样本。",
        )

    if group == "G3":
        return PredictionResult(
            predicted_lcst=None,
            model_group="G3（Benzyl methacrylate）建议单独建模",
            confidence_note="当前 G3 兼含纯相行为与应用型样本，不宜直接与 G2 混合预测。",
            engineering_score_10=None,
            thermal_window_label="仅提供分组提示，不建议直接输出数值预测",
            engineering_comment="建议先拆分 PBzMA/PBnMA 纯相行为样本与 PBMA 应用型样本，再分别评价。",
            next_step="优先建立 G3 纯相行为子模型，再考虑接入应用层评价。",
        )

    return PredictionResult(
        predicted_lcst=None,
        model_group="未知分组",
        confidence_note="当前平台最终版仅覆盖已形成研究主线的体系。",
        engineering_score_10=None,
        thermal_window_label="无法直接判断",
        engineering_comment="该聚合物尚未纳入当前研究框架。",
        next_step="建议先补文献样本并完成分组，再考虑接入模型。",
    )

# ========== Charts ==========
def make_temp_gauge(pred):
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=pred,
        number={'suffix': " ℃", 'font': {'size': 34, 'color': "#0f172a"}},
        title={'text': "热触发温区位置", 'font': {'size': 16, 'color': "#334155"}},
        gauge={
            'axis': {'range': [0, max(180, pred + 20)], 'tickcolor': "#94a3b8"},
            'bar': {'color': "#2563eb", 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#dbeafe",
            'steps': [
                {'range': [0, 90], 'color': "#e0f2fe"},
                {'range': [90, 120], 'color': "#bfdbfe"},
                {'range': [120, max(180, pred + 20)], 'color': "#eff6ff"},
            ],
            'threshold': {
                'line': {'color': "#1d4ed8", 'width': 4},
                'thickness': 0.75,
                'value': pred
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10))
    return fig

def make_engineering_radar(pred, system_form, has_salt):
    thermal_score = thermal_window_score(pred)
    app_score = application_proximity_score(system_form, has_salt)
    labels = ["热窗口匹配", "应用接近度", "模型适用性", "可解释性"]
    values = [thermal_score, app_score, 2, 2]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill='toself',
        line=dict(color="#2563eb", width=3),
        fillcolor="rgba(96, 165, 250, 0.28)",
        marker=dict(size=8, color="#1d4ed8")
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 2], tickvals=[0, 1, 2], tickfont=dict(color="#64748b")),
            bgcolor="rgba(255,255,255,0.0)"
        ),
        showlegend=False,
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig

def make_result_bar(pred):
    df = pd.DataFrame({
        "类别": ["预测温区", "目标窗口下限", "目标窗口上限"],
        "温度": [pred, 90, 120]
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["温度"],
        y=df["类别"],
        orientation="h",
        marker=dict(
            color=df["温度"],
            colorscale=[[0, "#dbeafe"], [0.5, "#93c5fd"], [1, "#2563eb"]],
            line=dict(color="#bfdbfe", width=1),
        ),
        text=[f"{v} ℃" for v in df["温度"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="热触发温区对照图",
        xaxis_title="温度 / ℃",
        yaxis_title="",
        height=240,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
    )
    return fig

# ========== UI ==========
st.markdown(
    """
<div class="app-hero">
    <div style="font-size:2rem;font-weight:800;margin-bottom:0.35rem;">热响应聚合物相变温区预测与工程筛选平台（最终版）</div>
    <div class="small-note">浅蓝科技感界面 + G2 真实随机森林模型 + 工程层图形化评价。当前优先支持 G2（PBA + PnBMA）体系。</div>
</div>
""",
    unsafe_allow_html=True,
)

with st.expander("平台说明", expanded=False):
    st.markdown("""
**当前平台采用两层结构：**

- **第一层：纯相行为主模型**
  - 用于解释结构—LCST 关系
  - 当前主模型聚焦 **G2（PBA + PnBMA）**
- **第二层：工程评价框架**
  - 用于温敏电解液、ionogel 等更接近应用环境样本的初步排序
  - 当前为原型工程评价，不替代实验验证
""")
    st.markdown(
        '<span class="info-pill">真实 G2 模型已接入</span><span class="info-pill">浅蓝科技感主题</span><span class="info-pill">高级图表输出</span>',
        unsafe_allow_html=True,
    )

left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">A. 参数输入</div>', unsafe_allow_html=True)
    polymer_main = st.selectbox("聚合物主体 Polymer_Main", ["PBA", "PnBMA", "PEO", "PBzMA", "PBMA", "PBnMA"], index=0)
    molecular_weight = st.number_input("分子量 Molecular_Weight (g/mol)", min_value=1000.0, max_value=500000.0, value=20000.0, step=1000.0)
    polymer_concentration = st.number_input("聚合物浓度 Polymer_Concentration (wt%)", min_value=0.1, max_value=95.0, value=10.0, step=0.5)
    il_cation = st.selectbox("离子液体阳离子 IL_Cation", ["[BMIM]", "[C3MIM]", "[EMIM]", "[DMIM]"], index=0)
    il_anion = st.selectbox("离子液体阴离子 IL_Anion", ["[NTf2]", "[TFSA]", "[BF4]", "[TFSI]"], index=0)
    has_salt = st.checkbox("是否含盐 Has_Salt", value=False)
    system_form = st.selectbox("体系形式 System_Form", ["Pure_Phase", "Ionogel", "Electrolyte"], index=0)
    run_btn = st.button("开始预测与筛选", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">B. 输出结果</div>', unsafe_allow_html=True)
    placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

if run_btn:
    inputs = {
        "polymer_main": polymer_main,
        "molecular_weight": molecular_weight,
        "polymer_concentration": polymer_concentration,
        "il_cation": il_cation,
        "il_anion": il_anion,
        "has_salt": has_salt,
        "system_form": system_form,
    }
    result = run_inference(inputs)

    with placeholder.container():
        st.markdown(f'**模型分组判断：** <span class="result-badge">{result.model_group}</span>', unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        with m1:
            if result.predicted_lcst is not None:
                st.metric("预测 LCST / 相变温区", f"{result.predicted_lcst} ℃")
            else:
                st.info("当前该分组暂不输出数值预测。")
        with m2:
            if result.engineering_score_10 is not None:
                st.metric("原型工程评分", f"{result.engineering_score_10} / 10")
            else:
                st.info("当前该分组暂不输出工程评分。")

        st.markdown(f"**模型适用性提示：** {result.confidence_note}")
        st.markdown(f"**热窗口判断：** {result.thermal_window_label}")
        st.markdown(f"**工程层说明：** {result.engineering_comment}")

        if result.predicted_lcst is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(make_temp_gauge(result.predicted_lcst), use_container_width=True)
            with c2:
                st.plotly_chart(make_engineering_radar(result.predicted_lcst, inputs["system_form"], inputs["has_salt"]), use_container_width=True)

            st.plotly_chart(make_result_bar(result.predicted_lcst), use_container_width=True)

        st.markdown("---")
        st.subheader("C. 下一步建议")
        st.write(result.next_step)

st.markdown('<div class="app-card">', unsafe_allow_html=True)
st.subheader("推荐使用方式")
st.markdown("""
1. 先用本平台做 **前端筛选**，快速判断体系是否值得进入下一轮实验。  
2. 对于 **G2 体系**，优先关注分子量与聚合物浓度的调节。  
3. 对于 **G1 / G3 体系**，先积累样本，再分别建立独立子模型。  
4. 对于更接近应用的体系，建议同时结合 **第二层工程评价** 进行判断。  
""")
st.markdown('</div>', unsafe_allow_html=True)
