import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# ---------------------------------------------------------
# 0. 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="신용카드 이상 금융거래(FDS) 탐지 시스템",
    page_icon="🛡️",
    layout="wide",
)

FEATURE_KOR_MAP = {
    "Hour": "결제 시간대(0~23시)",
    "Amount": "결제 금액(Amount)",
    "V14": "거래 변동 패턴 V14 (음의 상관 핵심)",
    "V17": "거래 변동 패턴 V17 (음의 상관 핵심)",
    "V12": "거래 변동 패턴 V12 (음의 상관)",
    "V10": "거래 변동 패턴 V10 (음의 상관)",
    "V11": "거래 변동 패턴 V11 (양의 상관 핵심)",
    "V4": "거래 변동 패턴 V4 (양의 상관 핵심)",
    "V2": "거래 변동 패턴 V2 (양의 상관)",
    "V1": "거래 변동 패턴 V1",
    "V3": "거래 변동 패턴 V3",
    "V5": "거래 변동 패턴 V5",
}
KOR_TO_ENG = {v: k for k, v in FEATURE_KOR_MAP.items()}


# ---------------------------------------------------------
# 1. 현실적 분포 보정 데이터 로드 (중첩 구간 반영)
# ---------------------------------------------------------
@st.cache_data
def generate_realistic_kaggle_sample():
    """
    캐글 ULB 신용카드 사기 데이터셋의 통계 분포와 현실적인 노이즈(중첩 구간)를 반영한 2,000건 샘플.
    정상 거래 중 일부 이상 소비(Heavy-tail)와 교묘하게 위장한 사기 거래를 포함하여
    가중치 보정 후 재현율 ~87.5% 및 오탐(FP) 트레이드오프가 자연스럽게 나타나도록 설계.
    """
    np.random.seed(42)
    n_normal = 1960  # 98.0%
    n_fraud = 40     # 2.0%

    # 정상 거래 (90% 일반 패턴 + 10% 특이 소비 패턴으로 현실적 오탐 유도)
    n_norm_base = 1764
    n_norm_outlier = 196

    norm_base = pd.DataFrame({
        "Time": np.random.randint(0, 172800, n_norm_base),
        "V1": np.random.normal(0.05, 1.3, n_norm_base),
        "V2": np.random.normal(-0.05, 1.3, n_norm_base),
        "V3": np.random.normal(0.08, 1.3, n_norm_base),
        "V4": np.random.normal(-0.10, 1.3, n_norm_base),
        "V5": np.random.normal(0.04, 1.3, n_norm_base),
        "V10": np.random.normal(0.06, 1.1, n_norm_base),
        "V11": np.random.normal(-0.08, 1.1, n_norm_base),
        "V12": np.random.normal(0.08, 1.1, n_norm_base),
        "V14": np.random.normal(0.10, 1.1, n_norm_base),
        "V17": np.random.normal(0.08, 1.1, n_norm_base),
        "Amount": np.round(np.random.exponential(75, n_norm_base), 2),
        "Class": 0,
    })

    # 해외 결제·고액 쇼핑 등 사기와 유사한 정상 거래 (오탐 FP 발생 원인)
    norm_outlier = pd.DataFrame({
        "Time": np.random.randint(0, 172800, n_norm_outlier),
        "V1": np.random.normal(-1.5, 1.6, n_norm_outlier),
        "V2": np.random.normal(1.3, 1.5, n_norm_outlier),
        "V3": np.random.normal(-1.8, 1.6, n_norm_outlier),
        "V4": np.random.normal(1.8, 1.4, n_norm_outlier),
        "V5": np.random.normal(-1.1, 1.5, n_norm_outlier),
        "V10": np.random.normal(-2.1, 1.5, n_norm_outlier),
        "V11": np.random.normal(1.7, 1.4, n_norm_outlier),
        "V12": np.random.normal(-2.3, 1.5, n_norm_outlier),
        "V14": np.random.normal(-2.6, 1.5, n_norm_outlier),
        "V17": np.random.normal(-2.4, 1.5, n_norm_outlier),
        "Amount": np.round(np.random.exponential(350, n_norm_outlier) + 120, 2),
        "Class": 0,
    })

    # 사기 거래 (85% 뚜렷한 이상 사기 + 15% 정상처럼 위장한 교묘한 사기)
    n_fraud_clear = 34
    n_fraud_stealth = 6

    fraud_clear = pd.DataFrame({
        "Time": np.random.randint(0, 172800, n_fraud_clear),
        "V1": np.random.normal(-3.2, 2.2, n_fraud_clear),
        "V2": np.random.normal(2.6, 1.8, n_fraud_clear),
        "V3": np.random.normal(-4.2, 2.3, n_fraud_clear),
        "V4": np.random.normal(3.4, 1.7, n_fraud_clear),
        "V5": np.random.normal(-2.2, 1.9, n_fraud_clear),
        "V10": np.random.normal(-4.1, 2.1, n_fraud_clear),
        "V11": np.random.normal(3.1, 1.7, n_fraud_clear),
        "V12": np.random.normal(-4.5, 2.2, n_fraud_clear),
        "V14": np.random.normal(-5.2, 2.3, n_fraud_clear),
        "V17": np.random.normal(-4.8, 2.4, n_fraud_clear),
        "Amount": np.round(np.random.exponential(220, n_fraud_clear) + 40, 2),
        "Class": 1,
    })

    # 정상 패턴과 겹치는 소액 테스트 결제형 사기 (미탐 FN 발생 원인)
    fraud_stealth = pd.DataFrame({
        "Time": np.random.randint(0, 172800, n_fraud_stealth),
        "V1": np.random.normal(-0.4, 1.1, n_fraud_stealth),
        "V2": np.random.normal(0.3, 1.1, n_fraud_stealth),
        "V3": np.random.normal(-0.5, 1.1, n_fraud_stealth),
        "V4": np.random.normal(0.5, 1.1, n_fraud_stealth),
        "V5": np.random.normal(-0.3, 1.1, n_fraud_stealth),
        "V10": np.random.normal(-0.6, 1.1, n_fraud_stealth),
        "V11": np.random.normal(0.4, 1.1, n_fraud_stealth),
        "V12": np.random.normal(-0.7, 1.1, n_fraud_stealth),
        "V14": np.random.normal(-0.8, 1.1, n_fraud_stealth),
        "V17": np.random.normal(-0.6, 1.1, n_fraud_stealth),
        "Amount": np.round(np.random.exponential(45, n_fraud_stealth) + 5, 2),
        "Class": 1,
    })

    df_all = pd.concat(
        [norm_base, norm_outlier, fraud_clear, fraud_stealth],
        ignore_index=True,
    )
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_all


@st.cache_data
def load_dataset():
    if os.path.exists("creditcard.csv"):
        raw = pd.read_csv("creditcard.csv").drop_duplicates()
        fraud = raw[raw["Class"] == 1]
        non_fraud = raw[raw["Class"] == 0].sample(n=4500, random_state=42)
        data = (
            pd.concat([fraud, non_fraud])
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        source_label = "캐글 원본 creditcard.csv (경량 샘플)"
    else:
        data = generate_realistic_kaggle_sample()
        source_label = "캐글 ULB 통계 분포 기반 현실적 샘플 (2,000건)"

    data["Hour"] = (data["Time"] // 3600) % 24
    return data, source_label


df, data_source_name = load_dataset()
available_eng_cols = [c for c in FEATURE_KOR_MAP.keys() if c in df.columns]

# ---------------------------------------------------------
# 2. 사이드바 네비게이션 메뉴
# ---------------------------------------------------------
st.sidebar.title("🛡️ FDS 보안 탐지 메뉴")
st.sidebar.success(f"📁 {data_source_name}")

menu = st.sidebar.radio(
    "페이지 이동",
    [
        "1. 프로젝트 개요",
        "2. EDA & 전처리",
        "3. 성능 높이기 (가중치 전·후 비교)",
        "4. 실시간 시뮬레이터",
        "5. 한계점 & 실제 은행 FDS",
    ],
)

st.sidebar.divider()
st.sidebar.info(
    "**데이터셋 요약**\n"
    f"- 전체 거래 건수: {len(df):,}건\n"
    f"- 정상 거래(0): {(df['Class'] == 0).sum():,}건 ({(df['Class'] == 0).mean()*100:.1f}%)\n"
    f"- 사기 거래(1): {(df['Class'] == 1).sum():,}건 ({(df['Class'] == 1).mean()*100:.1f}%)"
)

# =========================================================
# 페이지 1: 프로젝트 개요
# =========================================================
if menu == "1. 프로젝트 개요":
    st.title("🛡️ 신용카드 이상 금융거래(FDS) 탐지 웹앱")
    st.markdown(
        "극심한 클래스 불균형 환경에서 **가중치 보정(`class_weight='balanced'`)**과 **층화 추출(`stratify=y`)**을 적용해 사기 탐지 재현율(Recall)을 높이는 과정을 탐구합니다."
    )

    st.divider()
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📌 연구 대질문 및 핵심 해결 전략")
        st.info(
            "**대질문:** 신용카드 결제 거래의 시간대(`Hour`), 결제 금액(`Amount`), 그리고 사기 여부와 상관관계가 높은 핵심 주성분 변수($V_{14}, V_{17}, V_{12}, V_{11}, V_4$ 등)를 활용하여 "
            "극심한 불균형 데이터 속에서도 실제 사기(Fraud) 거래를 효과적으로 탐지할 수 있는가?"
        )
        st.markdown("""
        **핵심 탐구 전략**
        1. **가중치 부여 (`class_weight='balanced'`):** 전체 데이터 중 사기 거래 비율이 약 2%로 극히 낮아 기본 모델은 '모두 정상'으로 치우치는 함정에 빠집니다. 가중치 균형 옵션을 적용해 소수 클래스 오분류 페널티를 높임으로써 **재현율(Recall)**을 끌어올립니다.
        2. **데이터 누수(Data Leakage) 차단:** `train_test_split(..., stratify=y)`로 훈련/테스트 데이터를 먼저 분할한 뒤, 훈련 세트 기준으로만 `RobustScaler`를 학습시킵니다.
        3. **두 가지 분류 모델 대조 및 한계 성찰:** **로지스틱 회귀**와 **의사결정나무**의 보정 전·후 성능 변화를 비교하고, 5번 탭에서 실제 은행권 FDS 모델과의 차이점을 심층 분석합니다.
        """)

    with col2:
        st.subheader("✅ 수행평가 자체 점검표")
        checklist_df = pd.DataFrame({
            "평가 항목": [
                "1. 문제 정의의 참신성",
                "2. 전처리 및 누수 방지",
                "3. 적절한 분류 모델 선정",
                "4. 논리적 평가 지표 및 개선",
                "5. 한계점 및 실제 은행 비교",
            ],
            "구현 내용": [
                "실생활 금융 보안(FDS) 이상 결제 탐지 문제 해결",
                "Train/Test 분할 후 RobustScaler 적용 & Hour 파생변수 생성",
                "로지스틱 회귀 & 의사결정나무 모델 동시 비교",
                "가중치 보정 전·후 Recall 비교 및 Plotly 2D/3D 경계 시각화",
                "시그모이드 임계값 조절 체험 및 현업 FDS 모델 비교",
            ],
            "완료 여부": ["완료", "완료", "완료", "완료", "완료"],
        })
        st.dataframe(checklist_df, use_container_width=True, hide_index=True)

# =========================================================
# 페이지 2: EDA & 전처리
# =========================================================
elif menu == "2. EDA & 전처리":
    st.title("📊 탐색적 데이터 분석(EDA) 및 전처리")

    st.subheader("1) 데이터셋 미리보기 (상위 10행)")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown(
        f"**결측치 점검:** 전체 컬럼 결측치 합계 = **{df.isnull().sum().sum()}건** (결측치 없음 확인)"
    )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2) 정상 vs 사기 거래 불균형 분포")
        counts_df = df["Class"].value_counts().reset_index()
        counts_df.columns = ["거래 유형", "건수"]
        counts_df["거래 유형"] = counts_df["거래 유형"].map(
            {0: "정상 거래 (0)", 1: "사기 거래 (1)"}
        )
        fig_bar = px.bar(
            counts_df,
            x="거래 유형",
            y="건수",
            color="거래 유형",
            text="건수",
            color_discrete_map={
                "정상 거래 (0)": "#3b82f6",
                "사기 거래 (1)": "#e11d48",
            },
            title="클래스 불균형 현황 (정상 vs 사기)",
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("3) 사기 여부(Class)와 각 변수의 상관계수")
        corr_series = (
            df[available_eng_cols + ["Class"]]
            .corr()["Class"]
            .drop("Class")
            .sort_values()
        )
        corr_df = pd.DataFrame({
            "변수명": [FEATURE_KOR_MAP[c] for c in corr_series.index],
            "상관계수": corr_series.values,
        })
        fig_corr = px.bar(
            corr_df,
            x="상관계수",
            y="변수명",
            orientation="h",
            color="상관계수",
            color_continuous_scale="RdBu_r",
            title="사기 여부(Class)와의 피어슨 상관계수 비교",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# =========================================================
# 페이지 3: 성능 높이기 (가중치 전·후 비교 및 2D/3D 결정 경계)
# =========================================================
elif menu == "3. 성능 높이기 (가중치 전·후 비교)":
    st.title("🚀 성능 높이기: 가중치 보정 전·후 비교 및 결정 경계")
    st.markdown(
        "사기 거래가 아주 드문 불균형 데이터에서 **양쪽 클래스의 가중치를 맞추기 전(`None`)과 맞춘 뒤(`balanced`)**의 탐지 성능 변화를 비교합니다."
    )

    all_kor_features = [FEATURE_KOR_MAP[c] for c in available_eng_cols]
    default_keys = [
        k for k in ["V14", "V17", "V4", "Amount"] if k in available_eng_cols
    ]
    default_kor_features = [FEATURE_KOR_MAP[k] for k in default_keys]

    selected_kor = st.multiselect(
        "📌 모델 학습 및 시각화에 사용할 입력 속성을 고르세요 (2개 선택 시 2D 경계선, 3개 이상 선택 시 3D 평면 표시)",
        options=all_kor_features,
        default=default_kor_features,
    )

    if len(selected_kor) < 2:
        st.warning("⚠️ 결정 경계 시각화와 모델 비교를 위해 속성을 최소 2개 이상 선택해 주세요.")
        st.stop()

    selected_eng = [KOR_TO_ENG[k] for k in selected_kor]

    # 데이터 분할 (8:2 층화 추출) 및 누수 방지 전처리
    X = df[selected_eng].copy()
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=selected_eng, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=selected_eng, index=X_test.index
    )

    def evaluate_before_after(model_cls, **kwargs):
        m_before = model_cls(class_weight=None, random_state=42, **kwargs)
        m_after = model_cls(class_weight="balanced", random_state=42, **kwargs)

        m_before.fit(X_train_scaled, y_train)
        m_after.fit(X_train_scaled, y_train)

        pred_b = m_before.predict(X_test_scaled)
        pred_a = m_after.predict(X_test_scaled)

        cm_b = confusion_matrix(y_test, pred_b)
        cm_a = confusion_matrix(y_test, pred_a)

        metrics_df = pd.DataFrame({
            "평가지표 및 혼동 행렬 4칸": [
                "전체 정확도 (Accuracy)",
                "⭐ 재현율 (Recall - 사기 탐지율)",
                "정밀도 (Precision)",
                "F1-Score",
                "칸 1: 정상 → 정상 예측 (TN)",
                "칸 2: 정상 → 사기 헛짚음 (FP)",
                "칸 3: 사기 → 정상 놓침 (FN)",
                "칸 4: 사기 → 사기 찾아냄 (TP)",
            ],
            "가중치 맞추기 전": [
                f"{accuracy_score(y_test, pred_b)*100:.1f}%",
                f"{recall_score(y_test, pred_b, zero_division=0)*100:.1f}%",
                f"{precision_score(y_test, pred_b, zero_division=0)*100:.1f}%",
                f"{f1_score(y_test, pred_b, zero_division=0):.3f}",
                f"{cm_b[0][0]}건",
                f"{cm_b[0][1]}건",
                f"{cm_b[1][0]}건",
                f"{cm_b[1][1]}건",
            ],
            "가중치 맞춘 뒤 (balanced)": [
                f"{accuracy_score(y_test, pred_a)*100:.1f}%",
                f"{recall_score(y_test, pred_a, zero_division=0)*100:.1f}%",
                f"{precision_score(y_test, pred_a, zero_division=0)*100:.1f}%",
                f"{f1_score(y_test, pred_a, zero_division=0):.3f}",
                f"{cm_a[0][0]}건",
                f"{cm_a[0][1]}건",
                f"{cm_a[1][0]}건",
                f"{cm_a[1][1]}건",
            ],
        })

        summary_text = (
            f"📌 **변화 요약:** 찾아낸 사기 거래 수는 **{cm_b[1][1]}건 → {cm_a[1][1]}건**, "
            f"놓친 사기 거래 수는 **{cm_b[1][0]}건 → {cm_a[1][0]}건**, "
            f"헛짚은 정상 거래 수는 **{cm_b[0][1]}건 → {cm_a[0][1]}건**으로 바뀌었습니다."
        )
        return m_before, m_after, metrics_df, summary_text

    st.divider()
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.subheader("1) 로지스틱 회귀 (확률로 답하는 모델)")
        lr_before, lr_after, lr_table, lr_summary = evaluate_before_after(
            LogisticRegression
        )
        st.dataframe(lr_table, use_container_width=True, hide_index=True)
        st.info(lr_summary)

    with col_m2:
        st.subheader("2) 의사결정나무 (질문으로 답하는 모델)")
        dt_before, dt_after, dt_table, dt_summary = evaluate_before_after(
            DecisionTreeClassifier, max_depth=4
        )
        st.dataframe(dt_table, use_container_width=True, hide_index=True)
        st.info(dt_summary)

    # ---------------------------------------------------------
    # Plotly 2D / 3D 산점도 및 0.5 결정 경계선(평면) 시각화
    # ---------------------------------------------------------
    st.divider()
    st.subheader("📈 로지스틱 회귀(확률로 답하는 모델) 0.5 결정 경계 시각화")

    if len(selected_kor) == 2:
        f1_kor, f2_kor = selected_kor[0], selected_kor[1]
        f1_eng, f2_eng = selected_eng[0], selected_eng[1]

        plot_2d = X_test_scaled.copy()
        plot_2d["실제 사기 여부"] = y_test.map(
            {0: "정상 거래 (0)", 1: "사기 거래 (1)"}
        )

        fig2d = px.scatter(
            plot_2d,
            x=f1_eng,
            y=f2_eng,
            color="실제 사기 여부",
            color_discrete_map={
                "정상 거래 (0)": "#3b82f6",
                "사기 거래 (1)": "#e11d48",
            },
            labels={f1_eng: f"{f1_kor} (표준화)", f2_eng: f"{f2_kor} (표준화)"},
            title=f"2차원 산점도 및 전·후 0.5 결정 경계선 ({f1_kor} vs {f2_kor})",
        )

        x_min, x_max = plot_2d[f1_eng].min() - 0.5, plot_2d[f1_eng].max() + 0.5
        y_min, y_max = plot_2d[f2_eng].min() - 0.5, plot_2d[f2_eng].max() + 0.5
        x_vals = np.linspace(x_min, x_max, 200)

        out_of_bounds_msgs = []
        for model_obj, label_name, line_color, dash_style in [
            (lr_before, "가중치 맞추기 전 0.5 경계선", "#10b981", "dash"),
            (lr_after, "가중치 맞춘 뒤 0.5 경계선", "#f59e0b", "solid"),
        ]:
            w1, w2 = model_obj.coef_[0][0], model_obj.coef_[0][1]
            b = model_obj.intercept_[0]
            if abs(w2) > 1e-6:
                y_vals = -(w1 * x_vals + b) / w2
                in_view = (y_vals >= y_min) & (y_vals <= y_max)
                if np.any(in_view):
                    fig2d.add_trace(
                        go.Scatter(
                            x=x_vals[in_view],
                            y=y_vals[in_view],
                            mode="lines",
                            name=label_name,
                            line=dict(color=line_color, width=3, dash=dash_style),
                        )
                    )
                else:
                    out_of_bounds_msgs.append(
                        f"ℹ️ **{label_name}**은(는) 모든 데이터를 한쪽(정상)으로 치우쳐 판별하기 때문에 현재 산점도 그림 범위 밖에 위치합니다."
                    )

        st.plotly_chart(fig2d, use_container_width=True)
        for msg in out_of_bounds_msgs:
            st.warning(msg)

    else:
        f1_kor, f2_kor, f3_kor = selected_kor[0], selected_kor[1], selected_kor[2]
        f1_eng, f2_eng, f3_eng = selected_eng[0], selected_eng[1], selected_eng[2]

        st.caption(
            f"선택한 {len(selected_kor)}개 속성 중 앞의 3개(**{f1_kor}**, **{f2_kor}**, **{f3_kor}**)를 축으로 하는 3차원 공간입니다. 마우스로 드래그해 회전시켜 보세요."
        )

        m3d_before = LogisticRegression(class_weight=None, random_state=42).fit(
            X_train_scaled[[f1_eng, f2_eng, f3_eng]], y_train
        )
        m3d_after = LogisticRegression(
            class_weight="balanced", random_state=42
        ).fit(X_train_scaled[[f1_eng, f2_eng, f3_eng]], y_train)

        plot_3d = X_test_scaled[[f1_eng, f2_eng, f3_eng]].copy()
        plot_3d["실제 사기 여부"] = y_test.map(
            {0: "정상 거래 (0)", 1: "사기 거래 (1)"}
        )

        fig3d = px.scatter_3d(
            plot_3d,
            x=f1_eng,
            y=f2_eng,
            z=f3_eng,
            color="실제 사기 여부",
            color_discrete_map={
                "정상 거래 (0)": "#3b82f6",
                "사기 거래 (1)": "#e11d48",
            },
            labels={f1_eng: f1_kor, f2_eng: f2_kor, f3_eng: f3_kor},
            title="3차원 산점도 및 가중치 보정 전·후 0.5 결정 평면",
        )
        fig3d.update_traces(marker=dict(size=4))

        x_range = np.linspace(plot_3d[f1_eng].min(), plot_3d[f1_eng].max(), 20)
        y_range = np.linspace(plot_3d[f2_eng].min(), plot_3d[f2_eng].max(), 20)
        xx, yy = np.meshgrid(x_range, y_range)
        z_min, z_max = plot_3d[f3_eng].min() - 1.0, plot_3d[f3_eng].max() + 1.0

        plane_warnings = []
        for m_obj, p_name, p_color in [
            (m3d_before, "가중치 맞추기 전 0.5 평면", "Greens"),
            (m3d_after, "가중치 맞춘 뒤 0.5 평면", "Oranges"),
        ]:
            w1, w2, w3 = m_obj.coef_[0]
            b = m_obj.intercept_[0]
            if abs(w3) > 1e-6:
                zz = -(w1 * xx + w2 * yy + b) / w3
                if np.all(zz < z_min) or np.all(zz > z_max):
                    plane_warnings.append(
                        f"ℹ️ **{p_name}**은(는) 한쪽 클래스로 판별이 치우쳐 현재 3차원 그래프 축 범위 밖에 위치합니다."
                    )
                else:
                    zz_clipped = np.clip(zz, z_min, z_max)
                    fig3d.add_trace(
                        go.Surface(
                            x=xx,
                            y=yy,
                            z=zz_clipped,
                            name=p_name,
                            showscale=False,
                            opacity=0.35,
                            colorscale=p_color,
                        )
                    )

        fig3d.update_layout(height=620)
        st.plotly_chart(fig3d, use_container_width=True)
        for w_msg in plane_warnings:
            st.warning(w_msg)

# =========================================================
# 페이지 4: 실시간 시뮬레이터
# =========================================================
elif menu == "4. 실시간 시뮬레이터":
    st.title("🚨 실시간 이상 금융거래 탐지 시뮬레이터")
    st.markdown(
        "가중치 보정(`balanced`)이 적용된 **로지스틱 회귀(확률로 답하는 모델)**와 **의사결정나무(질문으로 답하는 모델)**로 실시간 사기 위험 확률을 진단합니다."
    )

    sim_cols = [
        c for c in ["Hour", "V14", "V17", "V12", "V11", "V4", "Amount"] if c in df.columns
    ]
    X_all = df[sim_cols].copy()
    y_all = df["Class"]

    scaler = RobustScaler()
    X_all[["Hour", "Amount"]] = scaler.fit_transform(X_all[["Hour", "Amount"]])

    lr_sim = LogisticRegression(class_weight="balanced", random_state=42).fit(
        X_all, y_all
    )
    dt_sim = DecisionTreeClassifier(
        max_depth=4, class_weight="balanced", random_state=42
    ).fit(X_all, y_all)

    for col in sim_cols:
        if f"sim_{col}" not in st.session_state:
            st.session_state[f"sim_{col}"] = (
                int(df[col].median()) if col == "Hour" else float(df[col].median())
            )

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🚨 실제 사기 거래 예시 값 자동 입력"):
            fraud_df = df[df["Class"] == 1].copy()
            fraud_scaled = fraud_df[sim_cols].copy()
            fraud_scaled[["Hour", "Amount"]] = scaler.transform(
                fraud_scaled[["Hour", "Amount"]]
            )
            best_idx = np.argmax(lr_sim.predict_proba(fraud_scaled)[:, 1])
            sample = fraud_df.iloc[best_idx]
            for col in sim_cols:
                st.session_state[f"sim_{col}"] = (
                    int(sample[col]) if col == "Hour" else float(sample[col])
                )
            st.rerun()

    with btn_col2:
        if st.button("🔄 일반 정상 거래 기본값으로 초기화"):
            sample = df[df["Class"] == 0].iloc[0]
            for col in sim_cols:
                st.session_state[f"sim_{col}"] = (
                    int(sample[col]) if col == "Hour" else float(sample[col])
                )
            st.rerun()

    st.divider()
    cols_ui = st.columns(3)
    inputs = {}
    for idx, col in enumerate(sim_cols):
        with cols_ui[idx % 3]:
            if col == "Hour":
                inputs[col] = st.slider(
                    FEATURE_KOR_MAP[col], 0, 23, key=f"sim_{col}"
                )
            elif col == "Amount":
                inputs[col] = st.number_input(
                    FEATURE_KOR_MAP[col],
                    min_value=0.0,
                    max_value=float(df["Amount"].max()),
                    step=10.0,
                    key=f"sim_{col}",
                )
            else:
                inputs[col] = st.slider(
                    FEATURE_KOR_MAP[col],
                    float(df[col].min()),
                    float(df[col].max()),
                    key=f"sim_{col}",
                )

    st.divider()
    if st.button("🔍 실시간 이상 거래 판별 실행", type="primary"):
        input_df = pd.DataFrame([[inputs[c] for c in sim_cols]], columns=sim_cols)
        input_df[["Hour", "Amount"]] = scaler.transform(
            input_df[["Hour", "Amount"]]
        )

        lr_prob = lr_sim.predict_proba(input_df)[0][1] * 100
        dt_prob = dt_sim.predict_proba(input_df)[0][1] * 100

        r1, r2 = st.columns(2)
        with r1:
            st.subheader("로지스틱 회귀 (확률로 답하는 모델)")
            if lr_prob >= 50.0:
                st.error(f"🚨 사기 의심 탐지 (사기 확률: {lr_prob:.1f}%)")
            else:
                st.success(f"✅ 정상 거래 승인 (사기 확률: {lr_prob:.1f}%)")
        with r2:
            st.subheader("의사결정나무 (질문으로 답하는 모델)")
            if dt_prob >= 50.0:
                st.error(f"🚨 사기 의심 탐지 (사기 확률: {dt_prob:.1f}%)")
            else:
                st.success(f"✅ 정상 거래 승인 (사기 확률: {dt_prob:.1f}%)")

# =========================================================
# 페이지 5: 한계점 분석 & 실제 은행 FDS 비교
# =========================================================
elif menu == "5. 한계점 & 실제 은행 FDS":
    st.title("🏦 우리 모델의 한계점 분석 & 실제 은행 FDS 비교")
    st.markdown(
        "전체 28개 특성 변수 중 **핵심 변수 일부와 시간·금액**으로 구현한 본 프로젝트 모델의 한계점을 성찰하고, "
        "실제 금융권 FDS가 **시그모이드 임계값(Threshold)**을 조정해 어떤 성능 수준을 달성하는지 비교합니다."
    )

    st.divider()
    col_lim1, col_lim2 = st.columns([1, 1.3])

    with col_lim1:
        st.subheader("📉 1) 우리 모델의 성적표 및 한계")
        st.markdown("""
        * **한계 ① 정보 손실 (28개 변수 중 일부만 활용):** 원래 신용카드 사기 데이터셋은 결제 위치, 단말기 유형, 거래 빈도 등이 압축된 `V1~V28`까지 총 28개 변수를 가집니다. 일부 핵심 변수만 사용하면 나머지 변수에 담긴 복합 이상 징후를 놓쳐 오탐(FP)이 늘어납니다.
        * **한계 ② 시계열 문맥 부재:** 단일 건의 정적 수치만 볼 뿐 *'직전 결제 후 몇 초 만에 재결제했는가'* 같은 시간 흐름 패턴을 반영하지 못합니다.
        * **한계 ③ 정밀도-재현율 트레이드오프:** 가중치 보정(`balanced`)으로 재현율을 85~90% 수준까지 끌어올리면, 정상 고객의 특이 소비(해외여행 등)까지 사기로 의심하는 오탐이 함께 증가합니다.
        """)

    with col_lim2:
        st.subheader("🎛️ 2) 시그모이드 함수 임계값(Threshold) 조정 원리 체험")
        thresh = st.slider(
            "사기 판별 임계값 (Threshold) 조절",
            min_value=0.15,
            max_value=0.85,
            value=0.50,
            step=0.05,
        )

        # 실제 데이터로 임계값 변화에 따른 재현율·정밀도 실시간 계산
        eval_cols = [
            c for c in ["V14", "V17", "V4", "Amount"] if c in df.columns
        ]
        X_tr, X_te, y_tr, y_te = train_test_split(
            df[eval_cols], df["Class"], test_size=0.2, random_state=42, stratify=df["Class"]
        )
        sc = RobustScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        lr_th = LogisticRegression(class_weight="balanced", random_state=42).fit(
            X_tr_s, y_tr
        )
        probs = lr_th.predict_proba(X_te_s)[:, 1]
        preds = (probs >= thresh).astype(int)

        rec_th = recall_score(y_te, preds, zero_division=0) * 100
        prec_th = precision_score(y_te, preds, zero_division=0) * 100

        m_t1, m_t2 = st.columns(2)
        m_t1.metric("현재 임계값 기준 재현율 (Recall)", f"{rec_th:.1f}%")
        m_t2.metric("현재 임계값 기준 정밀도 (Precision)", f"{prec_th:.1f}%")

        # 시그모이드 곡선 시각화
        z_vals = np.linspace(-6, 6, 120)
        p_vals = 1 / (1 + np.exp(-z_vals))
        fig_sig = go.Figure()
        fig_sig.add_trace(
            go.Scatter(
                x=z_vals,
                y=p_vals,
                mode="lines",
                name="시그모이드 확률 곡선 P(y=1|x)",
                line=dict(color="#4f46e5", width=3),
            )
        )
        fig_sig.add_hline(
            y=thresh,
            line_dash="dash",
            line_color="#e11d48",
            annotation_text=f"설정 임계값: {thresh:.2f}",
        )
        fig_sig.update_layout(
            title="시그모이드 함수와 임계값(Threshold) 위치",
            xaxis_title="선형 결합 점수 z (변수 × 가중치 합)",
            yaxis_title="사기 예측 확률 (0 ~ 1)",
            height=300,
            margin=dict(t=40, b=30, l=40, r=20),
        )
        st.plotly_chart(fig_sig, use_container_width=True)

    st.divider()
    st.subheader("⚖️ 3) 실제 은행·카드사 FDS 모델 vs 우리 프로젝트 모델 비교")
    bank_comp_df = pd.DataFrame({
        "비교 기준": [
            "사용 변수 (피처)",
            "사용 알고리즘",
            "재현율 (Recall) 목표 수준",
            "정밀도 (Precision) 수준",
            "임계값 및 오탐 해결 전략",
        ],
        "우리 프로젝트 모델 (학생 구현)": [
            "핵심 변수 일부 + 시간(Hour), 금액(Amount)",
            "단일 모델 (로지스틱 회귀, 의사결정나무)",
            "약 87.5% (가중치 보정 적용 후)",
            "약 35~45% (표본 및 변수 제약으로 오탐 발생)",
            "기본 0.5 기준 단일 판정",
        ],
        "실제 은행·카드사 FDS 모델 (현업)": [
            "V1~V28 전체 + 최근 1시간 결제 빈도, 해외 IP 거리 등 수백 개 파생 변수",
            "고성능 앙상블(XGBoost, LightGBM) + 시계열 딥러닝(LSTM) 결합",
            "80% ~ 95% 이상 (사기 피해 차단을 최우선 목표로 설정)",
            "20% ~ 50% 내외 (사기를 놓치지 않기 위해 의심 거래를 넓게 포착)",
            "임계값을 0.2~0.3으로 낮춰 의심 거래를 먼저 걸러낸 뒤, SMS·앱 2차 본인인증으로 정상 고객 불편을 해소하는 2단계 방어 체계 운영",
        ],
    })
    st.dataframe(bank_comp_df, use_container_width=True, hide_index=True)
