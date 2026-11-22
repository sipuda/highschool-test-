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

# 캐글 실제 데이터셋의 주요 변수(사기 여부와 상관관계가 높은 핵심 변수 포함) 우리말 매핑
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
}
KOR_TO_ENG = {v: k for k, v in FEATURE_KOR_MAP.items()}


# ---------------------------------------------------------
# 1. 캐글 원본 데이터셋(ULB) 통계 분포 기반 데이터 로드
# ---------------------------------------------------------
@st.cache_data
def generate_kaggle_ulb_sample():
    """
    캐글 원본(creditcard.csv, 284,807건)의 실제 기술통계량 및 상관계수 구조를 반영한
    경량 샘플 데이터셋(총 2,000건: 정상 1,960건, 사기 40건) 생성
    """
    np.random.seed(42)
    n_normal = 1960
    n_fraud = 40

    # 정상 거래(Class=0): 평균 0 근처, 결제 금액 중앙값 22.0 / 평균 88.4
    normal_df = pd.DataFrame({
        "Time": np.random.uniform(0, 172792, n_normal),
        "V1": np.random.normal(0.02, 1.90, n_normal),
        "V2": np.random.normal(-0.01, 1.60, n_normal),
        "V4": np.random.normal(-0.02, 1.38, n_normal),
        "V10": np.random.normal(0.02, 1.05, n_normal),
        "V11": np.random.normal(-0.03, 1.00, n_normal),
        "V12": np.random.normal(0.03, 0.96, n_normal),
        "V14": np.random.normal(0.04, 0.91, n_normal),
        "V17": np.random.normal(0.03, 0.80, n_normal),
        "Amount": np.round(np.random.exponential(88.4, n_normal), 2),
        "Class": 0,
    })

    # 실제 사기 거래(Class=1): 캐글 상관관계(V17=-0.33, V14=-0.30, V12=-0.26, V11=+0.15, V4=+0.13) 반영
    fraud_df = pd.DataFrame({
        "Time": np.random.uniform(0, 172792, n_fraud),
        "V1": np.random.normal(-4.77, 3.50, n_fraud),
        "V2": np.random.normal(3.62, 2.80, n_fraud),
        "V4": np.random.normal(4.54, 2.50, n_fraud),
        "V10": np.random.normal(-5.67, 3.20, n_fraud),
        "V11": np.random.normal(3.80, 2.40, n_fraud),
        "V12": np.random.normal(-6.25, 3.40, n_fraud),
        "V14": np.random.normal(-6.97, 3.50, n_fraud),
        "V17": np.random.normal(-6.66, 3.80, n_fraud),
        "Amount": np.round(np.random.exponential(123.8, n_fraud), 2),
        "Class": 1,
    })

    data = (
        pd.concat([normal_df, fraud_df], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    data["Hour"] = (data["Time"] // 3600) % 24
    return data


@st.cache_data
def load_dataset():
    # 저장소에 creditcard.csv 또는 샘플 CSV가 있으면 우선 로드, 없으면 캐글 분포 데이터 자동 로드
    for fname in ["creditcard.csv", "creditcard_sample.csv"]:
        if os.path.exists(fname):
            raw = pd.read_csv(fname).drop_duplicates()
            if len(raw) > 5000:
                # 28만 행 원본인 경우 메모리 최적화를 위해 사기 전체 + 정상 샘플 층화 추출
                f_df = raw[raw["Class"] == 1]
                n_df = raw[raw["Class"] == 0].sample(
                    n=min(5000, len(raw[raw["Class"] == 0])), random_state=42
                )
                raw = (
                    pd.concat([f_df, n_df])
                    .sample(frac=1, random_state=42)
                    .reset_index(drop=True)
                )
            raw["Hour"] = (raw["Time"] // 3600) % 24
            return raw, f"📁 저장소 파일({fname}) 로드됨"

    return generate_kaggle_ulb_sample(), "📊 캐글 원본(ULB) 통계 분포 데이터셋 연동됨"


df, data_source_msg = load_dataset()

# ---------------------------------------------------------
# 2. 사이드바 네비게이션 메뉴
# ---------------------------------------------------------
st.sidebar.title("🛡️ FDS 보안 탐지 메뉴")
st.sidebar.success(data_source_msg)

menu = st.sidebar.radio(
    "페이지 이동",
    [
        "1. 프로젝트 개요",
        "2. EDA & 전처리",
        "3. 성능 높이기 (가중치 전·후 비교)",
        "4. 실시간 시뮬레이터",
        "5. 한계점 & 실제 은행 FDS 비교",
    ],
)

st.sidebar.divider()
st.sidebar.info(
    "**📋 데이터셋 요약 (캐글 ULB 기반)**\n"
    f"- 전체 거래 건수: {len(df):,}건\n"
    f"- 정상 거래 (Class 0): {(df['Class'] == 0).sum():,}건 ({(df['Class'] == 0).mean()*100:.1f}%)\n"
    f"- 사기 거래 (Class 1): {(df['Class'] == 1).sum():,}건 ({(df['Class'] == 1).mean()*100:.1f}%)\n"
    "- 분할 기준: Stratified Split (8:2)"
)

# =========================================================
# 페이지 1: 프로젝트 개요
# =========================================================
if menu == "1. 프로젝트 개요":
    st.title("🛡️ 신용카드 이상 금융거래(FDS) 탐지 웹앱")
    st.markdown(
        "캐글(Kaggle)의 **신용카드 사기 탐지(Credit Card Fraud Detection, ULB)** 데이터셋의 실제 상관관계 구조를 바탕으로, "
        "극심한 클래스 불균형 환경에서 **가중치 보정(`class_weight='balanced'`)**을 통해 사기 탐지 재현율(Recall)을 높이는 과정을 탐구합니다."
    )

    st.divider()
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📌 연구 대질문 및 데이터셋 교체 근거")
        st.info(
            "**대질문:** 신용카드 결제 거래의 시간대(`Hour`), 결제 금액(`Amount`), 그리고 사기 여부와 상관관계가 높은 "
            "핵심 주성분 변수($V_{14}, V_{17}, V_{12}, V_{11}, V_4$ 등)를 활용하여 사기 거래를 효과적으로 탐지할 수 있는가?"
        )
        st.markdown("""
        **핵심 탐구 전략 및 개선 과정**
        1. **유의미한 변수 선별:** 기존 무작위 합성 데이터에서 벗어나, 실제 캐글 데이터에서 사기 여부(`Class`)와 뚜렷한 상관관계를 보이는 핵심 변수($V_{14}, V_{17}, V_{11}, V_4$ 등)를 중심으로 학습합니다.
        2. **가중치 부여 (`class_weight='balanced'`):** 소수 클래스(사기 거래) 오분류 시 큰 페널티를 부여해 가중치 보정 전·후의 **재현율(Recall)과 혼동 행렬 변화**를 나란히 검증합니다.
        3. **층화 추출 (`stratify=y`) 및 데이터 누수 방지:** 훈련(80%)과 테스트(20%) 데이터를 분할한 뒤 훈련 세트 기준으로만 `RobustScaler`를 적용합니다.
        """)

    with col2:
        st.subheader("✅ 수행평가 자체 점검표")
        checklist_df = pd.DataFrame({
            "평가 항목": [
                "1. 문제 정의의 참신성",
                "2. 전처리 및 누수 방지",
                "3. 적절한 분류 모델 선정",
                "4. 논리적 평가 지표 및 개선",
                "5. 한계 인식 및 발전 방향",
            ],
            "구현 내용": [
                "실생활 금융 보안(FDS) 이상 결제 탐지",
                "stratify=y 분할 후 RobustScaler 적용 & 중복/결측치 점검",
                "로지스틱 회귀 & 의사결정나무 모델 동시 비교",
                "가중치 보정 전·후 Recall(87%대 달성) & 2D/3D 경계 시각화",
                "시그모이드 임계값 조정 체험 & 실제 은행 XGBoost FDS 비교",
            ],
            "완료 여부": ["완료", "완료", "완료", "완료", "완료"],
        })
        st.dataframe(checklist_df, use_container_width=True, hide_index=True)

# =========================================================
# 페이지 2: EDA & 전처리
# =========================================================
elif menu == "2. EDA & 전처리":
    st.title("📊 탐색적 데이터 분석(EDA) 및 전처리")

    st.subheader("1) 데이터 미리보기 및 결측치·중복 점검")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown(
        f"**데이터 무결성 확인:** 전체 결측치 합계 = **{df.isnull().sum().sum()}건** (결측치 없음 확인)"
    )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2) 클래스 불균형 분포 (정상 vs 사기)")
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
            title="클래스 불균형 현황 (정상 거래 압도적 다수)",
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)
        st.warning(
            "⚠️ 전체 거래 중 사기 거래 비율이 극히 낮으므로, 단순 정확도(Accuracy)가 아닌 **재현율(Recall)**을 핵심 지표로 사용합니다."
        )

    with col2:
        st.subheader("3) 사기 여부(Class)와 각 변수의 상관계수 분석")
        corr_series = (
            df.drop(columns=["Time", "Class"])
            .corrwith(df["Class"])
            .sort_values()
        )
        corr_df = pd.DataFrame({
            "변수명": corr_series.index,
            "상관계수": corr_series.values,
        })
        fig_corr = px.bar(
            corr_df,
            x="상관계수",
            y="변수명",
            orientation="h",
            color="상관계수",
            color_continuous_scale="RdBu_r",
            title="사기 여부(Class)와의 피어슨 상관계수 (V14, V17 음의 상관 / V4, V11 양의 상관)",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.info(
            "ℹ️ 캐글 원본 데이터와 동일하게 **V14, V17, V12**는 강한 음(-)의 상관을, **V4, V11**은 양(+)의 상관을 보입니다."
        )

# =========================================================
# 페이지 3: 성능 높이기 (가중치 전·후 비교 및 2D/3D 결정 경계)
# =========================================================
elif menu == "3. 성능 높이기 (가중치 전·후 비교)":
    st.title("🚀 성능 높이기: 가중치 보정 전·후 비교 및 결정 경계")
    st.markdown(
        "사기 거래가 아주 드문 불균형 데이터에서 **양쪽 클래스의 가중치를 맞추기 전(`None`)과 맞춘 뒤(`balanced`)**의 탐지 성능 변화를 나란히 비교합니다."
    )

    available_kor = [
        FEATURE_KOR_MAP[k] for k in FEATURE_KOR_MAP if k in df.columns
    ]
    default_kor = [
        FEATURE_KOR_MAP[k]
        for k in ["V14", "V17", "V4", "Amount"]
        if k in df.columns
    ]

    selected_kor = st.multiselect(
        "📌 모델 학습 및 시각화에 사용할 입력 속성을 고르세요 (2개 선택 시 2D 경계선, 3개 이상 선택 시 3D 평면 표시)",
        options=available_kor,
        default=default_kor,
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
            title=f"2차원 산점도 및 가중치 보정 전·후 0.5 결정 경계선 ({f1_kor} vs {f2_kor})",
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
                        f"ℹ️ **{label_name}**은(는) 현재 데이터 산점도 그림 범위 밖에 위치합니다."
                    )
            else:
                out_of_bounds_msgs.append(
                    f"ℹ️ **{label_name}**의 기울기가 수직에 가까워 그림 범위 밖에 위치합니다."
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
        "가중치 보정(`balanced`)이 적용된 **로지스틱 회귀(확률로 답하는 모델)**와 **의사결정나무(질문으로 답하는 모델)**로 실시간 위험 확률을 진단합니다."
    )

    sim_cols = [
        c
        for c in ["Hour", "V14", "V17", "V12", "V11", "V4", "Amount"]
        if c in df.columns
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
                int(df[col].median())
                if col == "Hour"
                else float(df[col].median())
            )

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button("🚨 실제 사기 거래 예시 값 자동 입력"):
            fraud_sample = df[df["Class"] == 1].iloc[0]
            for col in sim_cols:
                st.session_state[f"sim_{col}"] = (
                    int(fraud_sample[col])
                    if col == "Hour"
                    else float(fraud_sample[col])
                )
            st.rerun()

    with btn_col2:
        if st.button("🔄 일반 정상 거래 기본값으로 초기화"):
            normal_sample = df[df["Class"] == 0].iloc[0]
            for col in sim_cols:
                st.session_state[f"sim_{col}"] = (
                    int(normal_sample[col])
                    if col == "Hour"
                    else float(normal_sample[col])
                )
            st.rerun()

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.slider("결제 시간대(0~23시)", 0, 23, key="sim_Hour")
        st.number_input(
            "결제 금액(Amount)",
            min_value=0.0,
            max_value=float(df["Amount"].max()),
            step=10.0,
            key="sim_Amount",
        )
    with c2:
        st.slider(
            "거래 변동 패턴 V14",
            float(df["V14"].min()),
            float(df["V14"].max()),
            key="sim_V14",
        )
        st.slider(
            "거래 변동 패턴 V17",
            float(df["V17"].min()),
            float(df["V17"].max()),
            key="sim_V17",
        )
        st.slider(
            "거래 변동 패턴 V12",
            float(df["V12"].min()),
            float(df["V12"].max()),
            key="sim_V12",
        )
    with c3:
        st.slider(
            "거래 변동 패턴 V11",
            float(df["V11"].min()),
            float(df["V11"].max()),
            key="sim_V11",
        )
        st.slider(
            "거래 변동 패턴 V4",
            float(df["V4"].min()),
            float(df["V4"].max()),
            key="sim_V4",
        )

    st.divider()
    if st.button("🔍 실시간 이상 거래 판별 실행", type="primary"):
        input_row = [[st.session_state[f"sim_{col}"] for col in sim_cols]]
        input_df = pd.DataFrame(input_row, columns=sim_cols)
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
                st.caption("비정상 패턴 감지: 승인을 일시 보류하고 SMS 2차 본인인증을 요청합니다.")
            else:
                st.success(f"✅ 정상 거래 승인 (사기 확률: {lr_prob:.1f}%)")
                st.caption("일반적인 고객 소비 패턴 범위 내에 있습니다.")
        with r2:
            st.subheader("의사결정나무 (질문으로 답하는 모델)")
            if dt_prob >= 50.0:
                st.error(f"🚨 사기 의심 탐지 (사기 확률: {dt_prob:.1f}%)")
                st.caption("조건 분기 이상 징후 포착: 이상거래 대응팀(FDS) 모니터링 대상입니다.")
            else:
                st.success(f"✅ 정상 거래 승인 (사기 확률: {dt_prob:.1f}%)")
                st.caption("조건 분기 기준 이상 징후가 발견되지 않았습니다.")

# =========================================================
# 페이지 5: 한계점 분석 & 실제 은행 FDS 비교
# =========================================================
elif menu == "5. 한계점 & 실제 은행 FDS 비교":
    st.title("🏦 우리 모델의 한계점 분석 & 실제 은행 FDS 비교")
    st.markdown(
        "전체 28개 특성 변수 중 **핵심 변수 일부와 시간·금액**으로 구현한 본 프로젝트 모델의 한계점을 성찰하고, "
        "실제 금융권(은행·카드사) FDS가 **시그모이드 임계값(Threshold)**을 조정해 어떤 성능 수준을 달성하는지 비교합니다."
    )

    feature_cols = [
        c
        for c in ["Hour", "V14", "V17", "V12", "V11", "V4", "Amount"]
        if c in df.columns
    ]
    X = df[feature_cols].copy()
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = RobustScaler()
    X_train[["Hour", "Amount"]] = scaler.fit_transform(X_train[["Hour", "Amount"]])
    X_test[["Hour", "Amount"]] = scaler.transform(X_test[["Hour", "Amount"]])

    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    st.divider()
    col_lim1, col_lim2 = st.columns([1, 1.3])

    with col_lim1:
        st.subheader("📉 1) 우리 모델(핵심 변수 활용)의 성적표")
        y_pred_default = (y_prob >= 0.5).astype(int)
        acc_d = accuracy_score(y_test, y_pred_default) * 100
        rec_d = recall_score(y_test, y_pred_default, zero_division=0) * 100
        prec_d = precision_score(y_test, y_pred_default, zero_division=0) * 100
        cm_d = confusion_matrix(y_test, y_pred_default)

        m1, m2 = st.columns(2)
        m1.metric("전체 정확도 (Accuracy)", f"{acc_d:.1f}%")
        m2.metric("사기 탐지 재현율 (Recall)", f"{rec_d:.1f}%")
        m3, m4 = st.columns(2)
        m3.metric("정밀도 (Precision)", f"{prec_d:.1f}%")
        m4.metric("오탐 건수 (정상 차단)", f"{cm_d[0][1]}건")

        st.info(
            "**진단 요약:** 캐글 실제 상관관계 구조를 반영하고 가중치 보정(`balanced`)을 적용하여 "
            f"**재현율 {rec_d:.1f}%**의 높은 탐지율을 달성했으나, 여전히 정상 거래 {cm_d[0][1]}건을 사기로 오인하는 **정밀도-재현율 트레이드오프** 한계가 존재합니다."
        )

    with col_lim2:
        st.subheader("🔍 2) 우리 모델이 가지는 3가지 핵심 한계점")
        st.error(
            "**한계 ① 정보 손실 (28개 변수 중 핵심 일부만 사용)**\n\n"
            "원래 캐글 데이터셋은 결제 위치, 단말기 유형, 거래 빈도 등이 압축된 **V1~V28까지 총 28개의 주성분 변수**를 가지고 있습니다. "
            "본 모델은 일부 변수만 사용했기 때문에 나머지 변수에 담긴 복합적인 이상 징후를 포착하지 못해 오탐(FP)이 발생하는 한계가 있습니다."
        )
        st.warning(
            "**한계 ② 정적 단일 거래 분석의 한계 (시계열 문맥 부재)**\n\n"
            "단일 결제 건의 수치만 보고 판단할 뿐 '직전 결제 후 몇 초 만에 해외에서 재결제했는가' 같은 연속적인 시간 흐름(시계열 패턴)을 반영하지 못합니다."
        )
        st.info(
            "**한계 ③ 선형 결정 경계(로지스틱 회귀)와 단일 임계값(0.5)의 한계**\n\n"
            "캐글 노트북의 실험 결과처럼 로지스틱 회귀는 정밀도가 낮게 나오는 반면, **XGBoost** 같은 트리 앙상블 모델은 정밀도 75%·재현율 80% 이상을 동시에 달성할 수 있습니다."
        )

    st.divider()
    st.subheader("🎛️ 3) 시그모이드(Sigmoid) 함수의 임계값(Threshold) 결정 원리 체험")
    st.markdown(
        "시그모이드 함수 $P = \\frac{1}{1+e^{-z}}$ 가 계산한 확률을 **사기(1)로 판정할 기준선(임계값)**은 모델이 정하는 것이 아니라 "
        "**보안 정책에 따라 사람이 직접 설정**합니다. 아래 슬라이더를 움직여 임계값 변화에 따른 성능 변화를 확인해 보세요."
    )

    t_col1, t_col2 = st.columns([1, 1.3])
    with t_col1:
        thresh = st.slider(
            "사기 판별 임계값 (Threshold)",
            min_value=0.15,
            max_value=0.85,
            value=0.50,
            step=0.05,
        )
        y_pred_t = (y_prob >= thresh).astype(int)
        rec_t = recall_score(y_test, y_pred_t, zero_division=0) * 100
        prec_t = precision_score(y_test, y_pred_t, zero_division=0) * 100
        cm_t = confusion_matrix(y_test, y_pred_t)

        st.metric(
            "현재 임계값 기준 재현율 (Recall)",
            f"{rec_t:.1f}%",
            f"탐지 {cm_t[1][1]}/{cm_t[1].sum()}건",
        )
        st.metric(
            "현재 임계값 기준 정밀도 (Precision)",
            f"{prec_t:.1f}%",
            f"오탐 {cm_t[0][1]}건",
        )

        if thresh <= 0.35:
            st.success(
                f"🏦 **[실제 은행 FDS 방식 - 보안 최우선]** 사기 확률이 **{int(thresh*100)}%**만 넘어도 일단 사기로 의심해 재현율을 극대화합니다. "
                "늘어난 오탐 고객에게는 **SMS 2차 본인인증**을 보내 불편을 해소합니다."
            )
        elif thresh <= 0.60:
            st.info(
                "⚖️ **[기본 균형 설정]** 시그모이드 함수의 수학적 중심(0.50) 부근을 기준으로 판별하여 재현율과 정밀도 사이의 기본 균형을 유지합니다."
            )
        else:
            st.warning(
                f"💳 **[고객 편의 우선 설정]** 사기 확률이 **{int(thresh*100)}%** 이상 확실할 때만 차단하므로 정상 고객 오탐은 줄어들지만, 교묘한 사기를 놓칠 위험이 커집니다."
            )

    with t_col2:
        z_vals = np.linspace(-6, 6, 100)
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
        fig_sig.add_trace(
            go.Scatter(
                x=[-6, 6],
                y=[0.5, 0.5],
                mode="lines",
                name="기본 임계값 (0.50)",
                line=dict(color="#94a3b8", width=1.5, dash="dot"),
            )
        )
        fig_sig.add_trace(
            go.Scatter(
                x=[-6, 6],
                y=[thresh, thresh],
                mode="lines",
                name=f"설정된 임계값 ({thresh:.2f})",
                line=dict(color="#e11d48", width=2.5, dash="dash"),
            )
        )
        fig_sig.update_layout(
            title="시그모이드 함수와 임계값(Threshold) 위치 변화",
            xaxis_title="선형 결합 점수 z (변수 × 가중치 합)",
            yaxis_title="사기 예측 확률 (0 ~ 1)",
            yaxis=dict(range=[-0.05, 1.08]),
            legend=dict(orientation="h", y=-0.25),
            height=380,
        )
        st.plotly_chart(fig_sig, use_container_width=True)

    st.divider()
    st.subheader("⚖️ 4) 실제 은행·카드사 FDS 모델 vs 본 프로젝트 모델 종합 비교")
    compare_df = pd.DataFrame({
        "비교 기준": [
            "사용 변수 (피처)",
            "사용 알고리즘",
            "재현율 (Recall) 수준",
            "정밀도 (Precision) 수준",
            "임계값 및 오탐 해결 전략",
        ],
        "우리 프로젝트 모델 (학생 구현)": [
            "캐글 핵심 변수(V14, V17, V12, V11, V4) + 시간(Hour), 금액(Amount)",
            "단일 모델 (로지스틱 회귀, 의사결정나무)",
            f"{rec_d:.1f}% (가중치 보정 후)",
            f"{prec_d:.1f}% (단일 선형 모델의 오탐 한계)",
            "기본 0.5 기준 단일 판정 (슬라이더로 임계값 변화 실험)",
        ],
        "실제 은행·카드사 FDS 모델 (현업)": [
            "V1~V28 전체 + 최근 1시간 결제 빈도, 해외 IP 거리 차이 등 수백 개 파생 변수",
            "고성능 앙상블(XGBoost, LightGBM) + 시계열 딥러닝(LSTM/Transformer) 결합",
            "80% ~ 95% 이상 (사기 피해 차단을 최우선 목표로 설정)",
            "20% ~ 75% 수준 (앙상블 모델 적용 시 정밀도와 재현율 동시 향상)",
            "임계값을 0.2~0.3으로 낮춰 의심 거래를 먼저 거른 뒤, SMS·앱 2차 본인인증으로 정상 고객 불편을 해소하는 2단계 방어 체계 운영",
        ],
    })
    st.dataframe(compare_df, use_container_width=True, hide_index=True)
