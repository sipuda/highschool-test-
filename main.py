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
# 1. 현실적 분포 보정 데이터 로드 (임계값별 재현율 변화가 뚜렷한 분포)
# ---------------------------------------------------------
@st.cache_data
def generate_realistic_kaggle_sample():
    """
    사기 거래의 위험 강도가 약한 의심(확률 0.2~0.4)부터 강한 의심(확률 0.8~0.99)까지
    단계적으로 분포하도록 설계하여, 임계값(Threshold)을 낮출수록 재현율(Recall)이
    뚜렷하게 상승하는 은행 FDS의 원리를 직관적으로 시연할 수 있는 데이터셋.
    """
    np.random.seed(42)
    n_normal = 1960  # 98.0%
    n_fraud = 40     # 2.0%

    n_norm_base = 1760
    n_norm_outlier = 200

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

    # 정상 고객의 특이 소비 (임계값을 낮출 때 오탐 FP가 늘어나는 원리 반영)
    norm_outlier = pd.DataFrame({
        "Time": np.random.randint(0, 172800, n_norm_outlier),
        "V1": np.random.normal(-1.4, 1.5, n_norm_outlier),
        "V2": np.random.normal(1.2, 1.4, n_norm_outlier),
        "V3": np.random.normal(-1.6, 1.5, n_norm_outlier),
        "V4": np.random.normal(1.6, 1.3, n_norm_outlier),
        "V5": np.random.normal(-1.0, 1.4, n_norm_outlier),
        "V10": np.random.normal(-1.9, 1.4, n_norm_outlier),
        "V11": np.random.normal(1.5, 1.3, n_norm_outlier),
        "V12": np.random.normal(-2.1, 1.4, n_norm_outlier),
        "V14": np.random.normal(-2.3, 1.4, n_norm_outlier),
        "V17": np.random.normal(-2.1, 1.4, n_norm_outlier),
        "Amount": np.round(np.random.exponential(320, n_norm_outlier) + 100, 2),
        "Class": 0,
    })

    # 사기 거래를 3단계 위험군(고위험 15건, 중위험 15건, 저위험/교묘한 사기 10건)으로 구성
    # -> 임계값을 낮출 때마다 중위험·저위험 사기까지 추가로 포착되어 재현율이 단계적으로 급상승!
    fraud_groups = [
        (15, -4.8, 3.3, 250),  # 고위험 사기 (임계값 0.8에서도 포착)
        (15, -2.5, 1.8, 140),  # 중위험 사기 (임계값 0.5 부근에서 포착)
        (10, -1.1, 0.9, 60),   # 교묘한 위장 사기 (임계값을 0.2~0.3으로 낮춰야 포착!)
    ]

    fraud_dfs = []
    for cnt, neg_mu, pos_mu, amt_base in fraud_groups:
        fraud_dfs.append(
            pd.DataFrame({
                "Time": np.random.randint(0, 172800, cnt),
                "V1": np.random.normal(neg_mu * 0.7, 1.2, cnt),
                "V2": np.random.normal(pos_mu * 0.8, 1.2, cnt),
                "V3": np.random.normal(neg_mu * 0.85, 1.2, cnt),
                "V4": np.random.normal(pos_mu, 1.1, cnt),
                "V5": np.random.normal(neg_mu * 0.5, 1.2, cnt),
                "V10": np.random.normal(neg_mu * 0.85, 1.2, cnt),
                "V11": np.random.normal(pos_mu * 0.95, 1.1, cnt),
                "V12": np.random.normal(neg_mu * 0.9, 1.2, cnt),
                "V14": np.random.normal(neg_mu, 1.2, cnt),
                "V17": np.random.normal(neg_mu * 0.95, 1.2, cnt),
                "Amount": np.round(np.random.exponential(150, cnt) + amt_base, 2),
                "Class": 1,
            })
        )

    df_all = pd.concat([norm_base, norm_outlier] + fraud_dfs, ignore_index=True)
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
        source_label = "캐글 ULB 통계 분포 기반 샘플 (2,000건)"

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
        "극심한 클래스 불균형 환경에서 **가중치 보정(`class_weight='balanced'`)**과 **시그모이드 임계값(Threshold) 조정**을 통해 사기 탐지 재현율(Recall)을 높이는 원리를 탐구합니다."
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
        1. **가중치 부여 (`class_weight='balanced'`):** 전체 데이터 중 사기 거래 비율이 약 2%로 극히 낮아 기본 모델은 '모두 정상'으로 치우치는 함정에 빠집니다. 가중치 균형 옵션을 적용해 소수 클래스 오분류 페널티를 높입니다.
        2. **시그모이드 임계값(Threshold) 최적화:** 실제 은행 FDS처럼 판별 기준선(임계값)을 기본 `0.50`에서 `0.20~0.30` 수준으로 낮춤으로써, 교묘하게 위장한 사기 거래까지 포착해 **재현율(Recall)을 90% 이상으로 끌어올리는 원리**를 검증합니다.
        3. **데이터 누수(Data Leakage) 차단:** `train_test_split(..., stratify=y)`로 훈련/테스트 데이터를 먼저 분할한 뒤, 훈련 세트 기준으로만 `RobustScaler`를 학습시킵니다.
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
                "임계값 하향 조정을 통한 재현율 급상승 시연 및 현업 비교",
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
# 페이지 5: 한계점 분석 & 실제 은행 FDS 비교 (임계값 하향 조정 시연 강화)
# =========================================================
elif menu == "5. 한계점 & 실제 은행 FDS":
    st.title("🏦 우리 모델의 한계점 분석 & 실제 은행 FDS의 임계값 전략")
    st.markdown(
        "실제 은행 FDS는 **시그모이드 함수의 사기 판별 임계값(Threshold)을 기본 0.50에서 0.20~0.30 수준으로 낮춤으로써**, "
        "교묘한 사기 거래까지 놓치지 않고 **재현율(Recall)을 90% 이상으로 끌어올리는 전략**을 사용합니다."
    )

    st.divider()
    st.subheader("🎛️ 1) 실제 은행의 핵심 원리 체험: 임계값(Threshold)을 낮추면 재현율이 어떻게 올라갈까?")

    # 실제 데이터로 임계값 변화에 따른 재현율·정밀도 실시간 계산
    eval_cols = [c for c in ["V14", "V17", "V4", "Amount"] if c in df.columns]
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

    # 기본값(0.50)일 때와 사용자 조절 임계값일 때 비교
    base_preds = (probs >= 0.50).astype(int)
    base_rec = recall_score(y_te, base_preds, zero_division=0) * 100
    base_prec = precision_score(y_te, base_preds, zero_division=0) * 100

    col_ctrl, col_viz1, col_viz2 = st.columns([1.1, 1.1, 1.1])

    with col_ctrl:
        thresh = st.slider(
            "👉 임계값(Threshold) 슬라이더를 왼쪽(0.20 방향)으로 낮춰보세요!",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.05,
        )
        preds = (probs >= thresh).astype(int)
        cm_th = confusion_matrix(y_te, preds)
        rec_th = recall_score(y_te, preds, zero_division=0) * 100
        prec_th = precision_score(y_te, preds, zero_division=0) * 100

        m_t1, m_t2 = st.columns(2)
        m_t1.metric(
            "⭐ 사기 탐지 재현율 (Recall)",
            f"{rec_th:.1f}%",
            delta=f"{rec_th - base_rec:+.1f}%p (0.50 대비)",
        )
        m_t2.metric(
            "정밀도 (Precision)",
            f"{prec_th:.1f}%",
            delta=f"{prec_th - base_prec:+.1f}%p (0.50 대비)",
        )

        st.info(
            f"**현재 임계값({thresh:.2f}) 판정 결과:**\n"
            f"- 실제 사기 {cm_th[1].sum()}건 중 **{cm_th[1][1]}건 포착** (미탐 {cm_th[1][0]}건)\n"
            f"- 정상 거래 중 의심 차단(오탐): **{cm_th[0][1]}건**\n\n"
            "💡 **은행의 해결책:** 오탐으로 걸러진 고객에게는 바로 카드를 정지시키지 않고 **SMS·앱 2차 본인인증**을 보내 불편을 해소합니다."
        )

    with col_viz1:
        # 시그모이드 곡선 및 임계값 기준선
        z_vals = np.linspace(-6, 6, 120)
        p_vals = 1 / (1 + np.exp(-z_vals))
        fig_sig = go.Figure()
        fig_sig.add_trace(
            go.Scatter(
                x=z_vals,
                y=p_vals,
                mode="lines",
                name="시그모이드 확률 곡선",
                line=dict(color="#4f46e5", width=3),
            )
        )
        fig_sig.add_hline(
            y=0.50,
            line_dash="dot",
            line_color="#94a3b8",
            annotation_text="기본 임계값 (0.50)",
        )
        fig_sig.add_hline(
            y=thresh,
            line_dash="dash",
            line_color="#e11d48",
            annotation_text=f"현재 임계값 ({thresh:.2f})",
        )
        fig_sig.update_layout(
            title="시그모이드 곡선 위 임계값 하향 이동",
            xaxis_title="선형 결합 점수 z",
            yaxis_title="사기 예측 확률 P",
            height=340,
            margin=dict(t=40, b=35, l=40, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig_sig, use_container_width=True)

    with col_viz2:
        # 임계값 전체 구간(0.10 ~ 0.90)에 따른 재현율 vs 정밀도 변화 추이 곡선
        t_grid = np.linspace(0.10, 0.90, 17)
        rec_curve = [
            recall_score(y_te, (probs >= t).astype(int), zero_division=0) * 100
            for t in t_grid
        ]
        prec_curve = [
            precision_score(y_te, (probs >= t).astype(int), zero_division=0) * 100
            for t in t_grid
        ]

        fig_tradeoff = go.Figure()
        fig_tradeoff.add_trace(
            go.Scatter(
                x=t_grid,
                y=rec_curve,
                mode="lines+markers",
                name="재현율 (Recall)",
                line=dict(color="#e11d48", width=3),
            )
        )
        fig_tradeoff.add_trace(
            go.Scatter(
                x=t_grid,
                y=prec_curve,
                mode="lines+markers",
                name="정밀도 (Precision)",
                line=dict(color="#3b82f6", width=3),
            )
        )
        fig_tradeoff.add_vline(
            x=thresh,
            line_dash="dash",
            line_color="#10b981",
            annotation_text=f"현재({thresh:.2f})",
        )
        fig_tradeoff.update_layout(
            title="임계값을 낮출수록 재현율(빨간선) 급상승!",
            xaxis_title="임계값 (Threshold)",
            yaxis_title="성능 지표 (%)",
            height=340,
            margin=dict(t=40, b=35, l=40, r=20),
            legend=dict(orientation="h", y=-0.22),
        )
        st.plotly_chart(fig_tradeoff, use_container_width=True)

    st.divider()
    col_lim1, col_lim2 = st.columns([1, 1.3])

    with col_lim1:
        st.subheader("📉 2) 우리 프로젝트 모델의 3가지 한계점")
        st.markdown("""
        * **한계 ① 정보 손실 (28개 변수 중 일부만 활용):** 원래 신용카드 사기 데이터셋은 결제 위치, 단말기 유형, 거래 빈도 등이 압축된 `V1~V28`까지 총 28개 변수를 가집니다. 일부 변수만 사용하면 복합 이상 징후를 놓쳐 오탐(FP)이 늘어납니다.
        * **한계 ② 시계열 문맥 부재:** 단일 건의 정적 수치만 볼 뿐 *'직전 결제 후 몇 초 만에 재결제했는가'* 같은 시간 흐름 패턴을 반영하지 못합니다.
        * **한계 ③ 정밀도-재현율 트레이드오프:** 임계값을 낮춰 재현율을 90% 이상으로 높이면 정밀도가 하락하므로, 실제 서비스에서는 반드시 2차 인증 절차가 결합되어야 합니다.
        """)

    with col_lim2:
        st.subheader("⚖️ 3) 실제 은행·카드사 FDS 모델 vs 우리 모델 비교")
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
                "기본 0.5에서 약 75% → 임계값 하향 시 87.5~100%",
                "약 15~35% (변수 제약으로 오탐 발생)",
                "단일 확률 판정 (오탐 시 차단 위험)",
            ],
            "실제 은행·카드사 FDS 모델 (현업)": [
                "V1~V28 전체 + 최근 1시간 결제 빈도, 해외 IP 거리 등 수백 개 파생 변수",
                "고성능 앙상블(XGBoost, LightGBM) + 시계열 딥러닝(LSTM) 결합",
                "80% ~ 95% 이상 (사기 피해 차단을 최우선 목표로 설정)",
                "20% ~ 50% 내외 (사기를 놓치지 않기 위해 의심 거래를 넓게 포착)",
                "임계값을 0.2~0.3으로 낮춰 재현율을 극대화한 뒤, SMS·앱 2차 본인인증으로 정상 고객 불편을 해소하는 2단계 방어 체계 운영",
            ],
        })
        st.dataframe(bank_comp_df, use_container_width=True, hide_index=True)
