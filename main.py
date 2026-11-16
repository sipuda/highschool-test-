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

# 변수 영문-우리말 매핑 딕셔너리
FEATURE_KOR_MAP = {
    "Hour": "결제 시간대(0~23시)",
    "Amount": "결제 금액(Amount)",
    "V1": "거래 변동 패턴 V1",
    "V2": "거래 변동 패턴 V2",
    "V3": "거래 변동 패턴 V3",
    "V4": "거래 변동 패턴 V4",
    "V5": "거래 변동 패턴 V5",
}
KOR_TO_ENG = {v: k for k, v in FEATURE_KOR_MAP.items()}


# ---------------------------------------------------------
# 1. 데이터 로드
# ---------------------------------------------------------
@st.cache_data
def load_dataset():
    file_path = "credit_card_fraud_synthetic.csv"
    if not os.path.exists(file_path):
        st.error(
            f"❌ '{file_path}' 파일을 찾을 수 없습니다. GitHub 저장소 최상위 경로에 파일이 있는지 확인해주세요."
        )
        st.stop()
    data = pd.read_csv(file_path)
    data["Hour"] = (data["Time"] // 3600) % 24
    return data


df = load_dataset()

# ---------------------------------------------------------
# 2. 사이드바 네비게이션 메뉴
# ---------------------------------------------------------
st.sidebar.title("🛡️ FDS 보안 탐지 메뉴")
st.sidebar.success("📁 credit_card_fraud_synthetic.csv 연동됨")

menu = st.sidebar.radio(
    "페이지 이동",
    [
        "1. 프로젝트 개요",
        "2. EDA & 전처리",
        "3. 성능 높이기 (가중치 전·후 비교)",
        "4. 실시간 시뮬레이터",
    ],
)

st.sidebar.divider()
st.sidebar.info(
    "**데이터셋 요약**\n"
    f"- 전체 거래 건수: {len(df):,}건\n"
    f"- 정상 거래(0): {(df['Class'] == 0).sum():,}건\n"
    f"- 사기 거래(1): {(df['Class'] == 1).sum():,}건"
)

# =========================================================
# 페이지 1: 프로젝트 개요
# =========================================================
if menu == "1. 프로젝트 개요":
    st.title("🛡️ 신용카드 이상 금융거래(FDS) 탐지 웹앱")
    st.markdown(
        "극심한 클래스 불균형(정상 98.2% vs 사기 1.8%) 환경에서 **가중치 보정(`class_weight='balanced'`)**과 **입력 속성 조합**을 통해 사기 탐지 성능을 높이는 과정을 탐구합니다."
    )

    st.divider()
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📌 연구 대질문 및 모델 선정 근거")
        st.info(
            "**대질문:** 신용카드 결제 거래의 시간대, 결제 금액, 거래 변동 특성을 분석하여 "
            "극심한 불균형 데이터 속에서도 실제 사기(Fraud) 거래를 효과적으로 탐지할 수 있는가?"
        )
        st.markdown("""
        **핵심 탐구 전략**
        1. **가중치 보정 전·후 비교:** 사기 거래가 매우 드문 데이터에서는 모델이 모든 거래를 '정상'으로 예측하는 편향이 발생합니다. 가중치를 맞추기 전과 맞춘 뒤의 **재현율(Recall)과 혼동 행렬 변화**를 나란히 비교합니다.
        2. **두 가지 분류 모델 대조:**
           * **로지스틱 회귀(확률로 답하는 모델):** 연속형 거래 데이터를 바탕으로 0~100% 사이의 사기 확률과 직선(평면) 결정 경계를 도출합니다.
           * **의사결정나무(질문으로 답하는 모델):** 스무고개 방식의 조건 분기를 통해 비선형 이상 거래 패턴을 포착합니다.
        """)

    with col2:
        st.subheader("✅ 수행평가 자체 점검표")
        checklist_df = pd.DataFrame({
            "평가 항목": [
                "1. 문제 정의의 참신성",
                "2. 전처리 및 누수 방지",
                "3. 적절한 분류 모델 선정",
                "4. 논리적 평가 지표 및 개선",
                "5. Streamlit 웹앱 완성도",
            ],
            "구현 내용": [
                "실생활 금융 보안(FDS) 사기 탐지 문제 해결",
                "Train/Test 분할 후 RobustScaler 적용 & 결측치 확인",
                "로지스틱 회귀 & 의사결정나무 모델 동시 비교",
                "가중치 보정 전·후 혼동 행렬 비교 및 2D/3D 결정 경계 시각화",
                "Plotly 인터랙티브 시각화 및 실시간 시뮬레이터 구현",
            ],
            "완료 여부": ["완료", "완료", "완료", "완료", "완료"],
        })
        st.dataframe(checklist_df, use_container_width=True, hide_index=True)

# =========================================================
# 페이지 2: EDA & 전처리
# =========================================================
elif menu == "2. EDA & 전처리":
    st.title("📊 탐색적 데이터 분석(EDA) 및 전처리")

    st.subheader("1) 원본 데이터 미리보기")
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
                "정상 거래 (0)": "#4A90E2",
                "사기 거래 (1)": "#E74C3C",
            },
            title="클래스 불균형 현황 (정상 982건 vs 사기 18건)",
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("3) 거래 유형별 결제 금액 분포")
        plot_df = df.copy()
        plot_df["거래 구분"] = plot_df["Class"].map(
            {0: "정상 거래 (0)", 1: "사기 거래 (1)"}
        )
        fig_box = px.box(
            plot_df,
            x="거래 구분",
            y="Amount",
            color="거래 구분",
            color_discrete_map={
                "정상 거래 (0)": "#4A90E2",
                "사기 거래 (1)": "#E74C3C",
            },
            labels={"Amount": "결제 금액(Amount)"},
            title="정상 vs 사기 거래 결제 금액 비교",
        )
        st.plotly_chart(fig_box, use_container_width=True)

# =========================================================
# 페이지 3: 성능 높이기 (가중치 전·후 비교 및 2D/3D 결정 경계)
# =========================================================
elif menu == "3. 성능 높이기 (가중치 전·후 비교)":
    st.title("🚀 성능 높이기: 가중치 보정 전·후 비교 및 결정 경계 시각화")
    st.markdown(
        "사기 거래가 아주 드문 불균형 데이터에서 **양쪽 클래스의 가중치를 맞추기 전(`None`)과 맞춘 뒤(`balanced`)**의 탐지 성능 변화를 비교합니다."
    )

    # 화면 맨 위 입력 속성 선택
    all_kor_features = list(FEATURE_KOR_MAP.values())
    default_kor_features = [
        FEATURE_KOR_MAP["Hour"],
        FEATURE_KOR_MAP["Amount"],
        FEATURE_KOR_MAP["V1"],
        FEATURE_KOR_MAP["V4"],
    ]

    selected_kor = st.multiselect(
        "📌 모델 학습 및 시각화에 사용할 입력 속성을 고르세요 (최소 2개 이상 선택)",
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

    # 모델 학습 함수 (가중치 전 vs 후)
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
            "구분 항목": [
                "정확도 (Accuracy)",
                "⭐ 재현율 (Recall - 사기 탐지율)",
                "정밀도 (Precision)",
                "F1-Score",
                "칸1: 정상 → 정상 예측 (TN)",
                "칸2: 정상 → 사기 헛짚음 (FP)",
                "칸3: 사기 → 정상 놓침 (FN)",
                "칸4: 사기 → 사기 찾아냄 (TP)",
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
            f"📌 **변화 요약:** 찾아낸 사기 건수는 **{cm_b[1][1]}건 → {cm_a[1][1]}건**, "
            f"놓친 사기 건수는 **{cm_b[1][0]}건 → {cm_a[1][0]}건**, "
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
        # 2개 속성 선택 시: 2D 산점도 + 전·후 0.5 경계선
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
                "정상 거래 (0)": "#4A90E2",
                "사기 거래 (1)": "#E74C3C",
            },
            labels={f1_eng: f"{f1_kor} (표준화)", f2_eng: f"{f2_kor} (표준화)"},
            title=f"2차원 산점도 및 가중치 보정 전·후 0.5 결정 경계선 ({f1_kor} vs {f2_kor})",
        )

        x_min, x_max = plot_2d[f1_eng].min() - 0.5, plot_2d[f1_eng].max() + 0.5
        y_min, y_max = plot_2d[f2_eng].min() - 0.5, plot_2d[f2_eng].max() + 0.5
        x_vals = np.linspace(x_min, x_max, 200)

        out_of_bounds_msgs = []

        for model_obj, label_name, line_color, dash_style in [
            (lr_before, "가중치 맞추기 전 0.5 경계선", "#2ECC71", "dash"),
            (lr_after, "가중치 맞춘 뒤 0.5 경계선", "#F39C12", "solid"),
        ]:
            w1, w2 = model_obj.coef_[0][0], model_obj.coef_[0][1]
            b = model_obj.intercept_[0]
            if abs(w2) > 1e-6:
                y_vals = -(w1 * x_vals + b) / w2
                # 그림 영역 안에 경계선이 들어오는지 확인
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
                        f"ℹ️ **{label_name}**은(는) 모든 데이터를 한쪽(정상)으로 치우쳐 판별하기 때문에 현재 데이터 산점도 그림 범위 밖에 위치합니다."
                    )
            else:
                out_of_bounds_msgs.append(
                    f"ℹ️ **{label_name}**의 기울기가 수직에 가까워 그림 범위 밖에 위치합니다."
                )

        st.plotly_chart(fig2d, use_container_width=True)
        for msg in out_of_bounds_msgs:
            st.warning(msg)

    else:
        # 3개 이상 속성 선택 시: 앞 3개 속성을 축으로 하는 3D 산점도 + 전·후 0.5 결정 평면
        f1_kor, f2_kor, f3_kor = selected_kor[0], selected_kor[1], selected_kor[2]
        f1_eng, f2_eng, f3_eng = selected_eng[0], selected_eng[1], selected_eng[2]

        st.caption(
            f"선택한 {len(selected_kor)}개 속성 중 앞의 3개(**{f1_kor}**, **{f2_kor}**, **{f3_kor}**)를 축으로 하는 3차원 공간입니다. 마우스로 드래그해 회전시켜 보세요."
        )

        # 3개 축만으로 시각화용 3D 모델 별도 적합
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
                "정상 거래 (0)": "#4A90E2",
                "사기 거래 (1)": "#E74C3C",
            },
            labels={
                f1_eng: f1_kor,
                f2_eng: f2_kor,
                f3_eng: f3_kor,
            },
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

        fig3d.update_layout(height=650)
        st.plotly_chart(fig3d, use_container_width=True)
        for w_msg in plane_warnings:
            st.warning(w_msg)

# =========================================================
# 페이지 4: 실시간 시뮬레이터
# =========================================================
elif menu == "4. 실시간 시뮬레이터":
    st.title("🚨 실시간 이상 금융거래 탐지 시뮬레이터")
    st.markdown(
        "가중치 보정(`balanced`)이 적용된 **로지스틱 회귀(확률로 답하는 모델)**와 **의사결정나무(질문으로 답하는 모델)**로 실시간 사기 위험도를 진단합니다."
    )

    feature_cols = ["Hour", "V1", "V2", "V3", "V4", "V5", "Amount"]
    X_all = df[feature_cols].copy()
    y_all = df["Class"]

    scaler = RobustScaler()
    X_all[["Hour", "Amount"]] = scaler.fit_transform(X_all[["Hour", "Amount"]])

    lr_sim = LogisticRegression(class_weight="balanced", random_state=42).fit(
        X_all, y_all
    )
    dt_sim = DecisionTreeClassifier(
        max_depth=4, class_weight="balanced", random_state=42
    ).fit(X_all, y_all)

    if "sim_hour" not in st.session_state:
        st.session_state.sim_hour = int(df["Hour"].median())
        st.session_state.sim_amount = float(df["Amount"].median())
        st.session_state.sim_v1 = float(df["V1"].median())
        st.session_state.sim_v2 = float(df["V2"].median())
        st.session_state.sim_v3 = float(df["V3"].median())
        st.session_state.sim_v4 = float(df["V4"].median())
        st.session_state.sim_v5 = float(df["V5"].median())

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button("🚨 실제 사기 거래 예시 값 자동 입력"):
            fraud_df = df[df["Class"] == 1].copy()
            fraud_scaled = fraud_df[feature_cols].copy()
            fraud_scaled[["Hour", "Amount"]] = scaler.transform(
                fraud_scaled[["Hour", "Amount"]]
            )
            best_idx = np.argmax(lr_sim.predict_proba(fraud_scaled)[:, 1])
            fraud_sample = fraud_df.iloc[best_idx]

            st.session_state.sim_hour = int(fraud_sample["Hour"])
            st.session_state.sim_amount = float(fraud_sample["Amount"])
            st.session_state.sim_v1 = float(fraud_sample["V1"])
            st.session_state.sim_v2 = float(fraud_sample["V2"])
            st.session_state.sim_v3 = float(fraud_sample["V3"])
            st.session_state.sim_v4 = float(fraud_sample["V4"])
            st.session_state.sim_v5 = float(fraud_sample["V5"])
            st.rerun()

    with btn_col2:
        if st.button("🔄 일반 정상 거래 기본값으로 초기화"):
            normal_sample = df[df["Class"] == 0].iloc[0]
            st.session_state.sim_hour = int(normal_sample["Hour"])
            st.session_state.sim_amount = float(normal_sample["Amount"])
            st.session_state.sim_v1 = float(normal_sample["V1"])
            st.session_state.sim_v2 = float(normal_sample["V2"])
            st.session_state.sim_v3 = float(normal_sample["V3"])
            st.session_state.sim_v4 = float(normal_sample["V4"])
            st.session_state.sim_v5 = float(normal_sample["V5"])
            st.rerun()

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        in_hour = st.slider(
            "결제 시간대(0~23시)", min_value=0, max_value=23, key="sim_hour"
        )
        in_amount = st.number_input(
            "결제 금액(Amount)",
            min_value=0.0,
            max_value=float(df["Amount"].max()),
            step=10.0,
            key="sim_amount",
        )
    with c2:
        in_v1 = st.slider(
            "거래 변동 패턴 V1",
            float(df["V1"].min()),
            float(df["V1"].max()),
            key="sim_v1",
        )
        in_v2 = st.slider(
            "거래 변동 패턴 V2",
            float(df["V2"].min()),
            float(df["V2"].max()),
            key="sim_v2",
        )
        in_v3 = st.slider(
            "거래 변동 패턴 V3",
            float(df["V3"].min()),
            float(df["V3"].max()),
            key="sim_v3",
        )
    with c3:
        in_v4 = st.slider(
            "거래 변동 패턴 V4",
            float(df["V4"].min()),
            float(df["V4"].max()),
            key="sim_v4",
        )
        in_v5 = st.slider(
            "거래 변동 패턴 V5",
            float(df["V5"].min()),
            float(df["V5"].max()),
            key="sim_v5",
        )

    st.divider()
    if st.button("🔍 실시간 이상 거래 판별 실행", type="primary"):
        input_df = pd.DataFrame(
            [[in_hour, in_v1, in_v2, in_v3, in_v4, in_v5, in_amount]],
            columns=feature_cols,
        )
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
