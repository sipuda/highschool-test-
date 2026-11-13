import os
import platform
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
import streamlit as st

# ---------------------------------------------------------
# 0. 페이지 기본 설정 및 한글 폰트 깨짐 방지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="신용카드 이상 금융거래(FDS) 탐지 시스템",
    page_icon="🛡️",
    layout="wide",
)


def set_korean_font():
    """운영체제(Windows, Mac, Linux/Streamlit Cloud)에 맞게 한글 폰트 자동 설정"""
    system_name = platform.system()
    if system_name == "Windows":
        plt.rc("font", family="Malgun Gothic")
    elif system_name == "Darwin":
        plt.rc("font", family="AppleGothic")
    else:
        font_dirs = ["/usr/share/fonts/truetype/nanum"]
        font_files = fm.findSystemFonts(fontpaths=font_dirs)
        if font_files:
            fm.fontManager.addfont(font_files[0])
            font_name = fm.FontProperties(fname=font_files[0]).get_name()
            plt.rc("font", family=font_name)
        else:
            plt.rc("font", family="DejaVu Sans")
    plt.rcParams["axes.unicode_minus"] = False


set_korean_font()


# ---------------------------------------------------------
# 1. GitHub 저장소의 CSV 데이터 직접 로드
# ---------------------------------------------------------
@st.cache_data
def load_dataset():
    file_path = "credit_card_fraud_synthetic.csv"
    if not os.path.exists(file_path):
        st.error(
            f"❌ '{file_path}' 파일을 찾을 수 없습니다. GitHub 저장소 최상위 경로에 파일명이 정확히 일치하는지 확인해주세요."
        )
        st.stop()
    data = pd.read_csv(file_path)
    # Time(초 단위)을 24시간 기준 시간대(Hour) 파생 변수로 생성
    data["Hour"] = (data["Time"] // 3600) % 24
    return data


df = load_dataset()

# ---------------------------------------------------------
# 2. 사이드바 네비게이션 메뉴
# ---------------------------------------------------------
st.sidebar.title("🛡️ FDS 보안 탐지 메뉴")
st.sidebar.success("📁 credit_card_fraud_synthetic.csv 연동 완료")

menu = st.sidebar.radio(
    "페이지 이동",
    [
        "1. 프로젝트 개요",
        "2. EDA & 전처리",
        "3. 로지스틱 회귀 학습 & 평가",
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
        "**로지스틱 회귀(Logistic Regression)** 분류 모델을 활용하여 신용카드 결제 로그의 이상 패턴을 탐지하고 실시간 사기 위험 확률을 진단합니다."
    )

    st.divider()
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📌 연구 대질문 및 모델 선정 근거")
        st.info(
            "**대질문:** 신용카드 결제 거래의 시간대(`Hour`), 결제 금액(`Amount`), 거래 변동 특성($V_1 \\sim V_5$)을 분석하여 "
            "극심한 클래스 불균형 속에서도 실제 사기(Fraud) 거래를 효과적으로 분류할 수 있는가?"
        )
        st.markdown("""
        **왜 선형회귀가 아닌 '로지스틱 회귀'를 선택했는가?**
        * 입력 변수(`Amount`, `V1~V5` 등)는 연속형 숫자이지만, 예측해야 할 종속 변수(`Class`)는 **0(정상)과 1(사기)로 나뉘는 이진 범주형 데이터**입니다.
        * 일반 선형회귀는 예측값이 0 미만이나 1 초과로 발산할 수 있는 반면, **로지스틱 회귀**는 시그모이드(Sigmoid) 함수를 통해 결과를 **0~100% 사이의 사기 발생 확률**로 변환해 주므로 보안 위험도 판별에 최적입니다.
        """)

    with col2:
        st.subheader("✅ 수행평가 자체 점검표")
        checklist_df = pd.DataFrame({
            "평가 항목": [
                "1. 문제 정의의 참신성",
                "2. 전처리 및 누수 방지",
                "3. 적절한 분류 모델 선정",
                "4. 논리적 평가 지표 활용",
                "5. Streamlit 웹앱 완성도",
            ],
            "구현 내용": [
                "실생활 금융 보안(FDS) 사기 탐지 문제 해결",
                "Train/Test 분할 후 RobustScaler 적용 & Hour 파생변수 반영",
                "로지스틱 회귀 (이진 확률 예측 및 class_weight='balanced' 적용)",
                "단순 정확도 외 재현율(Recall) 및 혼동 행렬 시각화",
                "데이터 탐색부터 실시간 탐지 시뮬레이터까지 통합 구현",
            ],
            "완료 여부": ["완료", "완료", "완료", "완료", "완료"],
        })
        st.dataframe(checklist_df, use_container_width=True, hide_index=True)

# =========================================================
# 페이지 2: EDA & 전처리
# =========================================================
elif menu == "2. EDA & 전처리":
    st.title("📊 탐색적 데이터 분석(EDA) 및 전처리")

    st.subheader("1) 원본 데이터 및 파생 변수(`Hour`) 미리보기")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown(
        f"**결측치 확인:** 전체 컬럼 결측치 합계 = **{df.isnull().sum().sum()}건** (결측치 없음 확인)"
    )

    st.divider()
    st.subheader("2) 클래스 불균형(Class Imbalance) 분포 확인")
    col1, col2 = st.columns([1, 1.2])

    with col1:
        class_counts = df["Class"].value_counts()
        fraud_count = class_counts.get(1, 0)
        normal_count = class_counts.get(0, 0)
        fraud_ratio = (fraud_count / len(df)) * 100

        st.metric("정상 거래 (Class = 0)", f"{normal_count:,}건")
        st.metric(
            "사기 거래 (Class = 1)",
            f"{fraud_count:,}건",
            f"전체의 {fraud_ratio:.2f}%",
            delta_color="inverse",
        )
        st.warning(
            "⚠️ **분석 인사이트:** 전체 1,000건 중 사기 거래가 18건(1.80%)으로 극심한 불균형을 보이므로, 학습 시 `class_weight='balanced'` 옵션으로 소수 클래스 가중치를 보정합니다."
        )

    with col2:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(
            x=["정상 거래 (0)", "사기 거래 (1)"],
            y=[normal_count, fraud_count],
            palette=["#4A90E2", "#E74C3C"],
            ax=ax1,
        )
        ax1.set_title("정상 vs 사기 거래 건수 비교 (클래스 불균형)")
        ax1.set_ylabel("거래 건수 (건)")
        for i, v in enumerate([normal_count, fraud_count]):
            ax1.text(
                i, v + (len(df) * 0.015), f"{v}건", ha="center", fontweight="bold"
            )
        st.pyplot(fig1)

    st.divider()
    st.subheader("3) 거래 금액(Amount) 및 시간대(Hour) 패턴 시각화")
    col3, col4 = st.columns(2)

    with col3:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=df,
            x="Class",
            y="Amount",
            palette=["#4A90E2", "#E74C3C"],
            ax=ax2,
        )
        ax2.set_xticklabels(["정상 거래 (0)", "사기 거래 (1)"])
        ax2.set_title("거래 유형별 결제 금액(Amount) 분포 비교")
        ax2.set_xlabel("거래 분류")
        ax2.set_ylabel("결제 금액 (Amount)")
        st.pyplot(fig2)

    with col4:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.histplot(
            data=df,
            x="Hour",
            hue="Class",
            bins=24,
            multiple="dodge",
            palette=["#4A90E2", "#E74C3C"],
            ax=ax3,
        )
        ax3.set_title("결제 시간대(Hour)별 정상 vs 사기 거래 분포")
        ax3.set_xlabel("결제 시간 (0시 ~ 23시)")
        ax3.set_ylabel("빈도수")
        st.pyplot(fig3)

# =========================================================
# 페이지 3: 로지스틱 회귀 학습 & 평가
# =========================================================
elif menu == "3. 로지스틱 회귀 학습 & 평가":
    st.title("⚙️ 로지스틱 회귀 모델 학습 및 성능 검증")
    st.markdown(
        "훈련 데이터와 테스트 데이터를 **층화 추출(Stratified Split)**로 먼저 분할한 뒤 스케일링을 적용해 데이터 누수(Data Leakage)를 방지합니다."
    )

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        test_size = st.slider(
            "테스트 데이터 비율 (Test Size)", 0.1, 0.4, 0.2, step=0.05
        )
    with col_opt2:
        threshold = st.slider(
            "사기 판별 확률 임계값 (Threshold)",
            0.1,
            0.9,
            0.5,
            step=0.05,
            help="사기 확률이 이 기준값 이상이면 사기(1)로 분류합니다. 낮출수록 사기 탐지 재현율(Recall)이 높아집니다.",
        )

    feature_cols = ["Hour", "V1", "V2", "V3", "V4", "V5", "Amount"]
    X = df[feature_cols].copy()
    y = df["Class"]

    # 데이터 누수 방지를 위해 분할(Split)을 먼저 수행 후 스케일링
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = RobustScaler()
    X_train[["Hour", "Amount"]] = scaler.fit_transform(X_train[["Hour", "Amount"]])
    X_test[["Hour", "Amount"]] = scaler.transform(X_test[["Hour", "Amount"]])

    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    st.divider()
    st.subheader(
        f"📈 테스트 데이터({len(X_test)}건) 검증 결과 — 로지스틱 회귀 (임계값 {threshold:.2f})"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("전체 정확도 (Accuracy)", f"{acc * 100:.1f}%")
    m2.metric(
        "⭐ 사기 탐지 재현율 (Recall)", f"{rec * 100:.1f}%", "보안 핵심 지표"
    )
    m3.metric("정밀도 (Precision)", f"{prec * 100:.1f}%")
    m4.metric("F1-Score", f"{f1:.3f}")

    st.divider()
    col_res1, col_res2 = st.columns([1, 1.2])

    with col_res1:
        st.subheader("🧩 혼동 행렬 (Confusion Matrix)")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax_cm,
            xticklabels=["정상 예측 (0)", "사기 예측 (1)"],
            yticklabels=["실제 정상 (0)", "실제 사기 (1)"],
        )
        ax_cm.set_xlabel("모델 예측값")
        ax_cm.set_ylabel("실제 정답")
        st.pyplot(fig_cm)

    with col_res2:
        st.subheader("📊 변수별 회귀 계수(가중치) 영향력")
        coef_df = pd.DataFrame({
            "변수명": feature_cols,
            "회귀 계수(Weight)": model.coef_[0],
        }).sort_values(by="회귀 계수(Weight)", ascending=False)

        fig_coef, ax_coef = plt.subplots(figsize=(5, 3))
        sns.barplot(
            data=coef_df,
            x="회귀 계수(Weight)",
            y="변수명",
            palette="coolwarm",
            ax=ax_coef,
        )
        ax_coef.axvline(0, color="black", linewidth=0.8)
        ax_coef.set_title("로지스틱 회귀 독립 변수별 영향력")
        st.pyplot(fig_coef)
        st.caption(
            f"실제 사기 {cm[1].sum()}건 중 {cm[1][1]}건 탐지 (미탐 {cm[1][0]}건, 정상 오탐 {cm[0][1]}건). 임계값 슬라이더를 조절해 트레이드오프를 확인할 수 있습니다."
        )

# =========================================================
# 페이지 4: 실시간 시뮬레이터
# =========================================================
elif menu == "4. 실시간 시뮬레이터":
    st.title("🚨 실시간 이상 금융거래 탐지 시뮬레이터")
    st.markdown(
        "거래 데이터를 직접 조절하거나 **[실제 사기 예시 자동 입력]** 버튼을 눌러 로지스틱 회귀 모델의 사기 확률을 진단하세요."
    )

    feature_cols = ["Hour", "V1", "V2", "V3", "V4", "V5", "Amount"]
    X_all = df[feature_cols].copy()
    y_all = df["Class"]

    scaler = RobustScaler()
    X_all[["Hour", "Amount"]] = scaler.fit_transform(X_all[["Hour", "Amount"]])

    sim_model = LogisticRegression(class_weight="balanced", random_state=42)
    sim_model.fit(X_all, y_all)

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
        if (df["Class"] == 1).sum() > 0:
            if st.button("🚨 실제 사기 거래 예시 값 자동 입력"):
                # 모델이 사기로 가장 명확히 잡아내는 사기 샘플 자동 선택
                fraud_df = df[df["Class"] == 1].copy()
                fraud_scaled = fraud_df[feature_cols].copy()
                fraud_scaled[["Hour", "Amount"]] = scaler.transform(
                    fraud_scaled[["Hour", "Amount"]]
                )
                best_idx = np.argmax(sim_model.predict_proba(fraud_scaled)[:, 1])
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
    st.subheader("1) 테스트할 결제 정보 입력")

    c1, c2, c3 = st.columns(3)
    with c1:
        in_hour = st.slider(
            "결제 시간대 (Hour, 0시~23시)",
            min_value=0,
            max_value=23,
            key="sim_hour",
        )
        in_amount = st.number_input(
            "결제 금액 (Amount)",
            min_value=0.0,
            max_value=float(df["Amount"].max()),
            step=10.0,
            key="sim_amount",
        )
    with c2:
        in_v1 = st.slider(
            "특성 변수 V1",
            float(df["V1"].min()),
            float(df["V1"].max()),
            key="sim_v1",
        )
        in_v2 = st.slider(
            "특성 변수 V2",
            float(df["V2"].min()),
            float(df["V2"].max()),
            key="sim_v2",
        )
        in_v3 = st.slider(
            "특성 변수 V3",
            float(df["V3"].min()),
            float(df["V3"].max()),
            key="sim_v3",
        )
    with c3:
        in_v4 = st.slider(
            "특성 변수 V4",
            float(df["V4"].min()),
            float(df["V4"].max()),
            key="sim_v4",
        )
        in_v5 = st.slider(
            "특성 변수 V5",
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

        pred_proba = sim_model.predict_proba(input_df)[0][1] * 100

        st.subheader("2) AI 보안 시스템 진단 결과")
        if pred_proba >= 50.0:
            st.error(
                f"🚨 **[경고] 사기 의심 거래로 탐지되었습니다! (사기 위험 확률: {pred_proba:.1f}%)**"
            )
            st.markdown("""
            * **조치 권고:** 비정상적인 결제 패턴이 감지되어 카드 승인을 일시 보류합니다.
            * **후속 절차:** 카드 소유자 본인 확인을 위한 SMS 2차 인증 및 이상거래 대응팀(FDS) 모니터링이 필요합니다.
            """)
        else:
            st.success(
                f"✅ **[정상] 안전한 일반 결제 거래입니다. (사기 위험 확률: {pred_proba:.1f}%)**"
            )
            st.markdown("""
            * **조치 권고:** 기존 정상 거래 패턴 범위 내에 있으므로 결제를 정상 승인합니다.
            """)
