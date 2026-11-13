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
from sklearn.tree import DecisionTreeClassifier, export_text
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
    """운영체제(Windows, Mac, Linux/Streamlit Cloud)에 맞게 한글 폰트를 자동 설정"""
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
# 1. 데이터 로드 및 사이드바 파일 업로더
# ---------------------------------------------------------
@st.cache_data
def generate_fallback_data():
    """CSV 미업로드 시 자동 작동하는 기본 FDS 데이터(1,000행)"""
    np.random.seed(42)
    n_normal = 950
    n_fraud = 50

    normal_df = pd.DataFrame({
        "Time": np.random.randint(0, 172800, n_normal),
        "V1": np.random.normal(0.2, 1.0, n_normal),
        "V2": np.random.normal(-0.1, 1.0, n_normal),
        "V3": np.random.normal(0.3, 1.0, n_normal),
        "V4": np.random.normal(-0.2, 1.0, n_normal),
        "V5": np.random.normal(0.1, 1.0, n_normal),
        "Amount": np.round(np.random.exponential(60, n_normal), 2),
        "Class": 0,
    })

    fraud_df = pd.DataFrame({
        "Time": np.random.randint(0, 172800, n_fraud),
        "V1": np.random.normal(-2.5, 1.5, n_fraud),
        "V2": np.random.normal(2.2, 1.4, n_fraud),
        "V3": np.random.normal(-3.1, 1.6, n_fraud),
        "V4": np.random.normal(2.8, 1.3, n_fraud),
        "V5": np.random.normal(-1.8, 1.5, n_fraud),
        "Amount": np.round(np.random.exponential(220, n_fraud) + 50, 2),
        "Class": 1,
    })

    df = pd.concat([normal_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


st.sidebar.title("🛡️ FDS 보안 탐지 메뉴")

st.sidebar.subheader("📂 데이터 파일 업로드")
uploaded_file = st.sidebar.file_uploader(
    "CSV 파일을 업로드하세요 (선택 사항)", type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ 업로드된 CSV 파일을 사용 중입니다!")
elif os.path.exists("credit_card_fraud_synthetic.csv"):
    df = pd.read_csv("credit_card_fraud_synthetic.csv")
    st.sidebar.info("📁 저장소의 기본 CSV 파일을 로드했습니다.")
else:
    df = generate_fallback_data()
    st.sidebar.warning("⚡ 내장된 샘플 데이터를 사용 중입니다.")

required_cols = {"Time", "V1", "V2", "V3", "V4", "V5", "Amount", "Class"}
if not required_cols.issubset(set(df.columns)):
    st.error(
        f"❌ 업로드한 CSV에 필수 컬럼({required_cols})이 모두 포함되어 있어야 합니다."
    )
    st.stop()

# [수정 반영 ②] EDA 인사이트를 모델링과 연결하기 위해 Hour 파생 변수 생성
df["Hour"] = (df["Time"] // 3600) % 24

# ---------------------------------------------------------
# 2. 사이드바 네비게이션 메뉴
# ---------------------------------------------------------
st.sidebar.divider()
menu = st.sidebar.radio(
    "페이지 이동",
    [
        "1. 프로젝트 개요",
        "2. EDA & 전처리",
        "3. 모델 학습 & 평가",
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
        "머신러닝 분류(Classification) 알고리즘을 활용하여 신용카드 결제 로그의 이상 패턴을 탐지하고 실시간 위험도를 진단합니다."
    )

    st.divider()
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📌 연구 대질문 및 핵심 목표")
        st.info(
            "**대질문:** 신용카드 결제 거래의 시간대(`Hour`), 결제 금액(`Amount`), 거래 변동 특성($V_1 \\sim V_5$)을 분석하여 "
            "극심한 클래스 불균형 속에서도 실제 사기(Fraud) 거래를 효과적으로 탐지할 수 있는가?"
        )
        st.markdown("""
        **핵심 해결 과제 및 데이터 과학적 설계**
        1. **데이터 누수(Data Leakage) 방지:** 훈련/테스트 데이터를 먼저 분할한 후 훈련 세트 기준으로만 스케일링을 학습시켜 공정한 성능 평가를 보장합니다.
        2. **클래스 불균형 극복:** 전체 데이터 중 사기 거래 비율이 극히 낮은 환경에서 `class_weight='balanced'` 옵션으로 소수 클래스 탐지력을 강화합니다.
        3. **보안 특화 평가지표 검증:** 단순 정확도(Accuracy)의 함정을 피하고, 실제 사기를 놓치지 않는 **재현율(Recall)**과 **혼동 행렬(Confusion Matrix)**을 중점 검증합니다.
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
                "의사결정나무 & 로지스틱 회귀 (가중치 균형 적용)",
                "정확도 외 재현율(Recall) 및 혼동 행렬 시각화",
                "파일 업로더 및 원클릭 실시간 시뮬레이터 구현",
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
            "⚠️ **분석 인사이트:** 사기 거래 비율이 매우 낮기 때문에, 학습 시 소수 클래스(사기)에 높은 가중치를 부여하는 보정 기법이 필수적입니다."
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
        st.caption(
            "결제 금액의 극단적 이상치(Outlier) 영향을 줄이기 위해 훈련 데이터 기준으로 RobustScaler를 적용합니다."
        )

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
        st.caption(
            "초 단위 Time을 24시간 주기(Hour) 파생 변수로 변환하여 모델 학습 독립 변수로 직접 활용합니다."
        )

# =========================================================
# 페이지 3: 모델 학습 & 평가
# =========================================================
elif menu == "3. 모델 학습 & 평가":
    st.title("⚙️ 분류 모델 학습 및 성능 검증")
    st.markdown(
        "훈련 데이터와 테스트 데이터를 **층화 추출(Stratified Split)**로 먼저 분할한 뒤 스케일링을 적용해 데이터 누수를 방지합니다."
    )

    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        model_type = st.selectbox(
            "분류 알고리즘 선택",
            [
                "의사결정나무 (Decision Tree)",
                "로지스틱 회귀 (Logistic Regression)",
            ],
        )
    with col_opt2:
        test_size = st.slider(
            "테스트 데이터 비율 (Test Size)", 0.1, 0.4, 0.2, step=0.05
        )
    with col_opt3:
        max_depth = st.slider(
            "의사결정나무 최대 깊이 (max_depth)",
            2,
            10,
            4,
            disabled=(model_type != "의사결정나무 (Decision Tree)"),
        )

    # [수정 반영 ②] EDA에서 도출한 Hour 파생 변수를 독립 변수에 포함
    feature_cols = ["Hour", "V1", "V2", "V3", "V4", "V5", "Amount"]
    X = df[feature_cols].copy()
    y = df["Class"]

    # [수정 반영 ①] 데이터 누수(Data Leakage) 방지를 위해 분할(Split)을 스케일링보다 먼저 수행
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = RobustScaler()
    X_train[["Hour", "Amount"]] = scaler.fit_transform(X_train[["Hour", "Amount"]])
    X_test[["Hour", "Amount"]] = scaler.transform(X_test[["Hour", "Amount"]])

    if model_type == "의사결정나무 (Decision Tree)":
        model = DecisionTreeClassifier(
            max_depth=max_depth, class_weight="balanced", random_state=42
        )
    else:
        model = LogisticRegression(class_weight="balanced", random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    st.divider()
    st.subheader(
        f"📈 테스트 데이터({len(X_test)}건) 검증 결과 — {model_type}"
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
        st.subheader("🔍 평가지표 해석 및 모델 설명")
        st.markdown(f"""
        * **왜 재현율(Recall)이 가장 중요한가?**
          불균형 데이터에서는 모든 거래를 '정상'으로 찍어도 정확도가 90% 이상 나옵니다. 금융 보안에서는 **실제 사기 거래({cm[1].sum()}건) 중 모델이 사기로 잡아낸 비율({cm[1][1]}건, {rec*100:.1f}%)**인 재현율이 핵심입니다.
        * **오탐(False Positive):** 정상 거래를 사기로 오인해 차단한 건수 = **{cm[0][1]}건**
        * **미탐(False Negative):** 실제 사기 거래를 놓치고 승인한 건수 = **{cm[1][0]}건**
        """)

        if model_type == "의사결정나무 (Decision Tree)":
            with st.expander(
                "🌳 의사결정나무 분류 규칙(Decision Rules) 텍스트 보기"
            ):
                tree_rules = export_text(model, feature_names=feature_cols)
                st.code(tree_rules, language="text")

# =========================================================
# 페이지 4: 실시간 시뮬레이터
# =========================================================
elif menu == "4. 실시간 시뮬레이터":
    st.title("🚨 실시간 이상 금융거래 탐지 시뮬레이터")
    st.markdown(
        "거래 데이터를 직접 조절하거나 **[실제 사기 예시 자동 입력]** 버튼을 눌러 모델의 실시간 위험도 판별 결과를 확인하세요."
    )

    feature_cols = ["Hour", "V1", "V2", "V3", "V4", "V5", "Amount"]
    X_all = df[feature_cols].copy()
    y_all = df["Class"]

    scaler = RobustScaler()
    X_all[["Hour", "Amount"]] = scaler.fit_transform(X_all[["Hour", "Amount"]])

    sim_model_name = st.radio(
        "진단에 사용할 모델 선택",
        ["의사결정나무 (Decision Tree)", "로지스틱 회귀 (Logistic Regression)"],
        horizontal=True,
    )

    if sim_model_name == "의사결정나무 (Decision Tree)":
        sim_model = DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=42
        )
    else:
        sim_model = LogisticRegression(class_weight="balanced", random_state=42)

    sim_model.fit(X_all, y_all)

    # [수정 반영 ③] session_state 초기화 및 원클릭 예시 자동 채우기 기능
    if "sim_hour" not in st.session_state:
        st.session_state.sim_hour = int(df["Hour"].median())
        st.session_state.sim_amount = 150.0
        st.session_state.sim_v1 = float(df["V1"].median())
        st.session_state.sim_v2 = float(df["V2"].median())
        st.session_state.sim_v3 = float(df["V3"].median())
        st.session_state.sim_v4 = float(df["V4"].median())
        st.session_state.sim_v5 = float(df["V5"].median())

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if (df["Class"] == 1).sum() > 0:
            if st.button("🚨 실제 사기 거래 예시 값 자동 입력"):
                fraud_sample = df[df["Class"] == 1].iloc[0]
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

        pred_class = sim_model.predict(input_df)[0]
        pred_proba = sim_model.predict_proba(input_df)[0][1] * 100

        st.subheader("2) AI 보안 시스템 진단 결과")
        if pred_class == 1 or pred_proba >= 50.0:
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
