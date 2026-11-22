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
    "V2": "거래
