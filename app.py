
import os
from datetime import date
from decimal import Decimal

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from datetime import timedelta
import pandas as pd

# 1. 환경 변수 불러오기
load_dotenv()

API_URL = os.getenv(
    "FRANKFURTER_BASE_URL",
    "https://api.frankfurter.dev/v2"
).rstrip("/")

# 2. Frankfurter 환율 조회 Tool
@tool
def get_exchange_rate(
    base: str,
    target: str,
    amount: float = 1.0,
    on_date: str = ""
) -> str:
    """
    Frankfurter API에서 환율을 조회하고 금액을 환산한다.

    Args:
        base: 기준 통화 코드 (예: USD, KRW, EUR, JPY)
        target: 대상 통화 코드 (예: KRW, USD, JPY)
        amount: 환전할 금액 (기본값 1)
        on_date: 과거 조회 날짜 YYYY-MM-DD. 최신 환율이면 빈 문자열.
    """
    try:
        base = base.upper().strip()
        target = target.upper().strip()

        if len(base) != 3 or len(target) != 3:
            return "오류: 통화 코드는 USD, KRW처럼 3자리여야 합니다."

        if not Decimal(str(amount)).is_finite() or amount < 0:
            return "오류: 금액은 0 이상의 유한한 숫자여야 합니다."

        if on_date:
            requested_date = date.fromisoformat(on_date)

            if requested_date > date.today():
                return "오류: 미래 날짜의 환율은 조회할 수 없습니다."

        if base == target:
            return (
                f"동일 통화입니다. 환율은 1입니다. "
                f"{amount:,.2f} {base} = {amount:,.2f} {target}"
            )

        # 3. API 엔드포인트 설정
        url = f"{API_URL}/rate/{base}/{target}"
        params = {}

        if on_date:
            params["date"] = on_date

        # 4. API 요청
        response = requests.get(
            url,
            params=params,
            timeout=10
        )
        response.raise_for_status()

        # 5. 응답 데이터 처리
        data = response.json(parse_float=Decimal)

        rate = data["rate"]
        converted = Decimal(str(amount)) * rate

        return (
            f"환율 기준일: {data['date']}\n"
            f"기준 통화: {base}\n"
            f"대상 통화: {target}\n"
            f"적용 환율: 1 {base} = {rate} {target}\n"
            f"환전 전 금액: {amount:,.2f} {base}\n"
            f"환전 결과: {converted:,.2f} {target}\n"
            f"참고: 공식 기준 환율이며, 실제 환전 수수료는 미포함"
        )

    except ValueError:
        return "오류: 날짜는 YYYY-MM-DD 형식으로 입력해야 합니다."

    except requests.RequestException as e:
        return f"환율 API 호출에 실패했습니다: {str(e)}"

    except (KeyError, TypeError) as e:
        return f"환율 데이터 처리에 실패했습니다: {str(e)}"


# ==============================================
# 기간별 환율 조회 (API 데이터 캐싱)
# ==============================================
@st.cache_data(ttl=3600)
def fetch_rate_history(base, target, days=30):
    base = base.upper().strip()
    target = target.upper().strip()

    if base == target:
        raise ValueError("서로 다른 두 통화를 선택하세요.")

    if days not in (7, 30, 90, 180, 365):
        raise ValueError("지원하지 않는 조회 기간입니다.")

    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    response = requests.get(
        f"{API_URL}/rates",
        params={
            "base": base,
            "quotes": target,
            "from": start_date.isoformat(),
            "to": end_date.isoformat()
        },
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list) or not data:
        raise ValueError("조회 가능한 환율 데이터가 없습니다.")

    df = pd.DataFrame(data)

    df["date"] = pd.to_datetime(df["date"])
    df["rate"] = pd.to_numeric(df["rate"])

    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .set_index("date")
    )

    return df[["rate"]]


# ==============================================
# AI가 호출하는 기간별 환율 분석 Tool
# ==============================================
@tool
def analyze_exchange_trend(
    base: str = "USD",
    target: str = "KRW",
    days: int = 30
) -> str:
    """
    특정 기간의 환율 추이를 분석한다.

    Args:
        base: 기준 통화 코드 (예: USD)
        target: 대상 통화 코드 (예: KRW)
        days: 분석 기간. 7, 30, 90, 180, 365 중 선택.
    """
    try:
        df = fetch_rate_history(base, target, days)

        first = float(df["rate"].iloc[0])
        last = float(df["rate"].iloc[-1])

        high = float(df["rate"].max())
        low = float(df["rate"].min())
        avg = float(df["rate"].mean())

        change = (last - first) / first * 100

        high_date = df["rate"].idxmax().date()
        low_date = df["rate"].idxmin().date()

        return f"""
환율 추이 분석 결과

통화: {base.upper()} → {target.upper()}
요청 기간: 최근 {days}일
실제 데이터 시작일: {df.index[0].date()}
실제 데이터 종료일: {df.index[-1].date()}
관측 데이터 수: {len(df)}개

시작 환율: {first:,.4f}
마지막 환율: {last:,.4f}
최고 환율: {high:,.4f} ({high_date})
최저 환율: {low:,.4f} ({low_date})
평균 환율: {avg:,.4f}
기간 변동률: {change:+.2f}%

주의:
- 위 변동률은 첫 관측일 대비 마지막 관측일 변화율임.
- 환율은 일별 기준 환율이며 거래 시세가 아님.
- 향후 환율의 상승이나 하락을 보장하지 않음.
"""

    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        return f"환율 추이 분석에 실패했습니다: {e}"


# 6. AI 모델 및 Agent 생성
@st.cache_resource
def create_exchange_agent():
    model = init_chat_model("gpt-5.4-mini")

    agent = create_agent(
        model,
        tools=[get_exchange_rate, analyze_exchange_trend],
        system_prompt="""
당신은 친절하고 정확한 환율 조회 AI 챗봇입니다.

규칙:
1. 환율이나 환전 금액을 질문받으면
   반드시 get_exchange_rate Tool을 사용하세요.
2. Tool이 반환한 환율 데이터를 기반으로 답변하세요.
3. 환율을 추측하거나 임의로 만들어내지 마세요.
4. 한국어로 쉽고 친절하게 답변하세요.
5. 달러는 기본적으로 USD, 원화는 KRW,
   엔화는 JPY, 유로는 EUR로 해석하세요.
6. 사용자가 '100달러는 몇 원이야?'라고 질문하면
   base=USD, target=KRW, amount=100으로 호출하세요.
7. 질문에 대상 통화가 생략되면
   이전 대화의 문맥을 참고하세요.
   문맥이 없으면 한국 원화(KRW)를 기본값으로 사용하세요.
8. 오류가 발생하면 환율을 지어내지 말고
   조회에 실패했다고 설명하세요.
9. 환율 기준일과 수수료 미포함 사실을 안내하세요.
10. Frankfurter 환율은 일별 기준 환율이므로
    초 단위 실시간 거래 시세라고 설명하지 마세요.
11. 사용자가 최근 N일, 최근 한 달, 환율 추이,
    상승률, 하락률, 최고 환율, 최저 환율 등을 질문하면
    analyze_exchange_trend Tool을 사용하세요.
12. '최근 한 달'은 기본적으로 30일로 해석하세요.
13. Tool에서 계산한 값을 그대로 사용하고,
    실제 데이터에 없는 값은 만들어내지 마세요.
14. 과거 환율 변화와 미래 환율 전망을 구분하세요.
15. 과거 환율이 상승했다고 해서 앞으로도
    상승할 것이라고 단정하지 마세요.
"""
    )

    return agent


agent = create_exchange_agent()


# ==================================================
# 7. Streamlit 페이지 구성
# ==================================================
st.set_page_config(
    page_title="Exchange AI | 환율 어시스턴트",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 8. 커스텀 디자인 CSS
# ==================================================
st.markdown("""
<style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.7px;
    }

    /* 페이지 대표 타이틀 */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin-bottom: 8px;
        line-height: 1.25;
    }

    .hero-accent {
        color: #2563EB;
    }

    .hero-desc {
        color: #64748B;
        font-size: 1rem;
        margin-bottom: 20px;
    }

    /* 상단 환율 카드 */
    .currency-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 22px;
        border-radius: 16px;
        margin-bottom: 10px;
        min-height: 145px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.025);
    }

    .currency-label {
        color: #64748B;
        font-size: 0.85rem;
        margin-bottom: 12px;
    }

    .currency-title {
        color: #0F172A;
        font-size: 1.5rem;
        font-weight: 750;
        margin-bottom: 6px;
    }

    .currency-desc {
        color: #64748B;
        font-size: 0.8rem;
    }

    /* 버튼 스타일 */
    div.stButton > button {
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        border-color: #2563EB;
        color: #2563EB;
    }

    /* 채팅 메시지 */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 14px;
        margin-bottom: 8px;
    }

    /* 입력창 */
    [data-testid="stChatInput"] {
        border-radius: 14px;
    }

    /* 하단 안내 */
    .footer-text {
        color: #94A3B8;
        font-size: 0.8rem;
        text-align: center;
        padding-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# 9. 세션 상태 초기화
# ==================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ==================================================
# 10. 빠른 질문 처리 함수
# ==================================================
def set_quick_question(question):
    st.session_state.pending_prompt = question


def clear_chat():
    st.session_state.messages = []
    st.session_state.pending_prompt = None


# ==================================================
# 11. 사이드바
# ==================================================
with st.sidebar:
    st.title("💱 Exchange AI")
    st.caption("Your Smart Currency Assistant")

    st.divider()

    st.subheader("📌 서비스 소개")

    st.write(
        "AI를 활용해 세계 주요 통화의 환율을 "
        "자연어로 조회하고 환전 금액을 계산합니다."
    )

    st.info("Frankfurter API 기반 일별 기준 환율")

    st.divider()

    st.subheader("💡 질문 가이드")

    st.markdown("""
    **이렇게 질문해 보세요.**

    - 지금 1달러는 몇 원이야?
    - 100유로를 원화로 환전해 줘.
    - 일본 엔화 환율 알려줘.
    - 2025년 10월 1일 달러 환율은?
    - 그럼 50달러는?
    """)

    st.divider()

    st.button(
        "🗑️ 새로운 대화 시작",
        on_click=clear_chat,
        use_container_width=True
    )

    st.caption("Powered by LangChain & OpenAI")


# ==================================================
# 12. 메인 타이틀
# ==================================================
st.markdown("""
<div class="hero-title">
    당신의 스마트한<br>
    <span class="hero-accent">AI 환율 어시스턴트</span>
</div>

<div class="hero-desc">
    복잡한 환율 계산도 간편하게.
    궁금한 환율을 자연어로 질문해 보세요.
</div>
""", unsafe_allow_html=True)

st.divider()



# ==============================================
# 13. 실시간 조회형 주요 환율 카드
# ==============================================
st.subheader("🌍 주요 환율 현황")

pairs = [
    ("🇺🇸 미국 달러", "USD"),
    ("🇯🇵 일본 엔", "JPY"),
    ("🇪🇺 유로", "EUR")
]

cols = st.columns(3)

for col, (label, currency) in zip(cols, pairs):
    with col:
        try:
            history = fetch_rate_history(
                currency, "KRW", days=7
            )

            latest = float(history["rate"].iloc[-1])
            latest_date = history.index[-1].date()

            if len(history) >= 2:
                previous = float(history["rate"].iloc[-2])
                change_pct = (
                    (latest - previous) / previous * 100
                )
                delta_text = f"{change_pct:+.2f}%"
                delta_desc = "직전 관측일 대비"
            else:
                delta_text = None
                delta_desc = ""

            st.metric(
                label=f"{label} ({currency}/KRW)",
                value=f"₩ {latest:,.2f}",
                delta=delta_text,
                delta_color="normal",
                help=delta_desc
            )

            st.caption(f"기준일: {latest_date}")

        except (requests.RequestException, ValueError, KeyError) as e:
            st.warning(f"{currency} 환율 조회 실패")
            st.caption(str(e))

st.caption(
    "환율은 기준 통화 1단위당 원화 금액입니다. "
    "변동률은 직전 관측일 대비입니다."
)



# ==================================================
# 14. 빠른 질문 버튼
# ==================================================
st.subheader("✨ 빠른 질문")

q1, q2, q3 = st.columns(3)

with q1:
    st.button(
        "💵 100달러는 몇 원?",
        on_click=set_quick_question,
        args=("지금 100달러는 몇 원이야?",),
        use_container_width=True
    )

with q2:
    st.button(
        "💴 1만 엔은 몇 원?",
        on_click=set_quick_question,
        args=("지금 10000엔은 몇 원이야?",),
        use_container_width=True
    )

with q3:
    st.button(
        "💶 50유로는 몇 원?",
        on_click=set_quick_question,
        args=("지금 50유로는 몇 원이야?",),
        use_container_width=True
    )

st.divider()



# ==============================================
# 환율 추이 시각화
# ==============================================
st.divider()
st.subheader("📈 환율 추이 분석")

st.caption(
    "관심 있는 통화와 기간을 선택하면 "
    "실제 환율 데이터를 시각화합니다."
)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    chart_currency = st.selectbox(
        "분석할 통화",
        ["USD", "JPY", "EUR", "CNY", "GBP"],
        format_func=lambda x: {
            "USD": "🇺🇸 미국 달러",
            "JPY": "🇯🇵 일본 엔",
            "EUR": "🇪🇺 유로",
            "CNY": "🇨🇳 중국 위안",
            "GBP": "🇬🇧 영국 파운드"
        }[x]
    )

with chart_col2:
    chart_period = st.selectbox(
        "조회 기간",
        [7, 30, 90, 180, 365],
        index=1,
        format_func=lambda x: f"최근 {x}일"
    )

try:
    chart_data = fetch_rate_history(
        chart_currency,
        "KRW",
        chart_period
    )

    # 환율 그래프
    st.line_chart(
        chart_data,
        y="rate",
        x_label="날짜",
        y_label=f"{chart_currency}/KRW",
        height=340
    )

    # 통계 요약
    latest_rate = chart_data["rate"].iloc[-1]
    avg_rate = chart_data["rate"].mean()
    highest_rate = chart_data["rate"].max()
    lowest_rate = chart_data["rate"].min()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("최근 환율", f"{latest_rate:,.2f}")
    c2.metric("평균 환율", f"{avg_rate:,.2f}")
    c3.metric("최고 환율", f"{highest_rate:,.2f}")
    c4.metric("최저 환율", f"{lowest_rate:,.2f}")

    st.caption(
        f"데이터 기간: "
        f"{chart_data.index[0].date()} ~ "
        f"{chart_data.index[-1].date()}"
    )

    # 데이터 내려받기
    csv_data = chart_data.to_csv().encode("utf-8-sig")

    st.download_button(
        label="📥 환율 데이터 CSV 다운로드",
        data=csv_data,
        file_name=f"{chart_currency}_KRW_{chart_period}days.csv",
        mime="text/csv"
    )

except (requests.RequestException, ValueError, KeyError) as e:
    st.error(f"차트 데이터를 가져오지 못했습니다: {e}")


# ==================================================
# 15. 채팅 영역
# ==================================================
st.subheader("💬 AI 환율 상담")

# 대화가 없는 경우 환영 메시지
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="💱"):
        st.markdown("""
        안녕하세요! 👋

        저는 **Exchange AI**입니다.

        세계 주요 통화의 최신 제공 환율과
        환전 금액을 확인해 드릴 수 있어요.

        위의 빠른 질문을 클릭하거나,
        아래 입력창에 자유롭게 질문해 보세요!
        """)


# 기존 대화 표시
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "💱"

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):
        st.write(message["content"])


# ==================================================
# 16. 질문 입력
# ==================================================
user_input = st.chat_input(
    "예: 지금 100달러는 몇 원이야?"
)

# 빠른 질문 버튼 우선 처리
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
else:
    prompt = user_input


# ==================================================
# 17. Agent 실행 및 응답 출력
# ==================================================
if prompt:

    # 사용자 메시지 저장
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user", avatar="👤"):
        st.write(prompt)

    with st.chat_message("assistant", avatar="💱"):

        with st.spinner("🔎 최신 환율을 확인하는 중..."):

            try:
                response = agent.invoke({
                    "messages": st.session_state.messages
                })

                answer = response["messages"][-1].content

                if not isinstance(answer, str):
                    answer = str(answer)

                st.markdown(answer)

            except Exception as e:
                answer = (
                    "죄송합니다. 환율 조회 중 "
                    "문제가 발생했습니다."
                )
                st.error(answer)
                # 자세한 오류는 서버 터미널에 출력
                print("Agent error:", repr(e))

        # AI 답변 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


# ==================================================
# 18. Footer
# ==================================================
st.markdown("""
<div class="footer-text">
    Exchange AI · Powered by OpenAI, LangChain & Frankfurter API<br>
    환율 정보는 참고용이며 실제 환전 환율과 차이가 있을 수 있습니다.
</div>
""", unsafe_allow_html=True)
