
import os
from datetime import date
from decimal import Decimal

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool

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


# 6. AI 모델 및 Agent 생성
@st.cache_resource
def create_exchange_agent():
    model = init_chat_model("gpt-5.4-mini")

    agent = create_agent(
        model,
        tools=[get_exchange_rate],
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


# ==================================================
# 13. 주요 통화 카드
# ==================================================
st.subheader("🌍 주요 통화 살펴보기")

col1, col2, col3 = st.columns(3)

currency_cards = [
    ("🇺🇸 미국 달러", "USD / KRW", "미국 달러 → 한국 원"),
    ("🇯🇵 일본 엔", "JPY / KRW", "일본 엔 → 한국 원"),
    ("🇪🇺 유로", "EUR / KRW", "유로 → 한국 원")
]

for col, (label, title, desc) in zip(
    [col1, col2, col3],
    currency_cards
):
    with col:
        st.markdown(f"""
        <div class="currency-card">
            <div class="currency-label">{label}</div>
            <div class="currency-title">{title}</div>
            <div class="currency-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)


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
