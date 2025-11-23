# app.py
import streamlit as st
from datetime import date
from common import init_session_state

st.set_page_config(
    page_title="말하는대로 · 회의록 서비스",
    layout="wide",
)

# 세션 초기화
init_session_state()


def login_view():
    st.title("말하는대로 · 로그인")

    email = st.text_input("Email")
    pw = st.text_input("Password", type="password")

    if st.button("Sign In", use_container_width=True):
        # TODO: 실제 인증 연동은 추후 구현
        st.session_state.logged_in = True
        st.success("로그인 되었습니다.")
        st.rerun()


def my_home_view():
    st.title("My Home")

    st.markdown("### 최근 회의")
    if not st.session_state.meetings:
        st.info("아직 생성된 회의가 없습니다. 좌측 sidebar에서 '01_Meeting_Home' 페이지로 이동해 회의를 생성해보세요.")
    else:
        # 최근 회의 3개 정도만
        for idx, m in reversed(list(enumerate(st.session_state.meetings[-3:]))):
            with st.expander(f"{m['date']} · {m['title']}"):
                st.write(f"장소: {m['place'] or '-'}")
                st.write(f"참석자: {', '.join(m['attendee']) if m['attendee'] else '-'}")
                if idx in st.session_state.meeting_results:
                    st.caption("분석 완료된 회의입니다. '01_Meeting_Home'에서 회의록을 자세히 볼 수 있습니다.")

    st.markdown("---")
    st.markdown("### To-do · Google Calendar (데모)")

    label = st.text_input("Label", key="todo_label_main")
    desc = st.text_input("Description", key="todo_desc_main")

    if st.button("To-do 추가", use_container_width=True):
        if label or desc:
            st.session_state.todos.append({"label": label, "desc": desc})

    if st.session_state.todos:
        st.write("#### 나의 To-do")
        for t in st.session_state.todos:
            st.write(f"- **{t['label']}**: {t['desc']}")

    st.markdown("---")
    st.caption("※ 실제 회의 생성/분석/회의록 기능은 좌측 sidebar의 '01_Meeting_Home' 페이지에서 이용할 수 있습니다.")


def main():
    # 간단한 상단 바 (로그아웃)
    cols = st.columns([6, 1])
    with cols[0]:
        st.empty()
    with cols[1]:
        if st.session_state.logged_in and st.button("로그아웃"):
            st.session_state.logged_in = False
            st.rerun()

    if not st.session_state.logged_in:
        login_view()
    else:
        my_home_view()


if __name__ == "__main__":
    main()


if st.button("📌 Print Session to Terminal"):
    print("=== SESSION STATE ===")
    print(st.session_state)