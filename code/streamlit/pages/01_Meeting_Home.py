# pages/01_Meeting_Home.py
import streamlit as st
from datetime import date
from common import init_session_state
from models import run_stt_diarization, run_meeting_llm

init_session_state()

if "meeting_page" not in st.session_state:
    st.session_state.meeting_page = "list"


def require_login():
    if not st.session_state.logged_in:
        st.error("이 페이지를 사용하려면 먼저 메인 페이지에서 로그인해야 합니다.")
        st.stop()


def view_meeting_list():
    st.title("Meeting Home")

    if st.button("새 회의 시작하기", use_container_width=True):
        st.session_state.meeting_page = "create"
        st.rerun()

    st.markdown("### 회의 목록")
    if not st.session_state.meetings:
        st.info("생성된 회의가 없습니다. '새 회의 시작하기'를 눌러 회의를 추가하세요.")
        return

    for idx, m in enumerate(st.session_state.meetings):
        cols = st.columns([4, 2, 3, 2])
        with cols[0]:
            st.write(f"{m['date']} · **{m['title']}**")
        with cols[1]:
            st.write(m["place"] or "-")
        with cols[2]:
            st.write(", ".join(m["attendee"]) if m["attendee"] else "-")
        with cols[3]:
            if st.button("열기", key=f"list_open_{idx}"):
                st.session_state.current_meeting_idx = idx
                # 분석 여부에 따라 어디로 갈지
                if idx in st.session_state.meeting_results:
                    st.session_state.meeting_page = "result"
                else:
                    st.session_state.meeting_page = "detail"
                st.rerun()


def view_create_meeting():
    st.title("새 회의 만들기")

    title = st.text_input("Title (한글, 영 대소문자, 숫자 30자 이내)")
    date_val = st.date_input("Date", value=date.today())
    place = st.text_input("Place (한글, 영 대소문자, 숫자 30자 이내)")

    attendee_options = [
        "전상아 (인사팀)",
        "임길진 (IT팀)",
        "이재은 (디자인팀)",
        "양송이 (홍보팀)",
        "조해리 (마케팅팀)",
        "김수현 (본부)",
    ]
    attendee = st.multiselect("Attendee", attendee_options)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Create", use_container_width=True):
            if not title:
                st.warning("Title은 필수입니다.")
            else:
                meeting = {
                    "title": title,
                    "date": str(date_val),
                    "place": place,
                    "attendee": attendee,
                }
                st.session_state.meetings.append(meeting)
                st.success("회의가 생성되었습니다.")
                st.session_state.meeting_page = "list"
                st.rerun()
    with col2:
        if st.button("취소", use_container_width=True):
            st.session_state.meeting_page = "list"
            st.rerun()


def view_meeting_detail():
    idx = st.session_state.current_meeting_idx
    if idx is None or idx >= len(st.session_state.meetings):
        st.error("선택된 회의가 없습니다.")
        if st.button("회의 목록으로 돌아가기"):
            st.session_state.meeting_page = "list"
            st.rerun()
        return

    m = st.session_state.meetings[idx]
    st.title(m["title"])
    st.write(f"일시: {m['date']}")
    st.write(f"장소: {m['place'] or '-'}")
    st.write(f"참석자: {', '.join(m['attendee']) if m['attendee'] else '-'}")

    st.markdown("---")
    st.subheader("회의 녹음 업로드")

    audio_file = st.file_uploader("회의 녹음 파일 업로드 (wav)", type=["wav"], key="audio_uploader")

    if audio_file is not None:
        st.info("파일이 업로드되었습니다. '음성 변환 및 회의 분석' 버튼을 눌러주세요.")
        if st.button("음성 변환 및 회의 분석", use_container_width=True):
            audio_bytes = audio_file.read()
            with st.spinner("음성 변환(STT + 화자 분리) 중입니다..."):
                stt_result = run_stt_diarization(audio_bytes)

            with st.spinner("회의 요약/태스크/회의록 생성 중입니다..."):
                llm_result = run_meeting_llm(stt_result["full_text"])

            # 결과 저장
            st.session_state.meeting_results[idx] = {
                "stt": stt_result,
                "llm": llm_result,
            }

            st.success("회의 분석이 완료되었습니다.")
            st.session_state.meeting_page = "result"
            st.rerun()

    st.markdown("---")
    if st.button("회의 목록으로 돌아가기", use_container_width=True):
        st.session_state.meeting_page = "list"
        st.rerun()


def view_meeting_result():
    idx = st.session_state.current_meeting_idx
    if idx is None or idx not in st.session_state.meeting_results:
        st.error("분석 결과가 있는 회의가 없습니다.")
        if st.button("회의 목록으로 돌아가기"):
            st.session_state.meeting_page = "list"
            st.rerun()
        return

    m = st.session_state.meetings[idx]
    result = st.session_state.meeting_results[idx]

    st.title(m["title"])
    st.write(f"일시: {m['date']}")
    st.write(f"장소: {m['place'] or '-'}")
    st.write(f"참석자: {', '.join(m['attendee']) if m['attendee'] else '-'}")

    st.markdown("---")

    # --------- 화자 매핑 ---------
    if result and "stt" in result and result["stt"].get("segments"):
        st.markdown("### 화자 이름 매핑")

        detected_speakers = list({seg['speaker'] for seg in result["stt"]["segments"]})
        attendee_list = m["attendee"]

        # speaker_map 존재여부 체크
        if "speaker_map" not in st.session_state.meeting_results[idx]:
            st.session_state.meeting_results[idx]["speaker_map"] = {}

        speaker_map = st.session_state.meeting_results[idx]["speaker_map"]

        # 전역 alias가 없다면 초기화
        if "global_speaker_alias" not in st.session_state:
            st.session_state.global_speaker_alias = {}

        temp_selection = {}

        for sp in detected_speakers:
            default_value = speaker_map.get(sp, st.session_state.global_speaker_alias.get(sp))
            options = ["(미지정)"] + attendee_list

            selected = st.selectbox(
                f"{sp}",
                options=options,
                index=options.index(default_value) if default_value in attendee_list else 0,
                key=f"speaker_select_{sp}"
            )

            temp_selection[sp] = selected

        # ---- 저장 버튼 추가 ----
        if st.button("저장하기", use_container_width=True):
            for sp, selected in temp_selection.items():
                if selected != "(미지정)":
                    speaker_map[sp] = selected
                    st.session_state.global_speaker_alias[sp] = selected  # 다른 회의에도 자동 적용

            st.session_state.meeting_results[idx]["speaker_map"] = speaker_map
            st.success("화자 매핑이 저장되었습니다.")  # <- 저장 후 메시지 뜸

        st.markdown("---")

    else:
        st.info("음성 분석 후에 화자 매핑을 할 수 있습니다.")

    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Task", "Full Text", "회의록"])

    with tab1:
        st.markdown("### 요약")
        st.write(result["llm"]["summary"])

        st.markdown("### 세부 안건")
        for ag in result["llm"]["agendas"]:
            st.markdown(f"- **{ag['title']}**: {ag['detail']}")

    with tab2:
        st.markdown("### 태스크")
        tasks = result["llm"]["tasks"]
        if not tasks:
            st.info("태스크가 없습니다.")
        else:
            for t in tasks:
                st.write(f"- **{t['who']}**: {t['what']} (기한: {t['when']})")

    with tab3:
        mapped_text = []
        for seg in result["stt"]["segments"]:
            real_name = result["speaker_map"].get(seg["speaker"], seg["speaker"])
            mapped_text.append(f"{real_name}: {seg['text']}")

        st.text("\n".join(mapped_text))

    with tab4:
        st.markdown("### 회의록")
        st.text(result["llm"]["minutes"])

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.button("pdf 다운로드 (향후 구현)", use_container_width=True)
        with col2:
            st.button("docs 다운로드 (향후 구현)", use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("회의 목록으로 돌아가기", use_container_width=True):
            st.session_state.meeting_page = "list"
            st.rerun()
    with col2:
        if st.button("다른 회의 선택", use_container_width=True):
            st.session_state.meeting_page = "list"
            st.session_state.current_meeting_idx = None
            st.rerun()



# --------- 메인 실행 ---------
def main():
    require_login()

    st.caption("My Home → (메인), Meeting Home → (현재 페이지)")

    page = st.session_state.meeting_page

    if page == "list":
        view_meeting_list()
    elif page == "create":
        view_create_meeting()
    elif page == "detail":
        view_meeting_detail()
    elif page == "result":
        view_meeting_result()
    else:
        view_meeting_list()


if __name__ == "__main__":
    main()
