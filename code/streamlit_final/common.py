# common.py
import streamlit as st

def init_session_state():
    defaults = {
        "logged_in": False,          # 로그인 여부
        "meetings": [],              # 회의 리스트
        "current_meeting_idx": None, # 선택된 회의 인덱스
        "meeting_results": {},       # {idx: {"stt": ..., "llm": ...}}
        "todos": [],                 # To-do 리스트
        "global_speaker_alias": {}   # 스피커 자동 매핑 저장
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

