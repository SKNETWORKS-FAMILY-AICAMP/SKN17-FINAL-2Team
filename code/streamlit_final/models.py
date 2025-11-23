# models.py
import streamlit as st
from typing import Dict, Any
import tempfile
import torch
import whisper
import json
import re
import os
from pyannote.audio import Pipeline
from datetime import timedelta
from dotenv import load_dotenv
from transformers import pipeline
from datetime import date
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel



# 토큰
load_dotenv()
token=os.getenv('HF_TOKEN')

# whisper + pyannote 모델로드
whisper_model = whisper.load_model("medium")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=token).to(torch.device("cuda"))

# 파튜 모델로드
adapter_name = "poketmon/ax-trained-model-v3-2"
base_name = "skt/A.X-4.0-Light"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    base_name,
    trust_remote_code=True,
    use_fast=False
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_name,
    torch_dtype="auto",
    trust_remote_code=True
).to(device) 

model = PeftModel.from_pretrained(
    base_model,
    adapter_name
).to(device)

def run_stt_diarization(audio_bytes: bytes):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    whisper_result = whisper_model.transcribe(tmp_path, language="ko")
    diarization_result = pipeline(tmp_path)
    annotation = diarization_result.speaker_diarization

    final_segments = []

    for ws in whisper_result["segments"]:
        w_start, w_end = ws["start"], ws["end"]

        best_speaker = None
        best_overlap = 0.0

        # diarization matching
        for item in annotation.itertracks(yield_label=True):
            if len(item) == 2:
                segment, speaker = item
            elif len(item) == 3:
                segment, _, speaker = item
            else:
                continue

            overlap = max(0, min(w_end, segment.end) - max(w_start, segment.start))

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        final_segments.append({
            "speaker": best_speaker or "UNKNOWN",
            "start": float(w_start),
            "end": float(w_end),
            "text": ws["text"].strip()
        })

    # ------------------------
    # STEP 2: 연속 화자 merge
    # ------------------------
    merged_segments = []
    for seg in final_segments:
        if merged_segments and merged_segments[-1]["speaker"] == seg["speaker"]:
            merged_segments[-1]["text"] += " " + seg["text"]
            merged_segments[-1]["end"] = seg["end"]  # end time 업데이트 (optional)
        else:
            merged_segments.append(seg)

    # formatted text 결과
    formatted_text = "\n".join(f"{seg['speaker']}: {seg['text']}" for seg in merged_segments)

    return {
        "full_text": formatted_text,
        "segments": merged_segments,
        "speakers": list({s['speaker'] for s in merged_segments}),
        "raw_transcript": whisper_result["text"]
    }




def run_meeting_llm() -> Dict[str, Any]:
    # 프롬프트에 들어갈 변수들 정의
    mapped_text_str = st.session_state.get("mapped_text_str", "")
    meeting = st.session_state.meetings[st.session_state.current_meeting_idx]
    title = meeting["title"]
    date_val = meeting["date"]
    record_id = f"{date_val}-{title}"
    
    # 프롬프트

    prompt = f"""
    [시스템]
    현재 날짜는 {date_val} 입니다. 이 날짜를 기준으로 'due_date'를 계산하십시오.
    '오늘' = {date_val}
    '내일' = (오늘 + 1일)
    '이번 주 금요일' = (오늘 기준 금요일 날짜)

    [지시]
    [입력 전문]만을 근거로 "summary", "agendas", "tasks" 필드를
    요청된 **JSON 형식**으로만 출력하라.
    다른 설명이나 텍스트, 마크다운(` ```json `)을 포함하지 말고, 순수한 JSON 객체 하나만 생성해야 한다.

    [입력 전문]
    {mapped_text_str}

    [출력 형식 (JSON)]
    아래 JSON 구조를 정확히 따르되, 실제 값은 전문 기반으로 채워라.

    {{
    "id": "{record_id}",
    "summary": {{
        "Who": "...",
        "What": "...",
        "When": "...",
        "Where": "...",
        "Why": "...",
        "How": "...",
        "How much": "...",
        "How many": "..."
    }},
    "agendas": [
        {{
        "title": "안건 1 제목",
        "description": "안건 1에 대한 모델의 설명"
        }},
        {{
        "title": "안건 2 제목",
        "description": "안건 2에 대한 모델의 설명"
        }}
    ],
    "tasks": [
        {{
        "description": "태스크 1 내용",
        "assignee": "담당자 이름",
        "due": "마감일 표현",
        "due_date": "YYYY-MM-DD"
        }}
    ]
    }}

    [규칙]
    1.  **전체 출력**: 반드시 유효한(Valid) JSON 객체 하나여야 한다.
    2.  `"summary"`: 전문의 핵심 내용을 5W3H 형식으로 표현한다
            * 전문의 핵심 내용을 **5W3H 형식(JSON)**으로만 채운다.
        * 각 항목은 전문을 기반으로 한 정보만 기입한다.
        * 전문에서 해당 항목의 내용이 확인되지 않으면 `null` 값을 넣는다.
        * 근거 없이 추론하거나 생성하지 않는다.
    3.  `"agendas"`:
        * 안건 리스트. 없으면 `[]`.
        * `"description"`: 안건 설명. 내용이 없으면 `null` 값을 사용한다.
    4.  `"tasks"`:
        * 태스크 리스트. 없으면 `[]`.
        * `"due"`: 전문에 나온 마감일 표현 (예: "오늘 오후", "다음 주 초"). 없으면 `*(언급 없음)*`.
        * `"due_date"`: **[시스템]의 {date_val}를 기준**으로 `"due"`의 날짜를 계산하여 "YYYY-MM-DD" 형식으로 기입. 계산이 불가능하거나 마감일 언급이 없으면 `null`.
    5.  **근거**: 전문에서 근거를 찾을 수 없는 정보는 절대 생성하지 않는다.

    위 규칙을 모두 지켜, 주어진 전문에 대한 JSON 결과만 출력하라.
    """

    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False
    )

    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(generated_text)
    raw_text = generated_text.strip()
    # 혹시 ```json ... ``` 이런 코드펜스가 붙어오면 제거
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.IGNORECASE | re.DOTALL).strip()

    print(raw_text)

        # JSON 파싱을 통해 필요한 값만 추출하기
    try:
        json_data = json.loads(raw_text)
        summary = json_data["summary"]
        agendas = json_data["agendas"]
        tasks = json_data["tasks"]
        
        # minutes (회의록) 생성
        minutes = create_minutes(summary, agendas, tasks)

        # 필요한 값만 리턴
        return {
            "summary": summary,
            "agendas": agendas,
            "tasks": tasks,
            "minutes": minutes,
            "raw": raw_text,  # 원문도 같이 저장해둠
            "parsed_ok": True
        }
    except Exception as e:
        print(f"Error parsing generated text: {e}")
        
        return {
            "summary": {},
            "agendas": [],
            "tasks": [],
            "minutes": "",
            "raw": raw_text,
            "parsed_ok": False
        }

def create_minutes(summary, agendas, tasks):
    # summary, agendas, tasks 정보를 종합하여 회의록 생성
    minutes = "\n"
    minutes += f"요약:\n{summary.get('Who', '')}가 {summary.get('What', '')}에 대해 논의했습니다.\n"
    minutes += f"일시: {summary.get('When', '')}\n"
    minutes += f"장소: {summary.get('Where', '')}\n"
    minutes += f"핵심 논의: {summary.get('Why', '')}\n"
    minutes += f"어떻게 진행되었는지: {summary.get('How', '')}\n"
    minutes += f"참석자 수: {summary.get('How many', '')}\n"
    
    minutes += "\n 안건:\n"
    for agenda in agendas:
        minutes += f"- {agenda['title']}: {agenda['description']}\n"

    minutes += "\n 태스크:\n"
    for task in tasks:
        minutes += f"- {task['description']} (담당자: {task['assignee']} - 마감일: {task['due_date']})\n"
    
    return minutes
    
