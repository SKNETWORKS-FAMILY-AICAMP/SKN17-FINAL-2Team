# models.py
from typing import Dict, Any
import tempfile
import torch
import whisper
import os
from pyannote.audio import Pipeline
from datetime import timedelta
from dotenv import load_dotenv


# 토큰
load_dotenv()
token=os.getenv('HF_TOKEN')

# 모델로드
whisper_model = whisper.load_model("medium")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=token).to(torch.device("cuda"))



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






def run_meeting_llm(transcript: str) -> Dict[str, Any]:
    """
    실제에서는 sLLM 호출해서
    요약, 안건, 태스크, 회의록을 생성하면 됨.
    여기서는 데모용 더미 데이터 반환.
    """
    summary = (
        "이번 회의에서는 AI 회의록 서비스의 타겟 사용자와 핵심 기능을 정의하였다. "
        "내부 임직원과 외부 파트너를 1차 타겟으로 설정하였고, "
        "Whisper 기반 STT와 sLLM 기반 요약/태스크 추출 파이프라인을 사용하기로 합의했다. "
        "또한 회의록을 바로 액션 아이템과 캘린더 연동에 활용하는 UX 방향성을 확인하였다."
    )

    agendas = [
        {
            "title": "타겟 사용자 정의",
            "detail": "내부 임직원과 외부 파트너를 1차 타겟으로 설정하고, 추후 확장 방안을 논의."
        },
        {
            "title": "STT 및 LLM 기술 스택",
            "detail": "Whisper + Pyannote 조합으로 전사 및 화자 분리, sLLM으로 요약/태스크/회의록 생성."
        },
        {
            "title": "UX 플로우",
            "detail": "회의 생성 → 음성 업로드 → 자동 분석 → 회의록/태스크/캘린더 연동 흐름을 확정."
        },
    ]

    tasks = [
        {"who": "재은", "what": "Streamlit UI 시안 다듬기 및 피그마와 싱크 맞추기", "when": "2025-11-20"},
        {"who": "길진", "what": "Whisper + Pyannote 파이프라인 프로토타입 구현", "when": "2025-11-18"},
        {"who": "송이", "what": "중간발표용 시나리오 및 데모 스크립트 작성", "when": "2025-11-19"},
    ]

    minutes = (
        "1. 타겟 사용자 정의\n"
        "- 내부 임직원과 외부 파트너를 1차 타겟으로 설정하였다.\n"
        "- 초기에는 사내 회의 중심으로 파일럿을 진행하고, 이후 외부 고객사로 확장하기로 하였다.\n\n"
        "2. STT 및 LLM 기술 스택\n"
        "- Whisper를 사용하여 음성을 텍스트로 전사하기로 결정하였다.\n"
        "- Pyannote를 활용해 화자 분리를 수행하고, 화자 태깅 정보를 프론트에서 매핑한다.\n"
        "- sLLM을 사용해 요약, 세부 안건, 태스크, 회의록을 한 번에 생성하는 구조를 채택하였다.\n\n"
        "3. UX 플로우\n"
        "- 사용자는 웹에서 회의를 생성한 후, 회의 종료 후 녹음 파일을 업로드한다.\n"
        "- 시스템은 자동으로 전사 및 분석을 수행하고, 결과를 Summary/Task/Full Text/회의록 탭으로 제공한다.\n"
        "- 태스크는 캘린더 및 To-do와 연동될 수 있도록 구조화한다.\n"
    )

    return {
        "summary": summary,
        "agendas": agendas,
        "tasks": tasks,
        "minutes": minutes,
    }
