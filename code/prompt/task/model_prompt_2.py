import torch
import os
from dotenv import load_dotenv
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# === 프롬프트 ===

USER_TEMPLATE = (
    "아래 회의록을 읽고, 업무와 관련된 태스크만 모두 추출하라.\n"
    "각 태스크는 위에서 정의한 JSON 스키마에 맞게 작성하고, 최종 출력은 오직 JSON 배열 형태로만 반환하라.\n"
    "JSON 이외의 설명, 문장, 마크다운, 표는 절대 출력하지 말 것.\n\n"
    "회의록:\n{input_text}"
)

SYSTEM_SUMMARY = (
    "너는 한국어 회의록에서 태스크만 추출하는 도우미다.\n"
    "태스크는 반드시 다음 정보를 가진다:\n"
    "- 해야할일(what): 수행해야 하는 구체적인 업무\n"
    "- 대상(who): 업무를 수행해야 하는 담당자, 팀 또는 역할\n"
    "- 마감일(when): 구체적인 마감 시점. 없으면 빈 문자열 \"\" 로 둔다.\n\n"
    "규칙:\n"
    "1) 사실만 유지하고, 원문에 없는 새로운 정보를 만들지 말 것.\n"
    "2) IT 기업에서 실제로 발생할 법한 업무(Task)만 추출할 것.\n"
    "3) 태스크가 아닌 논의, 잡담, 회고성 발언은 포함하지 말 것.\n"
    "4) 출력은 반드시 JSON 배열 형식만 사용하고, 그 외 어떤 설명, 문장, 마크다운, 표도 출력하지 말 것.\n"
    "5) 태스크가 하나도 없으면 빈 배열 [] 만 출력할 것.\n\n"
    "6) 태스크를 추출할 때는 앞,뒤 대화 문맥을 포함하여 추출할 것.\n"
    "출력 형식(JSON 스키마):\n"
    "[\n"
    "  {\n"
    "    \"해야할일\": \"...\",\n"
    "    \"대상\": \"...\",\n"
    "    \"마감일\": \"...\",\n"
    "    \"원문\": \"태스크를 뽑아낸 근거가 된 문장 또는 구간\"\n"
    "  }\n"
    "]\n\n"
    "예시:\n"
    "[\n"
    "  {\n"
    "    \"해야할일\": \"프로파일링으로 주문 목록 API N+1 쿼리 제거\",\n"
    "    \"대상\": \"백엔드\",\n"
    "    \"마감일\": \"\",\n"
    "    \"원문\": \"주문 목록 API가 느린데 N+1 의심돼요. 백엔드에서 프로파일링하고 N+1 제거해 주세요.\"\n"
    "  },\n"
    "  {\n"
    "    \"해야할일\": \"웹 접근성 기준 준수: 키보드 포커스 스타일 보강\",\n"
    "    \"대상\": \"프런트엔드\",\n"
    "    \"마감일\": \"다음 주 수요일\",\n"
    "    \"원문\": \"접근성 지적 있었죠. 프런트엔드에서 키보드 포커스 스타일 보강해서 다음 주 수요일까지 반영해 주세요.\"\n"
    "  }\n"
    "]"
)


def build_prompt(text) :
    return f"{SYSTEM_SUMMARY}\n\n{USER_TEMPLATE.format(input_text=text)}"


# --- 텍스트 파일 읽기 ---
def load_text_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# --- 모델 로딩 -- 
def load_model():
    model_name = "skt/A.X-4.0-Light"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

    return pipe


if __name__ == "__main__":
    pipe = load_model()
    FILE_PATH = "./text100.txt"

    if not os.path.exists(FILE_PATH):
        print(f"[에러] 파일을 찾을 수 없습니다: {FILE_PATH}")
        exit()

    text = load_text_from_file(FILE_PATH)
    prompt = build_prompt(text)

        
    print("\n[알림] 태스크 추출을 시작합니다...\n")
    t0 = time.time()

    response = pipe(
        prompt,
        max_new_tokens= 1024
        # config={
        #     "max_new_tokens": 512,
        #     "min_new_tokens": 150,
        #     "length_penalty": 1.0,
        #     "temperature": 0.2,
        #     "top_p": 0.9,
        #     "repetition_penalty": 1.05,
        #     "do_sample": False,  # 요약 안정성↑
        # },
    )[0]['generated_text']

    elapsed = time.time() - t0

    print(f"[완료] 생성 시간: {elapsed:.2f}초\n")
    print("=== 태스크 추출 결과 ===\n", response, "\n")