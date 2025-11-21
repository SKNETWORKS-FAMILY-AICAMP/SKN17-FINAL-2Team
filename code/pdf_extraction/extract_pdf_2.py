import pdfplumber, json, os
import re

PDF_PATH = "./data/최신ICT시사용어202522.pdf"  
OUTPUT_JSON = "ict_terms_2.json"

def has_korean(text: str) -> bool:
    return any("가" <= ch <= "힣" for ch in text)

def is_english_line(text: str) -> bool:
    if not text:
        return False

    if has_korean(text):
        return False

    letters = [ch for ch in text if ch.isalpha()]
    return len(letters) >= max(3, len(text) * 0.3)


def extract_term_and_def(lines):
    # 공백 제거 + 빈 줄 제거
    lines = [ln.strip() for ln in lines if ln.strip()]

    for i in range(len(lines) - 3):
        if (lines[i] == "최신"
                and lines[i + 1] == "ICT"
                and lines[i + 2] == "시사용어"
                and "2025" in lines[i + 3]):
            header_idx = i + 3
            break
    else:
        return None, None  

    if header_idx + 1 >= len(lines):
        return None, None

    term = lines[header_idx + 1].strip()
    if not has_korean(term):
        return None, None

    idx = header_idx + 2

    if idx < len(lines) and is_english_line(lines[idx]):
        idx += 1

    # 정의문 수집
    def_lines = []
    while idx < len(lines):
        line = lines[idx]

        # 이미지 캡션 또는 다른 블록 시작으로 보이는 것들은 중단
        if "출처" in line:
            break
        if not has_korean(line):
            # 영어/기호만 있는 줄 나오면 정의 끝으로 간주
            break

        def_lines.append(line)
        idx += 1

        if len(def_lines) >= 3:
            break

    if not def_lines:
        return None, None

    definition = " ".join(def_lines)
    return term, definition


def extract_qa_from_pdf(pdf_path: str):
    qa_list = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue

            lines = text.splitlines()
            term, definition = extract_term_and_def(lines)

            if term and definition:
                question = f"{term}란 무엇인가?"
                qa_list.append({
                    "page": page_idx,
                    "question": question,
                    "answer": definition
                })
                print(f"[{page_idx}쪽] {term} → 추출 완료")

    return qa_list


if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {PDF_PATH}")

    qa_data = extract_qa_from_pdf(PDF_PATH)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)

    print(f"\n총 {len(qa_data)}개 용어를 추출했습니다.")
    print(f"결과가 {OUTPUT_JSON} 파일로 저장되었습니다.") 