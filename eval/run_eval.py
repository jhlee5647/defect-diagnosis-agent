"""시스템의 성적표를 내는 자동 채점기 — SPEC-06.

두 질문에 숫자로 답한다:
① 유사 사례 few-shot 주입이 진단 정답률을 올리나 — (a) VLM 단독 vs (b) RAG 주입 비교표
② 질문마다 도구를 제대로 골랐나 — UC 5종 trace를 통과 제약과 기계 대조

pytest가 아니라 완성된 시스템 전체의 배치 채점이다. 사람이 쓰는 화면은 demo(SPEC-05).
실행: uv run python eval/run_eval.py [--limit 5]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.schema import LabelDoc, primary_defect, to_crop_coords

# ── VQA 문항 파싱 (결정적 — R5) ───────────────────────────

# 라벨 JSON의 문항 키 접두어 ↔ 채점 표의 문항 유형
_QUESTION_KINDS = {
    "detection": "defect_detection_q",
    "classification": "defect_classification_q",
    "localization": "defect_localization_q",
    "analysis": "defect_analysis_q",
}
_LETTERS = ("a", "b", "c", "d")


@dataclass(frozen=True)
class Question:
    """4지선다(유무는 2지선다) 문항 1개. options는 {글자: 보기 본문}."""

    kind: str
    prompt: str
    options: dict[str, str]
    answer: str


@dataclass(frozen=True)
class QuestionResult:
    """문항 1개의 채점 결과. format_fail이면 correct는 항상 False (오답 처리 + 별도 집계)."""

    condition: str  # "a"(VLM 단독) | "b"(RAG 주입)
    kind: str
    correct: bool
    format_fail: bool


def parse_vqa(doc: LabelDoc) -> list[Question]:
    """라벨의 vision_qa → 문항 목록. 없는 문항(정상 사진의 위치·유형·특징)은 그냥 빠진다."""
    questions = []
    for kind, prefix in _QUESTION_KINDS.items():
        prompt = doc.vqa.get(prefix)
        if not prompt:
            continue
        options = {
            letter: doc.vqa[key]
            for letter in _LETTERS
            if (key := f"{prefix}_option_{letter}") in doc.vqa
        }
        questions.append(Question(
            kind=kind, prompt=prompt, options=options, answer=doc.vqa[f"{prefix}_a"]))
    return questions


def verify_localization_label(doc: LabelDoc) -> bool:
    """위치 문항의 정답 보기 좌표가 to_crop_coords(대표결함 bbox)와 일치하는지 검증.

    채점기가 틀린 정답지로 채점하면 모든 숫자가 무의미하다 (함정 2).
    불일치 문서의 위치 문항은 '라벨 오류'로 채점에서 제외하고 카운트한다 (게이트 합의).
    """
    prefix = _QUESTION_KINDS["localization"]
    answer_key = doc.vqa.get(f"{prefix}_a")
    raw_option = doc.vqa.get(f"{prefix}_option_{answer_key}")
    main = primary_defect(doc)
    if raw_option is None or main is None or doc.cropped_bbox is None:
        return False
    expected = to_crop_coords(main.bbox, doc.cropped_bbox)
    labeled = json.loads(raw_option)
    return all(abs(e - v) < 0.5 for e, v in zip(expected, labeled, strict=True))


# ── 모델 응답 파싱·집계 (결정적 — R5) ─────────────────────

_ANSWER_RE = re.compile(r"^[\s(\[]*([abcd])[\s)\].:]*$", re.IGNORECASE)


def parse_model_answer(text: str) -> str | None:
    """'a/b/c/d 한 글자' 형식 강제. 벗어나면 None → 오답 + 형식실패 집계 (§5)."""
    m = _ANSWER_RE.match(text or "")
    return m.group(1).lower() if m else None


def aggregate(results: list[QuestionResult]) -> dict:
    """조건×문항유형 (정답 수, 문항 수) 표 + 조건별 형식실패 수.

    문항이 없던 유형은 분모에 아예 잡히지 않는다 — 정상 사진의 위치·유형·특징 등.
    """
    table: dict = {"format_fails": {}}
    for r in results:
        cond = table.setdefault(r.condition, {})
        for key in (r.kind, "전체"):
            correct, total = cond.get(key, (0, 0))
            cond[key] = (correct + int(r.correct), total + 1)
        fails = table["format_fails"]
        fails[r.condition] = fails.get(r.condition, 0) + int(r.format_fail)
    for cond in (c for c in table if c != "format_fails"):
        table["format_fails"].setdefault(cond, 0)
    return table


def main():
    raise SystemExit("평가 하네스는 SPEC-06 사이클 2에서 구현")


if __name__ == "__main__":
    main()
