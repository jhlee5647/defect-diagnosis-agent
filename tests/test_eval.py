"""SPEC-06 — 채점 로직 테스트 (R5: 채점기가 틀리면 모든 숫자가 무의미).

결정적 부분만 검증한다: VQA 파싱, 위치 정답 정합, 응답 파싱, 집계.
VLM 호출·평가 실행은 스모크(--limit 5) 담당.
"""

import dataclasses

from eval.run_eval import (
    QuestionResult,
    aggregate,
    parse_model_answer,
    parse_vqa,
    verify_localization_label,
)
from src.config import STUB_DATA_DIR
from src.schema import parse_label

DEFECT_DOC = parse_label(STUB_DATA_DIR / "2025_sungsan_5_A_LeadingEdge_001.json")
NORMAL_DOC = parse_label(STUB_DATA_DIR / "2025_sungsan_2_C_PressureSide_005.json")


# ── 사이클 1: 채점 코어 ──────────────────────────────────


def test_parse_vqa_extracts_four_questions_from_defect_photo():
    """1. 결함 사진 → 유무·유형·위치·특징 4문항, 보기와 정답 포함."""
    questions = parse_vqa(DEFECT_DOC)
    kinds = {q.kind for q in questions}
    assert kinds == {"detection", "classification", "localization", "analysis"}

    detection = next(q for q in questions if q.kind == "detection")
    assert set(detection.options) == {"a", "b"}  # 유무는 2지선다
    assert detection.answer == "a"

    classification = next(q for q in questions if q.kind == "classification")
    assert set(classification.options) == {"a", "b", "c", "d"}
    assert classification.options[classification.answer] == "Paint Damage"
    assert all(q.prompt for q in questions)


def test_parse_vqa_normal_photo_has_detection_only():
    """2. 정상 사진 → 유무 문항 1개만 (위치·유형·특징은 분모에서 제외)."""
    questions = parse_vqa(NORMAL_DOC)
    assert [q.kind for q in questions] == ["detection"]
    assert questions[0].answer == "b"  # "아니요"


def test_localization_answer_matches_shared_coord_transform():
    """3. 위치 정답 보기 좌표 == to_crop_coords(대표결함 bbox) — 공유 구현 정합 (함정 2)."""
    assert verify_localization_label(DEFECT_DOC) is True

    # 정답 키를 다른 보기로 조작 → 정합 실패를 감지해야 함
    tampered_vqa = {**DEFECT_DOC.vqa, "defect_localization_q_a": "b"}
    tampered = dataclasses.replace(DEFECT_DOC, vqa=tampered_vqa)
    assert verify_localization_label(tampered) is False


def test_parse_model_answer_accepts_letter_variants_only():
    """4. 'a'/'A'/' b)'/'(c)' 는 파싱, 형식 밖('정답은 아마도 a')은 None(형식실패)."""
    assert parse_model_answer("a") == "a"
    assert parse_model_answer("A") == "a"
    assert parse_model_answer(" b) ") == "b"
    assert parse_model_answer("(c)") == "c"
    assert parse_model_answer("d.") == "d"
    assert parse_model_answer("정답은 아마도 a") is None
    assert parse_model_answer("") is None
    assert parse_model_answer("e") is None


def test_aggregate_by_condition_and_kind_with_format_fails():
    """5. 조건×문항유형 정답률 집계 + 형식실패는 오답이면서 별도 카운트."""
    results = [
        QuestionResult(condition="a", kind="detection", correct=True, format_fail=False),
        QuestionResult(condition="a", kind="detection", correct=False, format_fail=True),
        QuestionResult(condition="b", kind="detection", correct=True, format_fail=False),
        QuestionResult(condition="b", kind="classification", correct=False, format_fail=False),
    ]
    table = aggregate(results)
    assert table["a"]["detection"] == (1, 2)      # (정답 수, 문항 수)
    assert table["b"]["detection"] == (1, 1)
    assert table["b"]["classification"] == (0, 1)
    assert table["a"]["전체"] == (1, 2) and table["b"]["전체"] == (1, 2)
    assert table["format_fails"] == {"a": 1, "b": 0}


def test_aggregate_denominator_excludes_absent_kinds():
    """6. 정상 사진처럼 문항이 없는 유형은 해당 조건 분모에 아예 안 잡힌다."""
    results = [
        QuestionResult(condition="a", kind="detection", correct=True, format_fail=False),
    ]
    table = aggregate(results)
    assert "localization" not in table["a"]
    assert table["a"]["전체"] == (1, 1)
