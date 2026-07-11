"""SPEC-06 — 채점 로직 테스트 (R5: 채점기가 틀리면 모든 숫자가 무의미).

결정적 부분만 검증한다: VQA 파싱, 위치 정답 정합, 응답 파싱, 집계, 하네스 가드.
VLM 호출·평가 실행은 스모크(--limit 5) 담당.
"""

import dataclasses

import pytest

import chromadb

from eval.run_eval import (
    QuestionResult,
    aggregate,
    assert_no_leak,
    build_fewshot_block,
    build_vqa_prompt,
    check_routing,
    estimate_calls,
    evaluate_photo,
    load_eval_docs,
    parse_model_answer,
    parse_vqa,
    render_report,
    verify_localization_label,
)
from src.config import STUB_DATA_DIR, V1_COLLECTION, V2_COLLECTION
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
    tampered_vqa = {**DEFECT_DOC.vqa, "defect_localization_a": "b"}
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


# ── 사이클 2: 하네스 결정 부분 ───────────────────────────


def test_leak_precheck_aborts_before_grading(tmp_path):
    """7. R1: 시험지 파일이 V1에 있으면 채점 시작 전 RuntimeError로 중단."""
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    v1 = client.create_collection(V1_COLLECTION, metadata={"hnsw:space": "cosine"})
    v1.add(ids=["2025_leak_1_A_LeadingEdge_001.jpg"], embeddings=[[0.1] * 8],
           documents=["몰래 들어간 시험지"])
    client.create_collection(V2_COLLECTION, metadata={"hnsw:space": "cosine"})

    with pytest.raises(RuntimeError, match="누수"):
        assert_no_leak(["2025_leak_1_A_LeadingEdge_001.jpg"], tmp_path / "chroma")

    # 깨끗한 상태면 통과
    assert_no_leak(["2025_clean_1_A_LeadingEdge_002.jpg"], tmp_path / "chroma")


def test_estimate_calls_counts_questions_twice():
    """8. R6: 예상 VLM 호출 수 = Σ(사진별 문항 수) × 2조건 — 001은 4문항, 005는 1문항."""
    assert estimate_calls([DEFECT_DOC, NORMAL_DOC]) == (4 + 1) * 2


def test_load_eval_docs_respects_limit(tmp_path):
    """9. R6: --limit 5 → 시험지 목록 앞 5장만 로드."""
    testset = tmp_path / "testset.txt"
    names = sorted(p.name for p in STUB_DATA_DIR.glob("*.jpg"))
    testset.write_text("\n".join(names) + "\n", encoding="utf-8")

    docs = load_eval_docs(STUB_DATA_DIR, testset, limit=5)
    assert len(docs) == 5
    assert len(load_eval_docs(STUB_DATA_DIR, testset, limit=None)) == 6


def test_eval_prompt_never_contains_eval_filename():
    """10. R2: (b) 프롬프트에 평가 사진의 파일명 미포함 — D1 우회 컨닝 차단. 과거 사례 파일명은 허용."""
    fewshot = build_fewshot_block([
        {"filename": "2024_other_1_A_LeadingEdge_009.jpg", "defect_type": "Paint Damage",
         "severity": 2, "similarity": 0.8, "description": "도장 박리 사례"}])
    question = parse_vqa(DEFECT_DOC)[0]
    prompt = build_vqa_prompt(question, fewshot)

    assert DEFECT_DOC.filename not in prompt
    assert "2024_other_1_A_LeadingEdge_009.jpg" in prompt
    assert "한 글자" in prompt  # 응답 형식 강제 문구


def test_no_visual_results_runs_without_fewshot():
    """11. visual_search 0건 → (b)를 few-shot 없이 진행 + 별도 카운트."""
    counters: dict = {}
    prompts = []

    def fake_asker(image, prompt):
        prompts.append(prompt)
        return "a"

    def empty_searcher(jpg_path, crop):
        return {"results": [], "count": 0}

    results = evaluate_photo(
        NORMAL_DOC, STUB_DATA_DIR / "2025_sungsan_2_C_PressureSide_005.jpg",
        asker=fake_asker, searcher=empty_searcher, counters=counters)

    assert counters["no_fewshot"] == 1
    assert {r.condition for r in results} == {"a", "b"}
    assert len(results) == 2  # 정상 사진: 유무 문항 × 2조건


def test_missing_cropped_bbox_falls_back_to_thumbnail():
    """12. cropped_bbox 없는 문서 → 긴 변 축소 폴백(SPEC-03 R5와 동일 규칙) + 별도 카운트."""
    counters: dict = {}
    sizes = []

    def fake_asker(image, prompt):
        sizes.append(image.size)
        return "b"

    doc = dataclasses.replace(NORMAL_DOC, cropped_bbox=None,
                              vqa={k: v for k, v in NORMAL_DOC.vqa.items()
                                   if k != "cropped_bbox"})
    evaluate_photo(doc, STUB_DATA_DIR / "2025_sungsan_2_C_PressureSide_005.jpg",
                   asker=fake_asker,
                   searcher=lambda jpg, crop: {"results": [], "count": 0},
                   counters=counters)

    assert counters["no_crop"] == 1
    assert sizes and all(max(s) <= 1024 for s in sizes), "원본 통짜가 VLM에 전달됨"


def test_render_report_contains_table_and_routing():
    """13. 결과 마크다운: 조건×유형 비교표 + 라우팅 판정 + 카운터가 전부 담긴다."""
    table = aggregate([
        QuestionResult("a", "detection", True, False),
        QuestionResult("b", "detection", True, False),
        QuestionResult("b", "classification", False, True),
    ])
    md = render_report(
        table,
        routing=[{"uc": "UC-2", "passed": True, "detail": "history만 호출"}],
        counters={"no_fewshot": 2, "no_crop": 1, "label_error": 0},
        models={"vlm": "gpt-4o", "image_embed": "clip"})

    assert "| (a)" in md and "| (b)" in md
    assert "결함유무" in md and "유형판별" in md
    assert "100%" in md
    assert "UC-2" in md and "PASS" in md
    assert "no_fewshot" in md or "few-shot 없이" in md
    assert "gpt-4o" in md  # 모델 버전 기록 (R3)


# ── 사이클 3: 라우팅 체커 (R7 — 합성 trace로 검증) ────────


def make_trace(*call_groups):
    """반복별 호출 목록으로 합성 trace 생성."""
    return [{"iteration": i, "reason": ["r"], "calls": list(calls), "blocked": [],
             "observation": "", "sufficiency": {}}
            for i, calls in enumerate(call_groups, 1)]


def tcall(tool, **params):
    return {"tool": tool, "params": params}


def test_uc2_passes_only_with_history_alone():
    """14. UC-2: history만 → 통과 / 금지 도구(knowledge 등) 하나라도 → 실패."""
    ok = check_routing("UC-2", make_trace([tcall("history_query", question="이력")]), "답")
    assert ok["passed"] is True

    bad = check_routing("UC-2", make_trace(
        [tcall("history_query", question="이력"), tcall("knowledge_search", query="기준")]), "답")
    assert bad["passed"] is False
    assert "금지" in bad["detail"]


def test_uc1_requires_all_tools_and_visual_before_fewshot_vlm():
    """15. UC-1: 4종 전부 + visual이 few-shot 재관찰보다 먼저 → 통과 / 누락·순서 위반 → 실패."""
    good = check_routing("UC-1", make_trace(
        [tcall("vlm_analyze", question="관찰"), tcall("visual_search", k=5)],
        [tcall("vlm_analyze", question="재관찰", few_shot=[{"filename": "x.jpg"}])],
        [tcall("history_query", question="이력"), tcall("knowledge_search", query="기준")],
    ), "리포트")
    assert good["passed"] is True

    missing_vlm = check_routing("UC-1", make_trace(
        [tcall("visual_search", k=5)],
        [tcall("history_query", question="이력"), tcall("knowledge_search", query="기준")],
    ), "리포트")
    assert missing_vlm["passed"] is False and "필수" in missing_vlm["detail"]

    wrong_order = check_routing("UC-1", make_trace(
        [tcall("vlm_analyze", question="재관찰", few_shot=[{"filename": "x.jpg"}])],
        [tcall("visual_search", k=5), tcall("history_query", question="이력")],
        [tcall("knowledge_search", query="기준")],
    ), "리포트")
    assert wrong_order["passed"] is False and "먼저" in wrong_order["detail"]


def test_uc4_requires_escalation_ending():
    """16. UC-4: 필수 3종을 다 불러도 종결이 에스컬레이션이 아니면 실패."""
    trace = make_trace(
        [tcall("vlm_analyze", question="관찰"), tcall("visual_search", k=5)],
        [tcall("knowledge_search", query="와류발생기 기준")],
    )
    confident = check_routing("UC-4", trace, "와류발생기 결함으로 확정 진단한다.")
    assert confident["passed"] is False and "에스컬레이션" in confident["detail"]

    honest = check_routing("UC-4", trace, "⚠ 전문가 확인 필요 — 시각 근거 약함. 참고 소견…")
    assert honest["passed"] is True


def test_uc5_history_must_precede_compare_vlm():
    """17. UC-5: history(과거 사진 경로 확보)가 2장 비교 vlm보다 뒤면 실패."""
    bad = check_routing("UC-5", make_trace(
        [tcall("vlm_analyze", question="관찰")],
        [tcall("vlm_analyze", question="비교", compare_image="past.jpg")],
        [tcall("history_query", question="이력")],
    ), "비교 리포트")
    assert bad["passed"] is False and "먼저" in bad["detail"]

    good = check_routing("UC-5", make_trace(
        [tcall("vlm_analyze", question="관찰"), tcall("visual_search", k=5)],
        [tcall("history_query", question="과거 사진")],
        [tcall("vlm_analyze", question="비교", compare_image="past.jpg")],
    ), "비교 리포트")
    assert good["passed"] is True
