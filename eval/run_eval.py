"""시스템의 성적표를 내는 자동 채점기 — SPEC-06.

두 질문에 숫자로 답한다:
① 유사 사례 few-shot 주입이 진단 정답률을 올리나 — (a) VLM 단독 vs (b) RAG 주입 비교표
② 질문마다 도구를 제대로 골랐나 — UC 5종 trace를 통과 제약과 기계 대조

pytest가 아니라 완성된 시스템 전체의 배치 채점이다. 사람이 쓰는 화면은 demo(SPEC-05).
실행: uv run python eval/run_eval.py [--limit 5]
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from src import config
from src import tools as toolbox
from src.schema import LabelDoc, parse_label, primary_defect, to_crop_coords

COST_PER_VLM_CALL_USD = 0.01  # gpt-4o 저해상 이미지+짧은 응답 기준 보수적 추정 (R6 예고용)

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


# ── 하네스 — 가드·로드·실행 ──────────────────────────────


def assert_no_leak(testset: list[str], chroma_dir: Path) -> None:
    """R1: 시험지 파일명이 V1·V2에 하나라도 있으면 채점을 시작조차 하지 않는다."""
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    for name in (config.V1_COLLECTION, config.V2_COLLECTION):
        leaked = client.get_collection(name).get(ids=testset)["ids"]
        if leaked:
            raise RuntimeError(f"테스트셋 누수: {name}에 {leaked} — 오염된 시험은 치르지 않는다 (R1)")


def load_eval_docs(data_dir: Path, testset_path: Path,
                   limit: int | None) -> list[tuple[LabelDoc, Path]]:
    """testset.txt 순서대로 (LabelDoc, 사진 경로) 로드. --limit이면 앞 N장만 (R6)."""
    names = Path(testset_path).read_text(encoding="utf-8").split()
    if limit is not None:
        names = names[:limit]
    docs = []
    for name in names:
        json_name = Path(name).with_suffix(".json").name
        jp = next(Path(data_dir).rglob(json_name), None)
        if jp is None:
            raise FileNotFoundError(f"시험지 라벨 없음: {json_name} (data_dir={data_dir})")
        docs.append((parse_label(jp), jp.with_suffix(".jpg")))
    return docs


def estimate_calls(docs: list[LabelDoc]) -> int:
    """R6: 예상 VLM 호출 수 = Σ(사진별 문항 수) × 2조건."""
    return sum(len(parse_vqa(doc)) for doc in docs) * 2


def build_fewshot_block(visual_results: list[dict]) -> str:
    """(b) 조건 주입 텍스트 — 과거 유사 사례의 라벨·설명문 (T4 few-shot과 같은 발상)."""
    lines = ["참고 — 과거 유사 점검 사례 (검색 결과):"]
    lines += [
        f"- {r['filename']} ({r['defect_type']}, 심각도 {r['severity']}): {r['description']}"
        for r in visual_results
    ]
    lines.append("위 사례들과 비교해 아래 문항에 답하라.")
    return "\n".join(lines)


def build_vqa_prompt(question: Question, fewshot_block: str | None) -> str:
    """문항 1개의 프롬프트. 평가 사진의 파일명·정답은 절대 넣지 않는다 (R2)."""
    lines = []
    if fewshot_block:
        lines += [fewshot_block, ""]
    lines.append("점검 사진에 대한 선다형 문항이다. 반드시 보기 글자 하나로만 답하라 "
                 "(예: a) — 설명 금지, 한 글자만.")
    lines.append(f"문항: {question.prompt}")
    lines += [f"{letter}) {text}" for letter, text in sorted(question.options.items())]
    return "\n".join(lines)


def _bump(counters: dict, key: str) -> None:
    counters[key] = counters.get(key, 0) + 1


def evaluate_photo(doc: LabelDoc, jpg_path: Path, *, asker, searcher,
                   counters: dict) -> list[QuestionResult]:
    """사진 1장을 (a)·(b) 두 조건으로 채점 — 같은 크롭·같은 문항·같은 순서 (R3).

    asker(크롭 PIL 이미지, 프롬프트) → 모델 응답 텍스트 / searcher(사진 경로, 크롭) → visual_search 반환.
    둘 다 주입 가능 — 테스트는 페이크, 실행은 gpt-4o·CLIP.
    """
    from src.tools import _prepare_vlm_image  # SPEC-03 R5와 동일 규칙 공유 (§5)

    if doc.cropped_bbox is None:
        _bump(counters, "no_crop")
    image = _prepare_vlm_image(jpg_path, doc.cropped_bbox)

    questions = parse_vqa(doc)
    if any(q.kind == "localization" for q in questions) and not verify_localization_label(doc):
        questions = [q for q in questions if q.kind != "localization"]
        _bump(counters, "label_error")  # 틀린 정답지로 채점하지 않는다 (게이트 합의)

    visual = searcher(jpg_path, doc.cropped_bbox)
    found = visual.get("results") or []
    fewshot = build_fewshot_block(found) if found else None
    if fewshot is None:
        _bump(counters, "no_fewshot")

    results = []
    for condition, block in (("a", None), ("b", fewshot)):
        for q in questions:
            letter = parse_model_answer(asker(image, build_vqa_prompt(q, block)))
            results.append(QuestionResult(
                condition=condition, kind=q.kind,
                correct=letter == q.answer, format_fail=letter is None))
    return results


# ── 라우팅 채점 (R7: 근거는 trace뿐, 통과 제약만 검사) ─────

# SPEC-04 §5의 UC별 통과 제약. 경로 완전 일치가 아니라 필수/금지/순서만 본다.
UC_CONSTRAINTS = {
    "UC-1": {"required": {"visual_search", "history_query", "knowledge_search", "vlm_analyze"},
             "order": "visual_before_fewshot_vlm"},
    "UC-2": {"required": {"history_query"},
             "forbidden": {"vlm_analyze", "visual_search", "knowledge_search"}},
    "UC-3": {"required": {"knowledge_search"},
             "forbidden": {"vlm_analyze", "visual_search", "history_query"}},
    "UC-4": {"required": {"vlm_analyze", "visual_search", "knowledge_search"},
             "must_escalate": True},  # history 생략이 기대 동작이나 금지는 아님 (§5)
    "UC-5": {"required": {"vlm_analyze", "history_query"},
             "order": "history_before_compare_vlm"},
}


def _first_index(calls: list[dict], tool: str) -> int | None:
    return next((i for i, c in enumerate(calls) if c["tool"] == tool), None)


def check_routing(uc: str, trace: list[dict], answer: str) -> dict:
    """trace의 도구 호출을 UC 통과 제약과 기계 대조 — 같은 로그면 같은 점수 (R7)."""
    spec = UC_CONSTRAINTS[uc]
    calls = [c for t in trace if "iteration" in t for c in t["calls"]]
    tools_called = [c["tool"] for c in calls]
    problems = []

    missing = spec.get("required", set()) - set(tools_called)
    if missing:
        problems.append(f"필수 도구 미호출: {sorted(missing)}")
    hit = set(tools_called) & spec.get("forbidden", set())
    if hit:
        problems.append(f"금지 도구 호출: {sorted(hit)}")

    order = spec.get("order")
    if order == "visual_before_fewshot_vlm":  # 검색 결과가 재관찰의 재료 (UC-1)
        first_visual = _first_index(calls, "visual_search")
        for i, c in enumerate(calls):
            if c["tool"] == "vlm_analyze" and c["params"].get("few_shot"):
                if first_visual is None or i < first_visual:
                    problems.append("few-shot 재관찰이 visual_search보다 먼저 — 순서 위반")
                break
    elif order == "history_before_compare_vlm":  # 과거 사진 경로가 비교의 입력 (UC-5)
        first_history = _first_index(calls, "history_query")
        for i, c in enumerate(calls):
            if c["tool"] == "vlm_analyze" and c["params"].get("compare_image"):
                if first_history is None or i < first_history:
                    problems.append("2장 비교 호출이 history_query보다 먼저 — 순서 위반")
                break

    if spec.get("must_escalate") and "전문가 확인 필요" not in answer:
        problems.append("종결이 에스컬레이션이 아님 — 억지 단정 (SPEC-04 함정 4)")

    detail = "; ".join(problems) if problems else f"호출 순서: {tools_called}"
    return {"uc": uc, "passed": not problems, "detail": detail}


def _uc_inputs(docs: list[tuple[LabelDoc, Path]]) -> list[tuple[str, str, Path | None]]:
    """UC 5종의 (질의, 사진) 자동 구성. UC-4 사진은 파일명 없는 무명 사본으로 전달."""
    import shutil
    import tempfile

    def klass(doc: LabelDoc) -> str:
        main = primary_defect(doc)
        return doc.categories[main.category_id] if main else "normal"

    defects = [(d, p) for d, p in docs if d.annotations]
    uc1_doc, uc1_jpg = defects[0] if defects else docs[0]
    uc4_doc, uc4_jpg = next(((d, p) for d, p in defects if klass(d) == "Vortex Generator"),
                            (uc1_doc, uc1_jpg))
    anonymous = Path(tempfile.mkdtemp(prefix="uc4_")) / "unknown_photo.jpg"
    shutil.copy(uc4_jpg, anonymous)

    return [
        ("UC-1", f"이 사진 점검해줘. 파일명 {uc1_doc.filename}", uc1_jpg),
        ("UC-2", "성산 5호기 A블레이드 심각도 3 이상 이력 보여줘", None),
        ("UC-3", "라미네이트 노출을 방치하면 어떻게 되나? 심각도 등급의 의미도 알려줘.", None),
        ("UC-4", "이 사진에 문제 있나?", anonymous),
        ("UC-5", f"이 설비, 작년보다 심해졌나? 파일명 {uc1_doc.filename}", uc1_jpg),
    ]


def run_routing_checks(docs: list[tuple[LabelDoc, Path]]) -> list[dict]:
    """UC 5종을 diagnose()로 실제 실행하고 trace를 통과 제약과 대조."""
    from src.orchestrator import diagnose

    results = []
    for uc, question, image in _uc_inputs(docs):
        print(f"  라우팅 {uc} 실행 중…")
        r = diagnose(question, image_path=image)
        results.append(check_routing(uc, r.trace, r.answer))
    return results


# ── 결과 보고 (R4: 유형별 분해, 불리한 숫자도 그대로) ──────

_KIND_LABELS = (("detection", "결함유무"), ("classification", "유형판별"),
                ("localization", "위치"), ("analysis", "특징"), ("전체", "전체"))
_CONDITION_LABELS = {"a": "(a) VLM 단독", "b": "(b) RAG 주입"}


def _pct(pair: tuple[int, int] | None) -> str:
    if not pair:
        return "—"
    correct, total = pair
    return f"{100 * correct / total:.0f}% ({correct}/{total})"


def render_report(table: dict, routing: list[dict], counters: dict, models: dict) -> str:
    """README에 붙일 마크다운 — VQA 비교표 + 라우팅 판정 + 엣지 카운터 + 모델 버전."""
    lines = ["## 평가 결과 (SPEC-06)", "", "### VQA: (a) VLM 단독 vs (b) 유사사례 few-shot 주입", ""]
    lines.append("| 조건 | " + " | ".join(label for _, label in _KIND_LABELS) + " |")
    lines.append("|---|" + "---|" * len(_KIND_LABELS))
    for cond in ("a", "b"):
        if cond not in table:
            continue
        cells = [_pct(table[cond].get(kind)) for kind, _ in _KIND_LABELS]
        lines.append(f"| {_CONDITION_LABELS[cond]} | " + " | ".join(cells) + " |")
    lines.append("")
    fails = table.get("format_fails", {})
    lines.append(f"형식실패(오답 처리): (a) {fails.get('a', 0)}건, (b) {fails.get('b', 0)}건 · "
                 f"few-shot 없이 진행(no_fewshot): {counters.get('no_fewshot', 0)}장 · "
                 f"크롭 폴백(no_crop): {counters.get('no_crop', 0)}장 · "
                 f"위치 라벨 오류 제외(label_error): {counters.get('label_error', 0)}장")
    lines += ["", "### 라우팅 검증 (UC 5종 통과 제약)", ""]
    if routing:
        for r in routing:
            verdict = "PASS" if r["passed"] else "FAIL"
            lines.append(f"- {r['uc']}: **{verdict}** — {r['detail']}")
    else:
        lines.append("- (미실행)")
    lines += ["", f"모델: {', '.join(f'{k}={v}' for k, v in models.items())}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="자동 채점기 (SPEC-06)")
    parser.add_argument("--limit", type=int, default=None, help="처음 N장만 (시험 가동용)")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    args = parser.parse_args()

    testset = Path(config.TESTSET_MANIFEST).read_text(encoding="utf-8").split()
    assert_no_leak(testset, config.CHROMA_DIR)  # R1 — 통과 못 하면 여기서 죽는다
    docs = load_eval_docs(args.data_dir, config.TESTSET_MANIFEST, args.limit)

    calls = estimate_calls([doc for doc, _ in docs])
    print(f"대상 {len(docs)}장 / VLM 약 {calls}회 호출 예정 "
          f"(추정 ${calls * COST_PER_VLM_CALL_USD:.2f})")  # R6

    def asker(image, prompt):
        # T4의 JSON 모드와 달리 한 글자 응답이 필요해 평가 전용 호출을 쓴다 (모델은 동일 고정)
        from dotenv import load_dotenv
        from openai import OpenAI

        from src.tools import _to_image_part

        load_dotenv()
        res = OpenAI().chat.completions.create(
            model=config.VLM_MODEL, temperature=0, max_tokens=8,
            messages=[{"role": "user",
                       "content": [{"type": "text", "text": prompt}, _to_image_part(image)]}])
        return res.choices[0].message.content or ""

    def searcher(jpg_path, crop):
        return toolbox.visual_search(jpg_path, crop=crop, k=config.TOP_K)

    counters: dict = {}
    results: list[QuestionResult] = []
    for i, (doc, jpg) in enumerate(docs, 1):
        results += evaluate_photo(doc, jpg, asker=asker, searcher=searcher, counters=counters)
        print(f"  [{i}/{len(docs)}] 채점 완료")

    table = aggregate(results)
    routing = run_routing_checks(docs)
    report = render_report(table, routing, counters, models={
        "vlm": config.VLM_MODEL, "image_embed": config.IMAGE_EMBED_MODEL,
        "orchestrator": config.ORCHESTRATOR_MODEL})
    print("\n" + report)
    out = Path(__file__).parent / "results.md"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
