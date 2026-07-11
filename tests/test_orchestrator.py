"""SPEC-04 — orchestrator 테스트.

LLM은 각본(Scripted) 페이크, 도구는 호출 기록 페이크를 주입해 **결정적 가드만** 검증한다:
R1 후보 제한 · R2 동일 호출 차단 · R3 상한 · R6 trace 구조 · 반복당 2개 제한 · evidence 축적.
라우팅 품질(어떤 도구를 고르는 게 좋았나)은 여기서 안 본다 — SPEC-06 채점 담당.
"""

from src import config
from src.orchestrator import diagnose


class ScriptedLLM:
    """stage별 정해진 응답을 돌려주고, 받은 payload를 전부 기록하는 페이크 A1 두뇌."""

    def __init__(self, interpret=None, selects=(), assesses=(), draft="초안 답변"):
        self.interpret = interpret or {
            "intent": "history", "output_format": "concise", "needed_info": ["설비 이력"]}
        self.selects = list(selects)
        self.assesses = list(assesses)
        self.draft = draft
        self.seen = []

    def __call__(self, stage, payload):
        self.seen.append((stage, payload))
        if stage == "interpret":
            return self.interpret
        if stage == "select":
            return self.selects.pop(0) if self.selects else {"calls": []}
        if stage == "assess":
            return self.assesses.pop(0) if self.assesses else {"sufficient": True, "missing": []}
        if stage == "draft":
            return {"draft": self.draft}
        raise AssertionError(f"알 수 없는 stage: {stage}")

    def payloads(self, stage):
        return [p for s, p in self.seen if s == stage]


def recording_tools(log):
    """호출을 log에 기록하고 정형 응답을 돌려주는 페이크 도구 4종."""
    canned = {
        "visual_search": {"results": [{"filename": "2024_sungsan_5_A_LeadingEdge_002.jpg",
                                       "defect_type": "Paint Damage", "severity": 2,
                                       "similarity": 0.86, "description": "도장 손상"}],
                          "count": 1, "params": {}},
        "history_query": {"rows": [{"filename": "2024_sungsan_5_A_LeadingEdge_002.jpg",
                                    "year": 2024, "defect_type": "Paint Damage"}],
                          "count": 1, "sql": "SELECT 1", "params": {}},
        "knowledge_search": {"results": [{"text": "페인트 손상은 방치 시 진행된다",
                                          "source": "2024_..._002.jpg의 설명문",
                                          "similarity": 0.8}], "count": 1, "params": {}},
        "vlm_analyze": {"observation": "도장 벗겨짐 관찰", "confidence": "high", "params": {}},
    }

    def make(name):
        def run(**params):
            log.append((name, params))
            return canned[name]
        return run

    return {name: make(name) for name in canned}


def call(tool, reason="부족한 정보 조사", **params):
    return {"tool": tool, "params": params, "reason": reason}


def iterations(result):
    return [t for t in result.trace if "iteration" in t]


# ── 사이클 1: 그래프 루프 + 결정적 가드 ──────────────────


def test_trace_records_four_fields_per_iteration():
    """1. R6: 매 반복에 (선택 이유, 호출 파라미터, 관찰 요약, 충분성 판단)이 남는다."""
    log = []
    llm = ScriptedLLM(
        selects=[{"calls": [call("history_query", question="sungsan 5호기 이력")]}],
        assesses=[{"sufficient": True, "missing": []}])
    result = diagnose("sungsan 5호기 이력 알려줘", llm=llm, tools=recording_tools(log))

    iters = iterations(result)
    assert len(iters) == 1
    for field in ("reason", "calls", "observation", "sufficiency"):
        assert iters[0][field], f"trace 반복 항목에 {field}가 비어 있음"
    assert iters[0]["calls"][0]["tool"] == "history_query"
    assert iters[0]["calls"][0]["params"]["question"] == "sungsan 5호기 이력"


def test_evidence_accumulates_in_call_order():
    """2. 도구 반환이 호출 순서대로 evidence에 쌓인다 (기록 주체는 오케스트레이터뿐)."""
    log = []
    llm = ScriptedLLM(
        selects=[{"calls": [call("history_query", question="이력"),
                            call("knowledge_search", query="기준")]}],
        assesses=[{"sufficient": True, "missing": []}])
    result = diagnose("이력과 기준", llm=llm, tools=recording_tools(log))

    tool_entries = [e for e in result.evidence if "tool" in e]
    assert [e["tool"] for e in tool_entries] == ["history_query", "knowledge_search"]
    assert [name for name, _ in log] == ["history_query", "knowledge_search"]
    assert tool_entries[0]["result"]["count"] == 1  # 도구 반환이 가공 없이 기록됨


def test_no_image_excludes_visual_tools():
    """3. R1: 이미지 없는 질의 — vlm·visual은 후보에서 빠지고, LLM이 골라도 실행이 차단된다."""
    log = []
    llm = ScriptedLLM(
        selects=[{"calls": [call("vlm_analyze", question="사진 봐줘")]}],
        assesses=[{"sufficient": True, "missing": []}])
    result = diagnose("성산 5호기 이력", llm=llm, tools=recording_tools(log))

    assert log == [], "이미지 없는 질의에서 vlm이 실행됨 (R1 위반)"
    assert set(llm.payloads("select")[0]["allowed_tools"]) == {
        "history_query", "knowledge_search"}
    assert iterations(result)[0]["blocked"], "차단 사실이 trace에 없음"


def test_identical_call_to_previous_iteration_blocked():
    """4. R2: 직전 반복과 동일한 (도구, 파라미터)는 실행되지 않고 차단이 trace에 남는다."""
    log = []
    same = {"question": "sungsan 5호기 심각도 3"}
    llm = ScriptedLLM(
        selects=[{"calls": [call("history_query", **same)]},
                 {"calls": [call("history_query", **same)]}],
        assesses=[{"sufficient": False, "missing": ["이력 부족"]},
                  {"sufficient": True, "missing": []}])
    result = diagnose("이력 질의", llm=llm, tools=recording_tools(log))

    assert len(log) == 1, "동일 호출이 중복 실행됨 (R2 위반)"
    iters = iterations(result)
    assert not iters[0]["blocked"] and iters[1]["blocked"]
    assert "R2" in str(iters[1]["blocked"])


def test_loop_stops_at_max_iterations_with_insufficiency_note():
    """5. R3: 충분성이 계속 '부족'이어도 정확히 6회에서 멈추고 '근거 불충분'을 명시한다."""
    log = []
    llm = ScriptedLLM(
        selects=[{"calls": [call("history_query", question=f"조건 변형 {i}")]}
                 for i in range(10)],
        assesses=[{"sufficient": False, "missing": ["여전히 부족"]}] * 10)
    result = diagnose("안 풀리는 질의", llm=llm, tools=recording_tools(log))

    assert len(iterations(result)) == config.MAX_ITERATIONS == 6
    assert len(log) == 6
    assert "근거 불충분" in result.answer


def test_at_most_two_tools_per_iteration():
    """6. 한 반복에 도구 2개까지 — 3개 선택 시 앞 2개만 실행되고 초과분은 차단 기록."""
    log = []
    llm = ScriptedLLM(
        selects=[{"calls": [call("history_query", question="a"),
                            call("knowledge_search", query="b"),
                            call("knowledge_search", query="c")]}],
        assesses=[{"sufficient": True, "missing": []}])
    result = diagnose("복합 질의", llm=llm, tools=recording_tools(log))

    assert len(log) == 2
    assert [name for name, _ in log] == ["history_query", "knowledge_search"]
    assert iterations(result)[0]["blocked"]


# ── 사이클 2: 종결(finalize) ─────────────────────────────


def test_finalize_never_executes_tools():
    """7. R7: 종결 진입 후 어떤 도구도 실행되지 않는다 — 루프에서 1회 실행이 전부."""
    log = []
    llm = ScriptedLLM(
        selects=[{"calls": [call("history_query", question="이력")]}],
        assesses=[{"sufficient": True, "missing": []}],
        draft="이력 조회 결과 요약. 추가로 knowledge_search를 호출해 보강하겠다.")
    result = diagnose("이력 질의", llm=llm, tools=recording_tools(log))

    assert len(log) == 1, "종결 단계에서 도구가 실행됨 (R7 위반)"
    assert llm.seen[-1][0] == "draft", "draft 이후에 다른 stage가 호출됨"
    assert result.answer


def test_uncited_filename_removed_from_draft():
    """8. R4: evidence에 없는 파일명 인용은 제거되고 '근거 부족' 표기가 남는다."""
    log = []
    fake = "2023_sungsan_9_Z_FakeCase_777.jpg"        # evidence에 없는 지어낸 인용
    real = "2024_sungsan_5_A_LeadingEdge_002.jpg"     # 페이크 history 결과에 실존
    llm = ScriptedLLM(
        selects=[{"calls": [call("history_query", question="이력")]}],
        assesses=[{"sufficient": True, "missing": []}],
        draft=f"유사 사례로 {fake}가 있다. 이력상 근거는 {real}이다.")
    result = diagnose("이력 질의", llm=llm, tools=recording_tools(log))

    assert fake not in result.answer, "지어낸 인용이 살아남음 (R4 위반)"
    assert real in result.answer, "실존 인용까지 지워짐"
    assert "근거 부족" in result.answer


def test_output_format_routed_by_intent():
    """9. R8: 진단 의도 → draft에 report 형식 지시 / 조회 의도 → concise 지시."""
    log = []
    concise_llm = ScriptedLLM(  # 기본 interpret: history/concise
        selects=[{"calls": [call("history_query", question="이력")]}],
        assesses=[{"sufficient": True, "missing": []}])
    diagnose("이력 질의", llm=concise_llm, tools=recording_tools(log))
    assert concise_llm.payloads("draft")[0]["output_format"] == "concise"

    report_llm = ScriptedLLM(
        interpret={"intent": "diagnosis", "output_format": "report",
                   "needed_info": ["관찰", "유사사례"]},
        selects=[{"calls": [call("vlm_analyze", question="관찰해줘")]}],
        assesses=[{"sufficient": True, "missing": []}])
    diagnose("이 사진 점검해줘", image_path="2025_sungsan_5_A_LeadingEdge_001.jpg",
             llm=report_llm, tools=recording_tools(log))
    assert report_llm.payloads("draft")[0]["output_format"] == "report"


def test_weak_visual_evidence_escalates():
    """10. R5: 유사도 전부 임계(0.75) 미달 + VLM 확신 low → '전문가 확인 필요'로 종결."""
    log = []
    weak_tools = recording_tools(log)
    weak_tools["visual_search"] = lambda **p: (log.append(("visual_search", p)) or {
        "results": [{"filename": "2023_yeongkwang_1_A_SuctionSide_006.jpg",
                     "defect_type": "Vortex Generator", "severity": 3,
                     "similarity": 0.55, "description": "와류발생기"}],
        "count": 1, "params": {}})
    weak_tools["vlm_analyze"] = lambda **p: (log.append(("vlm_analyze", p)) or {
        "observation": "확대해도 판별 어려움", "confidence": "low", "params": {}})

    llm = ScriptedLLM(
        interpret={"intent": "diagnosis", "output_format": "report",
                   "needed_info": ["관찰", "유사사례"]},
        selects=[{"calls": [call("vlm_analyze", question="관찰"),
                            call("visual_search", k=5)]}],
        assesses=[{"sufficient": True, "missing": []}],
        draft="관찰과 유사 사례가 모두 약해 확정 판정이 어렵다.")
    result = diagnose("이 사진 문제 있나?", image_path="unknown_photo.jpg",
                      llm=llm, tools=weak_tools)

    assert "전문가 확인 필요" in result.answer
    assert llm.payloads("draft")[0]["escalation"] is True


# ── 사이클 3: 엣지 케이스 (§6) ───────────────────────────


def test_diagnosis_without_image_returns_guidance():
    """11. 사진 없이 '이 사진 점검해줘' → 도구 호출·루프 없이 이미지 필요 안내."""
    log = []
    llm = ScriptedLLM(interpret={"intent": "diagnosis", "output_format": "report",
                                 "needed_info": ["관찰"]})
    result = diagnose("이 사진 점검해줘", llm=llm, tools=recording_tools(log))

    assert log == [], "이미지 없는 진단 질의에서 도구가 실행됨"
    assert llm.payloads("select") == [], "루프에 진입함"
    assert "첨부" in result.answer


def test_consecutive_all_error_iterations_stop_early():
    """12. 도구가 연속(2회 반복) error만 반환 → 상한(6) 전 조기 종료 + 불충분 명시."""
    log = []
    tools = recording_tools(log)

    def broken(**params):
        log.append(("history_query", params))
        return {"error": "D1 파일 없음"}

    tools["history_query"] = broken
    llm = ScriptedLLM(
        selects=[{"calls": [call("history_query", question=f"조건 변형 {i}")]}
                 for i in range(10)],
        assesses=[{"sufficient": False, "missing": ["설비 이력"]}] * 10)
    result = diagnose("이력 질의", llm=llm, tools=tools)

    assert len(iterations(result)) == 2, "연속 error인데 계속 반복함"
    assert "근거 불충분" in result.answer


def test_report_marks_zero_history_explicitly():
    """13. 이력 0건 → 리포트 ⑤항에 '이력 없음' 명기 (LLM 초안이 빠뜨려도 강제)."""
    log = []
    tools = recording_tools(log)

    def empty_history(**params):
        log.append(("history_query", params))
        return {"rows": [], "count": 0, "sql": "SELECT 1",
                "message": "조회 조건에 해당하는 이력 없음", "params": {}}

    tools["history_query"] = empty_history
    llm = ScriptedLLM(
        interpret={"intent": "diagnosis", "output_format": "report",
                   "needed_info": ["관찰", "이력"]},
        selects=[{"calls": [call("vlm_analyze", question="관찰"),
                            call("history_query", question="이력")]}],
        assesses=[{"sufficient": True, "missing": []}],
        draft="① Paint Damage ② 앞전 ③ 심각도 2 ④ 유사 사례 ⑤ (누락) ⑥ 재도장 권고")
    result = diagnose("점검해줘", image_path="2025_sungsan_5_A_LeadingEdge_001.jpg",
                      llm=llm, tools=tools)

    assert "이력 없음" in result.answer


def test_unparseable_filename_noted_in_evidence():
    """14. 파일명 형식이 아닌 사진 → '이력 조회 불가' note가 evidence에 남는다 (UC-4 재료)."""
    log = []
    llm = ScriptedLLM(
        interpret={"intent": "diagnosis", "output_format": "report", "needed_info": ["관찰"]},
        selects=[{"calls": [call("vlm_analyze", question="관찰")]}],
        assesses=[{"sufficient": True, "missing": []}])
    result = diagnose("문제 있나?", image_path="IMG_1234.jpg",
                      llm=llm, tools=recording_tools(log))

    notes = [e for e in result.evidence if e.get("type") == "note"]
    assert any("이력 조회 불가" in n["content"] for n in notes)
