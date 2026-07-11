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
