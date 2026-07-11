"""오케스트레이터 A1 — 도구를 골라 쓰고 스스로 검증하는 유일한 에이전트 (SPEC-04).

판독 데스크의 선임 엔지니어처럼: 질문을 해석해 필요 정보를 목록화하고, 모자란 정보만
골라 조사시키고, 근거가 충분한지 자문하고, 부족하면 방법을 바꿔 다시 조사한 뒤 답을 쓴다.

지능(무엇을 조사할지)은 LLM에, 규칙(해도 되는 것)은 결정적 가드에 둔다:
R1 이미지 없으면 vlm·visual 후보 제외 / R2 직전과 동일한 (도구, 파라미터) 차단 /
R3 루프 상한 6회 / 반복당 도구 2개. 가드는 LLM이 뭐라 답하든 코드로 강제된다.

Evidence Store(S0)는 질의 1건 동안만 사는 스크래치패드 — 기록 주체는 여기(execute)뿐이고
도구는 접근할 수 없다 (SPEC-00 §2).
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src import config
from src import tools as toolbox
from src.schema import parse_filename

TOOL_NAMES = ("visual_search", "history_query", "knowledge_search", "vlm_analyze")
IMAGE_TOOLS = frozenset({"visual_search", "vlm_analyze"})  # 이미지를 여는 도구 (SPEC-03 R6)
MAX_CALLS_PER_ITER = 2  # 한 반복에 도구 2개까지 (SPEC-04 §3, 순차 실행)


@dataclass
class AgentResult:
    """diagnose의 반환 — 답변 + 반복별 추론 로그 + 근거 원본."""

    answer: str
    trace: list[dict]
    evidence: list[dict]


class _State(TypedDict, total=False):
    question: str
    image_path: str | None
    intent: str            # diagnosis | history | knowledge | compare
    output_format: str     # report | concise
    needed_info: list[str]
    evidence: list[dict]
    trace: list[dict]
    iteration: int
    pending: list[dict]    # 이번 반복에 실행할 호출
    blocked: list[dict]    # 이번 반복에 차단된 호출 + 사유
    last_calls: list[str]  # 직전 반복 실행 호출의 지문 (R2)
    executed: list[dict]   # 이번 반복 실행 결과 요약 (trace 재료)
    error_streak: int      # 실행 도구가 전부 error였던 연속 반복 수 (조기 종료 §6)
    sufficient: bool
    missing: list[str]
    answer: str


# ── 결정적 헬퍼 ───────────────────────────────────────────


def _fingerprint(call: dict) -> str:
    """(도구, 파라미터) 동일성 비교용 지문 (R2)."""
    params = json.dumps(call.get("params", {}), sort_keys=True, ensure_ascii=False, default=str)
    return f"{call['tool']}|{params}"


def _summarize(tool: str, result: dict) -> str:
    """도구 반환 1건의 한 줄 요약 — trace 관찰 기록과 LLM 프롬프트 재료."""
    if "error" in result:
        return f"{tool} → error: {result['error']}"
    if "observation" in result:
        return f"{tool} → confidence={result.get('confidence')}: {str(result['observation'])[:80]}"
    if "count" in result:
        return f"{tool} → {result['count']}건"
    return f"{tool} → 결과 수신"


def _digest(evidence: list[dict]) -> list[str]:
    """evidence 전체의 압축 요약 — select/assess 프롬프트에 들어가는 현황판."""
    lines = []
    for e in evidence:
        if "tool" in e:
            lines.append(f"[반복{e['iteration']}] {_summarize(e['tool'], e['result'])}")
        else:
            lines.append(f"[{e['type']}] {e['content']}")
    return lines


def _allowed_tools(state: _State) -> list[str]:
    """R1: 이미지 없는 질의에서 이미지 도구를 후보에서 제외."""
    if state.get("image_path"):
        return list(TOOL_NAMES)
    return [t for t in TOOL_NAMES if t not in IMAGE_TOOLS]


_FILENAME_RE = re.compile(r"[\w\-]+\.jpg")


def _enforce_citations(draft: str, evidence: list[dict]) -> str:
    """R4 자기검증: 초안의 파일명 인용을 evidence와 기계 대조 — 없는 인용은 제거·표기.

    검색은 다 잘 해놓고 마지막 작문에서 지어내면 앞의 모든 노력이 무효다 (함정 3).
    """
    known = set(_FILENAME_RE.findall(json.dumps(evidence, ensure_ascii=False, default=str)))
    removed = []

    def check(m: re.Match) -> str:
        if m.group(0) in known:
            return m.group(0)
        removed.append(m.group(0))
        return "[근거 부족으로 인용 제거]"

    cleaned = _FILENAME_RE.sub(check, draft)
    if removed:
        cleaned += f"\n\n[자기검증: evidence에 없는 인용 {len(removed)}건 제거]"
    return cleaned


def _needs_escalation(state: _State) -> bool:
    """R5: 시각 근거 약함 = 유사도 전부 임계 미달 **그리고** 최근 VLM 확신 low (함정 4).

    틀린 확신보다 정직한 불확실성 — 진단·비교 의도에서만 발동한다.
    """
    if state.get("intent") not in ("diagnosis", "compare"):
        return False
    sims = [r.get("similarity", 0.0)
            for e in state["evidence"] if e.get("tool") == "visual_search"
            for r in e["result"].get("results", [])]
    confidences = [e["result"].get("confidence")
                   for e in state["evidence"] if e.get("tool") == "vlm_analyze"
                   if "error" not in e["result"]]
    visual_weak = bool(sims) and max(sims) < config.SIMILARITY_THRESHOLD
    vlm_weak = bool(confidences) and confidences[-1] == "low"
    return visual_weak and vlm_weak


# ── LangGraph 노드 (llm·tools는 diagnose에서 주입) ────────


def _build_graph(llm, tools):
    def interpret(state: _State) -> dict:
        decision = llm("interpret", {
            "question": state["question"], "has_image": bool(state.get("image_path"))})
        evidence = []
        if state.get("image_path"):
            name = Path(state["image_path"]).name
            try:
                meta = parse_filename(name)
                evidence.append({"type": "filename_meta",
                                 "content": dataclasses.asdict(meta)})
            except ValueError:
                evidence.append({"type": "note",
                                 "content": "파일명 형식 아님 — 설비 식별자가 없어 이력 조회 불가"})
        update = {
            "intent": decision.get("intent", "diagnosis"),
            "output_format": decision.get("output_format", "concise"),
            "needed_info": decision.get("needed_info", []),
            "evidence": state["evidence"] + evidence,
            "trace": state["trace"] + [{"stage": "interpret", **decision}],
        }
        # §6: 이미지가 필요한 의도인데 사진이 없다 — 루프에 들어가지 않고 안내로 종료
        if update["intent"] in ("diagnosis", "compare") and not state.get("image_path"):
            update["answer"] = ("이미지가 필요한 질의입니다 — 점검할 사진을 첨부해 주세요. "
                                "사진 없이 가능한 것은 설비 이력 조회와 결함 기준 질문입니다.")
        return update

    def route_after_interpret(state: _State) -> str:
        return END if state.get("answer") else "select"

    def select(state: _State) -> dict:
        allowed = _allowed_tools(state)
        resp = llm("select", {
            "question": state["question"],
            "needed_info": state.get("needed_info", []),
            "allowed_tools": allowed,
            "evidence_summary": _digest(state["evidence"]),
            "iteration": state["iteration"] + 1,
        })
        pending, blocked = [], []
        for c in resp.get("calls", []):
            if c.get("tool") not in TOOL_NAMES:
                blocked.append({**c, "why_blocked": f"알 수 없는 도구: {c.get('tool')}"})
            elif c["tool"] not in allowed:
                blocked.append({**c, "why_blocked": "이미지 없는 질의 — 후보 제외 (R1)"})
            elif _fingerprint(c) in state.get("last_calls", []):
                blocked.append({**c, "why_blocked": "직전과 동일한 (도구, 파라미터) — 바꿔서 재시도 (R2)"})
            elif len(pending) >= MAX_CALLS_PER_ITER:
                blocked.append({**c, "why_blocked": f"반복당 도구 {MAX_CALLS_PER_ITER}개 제한"})
            else:
                pending.append(c)
        return {"pending": pending, "blocked": blocked}

    def execute(state: _State) -> dict:
        new_evidence, executed = [], []
        for c in state["pending"]:
            params = dict(c.get("params", {}))
            if c["tool"] in IMAGE_TOOLS and "image_path" not in params:
                params["image_path"] = state["image_path"]
            try:
                result = tools[c["tool"]](**params)
            except Exception as e:  # 도구는 예외를 안 던지는 계약(SPEC-03 R4)이지만 루프는 어떤 경우에도 죽지 않는다
                result = {"error": f"도구 실행 예외: {e}"}
            new_evidence.append({"iteration": state["iteration"] + 1,
                                 "tool": c["tool"], "params": params, "result": result})
            executed.append({"tool": c["tool"], "params": params,
                             "reason": c.get("reason", ""),
                             "summary": _summarize(c["tool"], result)})
        # 실행이 없던 반복은 직전 지문 유지 — 같은 호출이 다음 반복에도 계속 차단되게 (R2)
        last = [_fingerprint(c) for c in state["pending"]] or state.get("last_calls", [])
        all_error = bool(new_evidence) and all("error" in e["result"] for e in new_evidence)
        streak = state.get("error_streak", 0) + 1 if all_error else 0
        return {"evidence": state["evidence"] + new_evidence,
                "executed": executed, "last_calls": last, "error_streak": streak}

    def assess(state: _State) -> dict:
        iteration = state["iteration"] + 1
        resp = llm("assess", {
            "question": state["question"],
            "needed_info": state.get("needed_info", []),
            "evidence_summary": _digest(state["evidence"]),
            "iteration": iteration,
        })
        entry = {
            "iteration": iteration,
            "reason": [c["reason"] for c in state["executed"]] or ["실행된 도구 없음"],
            "calls": [{"tool": c["tool"], "params": c["params"]} for c in state["executed"]],
            "blocked": state.get("blocked", []),
            "observation": "; ".join(c["summary"] for c in state["executed"]) or "새 근거 없음",
            "sufficiency": resp,
        }
        return {"iteration": iteration,
                "trace": state["trace"] + [entry],
                "sufficient": bool(resp.get("sufficient")),
                "missing": resp.get("missing", [])}

    def route_after_assess(state: _State) -> str:
        if state["sufficient"] or state["iteration"] >= config.MAX_ITERATIONS:
            return "finalize"
        if state.get("error_streak", 0) >= 2:  # §6: 도구가 계속 죽어 있다 — 더 돌아봐야 소용없음
            return "finalize"
        return "select"

    def finalize(state: _State) -> dict:
        # R7: 종결은 조립만 — 이 노드에는 도구 실행 경로 자체가 없다
        escalation = _needs_escalation(state)
        resp = llm("draft", {
            "question": state["question"],
            "output_format": state.get("output_format", "concise"),
            "evidence": state["evidence"],
            "escalation": escalation,
        })
        answer = _enforce_citations(resp.get("draft", ""), state["evidence"])  # R4
        # §6: 이력 0건은 리포트 ⑤항에 반드시 명기 — LLM 초안이 빠뜨려도 강제
        zero_history = any(e.get("tool") == "history_query" and e["result"].get("count") == 0
                           for e in state["evidence"])
        if state.get("output_format") == "report" and zero_history and "이력 없음" not in answer:
            answer += "\n\n⑤ 해당 설비 이력: 조회 조건에 해당하는 이력 없음 (0건)"
        if escalation:
            answer = ("⚠ 전문가 확인 필요 — 시각 근거 약함"
                      f"(유사 사례 유사도 전부 {config.SIMILARITY_THRESHOLD} 미만"
                      " + 관찰 확신 low). 아래는 참고용 소견이다.\n\n") + answer
        if not state.get("sufficient"):
            missing = ", ".join(state.get("missing", [])) or "미확보 정보 있음"
            answer += f"\n\n[근거 불충분: {missing} — 루프 상한 내 확보 실패, 해당 부분은 판단 보류]"
        return {"answer": answer}

    g = StateGraph(_State)
    g.add_node("interpret", interpret)
    g.add_node("select", select)
    g.add_node("execute", execute)
    g.add_node("assess", assess)
    g.add_node("finalize", finalize)
    g.add_edge(START, "interpret")
    g.add_conditional_edges("interpret", route_after_interpret, ["select", END])
    g.add_edge("select", "execute")
    g.add_edge("execute", "assess")
    g.add_conditional_edges("assess", route_after_assess, ["select", "finalize"])
    g.add_edge("finalize", END)
    return g.compile()


# ── 기본 구현 (실행 시 사용, 테스트는 페이크 주입) ─────────


_STAGE_PROMPTS = {
    "interpret": (
        "너는 풍력 점검 진단 에이전트의 해석 단계다. 질문과 이미지 유무를 보고 JSON으로 답한다: "
        '{"intent": "diagnosis|history|knowledge|compare", "output_format": "report|concise", '
        '"needed_info": ["답에 필요한 정보 항목", ...]}. '
        "이미지 진단·비교는 report, 이력·지식 질문은 concise."
    ),
    "select": (
        "너는 도구 선택 단계다. needed_info 중 evidence_summary에 아직 없는 정보를 채울 도구를 "
        "allowed_tools 안에서 최대 2개 고른다. JSON: "
        '{"calls": [{"tool": "이름", "params": {...}, "reason": "선택 이유"}]}. '
        "도구 파라미터 — visual_search: {k, defect_type, severity, crop} (이미지는 자동 주입) / "
        "history_query: {question: 자연어 조회 조건} / knowledge_search: {query, k} / "
        "vlm_analyze: {question, cropped_bbox, few_shot, compare_image}. "
        "직전과 똑같은 (도구, 파라미터) 재호출은 차단되니 파라미터를 바꿔 다르게 시도하라."
    ),
    "assess": (
        "너는 충분성 평가 단계다. needed_info 대비 evidence_summary의 빈칸을 확인하고 JSON으로 답한다: "
        '{"sufficient": true|false, "missing": ["아직 부족한 정보", ...]}. '
        "모든 필요 정보가 확보됐을 때만 sufficient=true."
    ),
    "draft": (
        "너는 답안 작성 단계다. evidence에 있는 정보만 사용해 답을 쓴다 — evidence에 없는 "
        "파일명·수치·사실을 지어내지 않는다. output_format이 report면 6항목 리포트"
        "(①결함 유무·종류 ②위치 ③심각도와 근거 ④유사 과거 사례 상위 3건(파일명) "
        "⑤해당 설비 이력 ⑥권고 조치(근거 인용)), concise면 간결 답변+출처. "
        'JSON: {"draft": "답변 전문"}'
    ),
}


def _default_llm(stage: str, payload: dict) -> dict:
    """gpt-4o-mini JSON 모드. 단계별 프롬프트는 _STAGE_PROMPTS."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    res = OpenAI().chat.completions.create(
        model=config.ORCHESTRATOR_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _STAGE_PROMPTS[stage]},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ],
    )
    return json.loads(res.choices[0].message.content or "{}")


def _default_tools() -> dict[str, Any]:
    return {
        "visual_search": toolbox.visual_search,
        "history_query": toolbox.history_query,
        "knowledge_search": toolbox.knowledge_search,
        "vlm_analyze": toolbox.vlm_analyze,
    }


def diagnose(question: str, image_path: Path | None = None, *, llm=None, tools=None) -> AgentResult:
    """진입점 (demo·eval이 호출). 질의 1건의 해석 → 루프 → 종결 전체를 수행한다."""
    llm = llm or _default_llm
    tools = tools or _default_tools()
    graph = _build_graph(llm, tools)
    final = graph.invoke(
        {
            "question": question,
            "image_path": str(image_path) if image_path else None,
            "evidence": [], "trace": [], "iteration": 0,
            "last_calls": [], "sufficient": False,
        },
        {"recursion_limit": 80},  # 6반복 × 3노드 + 해석·종결이면 충분
    )
    return AgentResult(answer=final.get("answer", ""),
                       trace=final["trace"], evidence=final["evidence"])
