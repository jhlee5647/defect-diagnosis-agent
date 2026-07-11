"""웹 데모 — 에이전트의 사고 과정을 보여주는 화면 (SPEC-05).

프론트엔드가 아니라 diagnose()를 웹 화면에 연결하는 어댑터다. 오케스트레이터가 엔진이라면
여기는 계기판과 핸들 — 판단·가공·재시도 로직을 두지 않는다 (R1). 이 시스템의 가치는
최종 답이 아니라 답에 도달하는 과정이므로, trace의 반복별 4요소를 전부 노출한다 (R2).

실행: uv run python -m src.demo → 브라우저 로컬 접속
"""

from __future__ import annotations

import json

from src.orchestrator import diagnose

PRESETS = {
    "UC-1 표준 진단": "이 사진 점검해줘.",
    "UC-2 이력 조회": "sungsan 5호기 A블레이드 심각도 3 이상 이력 보여줘",
    "UC-3 지식 질문": "라미네이트 노출을 방치하면 어떻게 되나? 심각도 등급의 의미도 알려줘.",
    "UC-4 희귀 결함": "이 사진에 문제 있나?",
    "UC-5 시계열 비교": "이 설비, 작년보다 심해졌나?",
}


def format_trace(trace: list[dict]) -> str:
    """trace → Markdown 표시 변환. 내용의 요약·미화·생략 금지 (R2·R4)."""
    lines = []
    for entry in trace:
        if entry.get("stage") == "interpret":
            lines.append("### 해석")
            lines.append(f"- 의도: {entry.get('intent')} / 출력 형식: {entry.get('output_format')}")
            needed = entry.get("needed_info") or []
            lines.append(f"- 필요 정보: {', '.join(needed) if needed else '(없음)'}")
        elif "iteration" in entry:
            lines.append(f"### 반복 {entry['iteration']}")
            for reason in entry["reason"]:
                lines.append(f"- 선택 이유: {reason}")
            for c in entry["calls"]:
                params = json.dumps(c["params"], ensure_ascii=False, default=str)
                lines.append(f"- 호출: `{c['tool']}` {params}")
            for b in entry.get("blocked", []):
                lines.append(f"- 차단: `{b.get('tool')}` — {b.get('why_blocked')}")
            lines.append(f"- 관찰: {entry['observation']}")
            suff = entry.get("sufficiency", {})
            verdict = "충분" if suff.get("sufficient") else "부족"
            missing = ", ".join(suff.get("missing") or [])
            lines.append(f"- 충분성: {verdict}" + (f" — 부족: {missing}" if missing else ""))
    return "\n".join(lines) or "(추론 기록 없음)"


def run(image_path: str | None, question: str) -> tuple[str, str]:
    """diagnose() 호출과 표시 변환이 전부 (R1). 에러·에스컬레이션도 그대로 (R4)."""
    if not (question or "").strip():
        return "", "질문을 입력해 주세요."
    result = diagnose(question, image_path=image_path or None)
    return format_trace(result.trace), result.answer


def build_app():
    import gradio as gr

    with gr.Blocks(title="발전설비 결함 진단 에이전트") as app:
        gr.Markdown("# 발전설비 결함 진단 — 멀티모달 Agentic RAG\n"
                    "사진(선택)과 질문을 넣으면 에이전트가 도구를 골라 조사하고, "
                    "그 추론 과정 전체를 아래에 표시합니다.")
        with gr.Row():
            image = gr.Image(label="점검 사진 (선택)", type="filepath")
            with gr.Column():
                question = gr.Textbox(label="질문", lines=3,
                                      placeholder="예: 이 사진 점검해줘.")
                submit = gr.Button("실행", variant="primary")
        with gr.Row():
            for label, preset in PRESETS.items():
                gr.Button(label, size="sm").click(lambda p=preset: p, outputs=question)
        gr.Markdown("## 추론 과정")
        trace_md = gr.Markdown("(실행 전)")
        gr.Markdown("## 답변")
        answer_md = gr.Markdown("(실행 전)")
        submit.click(run, inputs=[image, question], outputs=[trace_md, answer_md])
    return app


def main():
    build_app().launch()


if __name__ == "__main__":
    main()
