"""에이전트가 쓰는 도구 4종 — SPEC-03.

엔지니어의 네 가지 행동을 각각 함수 하나로: T1 visual_search(비슷한 사진 회상) ·
T2 history_query(설비 이력 조회) · T3 knowledge_search(결함 기준 검색) ·
T4 vlm_analyze(새 사진 직접 관찰 — 시스템의 유일한 눈).

도구는 판단하지 않는다 — 사실(검색 결과·관찰)만 dict로 반환하고, 해석은 오케스트레이터(SPEC-04) 몫.
실패도 예외 대신 {"error": 사유}로 반환해 에이전트 루프가 죽지 않게 한다 (R4).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src import config

# ── T2 history_query — "이 설비 기록 보여줘" ──────────────────

_D1_SCHEMA_FOR_LLM = """\
테이블: inspections (풍력 블레이드 점검 이력, 결함 1건 = 1행)
  filename TEXT     -- 사진 파일명 (예: 2025_sungsan_5_A_LeadingEdge_001.jpg)
  year INTEGER      -- 점검 연도
  site TEXT         -- 단지명, 소문자 영문 (예: sungsan, gangwon, yeongkwang)
  unit INTEGER      -- 호기 번호
  blade TEXT        -- 블레이드 (A/B/C)
  side TEXT         -- 부위 (LeadingEdge/TrailingEdge/PressureSide/SuctionSide)
  defect_type TEXT  -- 결함 종류 (예: Paint Damage, La Exposure). NULL = 정상
  severity INTEGER  -- 심각도 1~4. NULL = 정상
  file_path TEXT    -- 사진 파일 경로"""


def _default_sql_generator(question: str, schema: str) -> str:
    """gpt-4o-mini text-to-SQL. 테스트는 canned SQL 페이크를 주입한다."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    res = OpenAI().chat.completions.create(
        model=config.ORCHESTRATOR_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "다음 SQLite 스키마에 대한 SELECT 단일문만 출력한다. "
                "설명·마크다운 없이 SQL 텍스트만.\n\n" + schema,
            },
            {"role": "user", "content": question},
        ],
    )
    sql = (res.choices[0].message.content or "").strip()
    if sql.startswith("```"):  # 지시를 어기고 펜스를 두른 경우 벗긴다
        sql = sql.strip("`").removeprefix("sql").strip()
    return sql


def _reject_unless_single_select(sql: str) -> str | None:
    """R3 가드: SELECT 단일문만 통과. 위반 시 거부 사유를 돌려준다 (None = 통과)."""
    body = sql.strip().rstrip(";").strip()
    if not body:
        return "빈 SQL — 실행할 문장이 없음"
    if ";" in body:
        return "다중문 거부 — SELECT 단일문만 실행 가능 (R3)"
    if body.split()[0].upper() != "SELECT":
        return f"SELECT 문만 실행 가능 — '{body.split()[0]}' 구문 거부 (R3)"
    return None


def history_query(
    question: str,
    *,
    db_path: Path = config.SQLITE_PATH,
    sql_generator=None,
) -> dict:
    """T2: 자연어 조건 → LLM이 SQL 번역 → D1 조회. 실행된 SQL도 함께 반환 (검증 가능성).

    D1은 읽기 전용 모드로만 연다 — SELECT 가드가 뚫려도 쓰기가 물리적으로 불가능한 2차 방어선.
    """
    sql_generator = sql_generator or _default_sql_generator
    params = {"question": question}

    try:
        sql = sql_generator(question, _D1_SCHEMA_FOR_LLM)
    except Exception as e:
        return {"error": f"SQL 생성 실패: {e}", "params": params}

    rejection = _reject_unless_single_select(sql)
    if rejection:
        return {"error": rejection, "sql": sql, "params": params}

    try:
        con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
        try:
            con.row_factory = sqlite3.Row
            rows = [dict(r) for r in con.execute(sql).fetchall()]
        finally:
            con.close()
    except sqlite3.Error as e:
        return {"error": f"SQL 실행 오류: {e}", "sql": sql, "params": params}

    result = {"rows": rows, "count": len(rows), "sql": sql, "params": params}
    if not rows:
        result["message"] = "조회 조건에 해당하는 이력 없음"
    return result
