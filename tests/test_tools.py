"""SPEC-03 — tools.py 테스트.

LLM·임베더는 페이크 주입, 저장소는 스텁 데이터로 구축한 실제 인덱스를 쓴다 (완료 기준).
저장소는 모듈당 1회 구축(읽기 전용 사용이므로 공유 안전) — 도구는 질의 중 쓰기 금지(SPEC-00 §2).
"""

import sqlite3

import pytest

from src.config import STUB_DATA_DIR
from src.ingest import run_ingest
from src.tools import history_query


def fake_image_embedder(images):
    return [[0.1] * 8 for _ in images]


def fake_text_embedder(texts):
    return [[0.2] * 8 for _ in texts]


@pytest.fixture(scope="module")
def storage(tmp_path_factory):
    """스텁 6쌍 전부를 D1·V1·V2에 적재한 저장소 (testset_size=0 — 도구 테스트엔 제외 불필요)."""
    tmp = tmp_path_factory.mktemp("tools_storage")
    run_ingest(
        data_dir=STUB_DATA_DIR,
        storage_dir=tmp / "storage",
        testset_path=tmp / "testset.txt",
        image_embedder=fake_image_embedder,
        text_embedder=fake_text_embedder,
        testset_size=0,
    )
    return tmp / "storage"


def canned_sql(sql: str):
    """정해진 SQL을 돌려주는 페이크 LLM — text-to-SQL 부분을 결정적으로 만든다."""
    return lambda question, schema: sql


def d1_row_count(storage) -> int:
    con = sqlite3.connect(storage / "history.db")
    try:
        return con.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
    finally:
        con.close()


# ── 사이클 1: T2 history_query ──────────────────────────


def test_select_query_returns_rows(storage):
    """1. 정상 SELECT 단일문 → rows(dict 목록)와 실행 sql이 그대로 반환된다."""
    sql = (
        "SELECT year, defect_type, severity, filename FROM inspections "
        "WHERE site = 'sungsan' AND unit = 5"
    )
    result = history_query("sungsan 5호기 이력", db_path=storage / "history.db",
                           sql_generator=canned_sql(sql))
    assert "error" not in result
    assert result["count"] == len(result["rows"]) >= 1
    assert result["sql"] == sql
    first = result["rows"][0]
    assert set(first) == {"year", "defect_type", "severity", "filename"}


def test_non_select_rejected_and_db_intact(storage):
    """2. DELETE 시도 → error 반환 + D1 무변화 (R3 — DB 파괴 방지)."""
    before = d1_row_count(storage)
    result = history_query("이력 다 지워줘", db_path=storage / "history.db",
                           sql_generator=canned_sql("DELETE FROM inspections"))
    assert "error" in result and "SELECT" in result["error"]
    assert d1_row_count(storage) == before


def test_multi_statement_rejected(storage):
    """3. 'SELECT ...; DROP ...' 다중문 거부 — 테이블이 살아남는다 (R3)."""
    result = history_query(
        "이력 보여줘", db_path=storage / "history.db",
        sql_generator=canned_sql("SELECT * FROM inspections; DROP TABLE inspections"))
    assert "error" in result
    assert d1_row_count(storage) >= 1  # DROP이 실행됐다면 여기서 OperationalError


def test_zero_rows_returns_explicit_message(storage):
    """4. 0건 조회 → rows=[]·count=0에 '이력 없음' 메시지 명시 (R2 — 지어낼 재료 차단)."""
    result = history_query(
        "성산 9호기 이력", db_path=storage / "history.db",
        sql_generator=canned_sql("SELECT * FROM inspections WHERE unit = 9"))
    assert result["rows"] == [] and result["count"] == 0
    assert "없음" in result["message"]


def test_broken_sql_returns_error_not_exception(storage):
    """5. SQL 문법 오류 → 예외 대신 {"error": 사유} (R4 — 루프가 죽지 않는다)."""
    result = history_query("이력", db_path=storage / "history.db",
                           sql_generator=canned_sql("SELEC * FORM inspections"))
    assert "error" in result and result["error"]


def test_return_includes_query_params(storage):
    """6. 반환에 원 질의문 요약 포함 (R7 — 추론 로그 재구성 가능성)."""
    result = history_query("sungsan 5호기 이력", db_path=storage / "history.db",
                           sql_generator=canned_sql("SELECT filename FROM inspections"))
    assert result["params"]["question"] == "sungsan 5호기 이력"
