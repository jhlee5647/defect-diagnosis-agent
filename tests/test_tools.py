"""SPEC-03 — tools.py 테스트.

LLM·임베더는 페이크 주입, 저장소는 스텁 데이터로 구축한 실제 인덱스를 쓴다 (완료 기준).
저장소는 모듈당 1회 구축(읽기 전용 사용이므로 공유 안전) — 도구는 질의 중 쓰기 금지(SPEC-00 §2).
"""

import sqlite3

import pytest

import chromadb

from src.config import STUB_DATA_DIR, V2_COLLECTION
from src.ingest import run_ingest
from src.tools import history_query, knowledge_search, visual_search


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


# ── 사이클 2: T3 knowledge_search ────────────────────────


def test_knowledge_search_returns_sourced_snippets(storage):
    """7. 정상 질의 → {text, source, similarity} k건. 전 항목 출처 필수 + 파라미터 요약 (R7)."""
    result = knowledge_search("라미네이트 노출 기준", k=3, chroma_dir=storage / "chroma",
                              text_embedder=fake_text_embedder)
    assert "error" not in result
    assert result["count"] == len(result["results"]) == 3
    for item in result["results"]:
        assert item["text"], "설명문 본문이 비어 있음"
        assert ".jpg" in item["source"], "출처 없는 문단 반환 금지"
        assert -1.0 <= item["similarity"] <= 1.0
    assert result["params"] == {"query": "라미네이트 노출 기준", "k": 3}


def test_knowledge_search_empty_corpus(tmp_path):
    """8. 빈 코퍼스 질의 → 명시적 '결과 없음' 메시지 (R2 — 지어내기 금지)."""
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    client.create_collection(V2_COLLECTION, metadata={"hnsw:space": "cosine"})
    result = knowledge_search("아무 질의", chroma_dir=tmp_path / "chroma",
                              text_embedder=fake_text_embedder)
    assert result["results"] == [] and result["count"] == 0
    assert "없음" in result["message"]


# ── 사이클 2: T1 visual_search ───────────────────────────

QUERY_IMAGE = STUB_DATA_DIR / "2025_sungsan_5_A_LeadingEdge_001.jpg"


def test_visual_search_basic(storage):
    """9. 기본 검색 → {filename, defect_type, severity, similarity, description} k건."""
    result = visual_search(QUERY_IMAGE, k=3, chroma_dir=storage / "chroma",
                           image_embedder=fake_image_embedder)
    assert "error" not in result
    assert result["count"] == len(result["results"]) == 3
    for item in result["results"]:
        assert set(item) >= {"filename", "defect_type", "severity", "similarity", "description"}
        assert -1.0 <= item["similarity"] <= 1.0
    assert result["params"]["k"] == 3


def test_visual_search_filters(storage):
    """10. 필터 재검색(UC-4) — defect_type·severity 필터가 결과를 실제로 좁힌다."""
    by_type = visual_search(QUERY_IMAGE, k=6, defect_type="Paint Damage",
                            chroma_dir=storage / "chroma", image_embedder=fake_image_embedder)
    assert by_type["count"] == 2  # 스텁 001·002 (003의 대표결함은 La Exposure)
    assert all(i["defect_type"] == "Paint Damage" for i in by_type["results"])

    by_sev = visual_search(QUERY_IMAGE, k=6, severity=3,
                           chroma_dir=storage / "chroma", image_embedder=fake_image_embedder)
    assert by_sev["count"] == 2  # 스텁 003(La Exposure)·006(Vortex Generator)
    assert all(i["severity"] == 3 for i in by_sev["results"])


def test_visual_search_returns_fewer_when_corpus_small(storage):
    """11. k 미달 — 코퍼스(6건)보다 큰 k=50 요청 시 있는 만큼만, 지어내지 않는다."""
    result = visual_search(QUERY_IMAGE, k=50, chroma_dir=storage / "chroma",
                           image_embedder=fake_image_embedder)
    assert result["count"] == 6


def test_visual_search_embeds_crop_region(storage):
    """12. 크롭 영역 (x,y,w,h) 지정 시 그 영역만 임베딩에 들어간다 (원본 통짜 아님)."""
    seen_sizes = []

    def recording_embedder(images):
        seen_sizes.extend(img.size for img in images)
        return [[0.1] * 8 for _ in images]

    visual_search(QUERY_IMAGE, crop=(100, 200, 640, 480), chroma_dir=storage / "chroma",
                  image_embedder=recording_embedder)
    assert seen_sizes == [(640, 480)]
