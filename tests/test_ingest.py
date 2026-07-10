"""SPEC-02 — ingest.py 테스트. 임베딩은 페이크로 대체해 저장 로직만 검증한다 (완료 기준)."""

import json
import shutil
import sqlite3

import pytest

import chromadb

from src.config import STUB_DATA_DIR, V1_COLLECTION, V2_COLLECTION
from src.ingest import run_ingest, select_testset
from src.schema import parse_label

STUBS = sorted(p.name for p in STUB_DATA_DIR.glob("*.json"))


def fake_image_embedder(images):
    return [[0.1] * 8 for _ in images]


def fake_text_embedder(texts):
    return [[0.2] * 8 for _ in texts]


@pytest.fixture
def dataset(tmp_path):
    """스텁 6쌍(JSON+JPG)을 임시 데이터 폴더로 복사."""
    data = tmp_path / "data"
    data.mkdir()
    for p in STUB_DATA_DIR.iterdir():
        shutil.copy(p, data / p.name)
    return data


def ingest(tmp_path, data_dir, n=2, **kwargs):
    return run_ingest(
        data_dir=data_dir,
        storage_dir=tmp_path / "storage",
        testset_path=tmp_path / "testset.txt",
        image_embedder=kwargs.pop("image_embedder", fake_image_embedder),
        text_embedder=kwargs.pop("text_embedder", fake_text_embedder),
        testset_size=n,
        **kwargs,
    )


def d1(tmp_path):
    return sqlite3.connect(tmp_path / "storage" / "history.db")


# ── 사이클 1: 파싱·스킵·D1 적재 ──────────────────────────


def test_broken_file_skipped_with_reason(tmp_path, dataset):
    """R3: 깨진 JSON 1개가 배치를 죽이지 않고, 스킵 목록에 파일명+사유로 남는다."""
    (dataset / "2025_broken_1_A_LeadingEdge_777.json").write_text("{ not json", encoding="utf-8")
    report = ingest(tmp_path, dataset)
    assert report.total == 7
    assert len(report.skipped) == 1
    name, reason = report.skipped[0]
    assert "777" in name and reason


def test_category_mapping_conflict_aborts(tmp_path, dataset):
    """R4: 파일 간 카테고리 번호↔이름 매핑 불일치는 스킵이 아니라 전체 중단."""
    victim = dataset / STUBS[0]
    raw = json.loads(victim.read_text(encoding="utf-8"))
    raw["categories"][2]["name"] = "Contamination"  # id=3을 다른 이름으로 조작
    victim.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="categories"):
        ingest(tmp_path, dataset)


def test_multi_defect_photo_gets_multiple_rows(tmp_path, dataset):
    """다중 결함 사진(스텁 003: paint+la) → D1에 행 2개 (결함 1건=1행)."""
    ingest(tmp_path, dataset)
    rows = d1(tmp_path).execute(
        "SELECT COUNT(*) FROM inspections WHERE filename LIKE '%_003.jpg'"
    ).fetchone()
    assert rows[0] == 2


def test_normal_photo_gets_null_row(tmp_path, dataset):
    """정상 사진(스텁 005) → defect_type·severity NULL인 행 1개."""
    ingest(tmp_path, dataset)
    rows = d1(tmp_path).execute(
        "SELECT defect_type, severity FROM inspections WHERE filename LIKE '%_005.jpg'"
    ).fetchall()
    assert rows == [(None, None)]


def test_d1_columns_are_correct(tmp_path, dataset):
    """D1 필수 컬럼 값 정확성(스텁 001) — site는 소문자 정규화."""
    ingest(tmp_path, dataset)
    row = d1(tmp_path).execute(
        "SELECT year, site, unit, blade, side, defect_type, severity, file_path "
        "FROM inspections WHERE filename LIKE '%_001.jpg'"
    ).fetchone()
    year, site, unit, blade, side, defect_type, severity, file_path = row
    assert (year, site, unit, blade) == (2025, "sungsan", 5, "A")
    assert side == "LeadingEdge"
    assert (defect_type, severity) == ("Paint Damage", 2)
    assert file_path.endswith("2025_sungsan_5_A_LeadingEdge_001.jpg")


def test_testset_files_still_in_d1(tmp_path, dataset):
    """R2: 테스트셋으로 뽑힌 파일도 이력 표(D1)에는 포함된다 (벡터만 제외)."""
    ingest(tmp_path, dataset, n=2)
    testset = (tmp_path / "testset.txt").read_text(encoding="utf-8").split()
    assert len(testset) == 2
    con = d1(tmp_path)
    for fn in testset:
        count = con.execute(
            "SELECT COUNT(*) FROM inspections WHERE filename = ?", (fn,)
        ).fetchone()[0]
        assert count >= 1, f"{fn}이 D1에 없음"


# ── 사이클 2: 시험지 선정·V1/V2 적재·재구축 ──────────────


def all_stub_docs():
    return [parse_label(STUB_DATA_DIR / name) for name in STUBS]


def chroma(tmp_path, collection):
    client = chromadb.PersistentClient(path=str(tmp_path / "storage" / "chroma"))
    return client.get_collection(collection)


def test_select_testset_is_deterministic():
    """R1: 시드 고정 → 몇 번을 뽑아도 같은 시험지 (재현성)."""
    docs = all_stub_docs()
    assert select_testset(docs, n=3, seed=42) == select_testset(docs, n=3, seed=42)


def test_select_testset_stratified_includes_normal():
    """R1: 층화 선정 — 결함 클래스별로 고루 + 정상 사진 계층 포함."""
    picked = select_testset(all_stub_docs(), n=5, seed=42)
    assert len(picked) == 5
    assert any("_005.jpg" in name for name in picked), "정상 사진이 시험지에 없음"
    # 스텁 6개의 클래스는 5그룹(paint/la/contamination/vortex/normal) → 5장이면 전 그룹 커버
    from src.ingest import _class_of
    docs_by_name = {d.filename: d for d in all_stub_docs()}
    assert len({_class_of(docs_by_name[n]) for n in picked}) == 5


def test_first_run_creates_testset_file(tmp_path, dataset):
    """R1: testset.txt 없는 최초 실행 → 자동 생성 후 진행."""
    assert not (tmp_path / "testset.txt").exists()
    ingest(tmp_path, dataset, n=2)
    lines = (tmp_path / "testset.txt").read_text(encoding="utf-8").split()
    assert len(lines) == 2


def test_testset_absent_from_vectors(tmp_path, dataset):
    """R1 사후 검증: 시험지 파일명이 V1·V2에 0건 — 컨닝 3중 가드의 마지막."""
    report = ingest(tmp_path, dataset, n=2)
    testset = (tmp_path / "testset.txt").read_text(encoding="utf-8").split()
    for coll_name in (V1_COLLECTION, V2_COLLECTION):
        found = chroma(tmp_path, coll_name).get(ids=testset)["ids"]
        assert found == [], f"{coll_name}에 시험지 유입: {found}"
    assert report.v1_count == report.indexed == 4  # 6 - 시험지 2


def test_v1_embeds_crop_for_defect_and_thumbnail_for_normal(tmp_path, dataset):
    """R5: 결함 사진은 대표결함 bbox 크롭, 정상 사진은 긴 변 제한 축소본을 임베딩."""
    sizes = []

    def recording_embedder(images):
        sizes.extend(img.size for img in images)
        return [[0.1] * 8 for _ in images]

    ingest(tmp_path, dataset, n=0, image_embedder=recording_embedder)
    assert (400, 250) in sizes  # 스텁 001의 대표결함 bbox (w=400, h=250)
    assert all(max(s) <= 8256 // 2 for s in sizes), "원본 통짜(8256px)가 임베딩에 들어감"
    assert any(max(s) <= 512 for s in sizes), "정상 사진 축소본(≤512)이 없음"


def test_v2_metadata_has_filename_and_source(tmp_path, dataset):
    """V2: 설명문 1건=항목 1건, 메타에 파일명·출처 포함 — 근거 인용의 재료."""
    report = ingest(tmp_path, dataset, n=0)
    got = chroma(tmp_path, V2_COLLECTION).get()
    assert len(got["ids"]) == report.indexed == 6
    for meta in got["metadatas"]:
        assert meta["filename"].endswith(".jpg") and meta["source"] == "caption"


def test_rerun_rebuilds_without_duplicates(tmp_path, dataset):
    """R6: 재실행 = 전량 재구축 — 건수가 그대로여야 함 (중복 적재 없음)."""
    first = ingest(tmp_path, dataset, n=2)
    second = ingest(tmp_path, dataset, n=2)
    assert (first.d1_rows, first.v1_count, first.v2_count) == (
        second.d1_rows, second.v1_count, second.v2_count)


def test_report_numbers_are_consistent(tmp_path, dataset):
    """R7: 총계 = 벡터 적재 + 시험 제외 + 스킵 (보고서 숫자의 출처 추적 가능성)."""
    (dataset / "2025_broken_1_A_LeadingEdge_777.json").write_text("{ bad", encoding="utf-8")
    report = ingest(tmp_path, dataset, n=2)
    assert report.total == report.indexed + report.testset_excluded + len(report.skipped)
    assert (report.total, report.indexed, report.testset_excluded) == (7, 4, 2)
