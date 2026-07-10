"""SPEC-02 — ingest.py 테스트. 임베딩은 페이크로 대체해 저장 로직만 검증한다 (완료 기준)."""

import json
import shutil
import sqlite3

import pytest

from src.config import STUB_DATA_DIR
from src.ingest import run_ingest

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
