"""SPEC-01 — schema.py 테스트. 각 테스트 docstring에 대응 규칙 번호를 인용한다."""

import json

import pytest

from src.config import STUB_DATA_DIR
from src.schema import parse_filename, parse_label, primary_defect, to_crop_coords

STUB_NORMAL = STUB_DATA_DIR / "2025_sungsan_2_C_PressureSide_005.json"  # 정상 사진 (annotations 키 없음)
STUB_SINGLE = STUB_DATA_DIR / "2025_sungsan_5_A_LeadingEdge_001.json"   # 단일 결함 (페인트 sev2)
STUB_MULTI = STUB_DATA_DIR / "2025_sungsan_5_A_LeadingEdge_003.json"    # 다중 결함 (paint sev2 + la sev3)


def load_stub(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_doc(tmp_path, doc, name="2025_sungsan_5_A_LeadingEdge_999.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return p


# ── 사이클 1: parse_label 파싱·검증 ──────────────────────────


def test_normal_image_without_annotations_key():
    """R2: annotations 키가 아예 없는 정상 사진 → 예외 없이 annotations=()."""
    doc = parse_label(STUB_NORMAL)
    assert doc.annotations == ()


def test_normal_image_with_empty_annotations_list(tmp_path):
    """R2: annotations가 빈 리스트여도 키 부재와 동일 취급."""
    raw = load_stub(STUB_NORMAL)
    raw["annotations"] = []
    doc = parse_label(write_doc(tmp_path, raw))
    assert doc.annotations == ()


def test_bbox_wrong_length_raises(tmp_path):
    """R1: bbox 원소가 4개가 아니면 ValueError, 메시지에 파일명 포함."""
    raw = load_stub(STUB_SINGLE)
    raw["annotations"][0]["bbox"] = [100.0, 200.0, 300.0]
    path = write_doc(tmp_path, raw)
    with pytest.raises(ValueError, match=path.name.replace(".json", "")):
        parse_label(path)


def test_bbox_nonpositive_size_raises(tmp_path):
    """R1: w 또는 h가 0·음수인 bbox는 ValueError."""
    raw = load_stub(STUB_SINGLE)
    raw["annotations"][0]["bbox"] = [100.0, 200.0, 0.0, 250.0]
    with pytest.raises(ValueError):
        parse_label(write_doc(tmp_path, raw))
    raw["annotations"][0]["bbox"] = [100.0, 200.0, 400.0, -5.0]
    with pytest.raises(ValueError):
        parse_label(write_doc(tmp_path, raw))


def test_severity_out_of_range_raises(tmp_path):
    """R6: severity는 1~4만 허용 — 0과 5는 ValueError."""
    raw = load_stub(STUB_SINGLE)
    for bad in (0, 5):
        raw["annotations"][0]["severity"] = bad
        with pytest.raises(ValueError):
            parse_label(write_doc(tmp_path, raw))


def test_missing_required_block_raises(tmp_path):
    """R7: vision_qa 또는 categories 블록이 없으면 ValueError."""
    for missing in ("vision_qa", "categories"):
        raw = load_stub(STUB_SINGLE)
        del raw[missing]
        with pytest.raises(ValueError):
            parse_label(write_doc(tmp_path, raw))


def test_vqa_preserved_verbatim():
    """R8: 파싱 후 vqa dict는 원본 JSON의 vision_qa와 완전히 동일 (가공 금지)."""
    doc = parse_label(STUB_SINGLE)
    assert doc.vqa == load_stub(STUB_SINGLE)["vision_qa"]


def test_valid_file_parses_all_fields():
    """정상 파싱: 스텁 001의 주요 필드와 categories 매핑 확인."""
    doc = parse_label(STUB_SINGLE)
    assert doc.filename == "2025_sungsan_5_A_LeadingEdge_001.jpg"
    assert (doc.width, doc.height) == (8256, 5504)
    assert doc.location == "Sungsan"
    assert doc.categories[3] == "Paint Damage"
    assert len(doc.annotations) == 1
    ann = doc.annotations[0]
    assert ann.category_id == 3 and ann.severity == 2
    assert ann.bbox == (3500.0, 1200.0, 400.0, 250.0)
    assert "페인트" in doc.description
    assert doc.cropped_bbox == (3082.0, 668.0, 1920.0, 1080.0)


# ── 사이클 2: 파일명 파싱 · 대표결함 선정 · 좌표 변환 ──────────


def test_parse_filename_six_tokens():
    """R4: 6토큰 풍력 파일명 → year/site/unit/blade/side/seq 분해."""
    meta = parse_filename("2025_sungsan_5_A_LeadingEdge_056.jpg")
    assert (meta.year, meta.site, meta.unit) == (2025, "sungsan", 5)
    assert (meta.blade, meta.side, meta.seq) == ("A", "LeadingEdge", "056")


def test_parse_filename_normalizes_site_case():
    """R4: 대문자 단지명(Sungsan)은 소문자로 정규화 — 이력 DB에서 단지가 갈라지는 것 방지."""
    assert parse_filename("2022_Sungsan_3_A_LeadingEdge_045.jpg").site == "sungsan"


def test_parse_filename_rejects_five_tokens():
    """R4: 태양광 형식(5토큰)은 ValueError — 조용한 오파싱 차단."""
    with pytest.raises(ValueError):
        parse_filename("2023_solarsido_normal_panelFront_00001.jpg")


def test_primary_defect_severity_first():
    """R3: 다중 결함(스텁 003: paint sev2 + la sev3) → 심각도 높은 라미네이트 노출 선정."""
    doc = parse_label(STUB_MULTI)
    assert len(doc.annotations) == 2
    picked = primary_defect(doc)
    assert picked.category_id == 4 and picked.severity == 3


def test_primary_defect_area_breaks_tie(tmp_path):
    """R3: 심각도 동률이면 면적 큰 결함 선정 (VQA 출제 규칙과 동일)."""
    raw = load_stub(STUB_MULTI)
    for ann, area in zip(raw["annotations"], (70000.0, 150000.0)):
        ann["severity"] = 2
        ann["area"] = area
    doc = parse_label(write_doc(tmp_path, raw))
    assert primary_defect(doc).area == 150000.0


def test_primary_defect_none_for_normal():
    """R3: 결함 없는 정상 사진 → None."""
    assert primary_defect(parse_label(STUB_NORMAL)) is None


def test_to_crop_coords_translates():
    """R5: 원본 좌표 → 크롭 좌표 = (x−크롭x, y−크롭y, w·h 그대로)."""
    got = to_crop_coords((3500.0, 1200.0, 400.0, 250.0), (3082.0, 668.0, 1920.0, 1080.0))
    assert got == (418.0, 532.0, 400.0, 250.0)


def test_to_crop_coords_requires_crop():
    """R5: cropped_bbox가 None이면 ValueError — None 연산으로 엉뚱한 좌표 방지."""
    with pytest.raises(ValueError):
        to_crop_coords((3500.0, 1200.0, 400.0, 250.0), None)


def test_to_crop_coords_partial_vs_complete_departure():
    """R5: 크롭 영역 부분 이탈은 그대로 반환, 완전 이탈(교집합 없음)은 ValueError."""
    crop = (3082.0, 668.0, 1920.0, 1080.0)
    # 부분 이탈: bbox가 크롭 왼쪽 경계에 걸침 → 음수 좌표 그대로 반환
    assert to_crop_coords((3000.0, 600.0, 400.0, 250.0), crop) == (-82.0, -68.0, 400.0, 250.0)
    # 완전 이탈: 크롭 영역과 겹치지 않음 → 라벨 데이터 오류로 즉시 실패
    with pytest.raises(ValueError):
        to_crop_coords((100.0, 100.0, 50.0, 50.0), crop)
