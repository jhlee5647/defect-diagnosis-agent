"""SPEC-01 — schema.py 테스트. 각 테스트 docstring에 대응 규칙 번호를 인용한다."""

import json

import pytest

from src.config import STUB_DATA_DIR
from src.schema import parse_label

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
