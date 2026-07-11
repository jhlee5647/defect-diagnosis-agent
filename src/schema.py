"""라벨 JSON을 검증된 파이썬 객체로 변환하는 번역기 — SPEC-01.

데이터 함정(정상 사진의 annotations 부재, 좌표계 이원화, 다중 결함)을 이 모듈 한 곳에서 처리한다.
이후 모듈(ingest, tools, eval)은 JSON을 직접 만지지 않고 LabelDoc만 쓴다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_REQUIRED_BLOCKS = ("info", "image", "collection", "categories", "visionqa")
_FILENAME_TOKENS = 6  # 풍력: 연도_단지_호기_블레이드_부위_식별번호 (R4)


@dataclass(frozen=True)
class Annotation:
    """결함 1개 기록. bbox는 원본 해상도 좌표계의 (x, y, w, h) — COCO (R1)."""

    id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    area: float
    severity: int


@dataclass(frozen=True)
class FilenameMeta:
    """파일명에 박힌 설비 정보. site는 소문자 정규화 (R4)."""

    year: int
    site: str
    unit: int
    blade: str
    side: str
    seq: str


@dataclass(frozen=True)
class LabelDoc:
    """라벨 JSON 1개의 검증된 표현. 정상 사진이면 annotations=() (R2)."""

    filename: str
    width: int
    height: int
    location: str
    captured_at: str
    part_tag: str
    part_side_tag: str
    categories: dict[int, str]
    annotations: tuple[Annotation, ...]
    description: str
    vqa: dict
    cropped_bbox: tuple[float, float, float, float] | None


def _parse_bbox(raw_bbox, filename: str) -> tuple[float, float, float, float]:
    """bbox 검증 (R1): 길이 4의 [x, y, w, h], w·h 양수."""
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ValueError(f"{filename}: bbox는 [x, y, w, h] 4원소여야 함 — {raw_bbox!r}")
    x, y, w, h = (float(v) for v in raw_bbox)
    if w <= 0 or h <= 0:
        raise ValueError(f"{filename}: bbox의 w·h는 양수여야 함 — w={w}, h={h}")
    return (x, y, w, h)


def _parse_annotation(raw: dict, filename: str) -> Annotation:
    severity = int(raw["severity"])
    if not 1 <= severity <= 4:
        raise ValueError(f"{filename}: severity는 1~4여야 함 — {severity} (R6)")
    return Annotation(
        id=int(raw["id"]),
        category_id=int(raw["category_id"]),
        bbox=_parse_bbox(raw["bbox"], filename),
        area=float(raw["area"]),
        severity=severity,
    )


def parse_label(path: Path) -> LabelDoc:
    """라벨 JSON 파일 1개 → 검증된 LabelDoc. 검증 실패는 파일명 포함 ValueError."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    name = Path(path).name

    for block in _REQUIRED_BLOCKS:
        if block not in raw:
            raise ValueError(f"{name}: 필수 블록 '{block}' 누락 (R7)")

    vqa = raw["visionqa"]
    description = vqa.get("object_description")
    if not description:
        raise ValueError(f"{name}: visionqa.object_description 누락 (R7)")

    # R2: annotations 키 부재와 빈 리스트는 모두 정상 사진
    # 오류 메시지의 파일명은 파싱 대상 JSON 경로 기준 (내부 필드가 깨져도 항상 존재)
    annotations = tuple(_parse_annotation(a, name) for a in raw.get("annotations") or [])

    raw_crop = vqa.get("cropped_bbox")
    cropped_bbox = tuple(float(v) for v in raw_crop) if raw_crop else None

    return LabelDoc(
        filename=raw["image"]["filename"],
        width=int(raw["image"]["width"]),
        height=int(raw["image"]["height"]),
        location=raw["collection"]["location"],
        captured_at=raw["collection"]["datetime"],
        part_tag=raw["info"]["part_tag"],
        part_side_tag=raw["info"]["part_side_tag"],
        categories={int(c["id"]): c["name"] for c in raw["categories"]},
        annotations=annotations,
        description=description,
        vqa=vqa,  # R8: 원본 그대로 보존 (채점기 전용)
        cropped_bbox=cropped_bbox,
    )


def parse_filename(filename: str) -> FilenameMeta:
    """풍력 6토큰 파일명 분해 (R4). 형식 불일치는 ValueError — 스킵 여부는 호출자가 결정."""
    stem = filename.rsplit(".", 1)[0]
    tokens = stem.split("_")
    if len(tokens) != _FILENAME_TOKENS:
        raise ValueError(f"{filename}: 풍력 파일명은 6토큰이어야 함 — {len(tokens)}토큰 (R4)")
    year, site, unit, blade, side, seq = tokens
    try:
        return FilenameMeta(
            year=int(year), site=site.lower(), unit=int(unit), blade=blade, side=side, seq=seq
        )
    except ValueError as e:
        raise ValueError(f"{filename}: 연도·호기는 숫자여야 함 (R4)") from e


def primary_defect(doc: LabelDoc) -> Annotation | None:
    """대표 결함 선정 (R3): 심각도 최대 → 동률 시 면적 최대. VQA 출제 규칙과 동일해야 채점이 맞는다."""
    if not doc.annotations:
        return None
    return max(doc.annotations, key=lambda a: (a.severity, a.area))


def to_crop_coords(
    bbox: tuple[float, float, float, float],
    cropped_bbox: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float]:
    """원본 좌표 bbox → 크롭 좌표 (R5). 채점기(SPEC-06)와 vlm_analyze(SPEC-03)의 공유 구현.

    부분 이탈은 그대로 반환(음수 좌표 허용), 크롭 영역과 교집합이 없으면 라벨 오류로 ValueError.
    """
    if cropped_bbox is None:
        raise ValueError("cropped_bbox가 없는 문서 — 좌표 변환 불가 (R5)")
    x, y, w, h = bbox
    cx, cy, cw, ch = cropped_bbox
    no_overlap = x + w <= cx or x >= cx + cw or y + h <= cy or y >= cy + ch
    if no_overlap:
        raise ValueError(f"bbox {bbox}가 크롭 영역 {cropped_bbox}과 겹치지 않음 — 라벨 오류 (R5)")
    return (x - cx, y - cy, w, h)
