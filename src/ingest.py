"""P1 인덱싱 파이프라인 — SPEC-02.

data/의 사진+JSON 쌍을 세 저장소에 적재한다:
D1(SQLite 이력 표) · V1(Chroma 이미지 임베딩) · V2(Chroma 설명문 임베딩).
테스트셋(eval/testset.txt)은 V1·V2에서 하드 제외한다 — 누수는 평가 무효 (R1).

실행: uv run python -m src.ingest --data-dir data  (개발 중엔 stub_data)
"""

from __future__ import annotations

import argparse
import contextlib
import random
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from src import config
from src.schema import LabelDoc, parse_filename, parse_label, primary_defect

V1_NORMAL_MAX_SIDE = 512  # 정상 사진의 전체 축소본 긴 변 (R5)

_D1_DDL = """
CREATE TABLE inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    year INTEGER NOT NULL,
    site TEXT NOT NULL,
    unit INTEGER NOT NULL,
    blade TEXT NOT NULL,
    side TEXT NOT NULL,
    defect_type TEXT,          -- NULL = 정상
    severity INTEGER,          -- NULL = 정상
    file_path TEXT NOT NULL
)
"""


@dataclass
class IngestReport:
    """R7: '적재됐다'는 말 대신 증거가 되는 요약."""

    total: int = 0
    indexed: int = 0
    testset_excluded: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)
    d1_rows: int = 0
    v1_count: int = 0
    v2_count: int = 0
    elapsed_s: float = 0.0

    def summary(self) -> str:
        lines = [
            f"총 {self.total} = 벡터 적재 {self.indexed} + 시험 제외 {self.testset_excluded}"
            f" + 스킵 {len(self.skipped)}",
            f"D1 행 {self.d1_rows} / V1 {self.v1_count} / V2 {self.v2_count}"
            f" / 소요 {self.elapsed_s:.1f}s",
            f"모델: 이미지={config.IMAGE_EMBED_MODEL}, 텍스트={config.TEXT_EMBED_MODEL}",
        ]
        lines += [f"스킵: {name} — {reason}" for name, reason in self.skipped]
        return "\n".join(lines)


def _class_of(doc: LabelDoc) -> str:
    main = primary_defect(doc)
    return doc.categories[main.category_id] if main else "normal"


def select_testset(docs: list[LabelDoc], n: int, seed: int) -> list[str]:
    """시험지 선정 (R1): 결함 클래스별 + 정상 계층을 라운드로빈으로 고루. 시드 고정 = 재현 가능."""
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for doc in docs:
        groups.setdefault(_class_of(doc), []).append(doc.filename)
    for names in groups.values():
        names.sort()
        rng.shuffle(names)
    picked: list[str] = []
    while len(picked) < n and any(groups.values()):
        for cls in sorted(groups):
            if groups[cls] and len(picked) < n:
                picked.append(groups[cls].pop())
    return sorted(picked)


def _default_image_embedder(images):
    """CLIP ViT-B/32 로컬 CPU (ADR-002). 지연 임포트 — 테스트는 페이크 주입."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(config.IMAGE_EMBED_MODEL)
    processor = CLIPProcessor.from_pretrained(config.IMAGE_EMBED_MODEL)
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt")
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.tolist()


def _default_text_embedder(texts):
    """OpenAI text-embedding-3-small (ADR-002)."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    res = OpenAI().embeddings.create(model=config.TEXT_EMBED_MODEL, input=texts)
    return [d.embedding for d in res.data]


def _prepare_v1_image(doc: LabelDoc, image_path: Path):
    """R5: 대표결함 bbox 크롭, 정상 사진은 전체 축소본."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    main = primary_defect(doc)
    if main is not None:
        x, y, w, h = main.bbox
        return img.crop((int(x), int(y), int(x + w), int(y + h)))
    img.thumbnail((V1_NORMAL_MAX_SIDE, V1_NORMAL_MAX_SIDE))
    return img


def run_ingest(
    data_dir: Path,
    storage_dir: Path = config.STORAGE_DIR,
    testset_path: Path = config.TESTSET_MANIFEST,
    image_embedder=None,
    text_embedder=None,
    testset_size: int = config.TESTSET_SIZE,
    seed: int = config.TESTSET_SEED,
) -> IngestReport:
    """6단계: 스캔 → 파싱 → 시험지 분리 → 적재(D1 전부 / V1·V2는 시험지 제외) → 사후 검증 → 보고."""
    image_embedder = image_embedder or _default_image_embedder
    text_embedder = text_embedder or _default_text_embedder
    started = time.monotonic()
    report = IngestReport()

    # 1. 스캔
    json_paths = sorted(Path(data_dir).rglob("*.json"))
    report.total = len(json_paths)

    # 2. 파싱 — 깨진 파일은 스킵+사유 (R3), 카테고리 매핑은 파일 간 일관성 검증 (R4)
    parsed: list[tuple[Path, LabelDoc]] = []
    known_categories: dict[int, str] = {}
    for jp in json_paths:
        try:
            doc = parse_label(jp)
            parse_filename(doc.filename)  # 파일명 형식도 여기서 걸러 스킵 대상으로
        except ValueError as e:
            report.skipped.append((jp.name, str(e)))
            continue
        for cid, cname in doc.categories.items():
            if known_categories.setdefault(cid, cname) != cname:
                raise ValueError(
                    f"{jp.name}: categories 매핑 불일치 — id={cid}가 "
                    f"'{known_categories[cid]}'와 '{cname}'로 상충 (R4, 전체 중단)"
                )
        parsed.append((jp, doc))

    # 3. 시험지 분리 — 없으면 층화 생성 (R1). 파싱 성공분에서만 뽑는다
    testset_path = Path(testset_path)
    if testset_path.exists():
        testset = testset_path.read_text(encoding="utf-8").split()
    else:
        testset = select_testset([d for _, d in parsed], n=testset_size, seed=seed)
        testset_path.parent.mkdir(parents=True, exist_ok=True)
        testset_path.write_text("\n".join(testset) + "\n", encoding="utf-8")
    testset_set = set(testset)

    # 4. 적재 — 기존 인덱스는 전량 재구축 (R6).
    # 폴더 통삭제(rmtree)는 Windows에서 열린 핸들에 막히므로, DB 파일 교체 + 컬렉션 삭제로 재구축한다
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    db_path = storage_dir / "history.db"
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(db_path)
    con.execute(_D1_DDL)
    for jp, doc in parsed:  # D1에는 시험지 포함 전부 (R2)
        meta = parse_filename(doc.filename)
        file_path = str(jp.with_suffix(".jpg"))
        base = (doc.filename, meta.year, meta.site, meta.unit, meta.blade, meta.side)
        if doc.annotations:
            rows = [base + (doc.categories[a.category_id], a.severity, file_path)
                    for a in doc.annotations]
        else:
            rows = [base + (None, None, file_path)]
        con.executemany(
            "INSERT INTO inspections (filename, year, site, unit, blade, side,"
            " defect_type, severity, file_path) VALUES (?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    report.d1_rows = con.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
    con.close()

    import chromadb

    client = chromadb.PersistentClient(path=str(storage_dir / "chroma"))
    for name in (config.V1_COLLECTION, config.V2_COLLECTION):
        with contextlib.suppress(Exception):  # 없으면 통과 — 재구축 (R6)
            client.delete_collection(name)
    v1 = client.create_collection(config.V1_COLLECTION, metadata={"hnsw:space": "cosine"})
    v2 = client.create_collection(config.V2_COLLECTION, metadata={"hnsw:space": "cosine"})

    indexable = [(jp, doc) for jp, doc in parsed if doc.filename not in testset_set]
    report.testset_excluded = len(parsed) - len(indexable)
    report.indexed = len(indexable)

    if indexable:
        images = [_prepare_v1_image(doc, jp.with_suffix(".jpg")) for jp, doc in indexable]
        metadatas = []
        for _, doc in indexable:
            main = primary_defect(doc)
            meta = parse_filename(doc.filename)
            metadatas.append({
                "filename": doc.filename,
                "defect_type": _class_of(doc),
                "severity": main.severity if main else 0,
                "site": meta.site, "year": meta.year,
            })
        v1.add(
            ids=[doc.filename for _, doc in indexable],
            embeddings=image_embedder(images),
            metadatas=metadatas,
            documents=[doc.description for _, doc in indexable],
        )
        v2.add(
            ids=[doc.filename for _, doc in indexable],
            embeddings=text_embedder([doc.description for _, doc in indexable]),
            metadatas=[{**m, "source": "caption"} for m in metadatas],
            documents=[doc.description for _, doc in indexable],
        )
    report.v1_count = v1.count()
    report.v2_count = v2.count()

    # 5. 사후 검증 — 시험지 파일명이 V1·V2에 0건 (R1의 마지막 가드)
    for name, coll in (("V1", v1), ("V2", v2)):
        leaked = coll.get(ids=testset)["ids"] if testset else []
        if leaked:
            raise RuntimeError(f"테스트셋 누수: {name}에 {leaked} 존재 — 평가 무효 (R1)")

    report.elapsed_s = time.monotonic() - started
    return report


def main():
    parser = argparse.ArgumentParser(description="P1 인덱싱 파이프라인 (SPEC-02)")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    args = parser.parse_args()
    report = run_ingest(data_dir=args.data_dir)
    print(report.summary())


if __name__ == "__main__":
    main()
