"""전역 설정 — 경로·모델명·상수. 값 변경은 이 파일에서만 한다. (SPEC-00 §3)"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── 경로 ──────────────────────────────────────────────
DATA_DIR = ROOT / "data"            # 실데이터 (gitignore)
STUB_DATA_DIR = ROOT / "stub_data"  # 개발용 스텁 (커밋)
STORAGE_DIR = ROOT / "storage"      # 인덱스 산출물 (gitignore, 삭제 = 완전 초기화)
SQLITE_PATH = STORAGE_DIR / "history.db"          # D1 설비 이력 DB
CHROMA_DIR = STORAGE_DIR / "chroma"               # V1/V2 벡터 DB
TESTSET_MANIFEST = ROOT / "eval" / "testset.txt"  # 시험지 파일명 목록 (커밋 — SPEC-02 R1)

# ── 모델 (버전 고정 — 평가 재현성, ADR-005) ────────────
ORCHESTRATOR_MODEL = "gpt-4o-mini"   # A1 판단·자기검증·리포트, T2 text-to-SQL
VLM_MODEL = "gpt-4o"                 # T4 vlm_analyze (시스템의 유일한 눈)
TEXT_EMBED_MODEL = "text-embedding-3-small"         # V2 (ADR-002)
IMAGE_EMBED_MODEL = "openai/clip-vit-base-patch32"  # V1, 로컬 CPU (ADR-002)

# ── 에이전트 상수 (SPEC-04) ───────────────────────────
MAX_ITERATIONS = 6           # 루프 반복 상한 (R3)
TOP_K = 5                    # visual/knowledge 검색 기본 k
SIMILARITY_THRESHOLD = 0.75  # 미만이면 "유사 사례 빈약" 신호 → 에스컬레이션 판단 재료 (R5)

# ── 벡터 컬렉션 이름 (SPEC-02) ────────────────────────
V1_COLLECTION = "images"     # 대표결함 크롭 임베딩 + 라벨 메타
V2_COLLECTION = "knowledge"  # 설명문 임베딩 + 출처 메타

# ── 평가 (SPEC-06) ───────────────────────────────────
TESTSET_SIZE = 20  # 샘플 데이터(풍력 75장) 기준 — 전체 ~1,000장 확보 시 50으로 복원
TESTSET_SEED = 42
