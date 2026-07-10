# ADR-004 — 저장소: Chroma(벡터) + SQLite(이력)

**Status**: Accepted (2026-07-10)

## Context

벡터 DB는 원 지시서가 Qdrant 또는 Chroma를 권장했고, 이력 DB는 SQLite를 권장했다.
운영 환경 없이 로컬 단일 머신에서 전부 돌아야 한다.

## Decision

- **벡터 DB: Chroma** (임베디드 모드) — 별도 서버·도커 없이 pip 설치만으로 동작, 데이터가
  `storage/` 폴더 파일로 남아 재현·초기화가 쉽다. V1(이미지)·V2(텍스트)를 컬렉션 2개로 분리
- **이력 DB: SQLite** — 표준 라이브러리, 파일 하나(`storage/history.db`)

## Consequences

- (+) 인프라 제로 — `uv sync` 후 바로 실행 가능, storage/ 삭제 = 완전 초기화
- (+) 메타데이터 필터(결함종류·심각도)를 Chroma가 기본 지원 → visual_search 필터 요구 충족
- (−) 수십만 벡터 이상 규모·동시성에는 부적합 — 전량(10만) 확장 시 Qdrant 재검토
- 기각한 대안: Qdrant — 성능은 좋으나 서버 프로세스 관리가 MVP에 불필요한 무게
