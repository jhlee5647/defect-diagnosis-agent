# CLAUDE.md — 발전설비 결함 진단 멀티모달 Agentic RAG

풍력 점검 사진을 진단하고 근거를 인용하는 단일 에이전트 RAG 시스템. 현재 MVP 구축 단계.

## 문서 맵 (스펙이 곧 요구사항 — 지시서 원문은 폐기됨)

- **전역 참조**: `docs/specs/SPEC-00-overview.md` — 아키텍처 원칙·성공 기준·파일 트리 계약. 항상 먼저 읽는다
- **작업 단위별**: 해당 SPEC **하나만** 로드해 작업한다 (스펙↔소스↔테스트 1:1:1)
  - SPEC-01 `src/schema.py` · SPEC-02 `src/ingest.py` · SPEC-03 `src/tools.py`
  - SPEC-04 `src/orchestrator.py` · SPEC-05 `src/demo.py` · SPEC-06 `eval/run_eval.py`
- **용어**: `docs/GLOSSARY.md` / **기술 결정 근거**: `docs/adr/ADR-001~005`

## 방법론 규칙 (필수 준수)

1. **스펙이 계약이다.** 구현 착수 전 해당 SPEC을 읽는다. 스펙에 없는 결정이 필요해지면
   **진행을 멈추고 사용자에게 질문한다.** 임의로 정하고 나중에 보고하지 않는다.
2. **스펙 개정은 코드보다 먼저.** 구현 중 스펙과 현실이 충돌하면: 멈춤 → 개정안 제시 →
   사용자 승인 → 스펙 수정 → 그 다음 구현. 코드를 먼저 고치는 것 금지.
3. **파일 트리 계약** (SPEC-00 §5). 계약에 없는 파일을 만들기 전에 반드시 보고한다.
   개발 단위 = 파일 1개 — 한 단위 작업 중 다른 파일 수정이 필요하면 멈추고 보고.
4. **TDD 사이클**: red → green → refactor → commit.
   - 결정적 코드(파싱·좌표변환·채점기·SQL 가드): pytest로 진짜 red 먼저
   - LLM 경유 코드(라우팅·관찰·SQL 생성): 실행 가능한 검증 기준(트레이스 단언·스모크)을 먼저 작성
5. **게이트 = 파일 단위**: 단위 시작 시 테스트 목록을 사용자에게 확인받고, 완료 시
   실행 증거(테스트 출력·실행 로그)를 제시한다. "완료했다"는 말 대신 항상 실행 결과 첨부.
6. **커밋**: Conventional Commits (`feat:` `fix:` `test:` `docs:` `chore:` `refactor:`)

## 개발 순서 (Phase B, 의존성 순)

B-0 스캐폴딩(pyproject·config·스텁) → B-1 schema → B-2 ingest → B-3 tools(T2→T3→T1→T4)
→ B-4 orchestrator → B-5 demo → B-6 eval

## 명령어

```bash
uv sync                              # 의존성 설치
uv run pytest                        # 단위 테스트
uv run python -m src.ingest --data-dir data   # 인덱스 구축 (개발 중엔 stub_data)
uv run python -m src.demo            # 웹 데모
uv run python eval/run_eval.py --limit 5      # 평가 (소량 시험 실행)
```

## 데이터·시크릿 주의

- `data/`(원본 데이터셋)와 `storage/`(인덱스 산출물)는 **git 커밋 금지** (.gitignore 확인)
- 시크릿은 `.env`에만 (`OPENAI_API_KEY`). 코드·커밋·로그에 노출 금지
- **테스트셋 누수 금지** — `eval/testset.txt`의 파일은 벡터 인덱스(V1·V2)에 절대 안 들어감
  (SPEC-02 R1, 위반 = 평가 무효). 인덱싱 로직 수정 시 이 가드를 깨지 않는지 확인
