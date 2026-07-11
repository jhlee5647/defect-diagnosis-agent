# SPEC-00 — 시스템 개요 (전역 참조)

| 항목 | 내용 |
|---|---|
| 프로젝트 | 발전설비(풍력) 결함 진단 멀티모달 Agentic RAG |
| 목적 | 신규 점검 사진을 넣으면 과거 유사 사례·설비 이력·결함 기준을 스스로 검색해 근거로 제시하며 진단하고 리포트를 생성 |
| 상태 | Approved |

## 1. MVP 성공 기준 (완료 정의)

1. **E2E 데모**: 이미지 업로드 + 질의 → 중간 추론 로그 표시 → 6항목 리포트 출력 (Gradio)
2. **Agentic 증명**: UC 5종(SPEC-04 §5)의 도구 실행 경로가 서로 다름 + 재시도/보강 루프 최소 1개 동작
3. **미니 평가**: VQA 테스트 ~50장에서 (a) VLM 단독 vs (b) 유사사례 few-shot 주입 정답률 비교표 (SPEC-06)

## 2. 아키텍처 원칙 (위반 = 설계 오류)

- **단일 에이전트**: 판단(계획·도구 선택·충분성·검증)의 주체는 오케스트레이터 A1 하나뿐. 도구 4종은 자율성 없는 함수 — 스스로 계획하거나 다른 도구를 호출하지 않는다.
- **지능 배치는 3곳뿐**: ① A1 = 텍스트 LLM ② T4의 눈 = VLM ③ 임베딩 2종. 그 외 전부 결정적 코드.
- **이미지를 "보는" 지점은 T4뿐**. A1은 텍스트 전용 — 이미지 이해는 전부 T4의 관찰 텍스트를 경유.
- **금지 상호작용**: 도구 간 직접 호출(T→T) / 도구의 Evidence Store 직접 접근(기록은 A1만) / 질의 처리 중 V1·V2·D1 쓰기 / 사용자-도구 직접 상호작용.

## 3. 구성요소

| ID | 이름 | 유형 | 구현 |
|---|---|---|---|
| A1 | 오케스트레이터 | 유일 에이전트 | LangGraph + gpt-4o-mini (SPEC-04) |
| S0 | Evidence Store | 질의 단위 인메모리 상태 | SPEC-04 |
| T1 | visual_search | 도구(함수) | 로컬 CLIP ViT-B/32 CPU → Chroma V1 (SPEC-03) |
| T2 | history_query | 도구(함수) | gpt-4o-mini text-to-SQL → SQLite D1 (SPEC-03) |
| T3 | knowledge_search | 도구(함수) | text-embedding-3-small → Chroma V2 (SPEC-03) |
| T4 | vlm_analyze | 도구(함수) | gpt-4o vision, few-shot 주입 지원 (SPEC-03) |
| V1/V2 | 벡터 DB | 저장소 | Chroma (storage/chroma) |
| D1 | 설비 이력 DB | 저장소 | SQLite (storage/history.db) |
| P1 | 인덱싱 파이프라인 | 오프라인 배치 | SPEC-02, 온라인 플레인은 읽기 전용 |

모델 버전은 config.py에 상수로 고정 (평가 재현성).

## 4. 데이터 스코프

풍력만 ~1,000장 스토리 기반 서브셋 (ADR-003). 태양광 열화상 제외.
테스트셋 ~50장은 벡터 인덱스에서 **하드 제외** — 누수는 실격 사유 (SPEC-02).

## 5. 파일 트리 계약 (이 목록에 없는 파일 생성 전 사전 보고)

```
├── pyproject.toml, README.md, .env(비커밋), .gitignore, CLAUDE.md
├── docs/specs/SPEC-00~06.md, docs/adr/ADR-001~005.md
├── src/config.py, schema.py, ingest.py, tools.py, orchestrator.py, demo.py
├── eval/run_eval.py, eval/testset.txt, eval/results.md  # results.md = 평가 결과표 (SPEC-06 §3 산출물)
├── tests/test_schema.py, test_ingest.py, test_tools.py, test_orchestrator.py, test_eval.py
├── stub_data/   # 개발용 스텁 (커밋), data/  # 실데이터 (비커밋), storage/  # 산출물 (비커밋)
```

## 6. 비스코프 (요청되어도 착수 금지, 2차 과제)

태양광 파이프라인 / VLM 파인튜닝 / 시계열 결함 자동 매칭 / 멀티턴 메모리·계정 관리 /
MMR 다양성·타일링·폴리곤 활용 / 증분 인덱싱 / 컨테이너화·프로덕션 배포 / E2·E4·ablation 평가

## 7. 관련 문서

개발 시 해당 단위의 SPEC 1개만 로드: SPEC-01(schema) SPEC-02(ingest) SPEC-03(tools) SPEC-04(orchestrator) SPEC-05(demo) SPEC-06(eval). 기술 선택 근거는 ADR-001~005.
