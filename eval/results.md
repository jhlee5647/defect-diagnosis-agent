## 평가 결과 (SPEC-06)

### VQA: (a) VLM 단독 vs (b) 유사사례 few-shot 주입

| 조건 | 결함유무 | 유형판별 | 위치 | 특징 | 전체 |
|---|---|---|---|---|---|
| (a) VLM 단독 | 60% (12/20) | 42% (8/19) | 47% (8/17) | 37% (7/19) | 47% (35/75) |
| (b) RAG 주입 | 85% (17/20) | 63% (12/19) | 41% (7/17) | 58% (11/19) | 63% (47/75) |

형식실패(오답 처리): (a) 2건, (b) 11건 · few-shot 없이 진행(no_fewshot): 0장 · 크롭 폴백(no_crop): 1장 · 위치 라벨 오류 제외(label_error): 2장

### 라우팅 검증 (UC 5종 통과 제약)

- UC-1: **FAIL** — 필수 도구 미호출: ['history_query', 'knowledge_search']
- UC-2: **PASS** — 호출 순서: ['history_query']
- UC-3: **PASS** — 호출 순서: ['knowledge_search', 'knowledge_search', 'knowledge_search', 'knowledge_search', 'knowledge_search', 'knowledge_search', 'knowledge_search', 'knowledge_search']
- UC-4: **FAIL** — 필수 도구 미호출: ['knowledge_search']; 종결이 에스컬레이션이 아님 — 억지 단정 (SPEC-04 함정 4)
- UC-5: **FAIL** — 필수 도구 미호출: ['history_query']

모델: vlm=gpt-4o, image_embed=openai/clip-vit-base-patch32, orchestrator=gpt-4o-mini
