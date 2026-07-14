## 평가 결과 (SPEC-06)

### VQA: (a) VLM 단독 vs (b) 유사사례 few-shot 주입

| 조건 | 결함유무 | 유형판별 | 위치 | 특징 | 전체 |
|---|---|---|---|---|---|
| (a) VLM 단독 | 60% (12/20) | 42% (8/19) | 47% (8/17) | 37% (7/19) | 47% (35/75) |
| (b) RAG 주입 | 85% (17/20) | 63% (12/19) | 41% (7/17) | 58% (11/19) | 63% (47/75) |

형식실패(오답 처리): (a) 2건, (b) 11건 · few-shot 없이 진행(no_fewshot): 0장 · 크롭 폴백(no_crop): 1장 · 위치 라벨 오류 제외(label_error): 2장

### 라우팅 검증 (UC 5종 통과 제약)

프롬프트 개선(리포트 재료 충분성·비교 힌트, 83b5518) 후 UC-1·4·5 재실행 반영:

- UC-1: **PASS** — 호출 순서: ['vlm_analyze', 'visual_search', 'history_query', 'vlm_analyze', 'visual_search', 'knowledge_search', 'vlm_analyze']
- UC-2: **PASS** — 호출 순서: ['history_query']
- UC-3: **PASS** — 호출 순서: ['knowledge_search' × 8]
- UC-4: **FAIL** — 종결이 에스컬레이션이 아님 (필수 도구 미호출은 해소). 샘플 한계: 최희소 클래스(La Damage 1장)도 흔한 Paint Damage와 시각적으로 유사해 유사도·VLM 확신이 모두 높게 나옴 → R5 발동 조건 미충족 상태에서 확신 있는 오진(La Damage→Paint Damage). 에스컬레이션 트리거가 잡지 못하는 유형의 위험으로 2차 과제에 기록. 해결 경로: La Damage 사례 N장을 인덱스에 확보하면 visual_search 상위 결과에 클래스 경합이 생기므로, "1·2위 유사도 차 < δ이면서 클래스가 상이"를 R5 발동 조건에 추가하는 방식으로 검증 가능 (현재 1장으로는 경합 자체가 재현 불가)
- UC-5: **PASS** — 호출 순서: ['history_query', 'visual_search', 'history_query', 'vlm_analyze', 'history_query', 'vlm_analyze'] (history가 비교 관찰보다 먼저)

모델: vlm=gpt-4o, image_embed=openai/clip-vit-base-patch32, orchestrator=gpt-4o-mini
