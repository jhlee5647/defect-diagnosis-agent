# SPEC-01 — schema: 라벨 JSON을 믿을 수 있는 객체로 바꾸는 번역기

| 항목 | 내용 |
|---|---|
| 담당 파일 | `src/schema.py` / 테스트 `tests/test_schema.py` |
| 의존성 | 표준 라이브러리만 (LLM·DB·이미지 처리 없음 — 순수 함수) |
| 용어 | [GLOSSARY.md](../GLOSSARY.md) 참조 |
| 상태 | Draft |

## 1. 스토리 — 이 모듈이 하는 일

데이터셋의 단위는 **사진 1장 + 라벨 JSON 1개** 쌍이다. JSON에는 ① 무엇을 찍었나(성산 5호기 A날개 앞전),
② 결함이 어디에 몇 개(annotation 목록), ③ 전문가 설명문과 4지선다 문제(VQA)가 들어 있다:

```jsonc
{
  "info":        { "...데이터셋·버전 메타...": "" },
  "image":       { "filename": "2025_sungsan_5_A_LeadingEdge_056.jpg", "width": 8256, "height": 5504 },
  "collection":  { "...location·촬영일시·part_tag·part_side_tag 등 설비 정보...": "" },
  "categories":  [ { "id": 3, "name": "Paint Damage" } /* ...파일마다 들어 있는 번호↔이름 매핑... */ ],
  "annotations": [ { "bbox": [3500, 1200, 400, 250], "category_id": 3, "severity": 2 } ],
  "visionqa":    { "object_description": "앞전 중앙부에 페인트 박리가...",
                   "defect_classification_q": "이 결함의 유형은?",
                   "defect_classification_option": { "classification_option_a": "Paint Damage", "...": "" },
                   "defect_classification_a": "a",
                   "cropped_bbox": [3082, 668, 1920, 1080] }
}
```

(2026-07-11 실데이터 확인 반영: 블록명은 `visionqa`(언더스코어 없음), 보기는 `defect_*_option`
중첩 dict, 정답은 `defect_*_a`. 정상 사진의 visionqa에는 결함유무 문항·설명문만 있고
`cropped_bbox`가 없다. 위치 문항 보기 좌표는 크롭 영역과의 **교집합으로 클램프된 정수**다 —
검증 방식은 SPEC-06 참조.)

schema.py는 이 JSON을 읽어 **검증된 파이썬 객체(`LabelDoc`)로 변환**한다. 이후의 모든 모듈 —
DB 적재(SPEC-02), 도구(SPEC-03), 채점기(SPEC-06) — 는 JSON을 직접 만지지 않고 이 객체만 쓴다.
데이터의 함정 처리를 한 곳에 모으기 위해서다.

## 2. 함정 3개 — 왜 이 모듈이 따로 필요한가

1. **정상 사진에는 `annotations`가 아예 없다.** 대비 없이 파싱하면 죽거나, 더 나쁘게는 정상 사진이 조용히 누락된다.
2. **좌표계가 2개다.** 결함 위치(bbox)는 원본 8256×5504 기준인데, 4지선다 문제의 좌표 보기는 1920×1080 확대 영역(cropped_bbox) 기준이다. 이 변환을 채점기와 이미지 분석 도구가 **서로 다르게 구현하면 정답률 측정이 통째로 오염**된다 → 변환 함수를 여기 하나만 두고 두 모듈이 공유하도록 강제한다.
3. **결함이 여러 개인 사진이 있다.** 4지선다 문제는 "심각도 최고 → 동률이면 면적 최대" 결함 기준으로 출제되어 있다. 우리가 대표 결함을 다른 기준으로 고르면 문제와 어긋난 답을 채점하게 된다 → 같은 규칙의 선정 함수를 제공한다.

## 3. 인터페이스 — 입출력 예시로

**`parse_label(경로) → LabelDoc`** — JSON 파일 하나를 통째로 변환·검증

**`parse_filename(파일명) → FilenameMeta`** — 파일명에 박힌 설비 정보 추출
```
"2025_sungsan_5_A_LeadingEdge_056.jpg"
 → year=2025, site="sungsan", unit=5, blade="A", side="LeadingEdge", seq="056"
```

**`primary_defect(doc) → Annotation | None`** — 대표 결함 선정 (함정 3의 규칙)
```
[페인트손상 sev2 면적 8만, 라미네이트노출 sev3 면적 15만] → 라미네이트노출 (심각도 우선)
결함 없음 → None
```

**`to_crop_coords(bbox, cropped_bbox) → 크롭 기준 bbox`** — 좌표 변환 (함정 2의 공유 구현)
```
결함 (3500, 1200, 400, 250) + 크롭영역 (3082, 668, 1920, 1080) → (418, 532, 400, 250)
계산: (x−크롭x, y−크롭y, 가로·세로 그대로)
```

**`LabelDoc`의 주요 필드** (전부 읽기 전용):

| 필드 | 내용 |
|---|---|
| filename, width, height | 이미지 파일명·크기 |
| location, captured_at, part_tag, part_side_tag | 어디서 언제 무엇을 찍었나 |
| categories | 결함 id → 이름 매핑 (예: 3 → "Paint Damage") |
| annotations | 결함 목록. **정상 사진이면 빈 튜플** |
| description | 전문가 설명문 (검색 코퍼스·few-shot 재료) |
| vqa | 4지선다 원본 필드 그대로 (채점기 전용, 가공 안 함) |
| cropped_bbox | 확대 영역 좌표. 없는 문서도 있음 → None |

## 4. 규칙 — 각각이 막는 사고

| # | 규칙 | 왜 (없으면 나는 사고) |
|---|---|---|
| R1 | bbox는 길이 4의 [x,y,w,h], w>0·h>0. 위반 시 파일명 포함 `ValueError` | 원 문서에 [x최소,y최소,x최대,y최대]라는 상충 표기가 있었음 — 잘못 해석하면 **모든 크롭·좌표변환이 조용히 틀어짐** |
| R2 | `annotations` 키 부재 = 빈 리스트 = 정상 사진, 둘 다 `annotations=()` | 정상 사진(전체의 약 25%)이 파싱 단계에서 죽거나 누락됨 |
| R3 | `primary_defect` = 심각도 최대 → 동률 시 면적 최대 | 다른 기준으로 고르면 4지선다 문제와 어긋나 채점 왜곡 (함정 3) |
| R4 | 파일명은 6토큰 형식만. site 소문자 정규화, year·unit은 int. 불일치 시 `ValueError` | 원본에 Sungsan/sungsan 혼재 — 정규화 없으면 같은 단지가 이력 DB에서 둘로 갈라짐. 형식이 다른 태양광 파일명의 오파싱도 차단 |
| R5 | `to_crop_coords`는 cropped_bbox 없는 문서에서 `ValueError`. 변환 결과가 크롭 영역을 **부분** 이탈하면 그대로 반환, **완전** 이탈(교집합 없음)이면 `ValueError` | None과 연산해 엉뚱한 좌표가 나오는 대신 즉시 실패. 완전 이탈은 라벨 데이터 오류 — 조용히 틀린 좌표로 채점·크롭되는 것 방지 |
| R6 | severity는 1~4만 허용 | 심각도 기반 정렬·필터·대표선정이 전부 이 값을 신뢰함 — 범위 밖은 데이터 오류 |
| R7 | 필수 블록(info, image, collection, categories, visionqa)이나 설명문 없으면 `ValueError` | 깨진 파일이 인덱스에 들어가면 검색 결과에 빈 캡션이 섞임. categories 없으면 결함 번호를 이름으로 풀 수 없음 (SPEC-02 R4의 전제) |
| R8 | vqa dict는 원본 그대로 보존 (가공 금지) | 채점기가 원 문항·보기·정답을 그대로 써야 채점이 공정함 |

## 5. 엣지 케이스 (테스트 필수)

정상 사진(키 부재) / 빈 리스트 / 다중 결함의 대표 선정 — 심각도 우선 케이스와 동률·면적 케이스 각각 /
cropped_bbox 없는 문서의 to_crop_coords 호출 / 변환 결과가 크롭 영역을 부분 이탈·완전 이탈하는 bbox 각각 /
w나 h가 0·음수인 bbox / 5토큰 파일명(태양광 형식) 거부

## 6. 여기서 하지 않는 것 (→ 담당)

파일 **간** 카테고리 매핑 일관성 검증 (→ SPEC-02, 여긴 파일 1개만 봄) / 이미지 파일 열기 (→ SPEC-02·03) /
폴리곤(segmentation) 활용 (→ 비스코프) / 태양광 스키마 (→ 비스코프)

## 7. 완료 기준

`uv run pytest tests/test_schema.py` 전체 통과. R1~R8 각각에 대응 테스트 최소 1개 (docstring에 규칙 번호 인용).
