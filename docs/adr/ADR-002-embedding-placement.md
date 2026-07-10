# ADR-002 — 임베딩: 텍스트는 OpenAI API, 이미지는 로컬 CLIP(CPU)

**Status**: Accepted (2026-07-10)

## Context

원 지시서 권장은 텍스트 bge-m3 + 이미지 SigLIP/CLIP ViT-L을 **로컬 GPU**로 돌리는 것이었다
(10만 장 인덱싱의 API 비용 회피 목적). 그러나 개발 환경에 GPU가 없고(CPU만),
데이터 규모도 서브셋 ~1,000장(ADR-003)으로 줄어 전제가 달라졌다.
추가 제약: OpenAI에는 이미지 임베딩 API가 없다 — 이미지 임베딩은 어차피 로컬로 해야 한다.

## Decision

- **텍스트 임베딩**: OpenAI `text-embedding-3-small` (API) — 한국어 캡션 처리 품질 양호, GPU 불필요
- **이미지 임베딩**: CLIP ViT-B/32 (transformers, 로컬 CPU) — 1,000장 규모면 CPU로 수십 분 내 처리

## Consequences

- (+) GPU 없이 전체 파이프라인 동작
- (+) 이미지 임베딩은 로컬이므로 인덱싱 시 사진이 외부로 나가지 않음 (외부 전송은 T4 VLM 호출 시에만)
- (−) 텍스트 임베딩은 건당 API 비용 + 캡션이 외부로 전송됨
- (−) ViT-B/32는 권장(ViT-L)보다 표현력이 낮아 유사 검색 품질이 하한선일 수 있음
- 임베딩 호출부는 함수 하나로 격리해, GPU 확보 시 bge-m3/SigLIP로 교체가 쉬운 구조를 유지한다
