import pandas as pd
import numpy as np
import ast
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# CSV 파일 경로 상수화
CSV_PATH = "coursera_preprocessed.csv"
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# CSV 로드 (예외 처리 추가)
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# Skills: 문자열 → 리스트 변환
def parse_skills(x):
    if isinstance(x, str) and x.startswith('['):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return []

df['skills'] = df['Skills'].apply(parse_skills)

# 표시용 컬럼 만들기
df['rating_display'] = df['rating'].apply(
    lambda x: f"{x:.1f}" if pd.notna(x) else "평점 없음"
)
df['reviews_display'] = df['num_reviews'].apply(
    lambda x: f"{int(x)}명" if pd.notna(x) else "리뷰 없음"
)
df['Schedule_display'] = df['Schedule'].apply(
    lambda x: f"{x:.1f}시간"
              if pd.notna(x) and isinstance(x, (int, float))
              else (str(x) if pd.notna(x) else "학습 시간 정보 없음")
)

# 필터용 Sentinel 컬럼
df['rating_for_sort'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
df['Schedule_for_sort'] = pd.to_numeric(df['Schedule'], errors='coerce').fillna(0)

# 모델 로드 및 임베딩 생성 (예외 처리)
try:
    model = SentenceTransformer(MODEL_NAME)
except Exception as e:
    raise RuntimeError(f"임베딩 모델 로드 실패: {e}")

try:
    course_embeddings = model.encode(
        df['full_text'].tolist(), show_progress_bar=True, convert_to_numpy=True
    )
except Exception as e:
    raise RuntimeError(f"임베딩 생성 실패: {e}")

def recommend_courses(
    user_input: str,
    top_n: int = 5,
    min_rating: float = 0,
    skill_keywords: list = None,
    max_Schedule: float = None,
    weight_similarity: float = 0.7,
    weight_rating: float = 0.3,
    fallback: bool = True
) -> pd.DataFrame:
    """
    - user_input: 검색 키워드 or 과거 학습 텍스트
    - weight_similarity + weight_rating = 1.0
    """
    # 입력값 검증
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input은 비어있지 않은 문자열이어야 합니다.")
    if not (0 <= weight_similarity <= 1 and 0 <= weight_rating <= 1 and abs(weight_similarity + weight_rating - 1.0) < 1e-6):
        raise ValueError("weight_similarity와 weight_rating의 합은 1이어야 합니다.")
    if skill_keywords is not None and not isinstance(skill_keywords, list):
        raise ValueError("skill_keywords는 리스트여야 합니다.")
    if max_Schedule is not None and (not isinstance(max_Schedule, (int, float)) or max_Schedule < 0):
        raise ValueError("max_Schedule은 0 이상의 숫자여야 합니다.")

    # 사용자 임베딩 및 유사도 계산
    user_emb = model.encode([user_input], convert_to_numpy=True)[0]
    sims = cosine_similarity([user_emb], course_embeddings)[0]

    # 필터링 전용 복사본 생성
    filtered = df.copy()
    filtered['similarity'] = sims

    if min_rating > 0:
        filtered = filtered[filtered['rating_for_sort'] >= min_rating]
    if skill_keywords:
        filtered = filtered[
            filtered['skills'].apply(
                lambda skills: any(k.lower() in [s.lower() for s in skills] for k in skill_keywords)
            )
        ]
    if max_Schedule is not None:
        filtered = filtered[filtered['Schedule_for_sort'] <= max_Schedule]

    # 평점 정규화 (0~1)
    filtered = filtered.copy()  # SettingWithCopyWarning 방지
    filtered['rating_norm'] = filtered['rating_for_sort'] / 5.0

    # 결합 점수 계산
    filtered['combined_score'] = (
        filtered['similarity'] * weight_similarity +
        filtered['rating_norm'] * weight_rating
    )

    # combined_score 기준 정렬 후 상위 추출
    result = filtered.sort_values('combined_score', ascending=False).head(top_n)

    # 필터 조건 미충족 시 fallback
    if result.empty and fallback and (
        min_rating > 0 or skill_keywords or max_Schedule is not None
    ):
        print("❗조건 맞춘 강의가 없어 필터 해제 후 재추천합니다.")
        return recommend_courses(
            user_input, top_n, 0, None, None,
            weight_similarity, weight_rating, False
        )

    # 표시용 컬럼 및 URL 반환
    return result[[
        'title', 'rating_display', 'reviews_display',
        'Schedule_display', 'skills', 'URL', 'Instructor' if 'Instructor' in result.columns else result.columns[0]
    ]].reset_index(drop=True)

def parse_duration_weeks(schedule_str: str) -> int:
    """
    schedule_str 예시:
      - "12 hours to complete (3 weeks at 4 hours a week)"
      - "1 month (at 10 hours a week)"
      - "학습 시간 정보 없음"
    """
    if not isinstance(schedule_str, str):
        return 1
    s = schedule_str.lower()
    m = re.search(r'(\d+)\s*weeks?', s)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)\s*months?', s)
    if m:
        return int(m.group(1)) * 4
    return 1  # 파싱 실패 시 1주로 간주

def create_study_plan(recommended_df):
    """
    추천 강의 목록을 기반으로 주차별 학습 플랜을 생성합니다.
    각 코스의 'Schedule_display'에서 수강 기간(주 단위)을 파싱하여
    누적 주차(current_week)를 계산하도록 수정됨.
    """
    study_plan = {}
    current_week = 1

    for _, row in recommended_df.reset_index().iterrows():
        schedule_str = row.get('Schedule_display', '')
        duration_weeks = parse_duration_weeks(schedule_str)

        study_plan[f"Week {current_week}"] = {
            "title": row['title'],
            "instructor": row.get('Instructor', "강사 정보 없음"),
            "rating": row['rating_display'],
            "reviews": row['reviews_display'],
            "Schedule": schedule_str or "학습 시간 정보 없음",
            "skills": row['skills'],
            "url": row['URL']
        }

        current_week += duration_weeks

    return study_plan

# ------------------- Inference 코드 -------------------
import re

if __name__ == "__main__":
    print("🧠 SkillPath 추천 시스템 (자연어 입력 기반)\n")

    # 사용자 문장 입력
    user_input = input(
        "학습 목표를 자유롭게 문장으로 입력하세요:\n"
        "(예: Please recommend Python lectures for beginners. I hope the GPA is 4.0 or higher, and the study time is less than 10 hours.)\n\n> "
    )

    # 평점과 수강시간을 문장에서 추출
    min_rating_match = re.search(r'(\d\.\d)\s*or higher|GPA.*?(\d\.\d)', user_input, re.IGNORECASE)
    max_schedule_match = re.search(r'less than\s+(\d+)\s*hours?', user_input, re.IGNORECASE)

    min_rating = float(min_rating_match.group(1) or min_rating_match.group(2)) if min_rating_match else 0.0
    max_schedule = float(max_schedule_match.group(1)) if max_schedule_match else None

    print(f"\n🔍 추출된 최소 평점: {min_rating}")
    print(f"🔍 추출된 최대 수강 시간: {max_schedule if max_schedule else '제한 없음'}")

    # 스킬 키워드 (선택 입력)
    skill_input = input("\n중요한 스킬 키워드를 쉼표로 입력하세요 (없으면 Enter):\n> ")
    skills_filter = [s.strip() for s in skill_input.split(",") if s.strip()] if skill_input.strip() else None

    # 추천 실행
    recommended_df = recommend_courses(
        user_input=user_input,
        top_n=6,
        min_rating=min_rating,
        skill_keywords=skills_filter,
        max_Schedule=max_schedule
    )

    if recommended_df.empty:
        print("\n❌ 조건에 맞는 추천 강의가 없습니다.")
    else:
        print("\n📌 추천 강의 결과:\n")
        for _, row in recommended_df.iterrows():
            print(f"▶ {row['title']}")
            print(f"  평점: {row['rating_display']}")
            print(f"  리뷰 수: {row['reviews_display']}")
            print(f"  수강 시간: {row['Schedule_display']}")
            print(f"  스킬: {', '.join(row['skills']) if row['skills'] else '스킬 정보 없음'}")
            print(f"  링크: {row['URL']}\n")

        # 학습 플랜 출력
        study_plan = create_study_plan(recommended_df)
        print("📅 주차별 학습 플랜:")
        for week, content in study_plan.items():
            print(f"{week} - {content['title']} ({content['rating']}, {content['Schedule']})")
