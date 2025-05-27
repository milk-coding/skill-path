from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import ast
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

CSV_PATH = "coursera_preprocessed.csv"
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Node 서버 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    user_input: str
    skill_keywords: list[str] = []

df = pd.read_csv(CSV_PATH)

def parse_skills(x):
    if isinstance(x, str) and x.startswith('['):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return []

df['skills'] = df['Skills'].apply(parse_skills)
df['rating_display'] = df['rating'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "평점 없음")
df['reviews_display'] = df['num_reviews'].apply(lambda x: f"{int(x)}명" if pd.notna(x) else "리뷰 없음")
df['Schedule_display'] = df['Schedule'].apply(lambda x: f"{x:.1f}시간" if pd.notna(x) and isinstance(x, (int, float)) else (str(x) if pd.notna(x) else "학습 시간 정보 없음"))
df['rating_for_sort'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
df['Schedule_for_sort'] = pd.to_numeric(df['Schedule'], errors='coerce').fillna(0)

model = SentenceTransformer(MODEL_NAME)
course_embeddings = model.encode(df['full_text'].tolist(), show_progress_bar=True, convert_to_numpy=True)

def recommend_courses(user_input, top_n=6, min_rating=0.0, skill_keywords=None, max_Schedule=None, weight_similarity=0.7, weight_rating=0.3, fallback=True):
    user_emb = model.encode([user_input], convert_to_numpy=True)[0]
    sims = cosine_similarity([user_emb], course_embeddings)[0]
    filtered = df.copy()
    filtered['similarity'] = sims

    if min_rating > 0:
        filtered = filtered[filtered['rating_for_sort'] >= min_rating]
    if skill_keywords:
        filtered = filtered[filtered['skills'].apply(lambda skills: any(k.lower() in [s.lower() for s in skills] for k in skill_keywords))]
    if max_Schedule is not None:
        filtered = filtered[filtered['Schedule_for_sort'] <= max_Schedule]

    filtered['rating_norm'] = filtered['rating_for_sort'] / 5.0
    filtered['combined_score'] = filtered['similarity'] * weight_similarity + filtered['rating_norm'] * weight_rating
    result = filtered.sort_values('combined_score', ascending=False).head(top_n)

    if result.empty and fallback and (min_rating > 0 or skill_keywords or max_Schedule is not None):
        return recommend_courses(user_input, top_n, 0, None, None, weight_similarity, weight_rating, False)

    return result[[
        'title', 'rating_display', 'reviews_display', 'Schedule_display', 'skills', 'URL', 'Instructor' if 'Instructor' in result.columns else result.columns[0]
    ]].reset_index(drop=True)

def parse_duration_weeks(schedule_str):
    if not isinstance(schedule_str, str):
        return 1
    s = schedule_str.lower()
    m = re.search(r'(\d+)\s*weeks?', s)
    if m: return int(m.group(1))
    m = re.search(r'(\d+)\s*months?', s)
    if m: return int(m.group(1)) * 4
    return 1

def create_study_plan(recommended_df):
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

@app.post("/recommend")
async def recommend_api(req: RecommendRequest):
    user_input = req.user_input

    # 자연어에서 평점, 시간 추출
    min_rating_match = re.search(r'(\d\.\d)\s*or higher|GPA.*?(\d\.\d)', user_input, re.IGNORECASE)
    max_schedule_match = re.search(r'less than\s+(\d+)\s*hours?', user_input, re.IGNORECASE)

    min_rating = float(min_rating_match.group(1) or min_rating_match.group(2)) if min_rating_match else 0.0
    max_schedule = float(max_schedule_match.group(1)) if max_schedule_match else None

    df_result = recommend_courses(
        user_input=user_input,
        top_n=6,
        min_rating=min_rating,
        skill_keywords=req.skill_keywords or [],
        max_Schedule=max_schedule
    )
    study_plan = create_study_plan(df_result)
    return {
        "recommendations": df_result.to_dict(orient="records"),
        "study_plan": study_plan,
        "min_rating": min_rating,
        "max_schedule": max_schedule
    }
