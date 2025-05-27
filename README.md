

## 🧠 Skill Path - AI 강의 추천 웹 애플리케이션

> 사용자의 자연어 입력 기반으로 AI 강의를 추천하고, 맞춤형 학습 플랜까지 제공하는 로컬 기반 웹 애플리케이션입니다.

---

### ✅ 프로젝트 구성

```
[사용자 브라우저]
   ⇅ (HTML + JS)
[Node.js 서버 (localhost:8080)]
   ⇅ (POST 요청 - fetch)
[FastAPI 서버 (localhost:8000)]
   ⇅
[추천 로직 + CSV + NLP 모델]
```

---

### 📁 디렉토리 구조

```
vsAi/
├── server.js                   # Node.js 웹 서버 (정적 파일 서빙)
├── index02.html                # 사용자 UI 페이지
├── recommend_fastapi_final.py  # FastAPI 백엔드 추천 API
├── coursera_preprocessed.csv   # 강의 데이터셋
├── requirements.txt            # Python 패키지 목록
├── package.json                # Node.js 패키지 정보
```

---

### 🚀 실행 방법 (로컬 개발 환경)

#### 🔹 1. FastAPI 서버 실행 (추천 시스템)

```bash
uvicorn recommend_fastapi_final:app --reload
```

* 실행 주소: `http://localhost:8000`
* 엔드포인트: `POST /recommend`

#### 🔹 2. Node.js 서버 실행 (프론트)

```bash
node server.js
```

* 실행 주소: `http://localhost:8080`
* `index02.html`을 브라우저에 렌더링

---

### 💡 기능 요약

* 사용자는 자유롭게 **학습 목표**를 자연어로 입력
* 백엔드는 NLP 기반 유사도 분석으로 적합한 강의 추천
* 추천된 강의 목록을 기반으로 **맞춤형 학습 플랜** 자동 생성
* 프론트에서 결과를 카드 및 주차별 플래너로 시각화

---

### 🧪 예시 입력

```
"초보자를 위한 Python 강의 추천해주세요. 10시간 이내면 좋겠어요."
```

→ 예상 결과:

* Python 입문 강의 추천
* 수강 시간 필터 적용
* 추천 강의 기반 주차별 학습 플랜 출력

---

### ⚙️ 요구 사항

#### Python 패키지 (`requirements.txt`)

```txt
fastapi
uvicorn
pandas
numpy
scikit-learn
sentence-transformers
```

#### Node.js 패키지 (`package.json` 예시)

```json
{
  "scripts": {
    "start": "node server.js"
  },
  "dependencies": {
    "express": "^4.18.2"
  }
}
```

---

### 📌 주의사항

* `coursera_preprocessed.csv` 파일은 FastAPI 서버가 실행되는 디렉토리에 있어야 합니다.
* API 주소(`fetch()`)는 현재 로컬 주소(`http://localhost:8000/recommend`)로 설정되어 있습니다.

---

### 📦 향후 계획 (배포 전환 시)

* Render를 이용한 Node.js + FastAPI 서비스 분리 배포
* 프론트 코드에서 API 주소를 Render용으로 전환
* 도메인 연결 및 GitHub Actions CI 설정

---

### 관련 문의

* 문의: [GitHub Issues](https://github.com/milk-coding/skill-path/issues)


