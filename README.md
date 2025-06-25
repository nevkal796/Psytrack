# 🧠 Psytrack

Psytrack is a mental health journaling application that allows users to submit daily entries and moods. Behind the scenes, it uses a custom-trained machine learning model to analyze the journal text and provide insights (coming soon!). The project is built with:

- 🧰 Frontend: React
- ⚙️ Backend: FastAPI
- 🧠 Machine Learning: TensorFlow, Pandas
- 💾 Database: SQLite + SQLModel

---

## 📁 Project Structure

psytrack/
├── psytrack-frontend/ # React frontend
│ ├── src/
│ │ ├── components/
│ │ │ └── JournalEntry.js
│ │ ├── App.js
│ │ └── index.js
│ └── package.json
│
├── psytrack-backend/ # FastAPI backend
│ ├── main.py
│ ├── database.db
│ └── requirements.txt
│
├── psytrack-ml/ # Machine learning training script
│ ├── mental_health.csv
│ ├── train_model.py
│ └── venv/
│
└── README.md # ← You are here


---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/psytrack.git
cd psytrack
