from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Field, create_engine, Session, select

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the SQLModel
class Entry(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    text: str
    mood: str

# SQLite database setup
sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url, echo=True)

# THIS FUNCTION CREATES THE DB FILE + TABLE
@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)

@app.post("/entry")
def create_entry(entry: Entry):
    with Session(engine) as session:
        session.add(entry)
        session.commit()
        session.refresh(entry)
        return entry

@app.get("/entries")
def get_entries():
    with Session(engine) as session:
        statement = select(Entry)
        results = session.exec(statement)
        entries = results.all()
        print("Retrieved entries:", entries)  # <- Debug line
        return entries
