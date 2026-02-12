from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import date, datetime, timedelta

# ================= CONFIG =================
SECRET_KEY = "SECRET123"
ALGORITHM = "HS256"

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

engine = create_engine("sqlite:///app.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# ================= DB =================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ================= MODELS =================

class PublicUserOut(BaseModel):
    id: int
    name: str
    email: str | None = None  # можно скрыть позже


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)
    name = Column(String)
    password_hash = Column(String)

class Habit(Base):
    __tablename__ = "habits"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String)
    period_days = Column(Integer, default=1)
    times_per_period = Column(Integer, default=1)

class HabitCheckin(Base):
    __tablename__ = "habit_checkins"
    id = Column(Integer, primary_key=True)
    habit_id = Column(Integer, ForeignKey("habits.id"))
    date = Column(String)
    value = Column(Integer, nullable=True)

class Goal(Base):
    __tablename__ = "goals"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String)
    description = Column(String, nullable=True)
    goal_type = Column(String)
    target_value = Column(Integer, nullable=True)
    unit = Column(String, nullable=True)
    progress_value = Column(Integer, default=0)
    deadline = Column(String, nullable=True)
    priority = Column(Integer, default=3)
    status = Column(String, default="ACTIVE")

class GoalStep(Base):
    __tablename__ = "goal_steps"
    id = Column(Integer, primary_key=True)
    goal_id = Column(Integer, ForeignKey("goals.id"))
    title = Column(String)
    is_done = Column(Integer, default=0)

class UserAchievement(Base):
    __tablename__ = "user_achievements"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    code = Column(String)
    title = Column(String)
    earned_at = Column(String)



# ================= SCHEMAS =================



class RegisterBody(BaseModel):
    email: str
    password: str
    name: str

class LoginBody(BaseModel):
    email: str
    password: str

class HabitCreateBody(BaseModel):
    title: str
    periodDays: int
    timesPerPeriod: int

class CheckinBody(BaseModel):
    date: str
    value: int | None = None

class GoalCreateBody(BaseModel):
    title: str
    description: str | None = None
    goalType: str
    targetValue: int | None = None
    unit: str | None = None
    deadline: str | None = None
    priority: int = 3

class GoalStatusBody(BaseModel):
    status: str

class Friend(Base):
    __tablename__ = "friends"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    friend_id = Column(Integer, ForeignKey("users.id"), nullable=False)

Base.metadata.create_all(engine)

# ================= AUTH UTILS =================
def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(password: str, hash_: str):
    return pwd_context.verify(password, hash_)

def create_token(user_id: int):
    return jwt.encode({"user_id": user_id}, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user_id(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return int(payload["user_id"])
    except (JWTError, KeyError):
        raise HTTPException(status_code=401, detail="Invalid token")

# ================= GAMIFICATION =================
HABIT_STREAK_THRESHOLDS = [7, 14, 30, 60, 180, 360]
GOAL_DONE_THRESHOLDS = [3, 5, 7, 10, 12, 15, 20]

def parse_ymd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def compute_current_streak(dates: list[str]) -> int:
    if not dates:
        return 0
    ds = sorted({parse_ymd(d) for d in dates})
    today = date.today()
    if ds[-1] != today:
        return 0
    streak = 1
    cur = today
    for d in reversed(ds[:-1]):
        if d == cur - timedelta(days=1):
            streak += 1
            cur = d
        else:
            break
    return streak

def ensure_achievement(db: Session, user_id: int, code: str, title: str):
    if db.query(UserAchievement).filter_by(user_id=user_id, code=code).first():
        return
    db.add(UserAchievement(
        user_id=user_id,
        code=code,
        title=title,
        earned_at=date.today().isoformat()
    ))

# ================= AUTH =================
@app.post("/auth/register")
def register(body: RegisterBody, db: Session = Depends(get_db)):
    if db.query(User).filter_by(email=body.email).first():
        raise HTTPException(400, "User exists")
    u = User(
        email=body.email,
        name=body.name,
        password_hash=hash_password(body.password)
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return {"token": create_token(u.id)}

@app.post("/auth/login")
def login(body: LoginBody, db: Session = Depends(get_db)):
    u = db.query(User).filter_by(email=body.email).first()
    if not u or not verify_password(body.password, u.password_hash):
        raise HTTPException(401, "Invalid credentials")
    return {"token": create_token(u.id)}

# ================= HABITS =================
@app.get("/habits")
def get_habits(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return db.query(Habit).filter_by(user_id=user_id).all()

@app.post("/habits")
def create_habit(body: HabitCreateBody, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    h = Habit(
        user_id=user_id,
        title=body.title,
        period_days=body.periodDays,
        times_per_period=body.timesPerPeriod
    )
    db.add(h)
    db.commit()
    db.refresh(h)
    return h

@app.post("/habits/{habit_id}/checkin")
def checkin(habit_id: int, body: CheckinBody, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    if db.query(HabitCheckin).filter_by(habit_id=habit_id, date=body.date).first():
        return {"ok": True}
    db.add(HabitCheckin(habit_id=habit_id, date=body.date, value=body.value))
    db.commit()

    dates = [c.date for c in db.query(HabitCheckin).filter_by(habit_id=habit_id).all()]
    streak = compute_current_streak(dates)

    for n in HABIT_STREAK_THRESHOLDS:
        if streak >= n:
            ensure_achievement(db, user_id, f"HABIT_STREAK_{n}", f"{n} дней подряд")
    db.commit()
    return {"ok": True, "currentStreak": streak}

# ================= GOALS =================
@app.get("/goals")
def get_goals(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return db.query(Goal).filter_by(user_id=user_id).all()

@app.post("/goals")
def create_goal(body: GoalCreateBody, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    g = Goal(
        user_id=user_id,
        title=body.title,
        description=body.description,
        goal_type=body.goalType,
        target_value=body.targetValue,
        unit=body.unit,
        deadline=body.deadline,
        priority=body.priority
    )
    db.add(g)
    db.commit()
    db.refresh(g)
    return g

@app.put("/goals/{goal_id}/status")
def set_goal_status(goal_id: int, body: GoalStatusBody, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    g = db.query(Goal).filter_by(id=goal_id, user_id=user_id).first()
    if not g:
        raise HTTPException(404)
    prev = g.status
    g.status = body.status
    db.commit()

    if prev != "DONE" and body.status == "DONE":
        done_count = db.query(Goal).filter_by(user_id=user_id, status="DONE").count()
        for n in GOAL_DONE_THRESHOLDS:
            if done_count >= n:
                ensure_achievement(db, user_id, f"GOAL_DONE_{n}", f"Завершено {n} целей")
        db.commit()
    return {"ok": True}

# ================= PROFILE =================
@app.get("/profile")
def profile(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    habits = db.query(Habit).filter_by(user_id=user_id).all()
    goals_done = db.query(Goal).filter_by(user_id=user_id, status="DONE").count()

    max_streak = 0
    for h in habits:
        dates = [c.date for c in db.query(HabitCheckin).filter_by(habit_id=h.id).all()]
        max_streak = max(max_streak, compute_current_streak(dates))

    achievements = db.query(UserAchievement).filter_by(user_id=user_id).all()

    return {
        "currentHabitStreak": max_streak,
        "goalsCompleted": goals_done,
        "achievements": achievements
    }
@app.get("/users")
def list_users(q: str | None = None, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    query = db.query(User).filter(User.id != user_id)
    if q:
        like = f"%{q}%"
        query = query.filter((User.name.like(like)) | (User.email.like(like)))
    users = query.limit(50).all()

    result = []
    for u in users:
        me_to_him = db.query(Friend).filter(Friend.user_id == user_id, Friend.friend_id == u.id).first()
        him_to_me = db.query(Friend).filter(Friend.user_id == u.id, Friend.friend_id == user_id).first()

        status = "NONE"
        if me_to_him and him_to_me:
            status = "FRIEND"
        elif me_to_him:
            status = "OUTGOING"
        elif him_to_me:
            status = "INCOMING"

        result.append({
            "id": u.id,
            "name": u.name,
            "status": status
        })
    return result



@app.post("/friends/{friend_id}")
def add_friend(friend_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    if friend_id == user_id:
        raise HTTPException(400, "Cannot add yourself")

    exists_user = db.query(User).filter(User.id == friend_id).first()
    if not exists_user:
        raise HTTPException(404, "User not found")

    exists = db.query(Friend).filter(
        Friend.user_id == user_id,
        Friend.friend_id == friend_id
    ).first()

    if exists:
        return {"ok": True}

    db.add(Friend(user_id=user_id, friend_id=friend_id))
    db.commit()
    return {"ok": True}


@app.get("/friends")
def list_friends(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    my = db.query(Friend).filter(Friend.user_id == user_id).subquery()
    mutual = db.query(Friend).filter(
        Friend.user_id == my.c.friend_id,
        Friend.friend_id == user_id
    ).subquery()

    users = db.query(User).filter(User.id.in_(
        db.query(my.c.friend_id).join(mutual, mutual.c.user_id == my.c.friend_id)
    )).all()

    return [{"id": u.id, "name": u.name} for u in users]

@app.get("/profile/{other_id}")
def friend_profile(other_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    a = db.query(Friend).filter(Friend.user_id == user_id, Friend.friend_id == other_id).first()
    b = db.query(Friend).filter(Friend.user_id == other_id, Friend.friend_id == user_id).first()

    if not a or not b:
        raise HTTPException(403, "Not friends")

    habits = db.query(Habit).filter(Habit.user_id == other_id).all()
    max_streak = 0
    for h in habits:
        dates = [c.date for c in db.query(HabitCheckin).filter(HabitCheckin.habit_id == h.id).all()]
        max_streak = max(max_streak, compute_current_streak(dates))

    goals_done = db.query(Goal).filter(Goal.user_id == other_id, Goal.status == "DONE").count()
    achievements = db.query(UserAchievement).filter(UserAchievement.user_id == other_id).all()
    user = db.query(User).filter(User.id == other_id).first()

    return {
        "user": {"id": user.id, "name": user.name},
        "currentHabitStreak": max_streak,
        "goalsCompleted": goals_done,
        "achievements": achievements
    }


@app.delete("/friends/{friend_id}")
def remove_friend(friend_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    row = db.query(Friend).filter(Friend.user_id == user_id, Friend.friend_id == friend_id).first()
    if not row:
        return {"ok": True}
    db.delete(row)
    db.commit()
    return {"ok": True}


