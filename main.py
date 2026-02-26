from fastapi import FastAPI, Depends, HTTPException, Request, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
bearer_scheme = HTTPBearer()

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
    goal_type = Column(String)  # QUANT | STEPS | HABIT_AS_GOAL
    target_value = Column(Integer, nullable=True)
    unit = Column(String, nullable=True)
    progress_value = Column(Integer, default=0)
    deadline = Column(String, nullable=True)
    priority = Column(Integer, default=3)
    status = Column(String, default="ACTIVE")  # ACTIVE|PAUSED|DONE|CANCELED
    show_in_profile = Column(Integer, default=0)  # 0/1

class GoalStep(Base):
    __tablename__ = "goal_steps"
    id = Column(Integer, primary_key=True)
    goal_id = Column(Integer, ForeignKey("goals.id"))
    title = Column(String)
    is_done = Column(Integer, default=0)  # 0/1

class UserAchievement(Base):
    __tablename__ = "user_achievements"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    code = Column(String)
    title = Column(String)
    earned_at = Column(String)

class Friend(Base):
    __tablename__ = "friends"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    friend_id = Column(Integer, ForeignKey("users.id"), nullable=False)

Base.metadata.create_all(engine)

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

class HabitUpdateBody(BaseModel):
    title: str | None = None
    periodDays: int | None = None
    timesPerPeriod: int | None = None

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
    status: str = "ACTIVE"
    showInProfile: bool = False

class GoalUpdateBody(BaseModel):
    title: str | None = None
    description: str | None = None
    goalType: str | None = None
    targetValue: int | None = None
    unit: str | None = None
    deadline: str | None = None
    priority: int | None = None
    status: str | None = None
    showInProfile: bool | None = None

class GoalProgressBody(BaseModel):
    delta: int = 1  # насколько прибавить

class StepCreateBody(BaseModel):
    title: str

class StepUpdateBody(BaseModel):
    title: str | None = None
    isDone: bool | None = None

# ================= AUTH UTILS =================
def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(password: str, hash_: str):
    return pwd_context.verify(password, hash_)

def create_token(user_id: int):
    return jwt.encode({"user_id": user_id}, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user_id(creds: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = creds.credentials
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

# ================= DTO HELPERS =================
def habit_to_dto(h: Habit) -> dict:
    return {
        "id": h.id,
        "title": h.title,
        "periodDays": h.period_days,
        "timesPerPeriod": h.times_per_period,
    }

def goal_to_dto(g: Goal) -> dict:
    return {
        "id": g.id,
        "title": g.title,
        "description": g.description,
        "goalType": g.goal_type,
        "targetValue": g.target_value,
        "unit": g.unit,
        "progressValue": g.progress_value,
        "deadline": g.deadline,
        "priority": g.priority,
        "status": g.status,
        "showInProfile": bool(g.show_in_profile),
    }

def step_to_dto(s: GoalStep) -> dict:
    return {
        "id": s.id,
        "goalId": s.goal_id,
        "title": s.title,
        "isDone": bool(s.is_done),
    }

def achievement_to_dto(a: UserAchievement) -> dict:
    return {
        "code": a.code,
        "title": a.title,
        "earned_at": a.earned_at,
    }

# ================= POINTS =================
POINT_RATES = {
    "км": 15,
    "книг": 200,
    "страниц": 1,
    "кг": 50,
}

def compute_points(goals: list[Goal]) -> int:
    total = 0
    for g in goals:
        if g.goal_type != "QUANT":
            continue
        if g.status != "DONE":
            continue
        if not g.unit or g.unit not in POINT_RATES:
            continue
        if not g.target_value:
            continue
        total += int(g.target_value) * POINT_RATES[g.unit]
    return total

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
    habits = db.query(Habit).filter_by(user_id=user_id).all()
    return [habit_to_dto(h) for h in habits]

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
    return habit_to_dto(h)

@app.put("/habits/{habit_id}")
def update_habit(habit_id: int, body: HabitUpdateBody,
                 user_id: int = Depends(get_current_user_id),
                 db: Session = Depends(get_db)):
    h = db.query(Habit).filter_by(id=habit_id, user_id=user_id).first()
    if not h:
        raise HTTPException(404, "Habit not found")
    if body.title is not None:
        h.title = body.title
    if body.periodDays is not None:
        h.period_days = body.periodDays
    if body.timesPerPeriod is not None:
        h.times_per_period = body.timesPerPeriod
    db.commit()
    db.refresh(h)
    return habit_to_dto(h)

@app.delete("/habits/{habit_id}")
def delete_habit(habit_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    h = db.query(Habit).filter_by(id=habit_id, user_id=user_id).first()
    if not h:
        raise HTTPException(404, "Habit not found")
    db.query(HabitCheckin).filter(HabitCheckin.habit_id == habit_id).delete()
    db.delete(h)
    db.commit()
    return {"ok": True}

@app.post("/habits/{habit_id}/checkin")
def checkin(habit_id: int, body: CheckinBody, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    h = db.query(Habit).filter_by(id=habit_id, user_id=user_id).first()
    if not h:
        raise HTTPException(404, "Habit not found")

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

@app.post("/debug/habits/{habit_id}/add_streak")
def debug_add_habit_streak(
    habit_id: int,
    n: int = Query(7, ge=1, le=365),
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    h = db.query(Habit).filter_by(id=habit_id, user_id=user_id).first()
    if not h:
        raise HTTPException(404, "Habit not found")

    today = date.today()
    created = 0
    for i in range(n):
        d = (today - timedelta(days=i)).isoformat()
        exists = db.query(HabitCheckin).filter_by(habit_id=habit_id, date=d).first()
        if not exists:
            db.add(HabitCheckin(habit_id=habit_id, date=d, value=None))
            created += 1
    db.commit()

    dates = [c.date for c in db.query(HabitCheckin).filter_by(habit_id=habit_id).all()]
    streak = compute_current_streak(dates)

    for t in HABIT_STREAK_THRESHOLDS:
        if streak >= t:
            ensure_achievement(db, user_id, f"HABIT_STREAK_{t}", f"{t} дней подряд")
    db.commit()

    return {"ok": True, "habitId": habit_id, "createdCheckins": created, "currentStreak": streak}

# ================= GOALS =================
@app.get("/goals")
def get_goals(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    goals = db.query(Goal).filter_by(user_id=user_id).all()
    return [goal_to_dto(g) for g in goals]

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
        priority=body.priority,
        status=body.status,
        show_in_profile=1 if body.showInProfile else 0,
    )
    db.add(g)
    db.commit()
    db.refresh(g)
    return goal_to_dto(g)

@app.put("/goals/{goal_id}")
def update_goal(goal_id: int, body: GoalUpdateBody,
                user_id: int = Depends(get_current_user_id),
                db: Session = Depends(get_db)):
    g = db.query(Goal).filter_by(id=goal_id, user_id=user_id).first()
    if not g:
        raise HTTPException(404, "Goal not found")

    prev_status = g.status

    if body.title is not None: g.title = body.title
    if body.description is not None: g.description = body.description
    if body.goalType is not None: g.goal_type = body.goalType
    if body.targetValue is not None: g.target_value = body.targetValue
    if body.unit is not None: g.unit = body.unit
    if body.deadline is not None: g.deadline = body.deadline
    if body.priority is not None: g.priority = body.priority
    if body.status is not None: g.status = body.status
    if body.showInProfile is not None:
        g.show_in_profile = 1 if body.showInProfile else 0

    db.commit()

    if prev_status != "DONE" and g.status == "DONE":
        done_count = db.query(Goal).filter_by(user_id=user_id, status="DONE").count()
        for n in GOAL_DONE_THRESHOLDS:
            if done_count >= n:
                ensure_achievement(db, user_id, f"GOAL_DONE_{n}", f"Завершено {n} целей")
        db.commit()

    db.refresh(g)
    return goal_to_dto(g)

@app.delete("/goals/{goal_id}")
def delete_goal(goal_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    g = db.query(Goal).filter_by(id=goal_id, user_id=user_id).first()
    if not g:
        raise HTTPException(404, "Goal not found")
    db.query(GoalStep).filter(GoalStep.goal_id == goal_id).delete()
    db.delete(g)
    db.commit()
    return {"ok": True}

@app.put("/goals/{goal_id}/progress")
def add_goal_progress(
    goal_id: int,
    body: GoalProgressBody,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    g = db.query(Goal).filter_by(id=goal_id, user_id=user_id).first()
    if not g:
        raise HTTPException(404, "Goal not found")

    if body.delta <= 0:
        raise HTTPException(400, "delta must be > 0")

    g.progress_value = (g.progress_value or 0) + body.delta

    if g.target_value is not None:
        if g.progress_value >= g.target_value:
            g.progress_value = g.target_value
            g.status = "DONE"

    db.commit()
    db.refresh(g)
    return goal_to_dto(g)

# ================= GOAL STEPS =================
@app.get("/goals/{goal_id}/steps")
def get_steps(
    goal_id: int,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    g = db.query(Goal).filter_by(id=goal_id, user_id=user_id).first()
    if not g:
        raise HTTPException(404, "Goal not found")

    steps = db.query(GoalStep).filter(GoalStep.goal_id == goal_id).all()
    return [step_to_dto(s) for s in steps]

@app.post("/goals/{goal_id}/steps")
def add_step(
    goal_id: int,
    body: StepCreateBody,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    g = db.query(Goal).filter_by(id=goal_id, user_id=user_id).first()
    if not g:
        raise HTTPException(404, "Goal not found")

    title = body.title.strip()
    if not title:
        raise HTTPException(400, "Empty title")

    s = GoalStep(goal_id=goal_id, title=title, is_done=0)
    db.add(s)
    db.commit()
    db.refresh(s)
    return step_to_dto(s)

@app.put("/steps/{step_id}")
def update_step(
    step_id: int,
    body: StepUpdateBody,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    s = db.query(GoalStep).filter_by(id=step_id).first()
    if not s:
        raise HTTPException(404, "Step not found")

    g = db.query(Goal).filter_by(id=s.goal_id, user_id=user_id).first()
    if not g:
        raise HTTPException(403, "Forbidden")

    if body.title is not None:
        t = body.title.strip()
        if not t:
            raise HTTPException(400, "Empty title")
        s.title = t

    if body.isDone is not None:
        s.is_done = 1 if body.isDone else 0

    db.commit()
    db.refresh(s)
    return step_to_dto(s)

@app.delete("/steps/{step_id}")
def delete_step(
    step_id: int,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    s = db.query(GoalStep).filter_by(id=step_id).first()
    if not s:
        return {"ok": True}

    g = db.query(Goal).filter_by(id=s.goal_id, user_id=user_id).first()
    if not g:
        raise HTTPException(403, "Forbidden")

    db.delete(s)
    db.commit()
    return {"ok": True}

# ================= TEMPLATES =================
GOAL_TEMPLATES = [
    {
        "id": 1,
        "title": "Пробежать 5 км",
        "description": "Базовая цель на выносливость и дисциплину",
        "category": "Здоровье",
        "suggestedTarget": 5,
        "suggestedUnit": "км",
    },
    {
        "id": 2,
        "title": "Прочитать 10 книг",
        "description": "Цель на развитие и регулярное чтение",
        "category": "Саморазвитие",
        "suggestedTarget": 10,
        "suggestedUnit": "книг",
    },
    {
        "id": 3,
        "title": "Тренироваться 12 раз",
        "description": "Регулярные тренировки без перегруза",
        "category": "Спорт",
        "suggestedTarget": 12,
        "suggestedUnit": "раз",
    },
]

@app.get("/templates/goals")
def get_goal_templates(user_id: int = Depends(get_current_user_id)):
    return GOAL_TEMPLATES

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

    visible_goals = (
        db.query(Goal)
          .filter(Goal.user_id == user_id, Goal.show_in_profile == 1)
          .all()
    )

    all_goals = db.query(Goal).filter(Goal.user_id == user_id).all()
    points = compute_points(all_goals)

    return {
        "currentHabitStreak": max_streak,
        "goalsCompleted": goals_done,
        "achievements": [achievement_to_dto(a) for a in achievements],
        "goals": [goal_to_dto(g) for g in visible_goals],
        "points": points,
    }

# ================= USERS / FRIENDS =================
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

        result.append({"id": u.id, "name": u.name, "status": status})
    return result

@app.post("/friends/{friend_id}")
def add_friend(friend_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    if friend_id == user_id:
        raise HTTPException(400, "Cannot add yourself")

    if not db.query(User).filter(User.id == friend_id).first():
        raise HTTPException(404, "User not found")

    exists = db.query(Friend).filter(Friend.user_id == user_id, Friend.friend_id == friend_id).first()
    if exists:
        return {"ok": True}

    db.add(Friend(user_id=user_id, friend_id=friend_id))
    db.commit()
    return {"ok": True}

@app.delete("/friends/{friend_id}")
def remove_friend(friend_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    row = db.query(Friend).filter(Friend.user_id == user_id, Friend.friend_id == friend_id).first()
    if row:
        db.delete(row)
        db.commit()
    return {"ok": True}

@app.get("/friends/incoming")
def friends_incoming(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    incoming = db.query(Friend).filter(Friend.friend_id == user_id).all()
    result = []
    for f in incoming:
        me_to_him = db.query(Friend).filter(Friend.user_id == user_id, Friend.friend_id == f.user_id).first()
        if me_to_him:
            continue
        u = db.query(User).filter(User.id == f.user_id).first()
        if u:
            result.append({"id": u.id, "name": u.name})
    return result

@app.get("/friends/outgoing")
def friends_outgoing(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    outgoing = db.query(Friend).filter(Friend.user_id == user_id).all()
    result = []
    for f in outgoing:
        him_to_me = db.query(Friend).filter(Friend.user_id == f.friend_id, Friend.friend_id == user_id).first()
        if him_to_me:
            continue
        u = db.query(User).filter(User.id == f.friend_id).first()
        if u:
            result.append({"id": u.id, "name": u.name})
    return result

@app.get("/friends")
def friends_mutual(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    my = db.query(Friend.friend_id).filter(Friend.user_id == user_id).subquery()
    mutual_ids = db.query(Friend.user_id).filter(
        Friend.user_id.in_(my),
        Friend.friend_id == user_id
    ).subquery()

    friends = db.query(User).filter(User.id.in_(mutual_ids)).all()
    return [{"id": u.id, "name": u.name} for u in friends]

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
        "achievements": [achievement_to_dto(a) for a in achievements],
    }