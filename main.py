from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import date, datetime, timedelta
from sqlalchemy import and_, or_
from fastapi import Request
from fastapi import Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


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
    show_in_profile = Column(Integer, default=0)

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

class GoalUpdateBody(BaseModel):
    title: str | None = None
    description: str | None = None
    goalType: str | None = None
    targetValue: int | None = None
    unit: str | None = None
    deadline: str | None = None
    priority: int | None = None
    status: str | None = None
    showInProfile : bool | None = None

class GoalTemplateOut(BaseModel):
    id: int
    title: str
    description: str
    category: str
    suggestedTarget: int | None = None
    suggestedUnit: str | None = None

# ================= TEMPLATES =================

GOAL_TEMPLATES: list[GoalTemplateOut] = [
    GoalTemplateOut(
        id=1,
        title="10 000 шагов в день",
        description="Ежедневно проходить 10 000 шагов",
        category="Здоровье",
        suggestedTarget=10000,
        suggestedUnit="шагов",
    ),
    GoalTemplateOut(
        id=2,
        title="Пробежать 5 км",
        description="Пробежать дистанцию 5 км",
        category="Спорт",
        suggestedTarget=5,
        suggestedUnit="км",
    ),
    GoalTemplateOut(
        id=3,
        title="Прочитать 12 книг за год",
        description="Постепенно читать книги в течение года",
        category="Образование",
        suggestedTarget=12,
        suggestedUnit="книг",
    ),
    GoalTemplateOut(
        id=4,
        title="Медитация 30 дней",
        description="Держать привычку медитации 30 дней подряд",
        category="Психология",
        suggestedTarget=30,
        suggestedUnit="дней",
    ),
]




# ================= SCHEMAS =================

class GoalProgressBody(BaseModel):
    delta: int = 1  # насколько прибавить

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
    status: str = "ACTIVE"
    showInProfile: bool = False

class GoalStatusBody(BaseModel):
    status: str

class Friend(Base):
    __tablename__ = "friends"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    friend_id = Column(Integer, ForeignKey("users.id"), nullable=False)

Base.metadata.create_all(engine)

class HabitUpdateBody(BaseModel):
    title: str | None = None
    periodDays: int | None = None
    timesPerPeriod: int | None = None


# ================= AUTH UTILS =================
def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(password: str, hash_: str):
    return pwd_context.verify(password, hash_)

def create_token(user_id: int):
    return jwt.encode({"user_id": user_id}, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user_id(
    creds: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
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

    # инкремент
    g.progress_value = (g.progress_value or 0) + body.delta

    # если есть targetValue — можно ограничить и/или автозавершать
    if g.target_value is not None:
        if g.progress_value >= g.target_value:
            g.progress_value = g.target_value
            # опционально: авто DONE
            # prev_status = g.status
            g.status = "DONE"

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

    # Награды за DONE (если статус стал DONE через редактирование)
    if prev_status != "DONE" and g.status == "DONE":
        done_count = db.query(Goal).filter_by(user_id=user_id, status="DONE").count()
        for n in GOAL_DONE_THRESHOLDS:
            if done_count >= n:
                ensure_achievement(db, user_id, f"GOAL_DONE_{n}", f"Завершено {n} целей")
        db.commit()

    db.refresh(g)
    return goal_to_dto(g)

from fastapi import Query
#проверка что все работает
@app.post("/debug/habits/{habit_id}/add_streak")
def debug_add_habit_streak(
    habit_id: int,
    n: int = Query(7, ge=1, le=365),
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    # 1) проверяем, что привычка твоя
    h = db.query(Habit).filter_by(id=habit_id, user_id=user_id).first()
    if not h:
        raise HTTPException(404, "Habit not found")

    # 2) добавляем чек-ины за последние n дней (включая сегодня)
    today = date.today()
    created = 0
    for i in range(n):
        d = (today - timedelta(days=i)).isoformat()
        exists = db.query(HabitCheckin).filter_by(habit_id=habit_id, date=d).first()
        if not exists:
            db.add(HabitCheckin(habit_id=habit_id, date=d, value=None))
            created += 1

    db.commit()

    # 3) пересчитываем стрик и выдаём награды
    dates = [c.date for c in db.query(HabitCheckin).filter_by(habit_id=habit_id).all()]
    streak = compute_current_streak(dates)

    for t in HABIT_STREAK_THRESHOLDS:
        if streak >= t:
            ensure_achievement(db, user_id, f"HABIT_STREAK_{t}", f"{t} дней подряд")
    db.commit()

    return {"ok": True, "habitId": habit_id, "createdCheckins": created, "currentStreak": streak}
#проверка что все работает


@app.put("/habits/{habit_id}")
def update_habit(habit_id: int, body: HabitUpdateBody,
                 user_id: int = Depends(get_current_user_id),
                 db: Session = Depends(get_db)):
    h = db.query(Habit).filter_by(id=habit_id, user_id=user_id).first()
    if not h:
        raise HTTPException(404, "Habit not found")

    if body.title is not None: h.title = body.title
    if body.periodDays is not None: h.period_days = body.periodDays
    if body.timesPerPeriod is not None: h.times_per_period = body.timesPerPeriod

    db.commit()
    db.refresh(h)

    # ✅ важно: возвращаем camelCase DTO
    return habit_to_dto(h)



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
        "showInProfile": bool(g.show_in_profile)
    }

def habit_to_dto(h: Habit) -> dict:
    return {
        "id": h.id,
        "title": h.title,
        "periodDays": h.period_days,
        "timesPerPeriod": h.times_per_period,
    }


@app.get("/goals")
def get_goals(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    goals = db.query(Goal).filter_by(user_id=user_id).all()
    return [goal_to_dto(g) for g in goals]

@app.post("/goals")
async def create_goal(
    request: Request,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    # 1) RAW body
    raw_bytes = await request.body()
    raw_text = raw_bytes.decode("utf-8", errors="ignore")
    print("RAW:", raw_text)

    # 2) JSON dict
    try:
        data = await request.json()
    except Exception as e:
        print("JSON PARSE ERROR:", repr(e))
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    print("JSON:", data)

    # 3) Pydantic parse
    try:
        body = GoalCreateBody(**data)
    except Exception as e:
        print("Pydantic ERROR:", repr(e))
        raise HTTPException(status_code=422, detail=str(e))

    print("PARSED:", body.dict())

    # 4) DB insert
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

    # !!! важно: возвращаем DTO (camelCase), а не ORM модель
    return goal_to_dto(g)

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

    visible_goals = db.query(Goal).filter_bvisible_goals = (
    db.query(Goal)
      .filter(Goal.user_id == user_id, Goal.show_in_profile == 1)
      .all()
)

    return {
        "currentHabitStreak": max_streak,
        "goalsCompleted": goals_done,
        "achievements": achievements,
        "goals": [goal_to_dto(g) for g in visible_goals],
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

@app.get("/friends/incoming")
def friends_incoming(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    # те, кто добавил тебя, но ты ещё не добавил их
    incoming = db.query(Friend).filter(Friend.friend_id == user_id).all()
    result = []
    for f in incoming:
        me_to_him = db.query(Friend).filter(Friend.user_id == user_id, Friend.friend_id == f.user_id).first()
        if me_to_him:
            continue  # уже взаимка
        u = db.query(User).filter(User.id == f.user_id).first()
        if u:
            result.append({"id": u.id, "name": u.name})
    return result


@app.get("/friends/outgoing")
def friends_outgoing(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    # ты добавил, но тебя ещё не добавили
    outgoing = db.query(Friend).filter(Friend.user_id == user_id).all()
    result = []
    for f in outgoing:
        him_to_me = db.query(Friend).filter(Friend.user_id == f.friend_id, Friend.friend_id == user_id).first()
        if him_to_me:
            continue  # уже взаимка
        u = db.query(User).filter(User.id == f.friend_id).first()
        if u:
            result.append({"id": u.id, "name": u.name})
    return result


@app.get("/friends")
def friends_mutual(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    # только взаимные друзья + краткая статистика (награды/стрик/цели)
    my = db.query(Friend.friend_id).filter(Friend.user_id == user_id).subquery()
    mutual_ids = db.query(Friend.user_id).filter(
        Friend.user_id.in_(my),
        Friend.friend_id == user_id
    ).subquery()

    friends = db.query(User).filter(User.id.in_(mutual_ids)).all()

    out = []
    for u in friends:
        # награды
        ach_count = db.query(UserAchievement).filter(UserAchievement.user_id == u.id).count()

        # цели done
        goals_done = db.query(Goal).filter(Goal.user_id == u.id, Goal.status == "DONE").count()

        # max streak
        habits = db.query(Habit).filter(Habit.user_id == u.id).all()
        max_streak = 0
        for h in habits:
            dates = [c.date for c in db.query(HabitCheckin).filter(HabitCheckin.habit_id == h.id).all()]
            max_streak = max(max_streak, compute_current_streak(dates))

        out.append({
            "id": u.id,
            "name": u.name,
            "achievementsCount": ach_count,
            "currentHabitStreak": max_streak,
            "goalsCompleted": goals_done
        })
    return out
#видно цель не видно цель показывать, цели с этапами добавить,по количественым выбираем цель смотрим прогресс, в чт надо диаграммы

@app.post("/friends/{friend_id}")
def friends_add_or_accept(friend_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    # add = создать исходящий запрос, accept = создать ответный запрос -> взаимка
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
def friends_remove_or_cancel(friend_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    # удаляет ТВОЮ связь user->friend (это и отмена исходящего запроса, и "удалить друга")
    row = db.query(Friend).filter(Friend.user_id == user_id, Friend.friend_id == friend_id).first()
    if row:
        db.delete(row)
        db.commit()
    return {"ok": True}


@app.delete("/habits/{habit_id}")
def delete_habit(habit_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    h = db.query(Habit).filter_by(id=habit_id, user_id=user_id).first()
    if not h:
        raise HTTPException(404, "Habit not found")

    # можно удалить чек-ины, чтобы не оставлять мусор
    db.query(HabitCheckin).filter(HabitCheckin.habit_id == habit_id).delete()
    db.delete(h)
    db.commit()
    return {"ok": True}

@app.delete("/goals/{goal_id}")
def delete_goal(goal_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    g = db.query(Goal).filter_by(id=goal_id, user_id=user_id).first()
    if not g:
        raise HTTPException(404, "Goal not found")

    # если есть шаги
    db.query(GoalStep).filter(GoalStep.goal_id == goal_id).delete()
    db.delete(g)
    db.commit()
    return {"ok": True}

@app.get("/templates/goals")
def get_goal_templates(user_id: int = Depends(get_current_user_id)):
    # user_id тут нужен только чтобы эндпоинт был защищён токеном, как у тебя всё остальное
    return GOAL_TEMPLATES

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
def get_goal_templates(
    user_id: int = Depends(get_current_user_id)
):
    # user_id тут просто чтобы endpoint был защищён токеном
    return GOAL_TEMPLATES


