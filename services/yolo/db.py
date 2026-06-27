import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# SQLite for local dev; override with a full PostgreSQL URL in production.
# e.g. DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/dbname
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./predictions.db")

# check_same_thread is a SQLite-only connect arg; omit it for other backends.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def init_db():
    """Create all tables via the ORM metadata (replaces the old raw DDL)."""
    # Import models so their tables are registered on Base.metadata before create_all.
    import models  # noqa: F401
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
