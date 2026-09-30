"""
Database connection and session management for SnapForge.
Uses SQLite for local persistent storage of projects, benchmarks, and optimization history.
"""

import os
from pathlib import Path
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.database.schema import Base

DEFAULT_DB_DIR = Path("E:/snapdragon/data")
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "snapforge.db"


class Database:
    _instance = None

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
            db_path = DEFAULT_DB_PATH
        else:
            db_path = Path(db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            connect_args={"check_same_thread": False}
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        self.init_db()

    def init_db(self):
        """Create all tables if they do not exist."""
        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session: Session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @classmethod
    def get_instance(cls, db_path: str | Path | None = None) -> "Database":
        if cls._instance is None:
            cls._instance = Database(db_path)
        return cls._instance


def get_db():
    """Dependency helper for getting a db session."""
    db = Database.get_instance()
    with db.session_scope() as session:
        yield session
