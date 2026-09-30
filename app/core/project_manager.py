"""
Project Manager for SnapForge.
Coordinates project creation, persistence, model associations, and retrieval.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from app.database.database import Database
from app.database.schema import Project, ModelRecord, OptimizationRun, BenchmarkResult, ReportRecord


class ProjectManager:
    """Manages SnapForge projects in the SQLite database."""

    @staticmethod
    def create_project(name: str, description: str = "", model_path: str = "") -> Project:
        db = Database.get_instance()
        with db.session_scope() as session:
            project = Project(
                name=name,
                description=description,
                status="Created",
                original_model_path=model_path,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            session.add(project)
            session.flush()
            session.refresh(project)
            # Detach/copy data
            p_id = project.id
            p_name = project.name
            p_desc = project.description
            p_stat = project.status
            p_path = project.original_model_path

        return Project(id=p_id, name=p_name, description=p_desc, status=p_stat, original_model_path=p_path)

    @staticmethod
    def list_projects() -> List[Dict[str, Any]]:
        db = Database.get_instance()
        with db.session_scope() as session:
            projects = session.query(Project).order_by(Project.updated_at.desc()).all()
            res = []
            for p in projects:
                # Find speedup if benchmarked
                speedup_str = "Not benchmarked"
                b_results = session.query(BenchmarkResult).filter_by(project_id=p.id).all()
                orig_b = next((b for b in b_results if b.model_label == "Original"), None)
                opt_b = next((b for b in b_results if b.model_label == "Optimized"), None)
                if orig_b and opt_b and opt_b.median_latency_ms > 0:
                    speedup = round(orig_b.median_latency_ms / opt_b.median_latency_ms, 2)
                    speedup_str = f"{speedup}x ({opt_b.label_type})"

                model_name = "None"
                if p.models:
                    model_name = p.models[0].name
                elif p.original_model_path:
                    from pathlib import Path
                    model_name = Path(p.original_model_path).stem

                res.append({
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "status": p.status,
                    "model_name": model_name,
                    "model_path": p.original_model_path,
                    "speedup": speedup_str,
                    "created_at": p.created_at.strftime("%Y-%m-%d %H:%M") if p.created_at else "",
                    "updated_at": p.updated_at.strftime("%Y-%m-%d %H:%M") if p.updated_at else ""
                })
            return res

    @staticmethod
    def get_project(project_id: int) -> Optional[Dict[str, Any]]:
        db = Database.get_instance()
        with db.session_scope() as session:
            p = session.query(Project).filter_by(id=project_id).first()
            if not p:
                return None
            return {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "status": p.status,
                "original_model_path": p.original_model_path,
                "created_at": p.created_at.isoformat() if p.created_at else "",
                "updated_at": p.updated_at.isoformat() if p.updated_at else ""
            }

    @staticmethod
    def update_status(project_id: int, status: str):
        db = Database.get_instance()
        with db.session_scope() as session:
            p = session.query(Project).filter_by(id=project_id).first()
            if p:
                p.status = status
                p.updated_at = datetime.now(timezone.utc)

    @staticmethod
    def update_model_path(project_id: int, model_path: str):
        db = Database.get_instance()
        with db.session_scope() as session:
            p = session.query(Project).filter_by(id=project_id).first()
            if p:
                p.original_model_path = model_path
                p.updated_at = datetime.now(timezone.utc)
