"""
Model Manager for SnapForge.
Coordinates model ingestion, metadata extraction, database storage, and retrieval.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from app.database.database import Database
from app.database.schema import ModelRecord, CompatibilityReport
from app.models.model_inspector import ModelInspector, ModelMetadata
from app.models.compatibility import SnapdragonCompatibilityEngine, CompatibilityAnalysisResult


class ModelManager:
    """Handles model registration and caching."""

    @staticmethod
    def import_model(
        file_path: str | Path,
        project_id: Optional[int] = None,
        custom_name: Optional[str] = None
    ) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # 1. Genuine inspection
        metadata: ModelMetadata = ModelInspector.inspect(path)
        if custom_name:
            metadata.name = custom_name

        # 2. Compatibility Analysis
        compat: CompatibilityAnalysisResult = SnapdragonCompatibilityEngine.analyze(metadata)

        # 3. Store in SQLite
        db = Database.get_instance()
        with db.session_scope() as session:
            # Check input/output shape summary strings
            in_shape_str = ", ".join([f"{i.name}: {i.shape}" for i in metadata.inputs[:2]])
            out_shape_str = ", ".join([f"{o.name}: {o.shape}" for o in metadata.outputs[:2]])

            model_rec = ModelRecord(
                project_id=project_id,
                name=metadata.name,
                format=metadata.format,
                file_path=str(path.resolve()),
                file_size_bytes=metadata.file_size_bytes,
                param_count=metadata.total_params,
                input_shape=in_shape_str or "N/A",
                output_shape=out_shape_str or "N/A",
                data_type=metadata.primary_dtype,
                op_count=len(metadata.nodes),
                layer_count=len(metadata.nodes),
                is_optimized=False,
                created_at=datetime.now(timezone.utc)
            )
            session.add(model_rec)
            session.flush()

            compat_rec = CompatibilityReport(
                model_id=model_rec.id,
                npu_score=compat.npu_score,
                weighted_npu_score=compat.weighted_npu_score,
                gpu_score=compat.gpu_score,
                cpu_score=compat.cpu_score,
                supported_op_count=compat.supported_nodes_count,
                partial_op_count=compat.partial_nodes_count,
                unsupported_op_count=compat.unsupported_nodes_count,
                operators_json=json.dumps([n.to_dict() for n in compat.node_evaluations]),
                bottlenecks_json=json.dumps(compat.bottlenecks),
                recommendations_json=json.dumps(compat.recommendations),
                created_at=datetime.now(timezone.utc)
            )
            session.add(compat_rec)
            session.flush()
            model_id = model_rec.id

        return {
            "model_id": model_id,
            "metadata": metadata,
            "compatibility": compat
        }
