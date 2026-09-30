"""
Database schema for SnapForge.
Defines tables for projects, models, compatibility analyses,
optimization runs, benchmark results, hardware snapshots, and reports.
"""

from datetime import datetime, timezone

def utc_now():
    return datetime.now(timezone.utc)

from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, ForeignKey, Boolean
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    status = Column(String(50), default="Created")  # Created, Analyzed, Optimized, Benchmarked
    original_model_path = Column(String(1024), default="")
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    # Relationships
    models = relationship("ModelRecord", back_populates="project", cascade="all, delete-orphan")
    optimization_runs = relationship("OptimizationRun", back_populates="project", cascade="all, delete-orphan")
    benchmark_results = relationship("BenchmarkResult", back_populates="project", cascade="all, delete-orphan")
    reports = relationship("ReportRecord", back_populates="project", cascade="all, delete-orphan")


class ModelRecord(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    name = Column(String(255), nullable=False)
    format = Column(String(50), default="ONNX")  # ONNX, PyTorch, TorchScript
    file_path = Column(String(1024), nullable=False)
    file_size_bytes = Column(Integer, default=0)
    param_count = Column(Integer, default=0)
    input_shape = Column(String(255), default="")
    output_shape = Column(String(255), default="")
    data_type = Column(String(50), default="float32")
    op_count = Column(Integer, default=0)
    layer_count = Column(Integer, default=0)
    is_optimized = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    project = relationship("Project", back_populates="models")
    compatibility_reports = relationship("CompatibilityReport", back_populates="model", cascade="all, delete-orphan")


class CompatibilityReport(Base):
    __tablename__ = "compatibility_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    npu_score = Column(Float, default=0.0)             # 0.0 - 100.0%
    weighted_npu_score = Column(Float, default=0.0)    # 0.0 - 100.0% (FLOPs weighted)
    gpu_score = Column(Float, default=0.0)
    cpu_score = Column(Float, default=100.0)
    supported_op_count = Column(Integer, default=0)
    partial_op_count = Column(Integer, default=0)
    unsupported_op_count = Column(Integer, default=0)
    operators_json = Column(Text, default="[]")        # Detailed per-node breakdown
    bottlenecks_json = Column(Text, default="[]")      # NPU blockers
    recommendations_json = Column(Text, default="[]")
    created_at = Column(DateTime, default=utc_now)

    model = relationship("ModelRecord", back_populates="compatibility_reports")


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    target_backend = Column(String(100), default="Snapdragon NPU")
    strategy = Column(String(100), default="Balanced")  # Performance, Battery, Accuracy, Balanced
    precision = Column(String(50), default="INT8")      # FP32, FP16, INT8
    optimizations_applied_json = Column(Text, default="[]")
    optimized_model_path = Column(String(1024), default="")
    original_size_bytes = Column(Integer, default=0)
    optimized_size_bytes = Column(Integer, default=0)
    size_reduction_pct = Column(Float, default=0.0)
    status = Column(String(50), default="Completed")
    log_text = Column(Text, default="")
    created_at = Column(DateTime, default=utc_now)

    project = relationship("Project", back_populates="optimization_runs")


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    optimization_run_id = Column(Integer, ForeignKey("optimization_runs.id"), nullable=True)
    model_label = Column(String(100), default="Original")  # Original or Optimized
    backend_name = Column(String(100), default="CPU")
    device_name = Column(String(255), default="")
    warmup_runs = Column(Integer, default=10)
    measured_runs = Column(Integer, default=100)
    median_latency_ms = Column(Float, default=0.0)
    mean_latency_ms = Column(Float, default=0.0)
    p95_latency_ms = Column(Float, default=0.0)
    min_latency_ms = Column(Float, default=0.0)
    max_latency_ms = Column(Float, default=0.0)
    throughput_fps = Column(Float, default=0.0)
    peak_memory_mb = Column(Float, default=0.0)
    cpu_util_pct = Column(Float, default=0.0)
    accuracy_score = Column(Float, nullable=True)
    label_type = Column(String(50), default="Measured")  # Measured, Estimated, Simulated, Unsupported
    raw_metrics_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=utc_now)

    project = relationship("Project", back_populates="benchmark_results")


class HardwareSnapshot(Base):
    __tablename__ = "hardware_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cpu_model = Column(String(255), default="")
    cpu_arch = Column(String(50), default="")
    gpu_name = Column(String(255), default="")
    npu_name = Column(String(255), default="")
    npu_status = Column(String(100), default="Unavailable")
    total_ram_gb = Column(Float, default=0.0)
    os_version = Column(String(255), default="")
    qualcomm_runtime_status = Column(String(255), default="")
    snapshot_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=utc_now)


class ReportRecord(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    title = Column(String(255), default="SnapForge Optimization Report")
    report_path = Column(String(1024), default="")
    format = Column(String(50), default="PDF")  # PDF, JSON, CSV
    created_at = Column(DateTime, default=utc_now)

    project = relationship("Project", back_populates="reports")
