"""Artifact ORM 模型 — 伪影处理作业"""

from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
import json

# 使用 Text 类型兼容 SQLite 的 JSON 列
from app.database.session import Base


class ArtifactJob(Base):
    """伪影处理作业"""
    __tablename__ = "artifact_jobs"

    id = Column(String, primary_key=True, index=True)
    job_type = Column(String, nullable=False)
    status = Column(String, default="pending")

    _config = Column("config", Text, nullable=True)
    output_path = Column(String, nullable=True)
    output_format = Column(String, default="nrrd")
    _quality_metrics = Column("quality_metrics", Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    @property
    def config(self):
        if self._config:
            try: return json.loads(self._config)
            except: return {}
        return None

    @config.setter
    def config(self, value):
        self._config = json.dumps(value) if value is not None else None

    @property
    def quality_metrics(self):
        if self._quality_metrics:
            try: return json.loads(self._quality_metrics)
            except: return {}
        return None

    @quality_metrics.setter
    def quality_metrics(self, value):
        self._quality_metrics = json.dumps(value) if value is not None else None
