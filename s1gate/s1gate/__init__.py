"""s1gate: a System 1 decision gate (typed, calibrated, local classification)."""

from .schema import DecisionSchema, Question
from .records import Record, read_jsonl, write_jsonl

__all__ = ["DecisionSchema", "Question", "Record", "read_jsonl", "write_jsonl"]
__version__ = "0.1.0"
