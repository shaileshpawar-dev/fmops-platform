"""AutoML: dataset profiling, target recommendation and candidate search.

Everything here is deterministic and rule-based. There is no model choosing
models and no LLM in the loop -- recommendations come from observable dataset
characteristics and are explained in those terms, which is what makes them
reviewable and overridable.

The training itself is not reimplemented. Each candidate goes through
``app.training.train.train_model`` with a per-run settings override, and the
winner goes through the same registry and approval gate as any other model.
"""

from app.automl.profiler import (
    ColumnProfile,
    DatasetProfile,
    ProblemType,
    profile_frame,
)
from app.automl.recommend import CandidateRecommendation, recommend_candidates

__all__ = [
    "CandidateRecommendation",
    "ColumnProfile",
    "DatasetProfile",
    "ProblemType",
    "profile_frame",
    "recommend_candidates",
]
