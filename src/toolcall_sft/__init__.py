"""LoRA fine-tuning of small local LLMs on multi-turn tool-calling dialogues.

The root package stays free of torch; training lives in ``toolcall_sft.training``
and is pulled in only by the ``tcsft-train`` CLI. transformers is needed at
runtime only where a tokenizer is actually used (``masking``, ``stats``).
"""

from .anonymizer import AnonymizationReport, Anonymizer
from .config import (
    DEFAULT_CHAT_TEMPLATE_KWARGS,
    LR_SCHEDULERS,
    ConfigError,
    DatasetSettings,
    ExperimentConfig,
    LoraSettings,
    TrackingSettings,
    TrainingSettings,
    flatten_for_logging,
    load_experiment_config,
)
from .dataset import (
    DatasetSplit,
    DatasetStats,
    dataset_stats,
    dedup_dialogues,
    load_dialogues,
    split_dialogues,
    write_dialogues,
)
from .generate import BRANCH_WEIGHTS, GenerationError, branch_names, generate_dialogues
from .masking import (
    LABEL_IGNORE_INDEX,
    MaskedExample,
    TemplateCompatibilityError,
    render_example,
    tokenize_dialogue,
)
from .metrics import ToolCallComparison, ToolCallReport, aggregate_comparisons, compare_tool_calls
from .scenario import SYSTEM_PROMPT, TOOLS, tool_names
from .schema import (
    DatasetError,
    Dialogue,
    Message,
    Role,
    ToolCall,
    chat_messages,
    content_fingerprint,
    dialogue_from_json,
    dialogue_to_json,
    to_chat_record,
)
from .stats import (
    DEFAULT_THRESHOLDS,
    HISTOGRAM_EDGES,
    DialogueTokenCount,
    LengthFilterResult,
    TokenStatsReport,
    compute_token_stats,
    filter_by_length,
    histogram,
    percentile,
    threshold_fits,
)

__all__ = [
    "BRANCH_WEIGHTS",
    "DEFAULT_CHAT_TEMPLATE_KWARGS",
    "DEFAULT_THRESHOLDS",
    "HISTOGRAM_EDGES",
    "LABEL_IGNORE_INDEX",
    "LR_SCHEDULERS",
    "SYSTEM_PROMPT",
    "TOOLS",
    "AnonymizationReport",
    "Anonymizer",
    "ConfigError",
    "DatasetError",
    "DatasetSettings",
    "DatasetSplit",
    "DatasetStats",
    "Dialogue",
    "DialogueTokenCount",
    "GenerationError",
    "LengthFilterResult",
    "ExperimentConfig",
    "LoraSettings",
    "MaskedExample",
    "Message",
    "Role",
    "TemplateCompatibilityError",
    "TokenStatsReport",
    "ToolCall",
    "ToolCallComparison",
    "ToolCallReport",
    "TrackingSettings",
    "TrainingSettings",
    "aggregate_comparisons",
    "branch_names",
    "chat_messages",
    "compare_tool_calls",
    "compute_token_stats",
    "filter_by_length",
    "content_fingerprint",
    "dataset_stats",
    "dedup_dialogues",
    "dialogue_from_json",
    "dialogue_to_json",
    "flatten_for_logging",
    "generate_dialogues",
    "histogram",
    "load_dialogues",
    "load_experiment_config",
    "percentile",
    "render_example",
    "split_dialogues",
    "threshold_fits",
    "to_chat_record",
    "tokenize_dialogue",
    "tool_names",
    "write_dialogues",
]
