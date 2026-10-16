from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "benchmark"


def writable_path(path: str | Path) -> Path:
    """Never write into the read-only benchmark/reference tree, even via symlinks."""
    path = Path(path).expanduser()
    path = (ROOT / path if not path.is_absolute() else path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Output must be inside {ROOT}: {path}")
    return path


def dump_json(path: Path, data) -> None:
    path = writable_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = writable_path(path.with_suffix(path.suffix + ".tmp"))
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@dataclass
class ModelConfig:
    name: str = "gpt-4o"
    base_url_env: str = "OPENAI_BASE_URL"
    api_key_env: str = "OPENAI_API_KEY"
    # User/provider supplied rates. Do not silently assume current prices.
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None
    max_output_tokens: int = 6000
    token_limit_field: str = "max_completion_tokens"
    json_mode: bool = True
    timeout_seconds: float = 180
    extra_body: dict = field(default_factory=dict)


@dataclass
class BudgetConfig:
    # None means accounting only; finite limits can be set for evaluation.
    max_tokens: int | None = None
    max_calls: int | None = None
    max_vlm_calls: int | None = None
    max_usd: float | None = None
    content_fraction: float = 0.70
    image_token_reservation: int = 8192


@dataclass
class EvidenceSelectionConfig:
    max_figures: int = 4
    max_candidates: int = 28
    semantic_capacity: bool = False


@dataclass
class LayoutConfig:
    width_inches: float = 48
    height_inches: float = 36
    columns: int = 3
    margin_inches: float = 0.5
    gap_inches: float = 0.30
    body_font_pt: float = 36
    minimum_font_pt: float = 24
    minimum_image_dpi: float = 65
    render_dpi: int = 96
    renderer_timeout_seconds: int = 180
    design: str = "classic"
    editorial_flow: str = "columns"


@dataclass
class Config:
    text_model: ModelConfig = field(default_factory=ModelConfig)
    vision_model: ModelConfig = field(default_factory=ModelConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    evidence_selection: EvidenceSelectionConfig = field(default_factory=EvidenceSelectionConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    max_document_chars: int = 60000
    benchmark_root: str = str(REFERENCE)
    reference_root: str = str(ROOT / "evaluation")
    output_root: str = "outputs"
    eval_max_tokens: int = 2000000
    eval_max_calls: int = 400
    eval_judge_prompt: str = "compact-fill"
    eval_quiz_limit: int = 50
    eval_fill_attempts: int = 6

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        values = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
        # Accept older run configurations that included retired local metrics.
        for name in ('offline_ppl_path', 'offline_clip_path', 'offline_eval_python'):
            values.pop(name, None)
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        for name, kind in [("text_model", ModelConfig), ("vision_model", ModelConfig),
                           ("budget", BudgetConfig), ("evidence_selection", EvidenceSelectionConfig), ("layout", LayoutConfig)]:
            if name in values:
                values[name] = kind(**values[name])
        config = cls(**values)
        config.validate()
        return config

    def validate(self):
        if self.eval_judge_prompt not in ('compact', 'compact-fill', 'detailed'):
            raise ValueError('Unknown evaluation Judge prompt protocol')
        for name in ('eval_max_tokens', 'eval_max_calls', 'eval_quiz_limit', 'eval_fill_attempts'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        selection = self.evidence_selection
        if type(selection.semantic_capacity) is not bool or type(selection.max_candidates) is not int or selection.max_candidates < 1:
            raise ValueError('Invalid evidence selection settings')
        if type(selection.max_figures) is not int or selection.max_figures < 0:
            raise ValueError('Invalid visual selection limit')
        if self.layout.editorial_flow not in ('columns', 'bands'):
            raise ValueError('Unknown editorial flow')
        if self.layout.editorial_flow != 'columns' and self.layout.design != 'editorial':
            raise ValueError('Evidence bands require editorial layout')
        if self.layout.design not in ("classic", "balanced", "portfolio", "editorial"):
            raise ValueError("Unknown layout design")
        for model in (self.text_model, self.vision_model):
            if not isinstance(model.extra_body, dict) or set(model.extra_body) - {'thinking', 'reasoning_effort'}:
                raise ValueError('extra_body only accepts provider thinking/reasoning_effort controls')
            if not model.name or any(x in model.name.lower() for x in ("mini", "nano", "vllm", "local_")):
                raise ValueError("Generation requires a full-size hosted API model, not a small/local model")
            sizes=re.findall(r"(?<![0-9.])(\d+(?:\.\d+)?)b\b",model.name.lower())
            if any(float(size)<30 for size in sizes):
                raise ValueError("Generation model explicitly names fewer than 30B parameters; use a full-size hosted model")
            if model.token_limit_field not in ("max_tokens", "max_completion_tokens"):
                raise ValueError("Unsupported API token limit field")
            for price in (model.input_usd_per_million, model.output_usd_per_million):
                if price is not None and (not math.isfinite(price) or price < 0):
                    raise ValueError("Prices must be finite and nonnegative")
            if self.budget.max_usd is not None and None in (model.input_usd_per_million, model.output_usd_per_million):
                raise ValueError("max_usd requires explicit input/output prices for both API models")
            if model.max_output_tokens <= 0 or model.timeout_seconds <= 0:
                raise ValueError("Invalid model limits")
        b, l = self.budget, self.layout
        for name, value, minimum in [('budget.max_tokens', b.max_tokens, 1),
                                     ('budget.max_calls', b.max_calls, 1),
                                     ('budget.max_vlm_calls', b.max_vlm_calls, 0)]:
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f'{name} must be null or an integer >= {minimum}')
        if b.image_token_reservation <= 0:
            raise ValueError("Image token reservation must be positive")
        if not 0 < b.content_fraction < 1:
            raise ValueError("Invalid budget allocation")
        if b.max_usd is not None and (not math.isfinite(b.max_usd) or b.max_usd <= 0):
            raise ValueError("Invalid dollar budget")
        if l.columns < 1 or l.columns > 6:
            raise ValueError("Invalid visual capacity")
        if min(l.width_inches, l.height_inches, l.margin_inches, l.gap_inches, l.minimum_font_pt) <= 0:
            raise ValueError("Invalid geometry")
        if l.body_font_pt < l.minimum_font_pt:
            raise ValueError("Invalid layout limits")
        if l.width_inches <= 2*l.margin_inches + (l.columns-1)*l.gap_inches + l.columns:
            raise ValueError("Canvas too narrow for requested columns")
        if l.height_inches < 12 or l.render_dpi < 30 or self.max_document_chars < 2000:
            raise ValueError("Insufficient canvas/resolution/document capacity")
        writable_path(self.output_root)

    def to_dict(self):
        return asdict(self)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()
