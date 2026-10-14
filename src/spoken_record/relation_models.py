"""Data models for co-occurrence edges and GLiREL relations."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spoken_record.evaluation import EntityType
from spoken_record.models import Record


class RelationLabel(BaseModel):
    """One predicate and the entity types allowed on each side."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    prompt_label: str = Field(min_length=1)
    canonical_predicate: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    asymmetric: bool
    allowed_head: list[EntityType] = Field(min_length=1)
    allowed_tail: list[EntityType] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_types(self) -> Self:
        if len(set(self.allowed_head)) != len(self.allowed_head) or \
                len(set(self.allowed_tail)) != len(self.allowed_tail):
            raise ValueError("relation label repeats an allowed entity type")
        return self


class RelationExtractionConfig(BaseModel):
    """GLiREL checkpoint (pinned), thresholds and the list of predicates."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal[1] = 1
    framework: Literal["glirel"]
    framework_version: str
    repo_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    encoder_repo_id: str
    encoder_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    device: Literal["cpu", "mps"]
    offline_only: bool = True
    tokenizer: Literal["spacy-blank-en-v1"]
    context_unit: Literal["sentence_within_passage", "coreference_sentence_with_previous",
                          "coreference_antecedent_window"]
    max_context_sentences: int = Field(default=8, ge=1, le=32)
    max_context_tokens: int = Field(default=256, ge=2)
    encoder_max_tokens: int = Field(default=512, ge=8)
    include_linked_coreference: bool = False
    candidate_floor: float = Field(ge=0, le=1)
    acceptance_threshold: float = Field(ge=0, le=1)
    top_k: int = Field(gt=0)
    labels: list[RelationLabel] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_inventory(self) -> Self:
        prompts = [label.prompt_label for label in self.labels]
        predicates = [label.canonical_predicate for label in self.labels]
        if len(set(prompts)) != len(prompts) or len(set(predicates)) != len(predicates):
            raise ValueError("relation labels and predicates must be unique")
        if self.candidate_floor > self.acceptance_threshold:
            raise ValueError("candidate floor exceeds the acceptance threshold")
        if self.context_unit == "coreference_antecedent_window" and not self.include_linked_coreference:
            raise ValueError("antecedent windows require linked coreference")
        return self


class RelationContext(Record):
    """The transcript context given to the model, including what was left out for the budget."""

    context_id: str
    episode_id: str
    transcript_version_id: str
    passage_id: str
    evidence_passage_ids: list[str]
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    focal_start: int = Field(ge=0)
    focal_end: int = Field(gt=0)
    text_sha256: str
    endpoint_ids: list[str]
    selected_anchor_ids: list[str]
    included_anchor_ids: list[str]
    missing_anchor_reference_ids: list[str]
    token_count: int = Field(gt=0)
    encoder_token_count: int = Field(gt=0)
    sentence_count: int = Field(gt=0)
    status: Literal["ready", "budget_exceeded"]

    @model_validator(mode="after")
    def valid_context(self) -> Self:
        """The focal sentence must be inside the context and the anchors must be selected."""
        if not self.char_start <= self.focal_start < self.focal_end <= self.char_end:
            raise ValueError("focal sentence falls outside relation context")
        if not set(self.included_anchor_ids) <= set(self.selected_anchor_ids):
            raise ValueError("included anchor was not selected")
        return self


class RelationCandidate(Record):
    """One GLiREL prediction between two endpoints (named mentions or resolved pronouns).

    The old ``*_mention_id`` field names hold the endpoint's source ID. The matching kind field
    says whether that is an ExtractedMention ID or a CoreferenceLink ID.
    """

    candidate_id: str
    episode_id: str
    passage_id: str
    context_char_start: int = Field(ge=0)
    context_char_end: int = Field(gt=0)
    head_mention_id: str
    tail_mention_id: str
    head_endpoint_kind: Literal["mention", "coreference_reference"] = "mention"
    tail_endpoint_kind: Literal["mention", "coreference_reference"] = "mention"
    head_corpus_entity_id: str
    tail_corpus_entity_id: str
    head_text: str = Field(min_length=1)
    tail_text: str = Field(min_length=1)
    head_char_start: int = Field(ge=0)
    head_char_end: int = Field(gt=0)
    tail_char_start: int = Field(ge=0)
    tail_char_end: int = Field(gt=0)
    prompt_label: str
    canonical_predicate: str
    score: float = Field(ge=0, le=1)
    decision: Literal["accepted", "below_threshold", "type_rejected", "self_relation",
                      "direction_conflict"]
    reason: str

    @model_validator(mode="after")
    def valid_spans(self) -> Self:
        if self.context_char_end <= self.context_char_start:
            raise ValueError("relation candidate has an empty context span")
        if self.head_char_end <= self.head_char_start or self.tail_char_end <= self.tail_char_start:
            raise ValueError("relation candidate has an empty mention span")
        if not (self.context_char_start <= self.head_char_start < self.head_char_end <= self.context_char_end
                and self.context_char_start <= self.tail_char_start < self.tail_char_end <= self.context_char_end):
            raise ValueError("relation mentions fall outside their model context")
        if self.decision == "accepted" and self.head_corpus_entity_id == self.tail_corpus_entity_id:
            raise ValueError("accepted relation cannot be a self relation")
        return self


class RelationEvidence(BaseModel):
    """Passage and mention behind one piece of relation evidence."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    passage_id: str
    head_mention_id: str
    tail_mention_id: str
    head_endpoint_kind: Literal["mention", "coreference_reference"] = "mention"
    tail_endpoint_kind: Literal["mention", "coreference_reference"] = "mention"
    score: float = Field(ge=0, le=1)


class RelationAssertion(Record):
    """A provisional relation between two entities, with all of its accepted evidence."""

    assertion_id: str
    subject_corpus_entity_id: str
    predicate: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    object_corpus_entity_id: str
    evidence: list[RelationEvidence] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    derivation: Literal["glirel_zero_shot"] = "glirel_zero_shot"
    review_status: Literal["provisional"] = "provisional"

    @model_validator(mode="after")
    def valid_assertion(self) -> Self:
        if self.subject_corpus_entity_id == self.object_corpus_entity_id:
            raise ValueError("relation assertion cannot point to itself")
        if self.confidence != max(item.score for item in self.evidence):
            raise ValueError("assertion confidence must equal its strongest evidence")
        return self


class CooccurrenceEdge(Record):
    """Undirected edge between two entities mentioned in the same passage."""

    edge_id: str
    left_corpus_entity_id: str
    right_corpus_entity_id: str
    passage_ids: list[str] = Field(min_length=1)
    passage_count: int = Field(gt=0)

    @model_validator(mode="after")
    def valid_edge(self) -> Self:
        if self.left_corpus_entity_id >= self.right_corpus_entity_id:
            raise ValueError("co-occurrence endpoint IDs must use stable sorted order")
        if self.passage_count != len(self.passage_ids) or len(set(self.passage_ids)) != self.passage_count:
            raise ValueError("co-occurrence passage count differs from its evidence")
        return self


class RelationOutcome(Record):
    """Result for one passage, alignment failures included."""

    episode_id: str
    passage_id: str
    status: Literal["processed", "skipped", "failed"]
    input_mentions: int = Field(ge=0)
    aligned_mentions: int = Field(ge=0)
    predictions: int = Field(ge=0)
    accepted: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    reason: str | None = None
    error: str | None = None
