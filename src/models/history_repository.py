from dataclasses import dataclass
from typing import Any

from supabase import Client


class HistoryError(RuntimeError):
    """Safe persistence error for analysis history."""


@dataclass(frozen=True)
class AnalysisHistoryCreate:
    user_id: str
    input_text: str
    input_source: str
    prediction_label: str
    confidence: float
    red_flags: list[str]


@dataclass(frozen=True)
class HistoryPage:
    items: list[dict[str, Any]]
    has_more: bool


class HistoryRepository:
    PAGE_SIZE = 20

    def __init__(self, client: Client):
        self.client = client

    def create(self, record: AnalysisHistoryCreate) -> dict[str, Any]:
        self._validate(record)
        payload = {
            "user_id": record.user_id,
            "input_text": record.input_text,
            "input_source": record.input_source,
            "prediction_label": record.prediction_label,
            "confidence": record.confidence,
            "red_flags": record.red_flags,
        }
        try:
            response = self.client.table("analysis_history").insert(payload).execute()
        except Exception as exc:
            raise HistoryError("Analysis history could not be saved.") from exc
        if not response.data:
            raise HistoryError("Analysis history could not be saved.")
        return response.data[0]

    def list_page(self, user_id: str, offset: int = 0) -> HistoryPage:
        if not user_id:
            raise HistoryError("Authenticated user is required.")
        if offset < 0 or offset % self.PAGE_SIZE != 0:
            raise HistoryError("Invalid history offset.")
        try:
            response = (
                self.client.table("analysis_history")
                .select(
                    "id,input_text,input_source,prediction_label,"
                    "confidence,red_flags,created_at"
                )
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .range(offset, offset + self.PAGE_SIZE)
                .execute()
            )
        except Exception as exc:
            raise HistoryError("Analysis history could not be loaded.") from exc
        rows = list(response.data or [])
        return HistoryPage(
            items=rows[: self.PAGE_SIZE],
            has_more=len(rows) > self.PAGE_SIZE,
        )

    @staticmethod
    def _validate(record: AnalysisHistoryCreate) -> None:
        if not record.user_id:
            raise HistoryError("Authenticated user is required.")
        if not record.input_text.strip():
            raise HistoryError("Analysis text is required.")
        if record.input_source not in {"text", "image"}:
            raise HistoryError("Invalid analysis input source.")
        if record.prediction_label not in {"Legitimate Job", "Potential Scam"}:
            raise HistoryError("Invalid prediction label.")
        if not 0.0 <= record.confidence <= 1.0:
            raise HistoryError("Confidence must be between 0 and 1.")
        if not all(isinstance(flag, str) for flag in record.red_flags):
            raise HistoryError("Red flags must be strings.")
