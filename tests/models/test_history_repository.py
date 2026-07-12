import unittest
from types import SimpleNamespace

from src.models.history_repository import (
    AnalysisHistoryCreate,
    HistoryError,
    HistoryRepository,
)


class FakeQuery:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def insert(self, payload):
        self.calls.append(("insert", payload))
        return self

    def select(self, columns):
        self.calls.append(("select", columns))
        return self

    def eq(self, column, value):
        self.calls.append(("eq", column, value))
        return self

    def order(self, column, desc=False):
        self.calls.append(("order", column, desc))
        return self

    def range(self, start, end):
        self.calls.append(("range", start, end))
        return self

    def execute(self):
        return SimpleNamespace(data=self.data)


class FakeClient:
    def __init__(self, data):
        self.query = FakeQuery(data)

    def table(self, name):
        self.query.calls.append(("table", name))
        return self.query


class HistoryRepositoryTests(unittest.TestCase):
    def test_create_scopes_payload_to_authenticated_user(self):
        client = FakeClient([{"id": "history-1"}])
        repository = HistoryRepository(client)
        record = AnalysisHistoryCreate(
            user_id="user-a",
            input_text="Job description",
            input_source="text",
            prediction_label="Legitimate Job",
            confidence=0.92,
            red_flags=[],
        )

        result = repository.create(record)

        self.assertEqual(result["id"], "history-1")
        insert_call = next(call for call in client.query.calls if call[0] == "insert")
        self.assertEqual(insert_call[1]["user_id"], "user-a")

    def test_create_rejects_out_of_range_confidence(self):
        repository = HistoryRepository(FakeClient([]))
        with self.assertRaises(HistoryError):
            repository.create(
                AnalysisHistoryCreate(
                    user_id="user-a",
                    input_text="Job",
                    input_source="text",
                    prediction_label="Legitimate Job",
                    confidence=1.1,
                    red_flags=[],
                )
            )

    def test_list_page_fetches_one_extra_row_to_detect_next_page(self):
        rows = [{"id": str(index)} for index in range(21)]
        client = FakeClient(rows)
        page = HistoryRepository(client).list_page("user-a", offset=20)

        self.assertEqual(len(page.items), 20)
        self.assertTrue(page.has_more)
        self.assertIn(("eq", "user_id", "user-a"), client.query.calls)
        self.assertIn(("range", 20, 40), client.query.calls)


if __name__ == "__main__":
    unittest.main()
