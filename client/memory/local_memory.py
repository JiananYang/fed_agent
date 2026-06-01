from __future__ import annotations

from shared.schemas import MemoryRecord, new_id


class LocalMemory:
    """Simple private client memory.

    This stores records in memory for the prototype. Replace with SQLite plus
    a vector store when the demo needs persistence and semantic retrieval.
    """

    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        self.records: list[MemoryRecord] = []

    def write(
        self,
        *,
        text: str,
        memory_type: str,
        tags: list[str],
        source_task_id: str | None,
        sensitivity: str = "private",
    ) -> MemoryRecord:
        record = MemoryRecord(
            memory_id=new_id("mem"),
            client_id=self.client_id,
            memory_type=memory_type,
            text=text,
            tags=tags,
            source_task_id=source_task_id,
            sensitivity=sensitivity,
        )
        self.records.append(record)
        return record

    def search(self, query: str, limit: int = 3) -> list[MemoryRecord]:
        terms = set(query.lower().split())

        def score(record: MemoryRecord) -> int:
            searchable = f"{record.text} {' '.join(record.tags)}".lower()
            return sum(1 for term in terms if term in searchable)

        ranked = sorted(self.records, key=score, reverse=True)
        return [record for record in ranked[:limit] if score(record) > 0]

