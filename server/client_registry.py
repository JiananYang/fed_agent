from __future__ import annotations

from shared.schemas import ClientProfile


class ClientRegistry:
    """Server-side registry of clients available for task execution."""

    def __init__(self) -> None:
        self._clients: dict[str, ClientProfile] = {}

    def register(self, profile: ClientProfile) -> None:
        self._clients[profile.client_id] = profile

    def get(self, client_id: str) -> ClientProfile:
        return self._clients[client_id]

    def list_clients(self) -> list[ClientProfile]:
        return list(self._clients.values())

    def mark_load(self, client_id: str, load: float) -> None:
        profile = self._clients[client_id]
        profile.load = max(0.0, min(1.0, load))

