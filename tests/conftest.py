"""A suíte padrão não pode abrir conexões de rede."""

import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")

    def blocked(*args, **kwargs):
        raise AssertionError(
            "Testes offline não podem acessar a rede; injete um transporte simulado."
        )

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
