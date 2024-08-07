import pytest
from fastapi.testclient import TestClient
from server import app, get_corpus_service
from repositories.corpus.memory_corpus_repository import MemoryCorpusRepository
from repositories.corpus_entry.memory_corpus_entry_repository import (
    MemoryCorpusEntryRepository,
)
from services.corpus_management_service import CorpusManagementService


@pytest.fixture
def corpus_service():
    corpus_repo = MemoryCorpusRepository()
    corpus_entry_repo = MemoryCorpusEntryRepository()

    service = CorpusManagementService(corpus_repo, corpus_entry_repo)

    # Clear existing data
    for corpus in service.list_corpora():
        service.corpus_repo.delete(corpus.id)

    yield service

    # Cleanup after test
    for corpus in service.list_corpora():
        service.corpus_repo.delete(corpus.id)


@pytest.fixture
def test_client(corpus_service):
    app.dependency_overrides[get_corpus_service] = lambda: corpus_service
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_create_corpus(test_client, corpus_service):
    response = test_client.post(
        "/corpus", json={"name": "Test Corpus", "description": "Test Description"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Corpus"
    assert data["description"] == "Test Description"
    assert "id" in data
    assert "created_at" in data
    assert "updated_at" in data


def test_get_corpora(test_client, corpus_service):
    # Create some test corpora
    corpus_service.create_corpus(name="Corpus 1", description="Description 1")
    corpus_service.create_corpus(name="Corpus 2", description="Description 2")

    response = test_client.get("/corpus")

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2
    assert data["total"] == 2
    assert data["skip"] == 0
    assert data["limit"] == 100


def test_add_corpus_entry(test_client, corpus_service):
    # Create a test corpus
    corpus = corpus_service.create_corpus(
        name="Test Corpus", description="Test Description"
    )

    response = test_client.post(
        f"/corpus/{corpus.id}/entry",
        json={"content": "Test Content", "entry_type": "knowledge"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "Test Content"
    assert data["entry_type"] == "knowledge"
    assert data["corpus_id"] == corpus.id


def test_get_corpus_entries(test_client, corpus_service):
    # Create a test corpus and add some entries
    corpus = corpus_service.create_corpus(
        name="Test Corpus", description="Test Description"
    )
    corpus_service.add_entry_to_corpus(corpus.id, "Entry 1", "knowledge")
    corpus_service.add_entry_to_corpus(corpus.id, "Entry 2", "chat")

    response = test_client.get(f"/corpus/{corpus.id}/entries")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["content"] in ["Entry 1", "Entry 2"]
    assert data[1]["content"] in ["Entry 1", "Entry 2"]
