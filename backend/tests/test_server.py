import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock
from server import app, get_corpus_service
from domain.corpus import Corpus, CorpusEntry


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def mock_corpus_service():
    return Mock()


def test_create_corpus(test_client, mock_corpus_service):
    mock_corpus = Corpus(
        id="1",
        name="Test Corpus",
        description="Test Description",
        created_at=None,
        updated_at=None,
    )
    mock_corpus_service.create_corpus.return_value = mock_corpus

    app.dependency_overrides[get_corpus_service] = lambda: mock_corpus_service

    response = test_client.post(
        "/corpus", json={"name": "Test Corpus", "description": "Test Description"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": "1",
        "name": "Test Corpus",
        "description": "Test Description",
        "created_at": None,
        "updated_at": None,
    }
    mock_corpus_service.create_corpus.assert_called_once_with(
        name="Test Corpus", description="Test Description"
    )


def test_add_corpus_entry(test_client, mock_corpus_service):
    mock_entry = CorpusEntry(
        id="1",
        corpus_id="1",
        content="Test Content",
        entry_type="knowledge",
        created_at=None,
        metadata={},
    )
    mock_corpus_service.add_entry_to_corpus.return_value = mock_entry

    app.dependency_overrides[get_corpus_service] = lambda: mock_corpus_service

    response = test_client.post(
        "/corpus/1/entry", json={"content": "Test Content", "entry_type": "knowledge"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": "1",
        "corpus_id": "1",
        "content": "Test Content",
        "entry_type": "knowledge",
        "created_at": None,
        "metadata": {},
    }
    mock_corpus_service.add_entry_to_corpus.assert_called_once_with(
        corpus_id="1", content="Test Content", entry_type="knowledge"
    )


# 为其他路由添加类似的测试


def test_get_corpora(test_client, mock_corpus_service):
    mock_corpora = [
        Corpus(
            id="1",
            name="Corpus 1",
            description="Description 1",
            created_at=None,
            updated_at=None,
        ),
        Corpus(
            id="2",
            name="Corpus 2",
            description="Description 2",
            created_at=None,
            updated_at=None,
        ),
    ]
    mock_corpus_service.list_corpora.return_value = mock_corpora

    app.dependency_overrides[get_corpus_service] = lambda: mock_corpus_service

    response = test_client.get("/corpus")

    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0]["name"] == "Corpus 1"
    assert response.json()[1]["name"] == "Corpus 2"
    mock_corpus_service.list_corpora.assert_called_once_with(skip=0, limit=100)


# 清理依赖覆盖
@pytest.fixture(autouse=True)
def cleanup():
    yield
    app.dependency_overrides.clear()
