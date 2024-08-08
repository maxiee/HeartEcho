import pytest
from datetime import datetime
from domain.corpus import CorpusEntry


def test_corpus_entry_creation_knowledge():
    entry = CorpusEntry(
        id="1",
        corpus="test_corpus",
        content="Test knowledge content",
        entry_type="knowledge",
        created_at=datetime.now(),
    )
    assert entry.id == "1"
    assert entry.corpus == "test_corpus"
    assert entry.content == "Test knowledge content"
    assert entry.entry_type == "knowledge"
    assert entry.sha256 is not None


def test_corpus_entry_creation_chat():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    entry = CorpusEntry(
        id="2",
        corpus="test_corpus",
        messages=messages,
        entry_type="chat",
        created_at=datetime.now(),
    )
    assert entry.id == "2"
    assert entry.corpus == "test_corpus"
    assert entry.messages == messages
    assert entry.entry_type == "chat"
    assert entry.sha256 is not None


def test_corpus_entry_invalid_type():
    with pytest.raises(ValueError):
        CorpusEntry(
            id="3",
            corpus="test_corpus",
            content="Invalid type",
            entry_type="invalid",
            created_at=datetime.now(),
        )


def test_corpus_entry_missing_content():
    with pytest.raises(ValueError):
        CorpusEntry(
            id="4",
            corpus="test_corpus",
            entry_type="knowledge",
            created_at=datetime.now(),
        )


def test_corpus_entry_missing_messages():
    with pytest.raises(ValueError):
        CorpusEntry(
            id="5", corpus="test_corpus", entry_type="chat", created_at=datetime.now()
        )


def test_corpus_entry_sha256_consistency():
    entry1 = CorpusEntry(
        id="6",
        corpus="test_corpus",
        content="Test content",
        entry_type="knowledge",
        created_at=datetime.now(),
    )
    entry2 = CorpusEntry(
        id="7",
        corpus="test_corpus",
        content="Test content",
        entry_type="knowledge",
        created_at=datetime.now(),
    )
    assert entry1.sha256 == entry2.sha256
