import pytest

from app.chunking import chunk_text, extract_pages


def test_empty_or_whitespace_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n\t ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.", chunk_size=100, overlap=10) == ["Hello world."]


def test_no_chunk_exceeds_chunk_size():
    text = "The quick brown fox jumps over the lazy dog. " * 200
    chunks = chunk_text(text, chunk_size=150, overlap=30)
    assert len(chunks) > 1
    assert all(len(c) <= 150 for c in chunks)


def test_every_word_survives_chunking():
    words = [f"w{i}" for i in range(400)]
    chunks = chunk_text(" ".join(words), chunk_size=120, overlap=20)
    seen = {w for c in chunks for w in c.split()}
    assert seen == set(words)


def test_neighbouring_chunks_overlap():
    text = " ".join(f"token{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=120, overlap=40)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert set(prev.split()) & set(nxt.split()), "expected shared words between neighbours"


def test_zero_overlap_has_no_repeats():
    text = " ".join(f"t{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=100, overlap=0)
    words = [w for c in chunks for w in c.split()]
    assert len(words) == len(set(words)) == 300


def test_prefers_to_cut_at_sentence_end():
    text = ("This is sentence number one with padding words. " * 3) + "Final words here without end"
    chunks = chunk_text(text, chunk_size=120, overlap=0)
    assert chunks[0].endswith(".")


def test_chunks_do_not_end_mid_word():
    text = " ".join(["supercalifragilistic"] * 60)
    for c in chunk_text(text, chunk_size=100, overlap=20):
        assert all(w == "supercalifragilistic" for w in c.split())


def test_whitespace_is_normalised():
    chunks = chunk_text("a  \t b\n\n\n\n\nc", chunk_size=100, overlap=0)
    assert chunks == ["a b\n\nc"]


@pytest.mark.parametrize("size,overlap", [(0, 0), (-5, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_sizes_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=size, overlap=overlap)


def test_extract_pages_text_file():
    assert extract_pages("notes.TXT", "héllo".encode()) == [(None, "héllo")]


def test_extract_pages_markdown_file():
    assert extract_pages("readme.md", b"# Title") == [(None, "# Title")]


def test_extract_pages_rejects_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_pages("slides.pptx", b"...")
