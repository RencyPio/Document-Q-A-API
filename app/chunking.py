"""Turn uploaded files into text, and text into overlapping chunks."""
import io
import re

from pypdf import PdfReader


def extract_pages(filename: str, data: bytes) -> list[tuple[int | None, str]]:
    """Return [(page_number, text)] from a PDF, .txt or .md upload. Text files have page None."""
    name = filename.lower()
    if name.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        except Exception as e:  # pypdf raises many error types for broken or locked files
            raise ValueError(f"Couldn't read {filename}. It may be damaged or password-protected.") from e
    if name.endswith((".txt", ".md")):
        return [(None, data.decode("utf-8", errors="replace"))]
    raise ValueError(f"Unsupported file type: {filename}. Please upload a PDF, .txt or .md file.")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into chunks of about `chunk_size` characters with `overlap` characters shared.

    Cuts prefer a paragraph break, then a sentence end, then a space, so chunks
    rarely end mid-word. Whitespace is normalised first.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            for sep in ("\n\n", ". ", " "):
                cut = window.rfind(sep)
                if cut > chunk_size // 2:  # don't make tiny chunks
                    end = start + cut + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        # Start the next chunk at a sentence or paragraph start near the overlap point (looking back
        # up to 2x overlap), so no chunk opens mid-sentence. Fall back to a word boundary.
        lo = max(start + 1, end - 2 * overlap)
        best = max(text.rfind(sep, lo, end - 1) + len(sep) if text.rfind(sep, lo, end - 1) != -1 else -1
                   for sep in ("\n\n", ". ", "? ", "! ", "\n"))
        if best > start and best < end:
            start = best
        else:
            space = text.find(" ", next_start, end)
            start = space + 1 if space != -1 else next_start
    return chunks
