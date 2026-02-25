import os
import PyPDF2


def extract_text_from_file(path: str) -> str:
    """Read text from PDF or plain text file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        return _read_pdf(path)
    elif ext in ['.txt', '.md']:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ''
    else:
        return ''


def _read_pdf(path: str) -> str:
    text = []
    try:
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or '')
    except Exception:
        pass
    return '\n'.join(text)
