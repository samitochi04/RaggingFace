from transformers import pipeline

# initialize a text generation pipeline once
_llm = None

def load_llm():
    global _llm
    if _llm is None:
        # small local model
        _llm = pipeline('text-generation', model='gpt2', device=-1)
    return _llm


def _truncate_text(text: str, max_chars: int = 1000) -> str:
    if len(text) <= max_chars:
        return text
    # prefer cutting at newline boundary
    cut = text[:max_chars]
    last_nl = cut.rfind("\n")
    if last_nl > 0:
        return cut[:last_nl]
    return cut


def generate_answer(query: str, context: str):
    """Generate an answer using the local LLM conditioned on provided context."""
    llm = load_llm()
    # truncate to avoid exceeding model max length
    safe_context = _truncate_text(context, max_chars=800)
    prompt = (
        "You are an industrial quality assistant. "
        "Use only the provided context to answer the question.\n\n"
        f"Context:\n{safe_context}\n\nQuestion:\n{query}\n\nAnswer:"
    )
    out = llm(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)[0]['generated_text']
    # strip the prompt portion
    if prompt in out:
        return out.split(prompt, 1)[1].strip()
    return out


def generate_report(query: str, context: str):
    """Create an intelligent quality report as a summary/explanation."""
    llm = load_llm()
    safe_context = _truncate_text(context, max_chars=800)
    prompt = (
        "You are an expert industrial engineer. "
        "Based on the context below, write a concise quality report explaining possible root causes and recommendations.\n\n"
        f"Context:\n{safe_context}\n\n{query}\n\nReport:"
    )
    out = llm(prompt, max_new_tokens=500, do_sample=True, temperature=0.7)[0]['generated_text']
    if prompt in out:
        return out.split(prompt, 1)[1].strip()
    return out
