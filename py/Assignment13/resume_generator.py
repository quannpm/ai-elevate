#!/usr/bin/env python3
import os, sys, json, time
from typing import Dict, Any, List

# Allow tuning via environment variables
MODEL_PATH = os.getenv("MODEL_PATH")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
OUTPUT_LOG = os.path.join(OUTPUT_DIR, "console_output.txt")  # JSON summary (kept name for compatibility)

N_CTX = int(os.getenv("N_CTX", "2048"))
N_THREADS = int(os.getenv("N_THREADS", str(os.cpu_count() or 4)))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

SAMPLE_PROFILES = [
    {"name": "Nguyen Quan", "title": "Backend Software Engineer", "summary": "Backend engineer with 5+ years...", "skills": ["Python", "Django"]},
    {"name": "Tran Thi B", "title": "Data Scientist", "summary": "Data scientist experienced with ML...", "skills": ["Python", "PyTorch"]},
    {"name": "Alex Smith", "title": "Product Manager", "summary": "Product manager with 7 years...", "skills": ["Product Strategy"]},
]

if not MODEL_PATH:
    sys.exit("Please set MODEL_PATH to your local .gguf LLaMA model file (e.g., D:\\models\\llama-3.1-8b-instruct-q4_K_M.gguf)")

try:
    from llama_cpp import Llama
except Exception as e:
    raise RuntimeError("llama-cpp-python not installed. Install with: pip install llama-cpp-python") from e

if not os.path.exists(MODEL_PATH):
    sys.exit(f"Model file not found at {MODEL_PATH}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model (CPU example)
# TIP: set N_THREADS to available cores; N_CTX controls max context length.
llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX, n_threads=N_THREADS)

def _safe_name(name: str) -> str:
    """Make a filesystem-friendly filename from a display name."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())

def generate_resume(profile: Dict[str, Any]) -> str:
    """Generate a resume from a profile using llama-cpp completion."""
    prompt = (
        "You are a resume writer. Given the user profile below, produce a professional, concise, and well-structured resume in plain text.\n\n"
        f"Profile:\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
        "Resume:\n"
    )
    try:
        # llama-cpp Llama is callable; returns dict with choices[0].text
        out = llm(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        if isinstance(out, dict) and out.get("choices"):
            choice = out["choices"][0]
            return choice.get("text") or choice.get("message", {}).get("content", "")
        return str(out)
    except Exception as e:
        raise RuntimeError(f"Generation failed: {e}") from e

def main():
    results = []
    for i, profile in enumerate(SAMPLE_PROFILES, start=1):
        print("Generating resume for:", profile["name"])
        res_text = generate_resume(profile)
        fname = os.path.join(OUTPUT_DIR, f"resume_{i}_{_safe_name(profile['name'])}.txt")
        with open(fname, "w", encoding="utf-8") as fh:
            fh.write(res_text)
        results.append({"name": profile["name"], "file": fname})
        time.sleep(0.2)  # small pacing to keep console readable
    with open(OUTPUT_LOG, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(results, indent=2, ensure_ascii=False))
    print("Generated resumes in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
