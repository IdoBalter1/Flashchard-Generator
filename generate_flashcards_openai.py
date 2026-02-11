"""
Generate lecture flashcards from transcript text + a supporting PDF using OpenAI.

Architecture (3 stages):
1) PDF digest extraction (text + diagrams/visual cues)
2) Chunk-level candidate flashcard generation from long transcripts
3) Global merge/dedupe/rank to cap the final deck (default: <= 20 cards)
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI


DEFAULT_MODEL = "gpt-4.1"
DEFAULT_MAX_FLASHCARDS = 20

PDF_DIGEST_PROMPT = """
You are an expert study-material extractor.
Your job is to read a lecture PDF and build a compact, high-value digest.

Rules:
- Use BOTH textual and visual content from the PDF (figures/diagrams/charts/tables).
- If a diagram communicates a relationship/process, explain that relationship in words.
- Do not hallucinate unreadable details; omit anything uncertain.
- Prefer conceptual anchors, definitions, mechanisms, and formulas.
- Output STRICT JSON only. No markdown, no commentary.

Required JSON schema:
{
  "title": "string",
  "major_themes": ["string", "..."],
  "diagram_insights": [
    {
      "diagram_or_figure": "string",
      "insight": "string",
      "why_it_matters": "string"
    }
  ],
  "formulae": [
    {
      "name": "string",
      "expression": "string",
      "meaning": "string"
    }
  ],
  "terminology": [
    {
      "term": "string",
      "definition": "string"
    }
  ]
}
""".strip()

CHUNK_FLASHCARD_PROMPT = """
You create exam-quality flashcards from lecture material.

Rules:
- Produce at most {max_candidate_cards} cards for this chunk.
- Focus on high-yield ideas, not trivia.
- Questions must be self-contained and unambiguous.
- Answers should be concise (1-4 sentences) and correct.
- Favor concepts, comparisons, mechanisms, formulas, and diagram-driven understanding.
- Avoid near-duplicates and generic questions.
- If the chunk has little value, return fewer cards (including zero).
- Output STRICT JSON only. No markdown.

Required JSON schema:
{
  "cards": [
    {
      "question": "string",
      "answer": "string",
      "topic": "string",
      "difficulty": "easy|medium|hard",
      "source_basis": "transcript|slides|both"
    }
  ]
}
""".strip()

FINAL_MERGE_PROMPT = """
You are curating a final flashcard deck from noisy candidate cards.

Rules:
- Return at most {max_final_cards} cards total.
- Remove duplicates and near-duplicates.
- Keep broad coverage across the lecture.
- Prefer cards that test understanding, not rote memorization.
- Keep questions precise and answers concise.
- Preserve factual consistency with provided material; if uncertain, drop the card.
- Output STRICT JSON only. No markdown.

Required JSON schema:
{
  "cards": [
    {
      "question": "string",
      "answer": "string",
      "topic": "string",
      "difficulty": "easy|medium|hard",
      "source_basis": "transcript|slides|both"
    }
  ]
}
""".strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def load_env_file(env_path: Path) -> None:
    if not env_path.exists() or not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        # Do not override environment variables already set in shell.
        os.environ.setdefault(key, value)


def quizlet_safe(value: str) -> str:
    cleaned = normalize_whitespace(value)
    cleaned = cleaned.replace("\t", " ")
    return cleaned


def write_quizlet_tsv(cards: List[Dict[str, str]], out_path: Path) -> None:
    lines: List[str] = []
    for card in cards:
        question = quizlet_safe(card.get("question", ""))
        answer = quizlet_safe(card.get("answer", ""))
        if not question or not answer:
            continue
        lines.append(f"{question}\t{answer}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(text_len, start + max_chars)
        if end < text_len:
            # Prefer splitting near sentence boundaries.
            boundary = max(
                text.rfind(". ", start, end),
                text.rfind("? ", start, end),
                text.rfind("! ", start, end),
                text.rfind("; ", start, end),
            )
            if boundary <= start:
                boundary = text.rfind(" ", start, end)
            if boundary <= start:
                boundary = end
            else:
                boundary += 1
        else:
            boundary = end

        chunk = text[start:boundary].strip()
        if chunk:
            chunks.append(chunk)
        if boundary >= text_len:
            break

        new_start = max(0, boundary - overlap_chars)
        if new_start <= start:
            new_start = boundary
        start = new_start

    return chunks


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    pieces: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                text = getattr(content, "text", "")
                if text:
                    pieces.append(text)
    return "\n".join(pieces).strip()


def parse_json_from_response(raw_text: str) -> Dict[str, Any]:
    candidate = raw_text.strip()
    if not candidate:
        raise ValueError("Model returned empty output.")

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fallback: extract the first JSON object.
    match = re.search(r"\{[\s\S]*\}", candidate)
    if not match:
        raise ValueError("Could not find JSON object in model response.")
    return json.loads(match.group(0))


def call_responses_with_retry(client: OpenAI, **kwargs: Any) -> Any:
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            return client.responses.create(**kwargs)
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unexpected retry control flow.")


def sanitize_cards(raw_cards: Any) -> List[Dict[str, str]]:
    if not isinstance(raw_cards, list):
        return []

    cleaned: List[Dict[str, str]] = []
    seen = set()

    for card in raw_cards:
        if not isinstance(card, dict):
            continue

        question = normalize_whitespace(str(card.get("question", "")))
        answer = normalize_whitespace(str(card.get("answer", "")))
        topic = normalize_whitespace(str(card.get("topic", "")))
        difficulty = normalize_whitespace(str(card.get("difficulty", "medium")).lower())
        source_basis = normalize_whitespace(str(card.get("source_basis", "both")).lower())

        if not question or not answer:
            continue
        if difficulty not in {"easy", "medium", "hard"}:
            difficulty = "medium"
        if source_basis not in {"transcript", "slides", "both"}:
            source_basis = "both"

        dedupe_key = (question.lower(), answer.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        cleaned.append(
            {
                "question": question,
                "answer": answer,
                "topic": topic or "General",
                "difficulty": difficulty,
                "source_basis": source_basis,
            }
        )

    return cleaned


def load_transcripts(module_name: str) -> Dict[int, str]:
    module = importlib.import_module(module_name)
    possible_attrs = ("transcripts", "CLEANED_TRANSCRIPTS", "raw_transcripts")

    for attr in possible_attrs:
        maybe = getattr(module, attr, None)
        if isinstance(maybe, dict):
            output: Dict[int, str] = {}
            for key, value in maybe.items():
                if value is None:
                    continue
                text = normalize_whitespace(str(value))
                if not text:
                    continue
                try:
                    lecture_id = int(key)
                except Exception:
                    continue
                output[lecture_id] = text
            if output:
                return output

    raise ValueError(
        f"No transcript dictionary found in module '{module_name}'. "
        "Expected one of: transcripts, CLEANED_TRANSCRIPTS, raw_transcripts."
    )


def upload_pdf(client: OpenAI, pdf_path: Path) -> str:
    with pdf_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="user_data")
    client.files.wait_for_processing(uploaded.id)
    return uploaded.id


def create_pdf_digest(client: OpenAI, model: str, pdf_file_id: str) -> Dict[str, Any]:
    response = call_responses_with_retry(
        client,
        model=model,
        temperature=0.1,
        max_output_tokens=3000,
        instructions=PDF_DIGEST_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Build a study digest from this lecture PDF."},
                    {"type": "input_file", "file_id": pdf_file_id},
                ],
            }
        ],
    )
    return parse_json_from_response(extract_response_text(response))


def create_chunk_candidates(
    client: OpenAI,
    model: str,
    lecture_id: int,
    chunk_index: int,
    total_chunks: int,
    transcript_chunk: str,
    pdf_digest: Dict[str, Any],
    max_candidate_cards: int,
) -> List[Dict[str, str]]:
    user_text = (
        f"Lecture ID: {lecture_id}\n"
        f"Chunk: {chunk_index}/{total_chunks}\n\n"
        "PDF digest (JSON):\n"
        f"{json.dumps(pdf_digest, ensure_ascii=False)}\n\n"
        "Transcript chunk:\n"
        f"{transcript_chunk}"
    )

    response = call_responses_with_retry(
        client,
        model=model,
        temperature=0.2,
        max_output_tokens=2500,
        instructions=CHUNK_FLASHCARD_PROMPT.replace(
            "{max_candidate_cards}", str(max_candidate_cards)
        ),
        input=[{"role": "user", "content": [{"type": "input_text", "text": user_text}]}],
    )
    parsed = parse_json_from_response(extract_response_text(response))
    return sanitize_cards(parsed.get("cards", []))


def merge_candidates(
    client: OpenAI,
    model: str,
    lecture_id: int,
    all_candidates: List[Dict[str, str]],
    max_final_cards: int,
) -> List[Dict[str, str]]:
    response = call_responses_with_retry(
        client,
        model=model,
        temperature=0.1,
        max_output_tokens=3000,
        instructions=FINAL_MERGE_PROMPT.replace("{max_final_cards}", str(max_final_cards)),
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"Lecture ID: {lecture_id}\n\n"
                            "Candidate cards JSON:\n"
                            f"{json.dumps(all_candidates, ensure_ascii=False)}"
                        ),
                    }
                ],
            }
        ],
    )
    parsed = parse_json_from_response(extract_response_text(response))
    cards = sanitize_cards(parsed.get("cards", []))
    return cards[:max_final_cards]


def generate_for_lecture(
    client: OpenAI,
    model: str,
    lecture_id: int,
    transcript: str,
    pdf_digest: Dict[str, Any],
    max_final_cards: int,
    chunk_chars: int,
    overlap_chars: int,
) -> Dict[str, Any]:
    chunks = chunk_text(transcript, max_chars=chunk_chars, overlap_chars=overlap_chars)
    if not chunks:
        return {"lecture_id": lecture_id, "flashcards": [], "num_chunks": 0}

    dynamic_chunk_cap = min(8, max(3, math.ceil(max_final_cards / len(chunks)) + 2))

    all_candidates: List[Dict[str, str]] = []
    for index, chunk in enumerate(chunks, start=1):
        cards = create_chunk_candidates(
            client=client,
            model=model,
            lecture_id=lecture_id,
            chunk_index=index,
            total_chunks=len(chunks),
            transcript_chunk=chunk,
            pdf_digest=pdf_digest,
            max_candidate_cards=dynamic_chunk_cap,
        )
        all_candidates.extend(cards)

    final_cards = merge_candidates(
        client=client,
        model=model,
        lecture_id=lecture_id,
        all_candidates=all_candidates,
        max_final_cards=max_final_cards,
    )
    return {
        "lecture_id": lecture_id,
        "flashcards": final_cards,
        "num_chunks": len(chunks),
        "num_candidates_before_merge": len(all_candidates),
    }


def select_lectures(
    all_transcripts: Dict[int, str],
    lecture_id: int | None,
) -> List[Tuple[int, str]]:
    if lecture_id is not None:
        if lecture_id not in all_transcripts:
            raise KeyError(f"Lecture {lecture_id} not found in transcripts.")
        return [(lecture_id, all_transcripts[lecture_id])]
    return sorted(all_transcripts.items(), key=lambda x: x[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate flashcards from lecture transcripts + a PDF using OpenAI."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the lecture PDF (slides/notes with text + diagrams).",
    )
    parser.add_argument(
        "--module",
        default="generated_transcripts",
        help="Python module that exposes transcripts dictionary (default: generated_transcripts).",
    )
    parser.add_argument(
        "--lecture-id",
        type=int,
        default=None,
        help="Single lecture ID to process. If omitted, processes all lectures in the module.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "OpenAI model name. If omitted, uses OPENAI_MODEL from .env, "
            f"otherwise falls back to {DEFAULT_MODEL}."
        ),
    )
    parser.add_argument(
        "--max-cards",
        type=int,
        default=DEFAULT_MAX_FLASHCARDS,
        help=f"Max final flashcards per lecture (default: {DEFAULT_MAX_FLASHCARDS}).",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=18000,
        help="Transcript chunk size by characters (default: 18000).",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=1200,
        help="Overlap between chunks by characters (default: 1200).",
    )
    parser.add_argument(
        "--output-dir",
        default="flashcards_output",
        help="Directory for generated flashcard JSON files.",
    )
    parser.add_argument(
        "--keep-uploaded-pdf",
        action="store_true",
        help="Do not delete uploaded PDF file from OpenAI storage after generation.",
    )
    parser.add_argument(
        "--no-quizlet-export",
        action="store_true",
        help="Disable writing a Quizlet import TSV file for each lecture.",
    )
    return parser


def main() -> None:
    load_env_file(Path(".env"))
    parser = build_parser()
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your shell env or a local .env file."
        )
    model = args.model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path.name}")

    max_cards = max(1, min(int(args.max_cards), 20))
    chunk_chars = max(2000, int(args.chunk_chars))
    overlap_chars = max(200, min(int(args.overlap_chars), chunk_chars // 2))
    quizlet_export = not args.no_quizlet_export

    transcripts = load_transcripts(args.module)
    lecture_items = select_lectures(transcripts, args.lecture_id)

    client = OpenAI()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Uploading PDF: {pdf_path.name}")
    pdf_file_id = upload_pdf(client, pdf_path)
    print(f"Uploaded PDF file_id: {pdf_file_id}")

    try:
        print("Building PDF digest...")
        pdf_digest = create_pdf_digest(client, model, pdf_file_id)
        (output_dir / "pdf_digest.json").write_text(
            json.dumps(pdf_digest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        for lecture_id, transcript in lecture_items:
            print(f"\nGenerating flashcards for lecture {lecture_id}...")
            result = generate_for_lecture(
                client=client,
                model=model,
                lecture_id=lecture_id,
                transcript=transcript,
                pdf_digest=pdf_digest,
                max_final_cards=max_cards,
                chunk_chars=chunk_chars,
                overlap_chars=overlap_chars,
            )

            out_path = output_dir / f"lecture_{lecture_id}_flashcards.json"
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(
                f"Saved {len(result.get('flashcards', []))} cards "
                f"(chunks={result.get('num_chunks', 0)}) -> {out_path}"
            )
            if quizlet_export:
                quizlet_path = output_dir / f"lecture_{lecture_id}_quizlet_import.tsv"
                write_quizlet_tsv(result.get("flashcards", []), quizlet_path)
                print(f"Saved Quizlet import file -> {quizlet_path}")
    finally:
        if not args.keep_uploaded_pdf:
            try:
                client.files.delete(pdf_file_id)
                print(f"Deleted uploaded PDF from OpenAI storage: {pdf_file_id}")
            except Exception as exc:
                print(f"Warning: could not delete uploaded PDF {pdf_file_id}: {exc}")


if __name__ == "__main__":
    main()
