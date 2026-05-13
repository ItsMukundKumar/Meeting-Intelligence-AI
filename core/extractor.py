import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.summarize import get_llm, split_transcript


def build_chain(system_prompt: str, task: str):

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", f"Meeting transcript:\n\n{{transcript}}\n\n{task}:"),
        ]
    )

    return prompt | llm | StrOutputParser()


def parse_extraction(llm_output: str) -> dict:
    """
    FIX: Old parser only handled items 1-9 (hardcoded startswith checks).
    Now uses regex to handle any number (10, 11, 20+, etc.).
    Also uses more robust section header detection with regex.
    """

    sections = {
        "action_items": [],
        "decisions": [],
        "open_questions": [],
    }

    current_section = None
    current_item: list[str] = []

    # Regex: matches numbered list items like "1.", "12.", "100."
    numbered_item_re = re.compile(r"^\d+\.\s")

    # Regex: matches section headers regardless of surrounding whitespace/dashes
    section_patterns = {
        "action_items": re.compile(r"ACTION_ITEMS\s*:", re.IGNORECASE),
        "decisions": re.compile(r"DECISIONS\s*:", re.IGNORECASE),
        "open_questions": re.compile(r"OPEN_QUESTIONS\s*:", re.IGNORECASE),
    }

    def flush_item():
        if current_item and current_section:
            text = "\n".join(current_item).strip()
            if text and text.lower() != "none":
                sections[current_section].append(text)

    for line in llm_output.splitlines():

        stripped = line.strip()

        if not stripped:
            continue

        # ── Section Detection ─────────────────────────

        matched_section = None
        for section_key, pattern in section_patterns.items():
            if pattern.search(stripped):
                matched_section = section_key
                break

        if matched_section:
            flush_item()
            current_item = []
            current_section = matched_section
            continue

        # ── New Numbered Item ─────────────────────────

        if numbered_item_re.match(stripped):
            flush_item()
            current_item = [stripped]

        else:
            # Continuation line of current item
            current_item.append(stripped)

    # Flush the last item
    flush_item()

    return sections


def extract_items(transcript: str) -> dict:

    extraction_chain = build_chain(
        system_prompt="""
        You are an expert meeting analyst.

        Analyze the meeting transcript and extract structured information.

        ────────────────────────────────────────
        1. ACTION ITEMS
        ────────────────────────────────────────

        Extract every task, follow-up, or commitment.

        Format:

        ACTION_ITEMS:
        1. Task: <task>
        Owner: <owner or Unassigned>
        Deadline: <deadline or Not specified>
        Priority: <High / Medium / Low>

        ────────────────────────────────────────
        2. KEY DECISIONS
        ────────────────────────────────────────

        Extract every conclusion or agreement.

        Format:

        DECISIONS:
        1. <decision>

        ────────────────────────────────────────
        3. OPEN QUESTIONS
        ────────────────────────────────────────

        Extract unresolved discussions or follow-ups.

        Format:

        OPEN_QUESTIONS:
        1. <question>

        ────────────────────────────────────────
        Rules
        ────────────────────────────────────────

        - If nothing exists for a section write: None
        - Do NOT mix sections
        - Be specific
        - Infer priority from urgency
        - Keep exact section headers: ACTION_ITEMS: / DECISIONS: / OPEN_QUESTIONS:
        """,
        task="Extract all structured information from the meeting transcript",
    )

    dedup_chain = build_chain(
        system_prompt="""
        You are an expert meeting analyst.

        You will receive extracted meeting information from multiple transcript chunks.

        Your tasks:
        - Merge all sections
        - Remove duplicates
        - Renumber sequentially
        - Preserve exact formatting

        Required sections (use exactly these headers):

        ACTION_ITEMS:
        1. Task: <task>
        Owner: <owner>
        Deadline: <deadline>
        Priority: <priority>

        DECISIONS:
        1. <decision>

        OPEN_QUESTIONS:
        1. <question>

        If a section is empty write: None
        """,
        task="Merge and deduplicate extracted meeting information",
    )

    # ── Split Transcript ─────────────────────────────

    chunks = split_transcript(
        transcript=transcript,
        chunk_size=10000,
        chunk_overlap=1000,
    )

    # ── Extract Per Chunk ────────────────────────────

    raw_results = []

    for chunk in chunks:
        result = extraction_chain.invoke({"transcript": chunk})
        raw_results.append(result)

    combined = "\n\n".join(raw_results)

    # ── Single Chunk — no dedup needed ───────────────

    if len(chunks) == 1:
        return parse_extraction(combined)

    # ── Multi Chunk — deduplicate ────────────────────

    final_output = dedup_chain.invoke({"transcript": combined})

    return parse_extraction(final_output)