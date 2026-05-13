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

    sections = {
        "action_items": [],
        "decisions": [],
        "open_questions": [],
    }

    current_section = None
    current_item = []

    lines = llm_output.splitlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # ── Section Detection ─────────────────────

        if "ACTION_ITEMS:" in line:

            if current_item and current_section:
                sections[current_section].append("\n".join(current_item))
                current_item = []

            current_section = "action_items"
            continue

        elif "DECISIONS:" in line:

            if current_item and current_section:
                sections[current_section].append("\n".join(current_item))
                current_item = []

            current_section = "decisions"
            continue

        elif "OPEN_QUESTIONS:" in line:

            if current_item and current_section:
                sections[current_section].append("\n".join(current_item))
                current_item = []

            current_section = "open_questions"
            continue

        # ── New Numbered Item ────────────────────

        if (
            line.startswith("1.")
            or line.startswith("2.")
            or line.startswith("3.")
            or line.startswith("4.")
            or line.startswith("5.")
            or line.startswith("6.")
            or line.startswith("7.")
            or line.startswith("8.")
            or line.startswith("9.")
        ):

            if current_item and current_section:
                sections[current_section].append("\n".join(current_item))

            current_item = [line]

        else:
            current_item.append(line)

    # ── Final Item Save ─────────────────────────

    if current_item and current_section:
        sections[current_section].append("\n".join(current_item))

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
        - Keep exact section headers
        """,
        task="Extract all structured information from the meeting transcript",
    )

    dedup_chain = build_chain(
        system_prompt="""
        You are an expert meeting analyst.

        You will receive extracted meeting information
        from multiple transcript chunks.

        Your tasks:

        - Merge all sections
        - Remove duplicates
        - Renumber sequentially
        - Preserve exact formatting

        Required sections:

        ACTION_ITEMS:
        1. Task: <task>
        Owner: <owner>
        Deadline: <deadline>
        Priority: <priority>

        DECISIONS:
        1. <decision>

        OPEN_QUESTIONS:
        1. <question>

        If empty:
        write None
        """,
        task="Merge and deduplicate extracted meeting information",
    )

    # ── Split Transcript ─────────────────────────

    chunks = split_transcript(
        transcript=transcript,
        chunk_size=10000,
        chunk_overlap=1000,
    )

    # ── Extract Per Chunk ────────────────────────

    raw_results = []

    for chunk in chunks:

        result = extraction_chain.invoke(
            {"transcript": chunk}
        )

        raw_results.append(result)

    combined = "\n\n".join(raw_results)

    # ── Single Chunk ─────────────────────────────

    if len(chunks) == 1:
        return parse_extraction(combined)

    # ── Multi Chunk Deduplication ────────────────

    final_output = dedup_chain.invoke(
        {"transcript": combined}
    )

    return parse_extraction(final_output)