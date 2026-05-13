from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe
from core.summarize import generate_title, summarize
from core.extractor import extract_items
from core.rag_engine import build_rag_chain

load_dotenv()


def run_pipeline(source: str, language: str = "en") -> dict:
    print("Starting AI Video Assistant")

    chunks = process_input(source=source)
    transcript = transcribe(chunks=chunks, language=language)
    print(f"Raw transcription (first 600 character) :\n {transcript[:600]}")

    summary = summarize(transcript=transcript)
    title = generate_title(summary=summary)
    items = extract_items(transcript=transcript)
    action_items = items["action_items"]
    decision = items["decisions"]
    open_questions = items["open_questions"]

    rag_chain = build_rag_chain(transcript=transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "decision": decision,
        "open_questions": open_questions,
        "rag_chain": rag_chain,
    }


def print_section(title: str, content) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    if isinstance(content, list):
        if not content:
            print("  None identified.")
        else:
            for item in content:
                print(f"  {item}")
    else:
        print(f"  {content}")


def start_chat(rag_chain) -> None:
    print("\n" + "=" * 60)
    print("  💬 CHAT WITH YOUR MEETING  (type 'exit' to quit)")
    print("=" * 60)

    while True:
        question = input("\n  You: ").strip()
        if not question:
            continue
        if question.lower() in ["exit", "quit", "q"]:
            print("\n  Exiting chat. Goodbye!")
            break

        print("\n  Assistant: ", end="", flush=True)
        answer = rag_chain.invoke(question)
        print(answer)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   🎙️  AI MEETING ASSISTANT")
    print("=" * 60)

    source = input("\n  Enter YouTube URL or local file path: ").strip()
    language = (
        input("  Language — 'en' for English / 'hi' for Hindi [default: en]: ").strip()
        or "en"
    )

    print("\n  Processing... please wait ⏳")

    result = run_pipeline(source=source, language=language)

    print_section("📌 TITLE", result["title"])
    print_section("📋 SUMMARY", result["summary"])
    print_section("✅ ACTION ITEMS", result["action_items"])
    print_section("🔑 DECISIONS", result["decision"])
    print_section("❓ OPEN QUESTIONS", result["open_questions"])

    print("\n  Transcript preview:")
    print(f"  {result['transcript'][:300]}...")

    start_chat(result["rag_chain"])
