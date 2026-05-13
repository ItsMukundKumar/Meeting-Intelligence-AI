from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def get_llm():

    api_key = st.secrets["API"]["MISTRAL_API_KEY"]

    return ChatMistralAI(
        model_name="mistral-medium-latest",
        api_key=api_key,
        temperature=0.3,
    )  # type: ignore


def split_transcript(
    transcript: str, chunk_size: int = 3000, chunk_overlap: int = 400
) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    return splitter.split_text(text=transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert meeting analyst. Your job is to extract the most important information from a portion of a meeting transcript.

                From this transcript chunk, extract:
                - **Key points discussed** (main topics, ideas, or updates shared)
                - **Decisions made** (anything agreed upon or concluded)
                - **Action items** (tasks assigned, with owner and deadline if mentioned)
                - **Important numbers or dates** (metrics, deadlines, figures mentioned)

                Rules:
                - Be concise but don't lose important details
                - Preserve names of people and their roles if mentioned
                - Keep technical terms as-is, don't simplify them
                - If nothing important is in this chunk, just say "No key information in this segment"
                """,
            ),
            ("human", "Transcript chunk:\n\n{text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript=transcript)

    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]

    combined = "\n\n".join(chunk_summaries)

    combine_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert meeting analyst. You will receive summaries of different portions of a meeting transcript.
                Combine them into one final, well-structured meeting summary with these sections:

                ## 📋 Overview
                A 2-3 sentence summary of what the meeting was about.

                ## 🔑 Key Discussion Points
                Bullet points of the main topics discussed.

                ## ✅ Decisions Made
                Bullet points of conclusions or agreements reached.

                ## 📌 Action Items
                Each action item with owner and deadline in this format:
                - [ ] Task description — **Owner** — Due: deadline (or "Not specified")

                ## ❓ Open Questions / Follow-ups
                Anything left unresolved or requiring further discussion.

                Rules:
                - Remove duplicate information across summaries
                - Keep it professional and concise
                - If a section has no content, write "None identified"
                """,
            ),
            ("human", "Meeting summaries to combine:\n{text}"),
        ]
    )

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combine_prompt
        | llm
        | StrOutputParser()
    )

    return combined_chain.invoke(combined)


def generate_title(summary: str) -> str:
    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert meeting analyst. Generate a concise and descriptive title for the meeting based on the transcript summary.

            Rules:
            - Maximum 8 words
            - Be specific, not generic (avoid titles like "Team Meeting" or "Discussion")
            - Capture the main topic or outcome of the meeting
            - Professional tone
            - No punctuation at the end

            Examples of good titles:
            - Q3 Marketing Strategy Review and Budget Approval
            - Product Roadmap Planning for Mobile App Launch
            - Customer Churn Analysis and Retention Strategy Discussion
            """,
            ),
            ("human", "Meeting summary:\n\n{summary}\n\nGenerate a title:"),
        ]
    )

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"summary": x})
        | title_prompt
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke({"summary": summary})
