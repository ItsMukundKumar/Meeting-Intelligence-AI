from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import build_vector_store, get_retriever
from core.summarize import get_llm


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def _build_chain_from_retriever(retriever):

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an intelligent meeting assistant.

                You have access to the transcript of a meeting.

                Answer the user's question ONLY using the provided transcript context.

                Context:
                {context}

                Rules:
                - Do NOT use outside knowledge
                - If answer is missing, say:
                - "I couldn't find that information in the meeting transcript."
                - Be concise and direct
                - Use bullet points when needed
                - Quote relevant meeting details when useful
                """,
            ),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def build_rag_chain(transcript: str):

    vector_store = build_vector_store(transcript=transcript)

    retriever = get_retriever(
        vectorstore=vector_store,
        k=4,
    )

    return _build_chain_from_retriever(retriever)


def ask_question(rag_chain, question: str) -> str:

    print(f"Question: {question}")

    answer = rag_chain.invoke(question)

    print(f"Answer: {answer}")

    return answer
