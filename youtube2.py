from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import re
import os
import pickle


class YouTubeProcessor:
    """
    Processes YouTube transcripts into a FAISS vector store.
    """

    def __init__(self, video_id: str):
        self.video_id = video_id
        self.transcript = self._get_youtube_transcript()
        self.chunked_text = self._split_transcript()
        self.vector_store = self._create_vector_store()

    from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
    )

    def _get_youtube_transcript(self) -> str:
        try:
            ytt_api = YouTubeTranscriptApi()
            transcripts = ytt_api.fetch(self.video_id)
            return " ".join(t.text for t in transcripts)

        except (TranscriptsDisabled, NoTranscriptFound):
            raise RuntimeError(
                "This video does not have captions enabled."
            )

        except VideoUnavailable:
            raise RuntimeError(
                "The video is unavailable or restricted."
            )



    def _split_transcript(self) -> list:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return splitter.split_text(self.transcript)


    def _create_vector_store(self):
        cache_dir = f".cache/{self.video_id}"
        os.makedirs(".cache", exist_ok=True)

        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        if os.path.exists(cache_dir):
            return FAISS.load_local(
                cache_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )

        store = FAISS.from_texts(self.chunked_text, embeddings)
        store.save_local(cache_dir)

        return store




class YouTubeChatbot:
    """
    Chatbot that answers questions using retrieved transcript chunks.
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOllama(model="llama3", temperature=0, streaming=True)

    def answer_stream(self, question: str, k: int = 3):
        docs = self.vector_store.similarity_search(question, k=k)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = PROMPT.format(context=context, question=question)

        for chunk in self.llm.stream(prompt):
            if chunk.content:
                print(chunk.content, end="", flush=True)


def extract_video_id(url: str) -> str:
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError("Invalid YouTube URL")


PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a concise assistant.
Answer ONLY using the context below.
If the answer is not present, say:
"The provided video transcript does not mention this."

Context:
{context}

Question:
{question}
"""
)



def build_chatbot(video_url: str) -> YouTubeChatbot:
    video_id = extract_video_id(video_url)
    processor = YouTubeProcessor(video_id)
    return YouTubeChatbot(processor.vector_store)


def main():
    video_url = input("Enter YouTube Video URL: ")
    question = input("Enter your question: ")

    bot = build_chatbot(video_url)

    print("\nAnswer:\n")
    bot.answer_stream(question)


if __name__ == "__main__":
    main()
