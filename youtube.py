from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
    )
import os
import re
import math


class YouTubeProcessor:
    """
    Processes YouTube transcripts into a FAISS vector store.
    """

    def __init__(self, video_id: str):
        self.video_id = video_id
        self.transcript = self._get_youtube_transcript()
        self.chunked_text = self._split_transcript()
        self.vector_store = self._create_vector_store(self.chunked_text, "main")
        self.summary = self._summarize_transcript()
        self.summary_vector_store = self._create_vector_store(
            [self.summary], "summary"
        )
  

    def _get_youtube_transcript(self) -> str:
        try:
            ytt_api = YouTubeTranscriptApi()
            transcripts = ytt_api.fetch(self.video_id)
            return " ".join(t.text for t in transcripts)

        except (TranscriptsDisabled, NoTranscriptFound):
            raise RuntimeError(
                f"This video does not have captions enabled: {e}"
            )

        except VideoUnavailable:
            raise RuntimeError(
                "The video is unavailable or restricted."
            )

    def _summarize_transcript(self) -> str:
        llm = ChatOllama(model="llama3", temperature=0)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=300
        )
        chunks = splitter.split_text(self.transcript)

        # MAP
        summaries = []
        for chunk in chunks:
            prompt = f"""
            Summarize the following part of a video transcript concisely.

            Text:
            {chunk}
            """
            response = llm.invoke(prompt)
            summaries.append(response.content)

        # REDUCE
        combined = "\n\n".join(summaries)
        
        final_prompt = f"""
        You are given summaries of parts of a video.
        Combine them into ONE clear, structured summary.

        Summaries:
        {combined}
        """
        final_response = llm.invoke(final_prompt)
        return final_response.content



    def _split_transcript(self) -> list:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return splitter.split_text(self.transcript)


    def _create_vector_store(self, texts, sub_dir="main"):
        cache_dir = f".cache/{self.video_id}/{sub_dir}"
        os.makedirs(cache_dir, exist_ok=True)

        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        if os.path.exists(os.path.join(cache_dir, "index.faiss")):
            return FAISS.load_local(
                cache_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )

        store = FAISS.from_texts(texts, embeddings)
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


def retrieval_confidence(vector_store, question: str, k: int = 3) -> float:
    results = vector_store.similarity_search_with_score(question, k=k)
    if not results:
        return 100.0  # High distance, poor match
    
    # Simple average of L2 distancesa
    # Lower usage means closer match.
    total_score = sum(score for _, score in results)
    return total_score / len(results)


def build_chatbot(video_url: str, question: str) -> YouTubeChatbot:
    video_id = extract_video_id(video_url)
    processor = YouTubeProcessor(video_id)
    
    # Check if the question is specific enough to answer from chunks
    # or if we should fall back to the summary.
    # L2 Distance: Lower is better.
    # Threshold 0.5 is heuristic; adjust based on embeddings model.
    score = retrieval_confidence(processor.vector_store, question)
    
    print(f"Retrieval Score (Distance): {score:.4f}")

    if score < 0.8:
        # Good match found in chunks -> Use detailed chunks
        # This roughly corresponds to "Global" search in specific content? 
        # Or "Local" in the sense of finding specific locations.
        # Naming convention: Detailed/Specific vs Summary/General.
        print("Using detailed transcript (Specific Query)")
        return YouTubeChatbot(processor.vector_store)
    else:
        # Poor match in chunks -> Use summary
        print("Using summary (General/Global Query)")
        return YouTubeChatbot(processor.summary_vector_store)




def main():
    video_url = input("Enter YouTube Video URL: ")
    question = input("Enter your question: ")

    bot = build_chatbot(video_url, question)

    print("\nAnswer:\n")
    bot.answer_stream(question)


if __name__ == "__main__":
    main()
