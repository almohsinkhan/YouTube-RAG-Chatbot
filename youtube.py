from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoTokenizer
from langchain_core.prompts import PromptTemplate


class YouTubeProcessor:
    """
    A class to process YouTube video transcripts for LLM applications.
    
    Key functionalities:
    1. fetch transcripts using youtube-transcript-api.
    2. split transcripts into manageable chunks using langchain's RecursiveCharacterTextSplitter.
    3. generate embeddings using langchain's HuggingFaceEmbeddings for indexing.
    4. store in a vector store using FAISS for efficient retrieval.
    """

    def __init__(self, video_id: str):
        self.video_id = video_id

        # Fetch and store the transcript
        self.transcript = self._get_youtube_transcript()

        # Split the transcript into chunks
        self.chunked_text = self._split_transcript()

        # Create the vector store
        self.vector_store = self._create_vector_store()
        

    def _get_youtube_transcript(self) -> str:
        """Fetches the transcript for the YouTube video ID and returns it as a single string."""
        ytt_api = YouTubeTranscriptApi()
        transcripts = ytt_api.fetch(self.video_id)
        full_transcript = " ".join([t.text for t in transcripts])
        return full_transcript
    
    def _split_transcript(self) -> list:
        """Splits the transcript into chunks for further processing."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        return splitter.split_text(self.transcript)
    
    def _create_vector_store(self):
        """Creates a FAISS vector store from the chunked text."""
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vector_store = FAISS.from_texts(self.chunked_text, embedding=embeddings)
        return vector_store
    

class YouTubeChatbot:
    """
    A class to create a chatbot that can answer questions based on YouTube video transcripts.
    
    Key functionalities:
    1. Create an LLM model using HuggingFace.
    2. Retrieve relevant chunks from the vector store based on user queries.
    3. Generate responses using the LLM model based on the retrieved context.
    """

    def __init__(self, vector_store, llm_model=None):
        self.vector_store = vector_store
        self.llm_model = llm_model if llm_model else self._create_llm_model()


    def _create_llm_model(self):
        """Function to create and return the LLM model."""
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        # 1. Explicitly load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # 2. Setup the pipeline
        llm = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            pipeline_kwargs={
                "temperature": 0.7,
                "max_new_tokens": 256, # Added to prevent the response from cutting off
            },
            model_kwargs={
                "torch_dtype": "auto",
                "low_cpu_mem_usage": True,
            }
        )

        # 3. Initialize ChatHuggingFace with the tokenizer
        # This ensures the Chat Template for TinyLlama is applied correctly
        model = ChatHuggingFace(llm=llm, tokenizer=tokenizer)
        return model
    

    


prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""<|system|>
You are a concise assistant. Use ONLY the provided context to answer. 
If the answer is not in the context, say "The provided video transcript does not mention this."
Do not provide any explanation, headers, or extra text. Give only the direct answer.</s>
<|user|>
Context: {context}

Question: {question}</s>
<|assistant|>
"""
)

def build_chatbot(video_id: str) -> YouTubeChatbot:
    """Create a YouTubeChatbot instance for a given YouTube video ID."""
    processor = YouTubeProcessor(video_id)
    return YouTubeChatbot(processor.vector_store)


def answer_question(bot: YouTubeChatbot, question: str, k: int = 3) -> str:
    retrieval = bot.vector_store.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in retrieval])
    
    raw_response = bot.llm_model.invoke([
        ("system", "You are a helpful assistant."),
        ("human", prompt_template.format(context=context, question=question))
    ])
    
    # TinyLlama cleanup: sometimes it repeats the prompt. 
    # This logic helps extract just the new text.
    content = raw_response.content
    return content

def main():
    video_id = input("Enter YouTube Video ID: ")
    question = input("Enter your question: ")
    bot = build_chatbot(video_id)
    answer = answer_question(bot, question)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
