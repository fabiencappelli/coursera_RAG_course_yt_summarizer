# Import necessary libraries for the YouTube bot
import gradio as gr
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import IpBlocked, RequestBlocked
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# ----------------------------
# Local model configuration
# ----------------------------
LLM_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"

# ----------------------------
# Global cache
# ----------------------------
processed_transcript = ""
current_video_url = ""
faiss_index_cache = None


def get_video_id(url: str):
    """
    Extract YouTube video ID from multiple common URL formats.
    Supports:
    - https://www.youtube.com/watch?v=VIDEOID
    - https://youtu.be/VIDEOID
    - additional query params after the id
    """
    if not url:
        return None

    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def get_transcript(url: str):
    """
    Fetch transcript, prioritizing manual English transcript over generated one.
    """
    video_id = get_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL or unsupported URL format.")

    ytt_api = YouTubeTranscriptApi()

    try:
        transcripts = ytt_api.list(video_id)
    except (IpBlocked, RequestBlocked) as e:
        raise RuntimeError(
            "YouTube blocked requests from this environment. "
            "Try running locally on your machine."
        ) from e

    transcript = None
    generated_fallback = None

    for t in transcripts:
        if t.language_code == "en":
            if t.is_generated:
                if generated_fallback is None:
                    generated_fallback = t.fetch()
            else:
                transcript = t.fetch()
                break

    transcript = transcript or generated_fallback

    if not transcript:
        raise RuntimeError("No English transcript available for this video.")

    return transcript


def process(transcript):
    """
    Convert transcript entries into a text block with timestamps.
    """
    txt = ""
    for i in transcript:
        try:
            txt += f"Text: {i.text} Start: {i.start}\n"
        except AttributeError:
            # Fallback if entries come as dict-like objects
            try:
                txt += f"Text: {i['text']} Start: {i['start']}\n"
            except (KeyError, TypeError):
                pass
    return txt.strip()


def chunk_transcript(processed_transcript, chunk_size=800, chunk_overlap=120):
    """
    Split transcript into chunks large enough to preserve meaning.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(processed_transcript)


def initialize_local_llm():
    """
    Initialize local Ollama LLM.
    """
    return Ollama(model=LLM_MODEL)


def setup_embedding_model():
    """
    Initialize local Ollama embeddings model.
    """
    return OllamaEmbeddings(model=EMBED_MODEL)


def create_faiss_index(chunks, embedding_model):
    """
    Create a FAISS index from text chunks using local embeddings.
    """
    return FAISS.from_texts(chunks, embedding_model)


def format_retrieved_context(docs):
    """
    Turn retrieved LangChain Document objects into a plain text context block.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def create_summary_prompt():
    """
    Prompt template for transcript summarization.
    """
    template = """
You are an AI assistant tasked with summarizing YouTube video transcripts.

Instructions:
1. Summarize the transcript in a single concise paragraph.
2. Ignore timestamps in the final summary.
3. Focus on the actual spoken content.
4. If the transcript is noisy or repetitive, clean it up mentally before summarizing.

Transcript:
{transcript}

Summary:
"""
    return PromptTemplate(input_variables=["transcript"], template=template)


def create_summary_chain(llm, prompt, verbose=False):
    return LLMChain(llm=llm, prompt=prompt, verbose=verbose)


def retrieve(query, faiss_index, k=5):
    """
    Retrieve relevant context from FAISS.
    """
    return faiss_index.similarity_search(query, k=k)


def create_qa_prompt_template():
    """
    Prompt template for question answering using retrieved transcript chunks.
    """
    qa_template = """
You are an expert assistant answering questions about a YouTube video.

Rules:
1. Answer only from the provided context.
2. Be precise, clear, and non-repetitive.
3. If the answer is not in the context, say you do not have enough information from the transcript.
4. When relevant, mention approximate timestamps present in the context.

Relevant Video Context:
{context}

Question:
{question}

Answer:
"""
    return PromptTemplate(input_variables=["context", "question"], template=qa_template)


def create_qa_chain(llm, prompt_template, verbose=False):
    return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)


def generate_answer(question, faiss_index, qa_chain, k=5):
    """
    Retrieve context and generate answer.
    """
    relevant_docs = retrieve(question, faiss_index, k=k)
    context_text = format_retrieved_context(relevant_docs)
    answer = qa_chain.predict(context=context_text, question=question)
    return answer


def ensure_transcript_and_index(video_url):
    """
    Build or reuse processed transcript and FAISS index if the URL has not changed.
    """
    global processed_transcript, current_video_url, faiss_index_cache

    if not video_url:
        raise ValueError("Please provide a valid YouTube URL.")

    # Rebuild only if video changed or cache is empty
    if (
        video_url != current_video_url
        or not processed_transcript
        or faiss_index_cache is None
    ):
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)

        if not processed_transcript:
            raise RuntimeError("Transcript could not be processed.")

        chunks = chunk_transcript(processed_transcript)
        embedding_model = setup_embedding_model()
        faiss_index_cache = create_faiss_index(chunks, embedding_model)
        current_video_url = video_url

    return processed_transcript, faiss_index_cache


def summarize_video(video_url):
    """
    Generate a summary of the video transcript.
    """
    try:
        transcript_text, _ = ensure_transcript_and_index(video_url)
        llm = initialize_local_llm()
        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)
        summary = summary_chain.run({"transcript": transcript_text})
        return summary
    except Exception as e:
        return f"Error: {e}"


def answer_question(video_url, user_question):
    """
    Answer a user's question about the video.
    """
    try:
        if not user_question or not user_question.strip():
            return "Please provide a valid question."

        _, faiss_index = ensure_transcript_and_index(video_url)

        llm = initialize_local_llm()
        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)

        answer = generate_answer(user_question, faiss_index, qa_chain)
        return answer
    except Exception as e:
        return f"Error: {e}"


def get_transcript_status(video_url):
    """
    Simple status check / prefetch helper.
    """
    try:
        ensure_transcript_and_index(video_url)
        return "Transcript fetched and indexed successfully."
    except Exception as e:
        return f"Transcript error: {e}"


with gr.Blocks() as interface:
    gr.Markdown(
        "<h2 style='text-align: center;'>YouTube Video Summarizer and Q&A (Local Qwen)</h2>"
    )

    video_url = gr.Textbox(
        label="YouTube Video URL", placeholder="Enter the YouTube Video URL"
    )

    transcript_status = gr.Textbox(label="Transcript Status", interactive=False)
    summary_output = gr.Textbox(label="Video Summary", lines=6)
    question_input = gr.Textbox(
        label="Ask a Question About the Video", placeholder="Ask your question"
    )
    answer_output = gr.Textbox(label="Answer to Your Question", lines=8)

    fetch_btn = gr.Button("Fetch Transcript")
    summarize_btn = gr.Button("Summarize Video")
    question_btn = gr.Button("Ask a Question")

    fetch_btn.click(get_transcript_status, inputs=video_url, outputs=transcript_status)
    summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
    question_btn.click(
        answer_question, inputs=[video_url, question_input], outputs=answer_output
    )

interface.launch(server_name="0.0.0.0", server_port=7860)
