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
    Fetch transcript, prioritizing manual French transcript over generated one.
    """
    video_id = get_video_id(url)
    if not video_id:
        raise ValueError("URL YouTube invalide ou format non pris en charge.")

    ytt_api = YouTubeTranscriptApi()

    try:
        transcripts = ytt_api.list(video_id)
    except (IpBlocked, RequestBlocked) as e:
        raise RuntimeError(
            "YouTube bloque les requêtes depuis cet environnement. "
            "Essaie d'exécuter le script en local sur ta machine."
        ) from e

    transcript = None
    generated_fallback = None

    for t in transcripts:
        if t.language_code == "fr":
            if t.is_generated:
                if generated_fallback is None:
                    generated_fallback = t.fetch()
            else:
                transcript = t.fetch()
                break

    transcript = transcript or generated_fallback

    if not transcript:
        raise RuntimeError(
            "Aucune transcription française n'est disponible pour cette vidéo."
        )

    return transcript


def process(transcript):
    """
    Convert transcript entries into a text block with timestamps.
    """
    txt = ""
    for i in transcript:
        try:
            txt += f"Texte : {i.text} Début : {i.start}\n"
        except AttributeError:
            # Fallback if entries come as dict-like objects
            try:
                txt += f"Texte : {i['text']} Début : {i['start']}\n"
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
Tu es un assistant IA chargé de résumer la transcription d'une vidéo YouTube en français.

Instructions :
1. Résume la transcription en un seul paragraphe concis.
2. Ignore les horodatages dans le résumé final.
3. Concentre-toi sur le contenu réellement prononcé.
4. Si la transcription est bruitée ou répétitive, nettoie-la mentalement avant de résumer.
5. Réponds uniquement en français.

Transcription :
{transcript}

Résumé :
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
Tu es un assistant expert qui répond à des questions sur une vidéo YouTube.

Règles :
1. Réponds uniquement à partir du contexte fourni.
2. Sois précis, clair et non répétitif.
3. Si la réponse ne se trouve pas dans le contexte, dis que tu n'as pas assez d'informations dans la transcription.
4. Si c'est pertinent, mentionne les horodatages approximatifs présents dans le contexte.
5. Réponds uniquement en français.

Contexte vidéo pertinent :
{context}

Question :
{question}

Réponse :
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
        raise ValueError("Merci de fournir une URL YouTube valide.")

    # Rebuild only if video changed or cache is empty
    if (
        video_url != current_video_url
        or not processed_transcript
        or faiss_index_cache is None
    ):
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)

        if not processed_transcript:
            raise RuntimeError("La transcription n'a pas pu être traitée.")

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
        return f"Erreur : {e}"


def answer_question(video_url, user_question):
    """
    Answer a user's question about the video.
    """
    try:
        if not user_question or not user_question.strip():
            return "Merci de poser une question valide."

        _, faiss_index = ensure_transcript_and_index(video_url)

        llm = initialize_local_llm()
        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)

        answer = generate_answer(user_question, faiss_index, qa_chain)
        return answer
    except Exception as e:
        return f"Erreur : {e}"


def get_transcript_status(video_url):
    """
    Simple status check / prefetch helper.
    """
    try:
        ensure_transcript_and_index(video_url)
        return "Transcription récupérée et indexée avec succès."
    except Exception as e:
        return f"Erreur de transcription : {e}"


with gr.Blocks() as interface:
    gr.Markdown(
        "<h2 style='text-align: center;'>Résumé et Questions/Réponses sur une vidéo YouTube (Local Qwen - Français)</h2>"
    )

    video_url = gr.Textbox(
        label="URL de la vidéo YouTube", placeholder="Entre l'URL de la vidéo YouTube"
    )

    transcript_status = gr.Textbox(
        label="Statut de la transcription", interactive=False
    )
    summary_output = gr.Textbox(label="Résumé de la vidéo", lines=6)
    question_input = gr.Textbox(
        label="Poser une question sur la vidéo",
        placeholder="Pose ta question en français",
    )
    answer_output = gr.Textbox(label="Réponse à ta question", lines=8)

    fetch_btn = gr.Button("Récupérer la transcription")
    summarize_btn = gr.Button("Résumer la vidéo")
    question_btn = gr.Button("Poser la question")

    fetch_btn.click(get_transcript_status, inputs=video_url, outputs=transcript_status)
    summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
    question_btn.click(
        answer_question, inputs=[video_url, question_input], outputs=answer_output
    )

interface.launch(server_name="0.0.0.0", server_port=7860)
