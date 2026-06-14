import os
import sqlite3
import getpass
import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================
if "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = getpass.getpass("Enter your Hugging Face API Token (hf_...): ")

PDF_PATH = "./Smile_Rehab_Simulator_final.pdf"
DB_FILE = "smile_rehab.db"

# ==========================================
# 2. AUTOMATIC CLINICAL DATABASE INITIALIZATION
# ==========================================
def initialize_database():
    """Builds and populates the tracking matrix from scratch if not present."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS resources (
        resource_id INTEGER PRIMARY KEY AUTOINCREMENT,
        resource_name TEXT,
        status TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bookings (
        booking_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name TEXT,
        resource_id INTEGER
    )
    """)
    conn.commit()

    # Pre-populate rows only if the resources schema is empty
    cursor.execute("SELECT COUNT(*) FROM resources")
    if cursor.fetchone()[0] == 0:
        resources = [
            ("Smile Rehabilitation Simulator", "available"),
            ("Facial Nerve Rehabilitation Unit", "available"),
            ("Smile Symmetry Training Center", "available"),
            ("Lip Mobility Rehabilitation Module", "available"),
            ("Cheek Muscle Rehabilitation Unit", "available"),
            ("Facial Rehabilitation Sentinel Center", "available"),
            ("Synkinesis Monitoring Laboratory", "available"),
            ("Tele-Facial Rehabilitation Center", "available")
        ]
        cursor.executemany("INSERT INTO resources(resource_name, status) VALUES (?, ?)", resources)
        conn.commit()
        print("[⚡] Clinical Database Initialized and Populated Automatically.")
    else:
        print("[✔] Clinical Database Connected Successfully.")
    
    conn.close()

# Initialize database array before starting UI
initialize_database()

# ==========================================
# 3. KAG DOCUMENT PROCESSING & EMBEDDINGS
# ==========================================
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"Missing core reference text: Please place '{PDF_PATH}' in the root directory.")

print("Loading documentation matrix...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
chunks = text_splitter.split_documents(documents)

print("Generating FAISS vector space mapping...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(chunks, embedding_model)
retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5})

# ==========================================
# 4. SERVERLESS LLM ENDPOINT SETUP
# ==========================================
print("Connecting serverless inference layer endpoints...")
base_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.1,
    max_new_tokens=1024,
    huggingfacehub_api_token=os.environ["HF_TOKEN"]
)
llm = ChatHuggingFace(llm=base_endpoint)

# ==========================================
# 5. RETRIEVAL AND EXTRACTION ARCHITECTURE
# ==========================================
advanced_prompt = ChatPromptTemplate.from_template(
"""
You are an AI-Powered Interactive 3D Robotic Smile Rehabilitation Assistant.

Your purpose is to provide highly accurate, technically detailed,
clinically structured answers strictly based on the supplied
Smile Rehabilitation knowledge base.

CORE DOMAIN:
* Facial Nerve Rehabilitation
* Smile Symmetry Training
* Lip Mobility Rehabilitation
* Cheek Muscle Rehabilitation
* Facial Palsy Rehabilitation
* Bell's Palsy Rehabilitation
* Stroke-Related Facial Weakness Rehabilitation
* Neuro-Facial Rehabilitation
* Facial Rehabilitation Sentinel Framework
* Tele-Rehabilitation Workflows
* Rehabilitation Engineering
* Clinical Rehabilitation Monitoring

RESPONSE RULES:
1. Always prioritize information from the provided context.
2. If information is unavailable in the context, state: "Information is not available within the Smile Rehabilitation knowledge base."
3. Never fabricate clinical data.
4. Never generate medical diagnoses.
5. Never generate treatment prescriptions.
6. Never reveal: inventor names, personal addresses, email addresses, phone numbers, patent filing details, or confidential personal information.
7. When users ask technical questions, provide structured explanations using: System Blocks, Workflow Pipelines, Monitoring Layers, Rehabilitation Stages, Feedback Loops, Clinical Interaction Flow, Sentinel Monitoring Layers, or Safety Intervention Pathways.
8. When discussing simulator operation, focus on: smile symmetry correction, facial muscle rehabilitation, facial nerve activation, visual biofeedback, rehabilitation monitoring, and sentinel safety mechanisms.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | advanced_prompt
    | llm
    | StrOutputParser()
)

print("[✔] Knowledge Pipeline Ready.")

# ==========================================
# 6. AGENTIC RETRIEVAL TOOLS
# ==========================================
@tool
def smile_rehabilitation_knowledge_tool(question: str) -> str:
    """Use ONLY for technical questions about robotic smile systems and tracking mechanisms."""
    return rag_chain.invoke(question)

@tool
def check_availability_tool() -> str:
    """Use ONLY for checking available smile rehabilitation assets."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT resource_id, resource_name FROM resources WHERE status='available'")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No smile rehabilitation resources available."
    return "\n".join([f"Resource {r[0]}: {r[1]}" for r in rows])

@tool
def book_resource_tool(patient_name: str, resource_id: int) -> str:
    """Use ONLY when a user explicitly requests resource booking allocations."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM resources WHERE resource_id=? AND status='available'", (resource_id,))
    resource = cursor.fetchone()

    if not resource:
        conn.close()
        return f"Resource {resource_id} is currently occupied or unavailable."

    cursor.execute("INSERT INTO bookings (patient_name, resource_id) VALUES (?, ?)", (patient_name, resource_id))
    cursor.execute("UPDATE resources SET status='occupied' WHERE resource_id=?", (resource_id,))
    conn.commit()
    conn.close()

    return f"Smile rehabilitation resource {resource_id} successfully booked for {patient_name}."

# ==========================================
# 7. ROUTING LAYER (KAG + YUOM ENGINE)
# ==========================================
def healthcare_suite_router(user_query):
    query = user_query.lower()

    sensitive_terms = [
        "inventor", "invented", "creator", "created", "developer", "developed",
        "author", "patent applicant", "contact", "email", "phone", "mobile",
        "address", "who made", "who built", "who designed", "who owns"
    ]

    if any(term in query for term in sensitive_terms):
        return (
            "I cannot provide inventor, applicant, author, developer, creator, "
            "ownership, or personal contact information. I can only discuss the "
            "technical, clinical, and rehabilitation aspects of the system."
        )

    elif "book" in query:
        return "Resource booking functionality is currently under development."

    elif "available" in query or "availability" in query:
        return check_availability_tool.invoke({})

    elif any(
        x in query
        for x in [
            "smile rehabilitation", "facial nerve", "smile symmetry", "lip mobility",
            "cheek muscle", "bell's palsy", "facial palsy", "synkinesis", "sentinel",
            "facial rehabilitation", "neuro-facial", "resource"
        ]
    ):
        return """
Facial Rehabilitation Sentinel Center
A specialized smile rehabilitation facility designed
to support facial nerve rehabilitation,
smile symmetry training,
lip mobility rehabilitation,
cheek muscle rehabilitation,
facial palsy recovery,
synkinesis monitoring,
neuro-facial rehabilitation,
tele-rehabilitation workflows,
and facial rehabilitation safety monitoring.
"""
    else:
        return rag_chain.invoke(user_query)

# ==========================================
# 8. PRODUCTION GRADIO WEB INTERFACE
# ==========================================
def respond(message, history):
    try:
        user_message = message.get("content", "") if isinstance(message, dict) else str(message)
        result = healthcare_suite_router(user_message)
        return {"role": "assistant", "content": str(result)}
    except Exception as e:
        return {"role": "assistant", "content": f"Core Execution Exception: {str(e)}"}

description_html = """
<div style='text-align: center; padding: 10px;'>
    <h3 style='margin: 0; color: #4A4A4A;'>Built by Dr. Lakshmi Gandi</h3>
    <p style='margin-top: 5px;'>Clinical Intelligence Platform specialized in Robotic Facial Rehabilitation,
    Biomechanical Smile Symmetry Tracking, and Neurological Nerve Recovery Telemetry using KAG + YUOM Framework.</p>
</div>
"""

demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="😊 Interactive 3D Robotic Smile Rehabilitation Simulator",
    description=description_html,
    examples=[
        "How does the smile rehabilitation simulator work?",
        "Explain the Facial Rehabilitation Sentinel Framework.",
        "What is smile symmetry training?",
        "Explain facial nerve rehabilitation workflows.",
        "How does synkinesis monitoring work?",
        "Which rehabilitation resources are available?",
        "What is the future scope of the AI-powered smile rehabilitation simulator?"
    ]
)

if __name__ == "__main__":
    gr.close_all()
    # Share set to False for standardized local network deployment
    demo.launch(share=False)
