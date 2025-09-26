# Install dependencies first:
# pip install streamlit langchain-community faiss-cpu sentence-transformers transformers

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import streamlit as st

# ---------------- Enhanced FAQ Data ----------------
faq_data = [
    # UG Courses
    {"question": "Tell me about B.Com at SIWS",
     "answer": "Eligibility: Passed Std. XII (HSC) from any recognized board. Fees: ₹10,265/year. Duration: 3 years. Timings: 9 AM – 12 PM."},

    {"question": "Tell me about BMS at SIWS",
     "answer": "Eligibility: Std. XII with 45% (Open) or 40% (Reserved), first attempt. Admission via MAH-CET. Fees: ₹47,765/year. Duration: 3 years. Timings: 12 PM – 3 PM."},

    {"question": "Tell me about B.Sc IT at SIWS",
     "answer": "Eligibility: Std. XII Science with Mathematics, 45% (Open) or 40% (Reserved). Fees: ₹33,765/year. Duration: 3 years. Timings: 1 PM – 4 PM."},

    {"question": "Tell me about BA at SIWS",
     "answer": "Eligibility: Std. XII (HSC) from any recognized board. Fees: ₹10,000/year. Duration: 3 years. Timings: 9 AM – 12 PM."},

    # PG Courses
    {"question": "Tell me about M.Com at SIWS",
     "answer": "Eligibility: Passed B.Com, BMS, BBI, or BAF with 50% aggregate. Fees: ₹16,050/year. Duration: 2 years. Timings: 3 PM – 6 PM."},

    {"question": "Tell me about M.Sc IT at SIWS",
     "answer": "Eligibility: Passed B.Sc in IT, CS, Data Science, or AI. Fees: ₹41,150/year. Duration: 2 years. Timings: 1 PM – 4 PM."},

    # Admission Info
    {"question": "How can I apply for admission?",
     "answer": "Apply online via the SIWS College website. Admission is merit-based or via entrance exams. Forms release in May, merit list in June, classes begin in July."},

    {"question": "What documents are required for admission?",
     "answer": "Required documents: Std. XII marksheet, leaving certificate, caste certificate (if applicable), passport-size photos, and Aadhaar card."},

    # Facilities
    {"question": "What facilities does SIWS College provide?",
     "answer": "Facilities include library (15,000+ books, 80,000+ e-resources), computer labs, gymkhana, canteen, smart classrooms, and Wi-Fi campus."},

    {"question": "Does SIWS College have hostels?",
     "answer": "No, SIWS does not offer on-campus hostels. Students can find PGs and private accommodations nearby."},

    # Placements
    {"question": "How is the placement at SIWS College?",
     "answer": "Placement rate ~70%. Average package ₹2 LPA, highest ₹7 LPA. Recruiters include TCS, L&T, Capgemini, Oracle, and media.net."},

    {"question": "Are internships available at SIWS?",
     "answer": "Yes, the placement cell helps students find internships in IT, finance, media, and biotech sectors."},

    # Campus Life
    {"question": "What clubs and events are at SIWS?",
     "answer": "SIWS has NSS, cultural club, tech club, sports teams, and annual fests like 'Synergy'. Seminars and workshops are held regularly."},

    {"question": "Where is SIWS College located?",
     "answer": "SIWS is located at Plot No. 337, Sewree-Wadala Estate, Major R. Parameshwaran Marg, Wadala, Mumbai – 400031."},

    {"question": "What are the college timings?",
     "answer": "College runs Monday to Saturday, 9 AM to 5 PM. Each department has specific lecture slots."}
]

# ---------------- Prepare Documents ----------------
docs = [Document(page_content=f["answer"], metadata={"question": f["question"]}) for f in faq_data]

# Embedding and Vector Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# QA Model
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# RetrievalQA Chain with source documents
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    return_source_documents=True
)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="SIWS College FAQ Chatbot", page_icon="🎓", layout="centered")

st.title("🎓 SIWS College FAQ Chatbot")
st.markdown("Ask anything about **courses, fees, timings, placements, or facilities** at SIWS College.")

# Sidebar Quick Links
st.sidebar.title("📌 Quick FAQs")
quick_links = [
    "B.Com course details",
    "M.Sc IT eligibility",
    "Placement stats",
    "Hostel info",
    "Admission process",
    "Campus clubs and events"
]
for item in quick_links:
    st.sidebar.markdown(f"- {item}")

# Main Input
query = st.text_input("💬 Type your question here:")

if st.button("Get Answer") and query:
    result = qa_chain(query)
    answer = result["result"]
    source_docs = result.get("source_documents", [])

    if source_docs:
        matched_question = source_docs[0].metadata["question"]
        st.success(f"**Answer:** {answer}")
        st.info(f"📌 Based on: *{matched_question}*")
    else:
        st.warning("Sorry, I couldn't find an answer to that. Try rephrasing your question.")
