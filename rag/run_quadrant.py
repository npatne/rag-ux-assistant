from haystack.components.preprocessors import DocumentCleaner, RecursiveDocumentSplitter
from haystack import Document, Pipeline
from qdrant_manager import QdrantManager
from haystack.dataclasses import Document

import os

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Detailed documents data provided by you
documents_data = [
    {
        "id": "1",
        "path": "./data/ceiling_fan_study.txt",
        "title": "Ceiling Fan Pull-Chain Controls: A Human Factors Study",
        "category": "UX Research Study",
        "description": "A deep-dive academic research project analyzing the usability flaws in ceiling fan pull-chain controls, combining cognitive psychology with empirical testing.",
        "keywords": ["Human Factors", "Usability", "UX Research", "Cognitive Psychology", "Ceiling Fans"],
        "external_link": "www.nishadpatne.com"
    },
    {
        "id": "2",
        "path": "./data/istart_early.txt",
        "title": "iSTART Early: Gamified Learning Ecosystem",
        "category": "UX Case Study",
        "description": "A comprehensive case study of a large-scale EdTech platform redesign, covering student and teacher UX, module development, and AI integration.",
        "keywords": ["UX Design", "Gamification", "Education Technology", "NLP", "AI Integration"],
        "external_link": "www.nishadpatne.com"
    },
    {
        "id": "3",
        "path": "./data/jerry.txt",
        "title": "Jerry the Junior Research AIDE",
        "category": "Enterprise UX Design",
        "description": "An enterprise design project for academic research management, featuring complex information architecture and design system engineering.",
        "keywords": ["Enterprise UX", "Design Systems", "Research Platforms", "Information Architecture", "Academic Tools"],
        "external_link": "www.nishadpatne.com"
    },
    {
        "id": "4",
        "path": "./data/overview_general.txt",
        "title": "Nishad Patne – General Portfolio Overview",
        "category": "Portfolio Summary",
        "description": "An overview of Nishad Patne’s UX, engineering, and research projects, summarizing major work and skill areas across industries.",
        "keywords": ["Portfolio", "UX Engineer", "Product Design", "AI", "Enterprise Software"],
        "external_link": "www.nishadpatne.com"
    },
    {
        "id": "5",
        "path": "./data/paani_foundation.txt",
        "title": "Paani Foundation Website Redesign",
        "category": "UX Research Project",
        "description": "An academic UX research project analyzing navigation issues in a nonprofit's website, supported by heuristic evaluation and user testing.",
        "keywords": ["Nonprofit UX", "Heuristic Evaluation", "User Testing", "Information Architecture", "Web Design"],
        "external_link": "www.nishadpatne.com"
    },
    {
        "id": "6",
        "path": "./data/rag_ai_portfolio.txt",
        "title": "AI-Powered Portfolio Chatbot",
        "category": "AI Engineering Case Study",
        "description": "A technical deep dive into building a conversational RAG AI chatbot using Gemini Pro and Qdrant, with detailed engineering trade-offs and design decisions.",
        "keywords": ["RAG", "LLM", "Qdrant", "Haystack", "AI Portfolio"],
        "external_link": "www.nishadpatne.com"
    },
    {
        "id": "7",
        "path": "./data/sandbox.txt",
        "title": "Project Sandbox: Mini Projects Collection",
        "category": "Multi-Project Portfolio",
        "description": "A set of focused UX and engineering mini-projects including a Figma plugin, academic tool UI, and a job scam prevention website.",
        "keywords": ["Figma Plugin", "Academic Tools", "Job Scam Prevention", "Mini Projects", "Design Engineering"],
        "external_link": "www.nishadpatne.com"
    },
    {
        "id": "8",
        "path": "./data/wat_researcher.txt",
        "title": "WAT Researcher: Desktop-First Text Analytics",
        "category": "Product Pivot Case Study",
        "description": "A case study in strategic product pivoting, describing the redesign and engineering of a desktop-first academic NLP tool to replace an aging web platform.",
        "keywords": ["Desktop App", "UX Engineering", "NLP", "Electron", "Strategic Pivot"],
        "external_link": "www.nishadpatne.com"
    },
    {
        "id": "9",
        "path": "./data/wat.txt",
        "title": "Writing Analytics Tool (WAT)",
        "category": "UX Redesign and Research",
        "description": "A full UX redesign and design system implementation for an NLP-based educational platform, backed by teacher focus groups and iterative prototyping.",
        "keywords": ["Writing Analytics", "UX Research", "Design Systems", "Focus Groups", "Education Technology"],
        "external_link": "www.nishadpatne.com"
    },
]


def load_document(doc_meta):
    with open(doc_meta["path"], "r", encoding="utf-8") as f:
        content = f.read()
    return Document(content=content, meta=doc_meta)


def process_and_upload_all():
    docs = [load_document(meta) for meta in documents_data]
    print(f"Loaded {len(docs)} full documents.")

    cleaner = DocumentCleaner(remove_empty_lines=True,
                              remove_extra_whitespaces=True)
    splitter = RecursiveDocumentSplitter(
        split_length=500, split_overlap=80, split_unit="word",
        separators=["\n\n", "\n", ".", " "]
    )

    pipeline = Pipeline()
    pipeline.add_component("cleaner", cleaner)
    pipeline.add_component("splitter", splitter)
    pipeline.connect("cleaner.documents", "splitter.documents")

    cleaned_and_split = pipeline.run({"cleaner": {"documents": docs}})[
        "splitter"]["documents"]
    print(f"Cleaned and split into {len(cleaned_and_split)} chunks.")

    # Enhance chunk metadata with external link anchors if necessary
    for i, chunk in enumerate(cleaned_and_split):
        chunk.meta["chunk_index"] = i
        chunk.meta["external_link"] = f"{chunk.meta['external_link']}#chunk-{i+1}"

    print("Example chunk content:\n", cleaned_and_split[0].content[:500])
    print("Metadata:\n", cleaned_and_split[0].meta)

    qdrant_manager = QdrantManager()
    qdrant_manager.embed_and_upload(cleaned_and_split)


if __name__ == "__main__":
    process_and_upload_all()
