from haystack.components.preprocessors import DocumentCleaner, RecursiveDocumentSplitter
from haystack import Document, Pipeline
from qdrant_manager import QdrantManager
from haystack.dataclasses import Document


import os

# Detailed documents data provided by you
documents_data = [
    {
        "id": "1",
        "path": "./data/FanResearchPaper.txt",
        "title": "Pull-Chain Fan Usability Research",
        "category": "UX Research Study",
        "description": "A human factors engineering research paper analyzing usability challenges with pull-chain ceiling fans.",
        "keywords": ["UX Research", "Human Factors", "Usability", "Ceiling Fans", "User Psychology"],
        "external_link": "https://nishadpatne.framer.website/project/cieling-fan"
    },
    {
        "id": "2",
        "path": "./data/iSTART_Early_Case_Study_Cleaned.txt",
        "title": "iSTART Early Mobile App Case Study",
        "category": "UX Case Study",
        "description": "A comprehensive case study detailing the design and development process of the iSTART Early mobile application.",
        "keywords": ["UX Case Study", "Mobile App Design", "iSTART Early", "User Experience", "UI/UX"],
        "external_link": "https://nishadpatne.framer.website/project/iSTARTEarly"
    },
    {
        "id": "3",
        "path": "./data/iSTARTEarlyInternalUsabilityTest.txt",
        "title": "iSTART Early Internal Usability Test",
        "category": "Usability Testing Report",
        "description": "Internal usability test report for the iSTART Early application, highlighting test procedures, findings, and recommendations.",
        "keywords": ["Usability Testing", "iSTART Early", "Internal Test", "UX Analysis", "User Feedback"],
        "external_link": "https://nishadpatne.framer.website/project/iSTARTEarly"
    },
    {
        "id": "4",
        "path": "./data/Nishad_LEI_Contributions.txt",
        "title": "Nishad LEI Contributions",
        "category": "UX Contributions",
        "description": "A detailed account of contributions to LEI projects, showcasing UX design, research, and strategic improvements.",
        "keywords": ["UX Contributions", "Design", "Research", "LEI", "Product Design"],
        "external_link": "https://nishadpatne.framer.website/project/lei-contributions"
    },
    {
        "id": "5",
        "path": "./data/resume.txt",
        "title": "Nishad Patne Resume",
        "category": "Professional Resume",
        "description": "The professional resume of Nishad Patne, detailing work experience, education, and skills in UX and product design.",
        "keywords": ["Resume", "Professional", "UX Design", "Product Design", "Career"],
        "external_link": "https://nishadpatne.framer.website/resume"
    },
    {
        "id": "6",
        "path": "./data/UX_Research_and_Recommendations_V2_raw.txt",
        "title": "Paani Foundation UX Research and Recommendations",
        "category": "UX Research Report",
        "description": "A comprehensive UX research report with recommendations based on heuristic evaluations, user testing, and survey findings about paani foundation website.",
        "keywords": ["UX Research", "Recommendations", "Heuristic Evaluation", "User Testing", "UX Strategy"],
        "external_link": "https://nishadpatne.framer.website/project/ux-research"
    },
    {
        "id": "7",
        "path": "./data/WAT_UI_Test_and_Results_UX_raw.txt",
        "title": "WAT UI Test and Results",
        "category": "UX Testing Report",
        "description": "Detailed results and analysis from the UI/UX testing of the WAT application, including user feedback and performance insights.",
        "keywords": ["UI Testing", "UX Report", "WAT", "Usability", "User Feedback"],
        "external_link": "https://nishadpatne.framer.website/project/wat-ui-test"
    }
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
