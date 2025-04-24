import asyncio
import aiohttp
import datetime
import os

API_URL = "http://127.0.0.1:8000/chat"

# Define sessions and chats
sessions = {
    "session_1": [
        {"query": "Explain Nishad's professional background."},
        {"query": "Can you summarize the iSTART Early Mobile App Case Study?", "mode": "specific", "active_case_study": "iSTART Early Mobile App Case Study", "section_context": None},
        {"query": "What were the UI improvements?", "mode": "specific", "active_case_study": "iSTART Early Mobile App Case Study", "section_context": "UI design"},
        {"query": "Describe Nishad’s role in LEI project.", "mode": "specific", "active_case_study": "Nishad LEI Contributions"},
        {"query": "List all UX research conducted by Nishad."}
    ],
    "session_2": [
        {"query": "Provide an overview of Nishad's UX projects."},
        {"query": "Explain heuristic evaluations in the Paani Foundation study.", "mode": "specific", "active_case_study": "Paani Foundation UX Research and Recommendations", "section_context": "heuristic evaluation"},
        {"query": "Summarize Pull-Chain Fan Usability Research.", "mode": "specific", "active_case_study": "Pull-Chain Fan Usability Research", "section_context": None},
        {"query": "What insights about user psychology were found?", "mode": "specific", "active_case_study": "Pull-Chain Fan Usability Research", "section_context": "user psychology"},
        {"query": "Detail Nishad’s contributions to WAT UI Test."}
    ],
    "session_3": [
        {"query": "Summarize Nishad's resume."},
        {"query": "Explain UI improvements from the iSTART Early study.", "mode": "specific", "active_case_study": "iSTART Early Mobile App Case Study", "section_context": "UI design"},
        {"query": "Discuss user psychology in fan usability.", "mode": "specific", "active_case_study": "Pull-Chain Fan Usability Research", "section_context": "user psychology"},
        {"query": "Describe Nishad's UX recommendations for Paani Foundation.", "mode": "specific", "active_case_study": "Paani Foundation UX Research and Recommendations"},
        {"query": "What methodologies did Nishad use in UX research?"}
    ]
}

async def fetch_chat(session, payload):
    start_time = datetime.datetime.now()
    async with session.post(API_URL, json=payload) as response:
        response_json = await response.json()
        latency = (datetime.datetime.now() - start_time).total_seconds()
        return payload, response_json, latency

async def run_tests():
    os.makedirs("test_results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%I-%M-%S%p")
    results_file = f"test_results/chat_results_{timestamp}.txt"

    async with aiohttp.ClientSession() as session:
        tasks = []
        for session_id, queries in sessions.items():
            for query_data in queries:
                payload = {
                    "query": query_data["query"],
                    "session_id": session_id,
                    "mode": query_data.get("mode", "general"),
                    "active_case_study": query_data.get("active_case_study"),
                    "section_context": query_data.get("section_context")
                }
                tasks.append(fetch_chat(session, payload))

        responses = await asyncio.gather(*tasks)

    with open(results_file, "w", encoding="utf-8") as f:
        for payload, response, latency in responses:
            f.write(f"Query: {payload['query']}\n")
            f.write(f"Session ID: {payload['session_id']}\n")
            f.write(f"Latency: {latency:.3f}s\n")
            f.write(f"Response: {response}\n")
            f.write("="*80 + "\n")

    print(f"Testing complete. Results saved in {results_file}")

if __name__ == "__main__":
    asyncio.run(run_tests())
