import asyncio
import aiohttp
import datetime
import os

# API_URL = "http://127.0.0.1:8000/chat"
API_URL = "http://137.131.30.181:8000/chat" # VM url

sessions = {
    "session_1": [
        {"query": "Walk me through Nishad's professional background—roles, timeline, and areas of focus."},
        {"query": "Which companies or labs has Nishad worked with, and what were his responsibilities in each?"},
        {"query": "What kind of design or research impact did Nishad have in these roles?"},
        {"query": "Tell me about Nishad's educational background and how it informs his work as a UX designer."},
        {"query": "In your opinion, what is Nishad’s strongest area—research, design, testing, or something else? Why?"}
    ],
    "session_2": [
        {"query": "Summarize the Pull-Chain Fan Usability Research—motivation and setup.", "mode": "specific", "active_case_study": "Pull-Chain Fan Usability Research"},
        {"query": "What user behaviors or usability challenges emerged?", "mode": "specific", "active_case_study": "Pull-Chain Fan Usability Research"},
        {"query": "Which psychological principles influenced design insights?", "mode": "specific", "active_case_study": "Pull-Chain Fan Usability Research", "section_context": "psychological principles"},
        {"query": "Were there recommended design improvements?", "mode": "specific", "active_case_study": "Pull-Chain Fan Usability Research", "section_context": "design improvements"},
        {"query": "What are the logical next steps for this research?", "mode": "specific", "active_case_study": "Pull-Chain Fan Usability Research"}
    ],
    "session_3": [
        {"query": "Introduce the iSTART Early Mobile App Case Study and its goals.", "mode": "specific", "active_case_study": "iSTART Early Mobile App Case Study"},
        {"query": "What major UI/UX improvements were made from desktop to mobile?", "mode": "specific", "active_case_study": "iSTART Early Mobile App Case Study", "section_context": "UI design"},
        {"query": "How did AI agent Matty improve the user experience?", "mode": "specific", "active_case_study": "iSTART Early Mobile App Case Study", "section_context": "AI/Matty"},
        {"query": "What heuristics guided the UI design for children?", "mode": "specific", "active_case_study": "iSTART Early Mobile App Case Study", "section_context": "UI design"},
        {"query": "What enhancements or next steps are planned?", "mode": "specific", "active_case_study": "iSTART Early Mobile App Case Study", "section_context": "future directions"}
    ]
}

async def fetch_chat(session, payload):
    start_time = datetime.datetime.now()
    try:
        async with session.post(API_URL, json=payload) as response:
            response_json = await response.json()
            latency = (datetime.datetime.now() - start_time).total_seconds()
            return payload, response_json, latency
    except Exception as e:
        return payload, {"error": str(e)}, None

async def run_session(session_id, queries, session):
    session_results = []
    for query_data in queries:
        payload = {
            "query": query_data["query"],
            "session_id": session_id,
            "mode": query_data.get("mode", "general"),
            "active_case_study": query_data.get("active_case_study"),
            "section_context": query_data.get("section_context")
        }
        result = await fetch_chat(session, payload)
        session_results.append(result)
        await asyncio.sleep(0.1)
    return session_results

async def main():
    os.makedirs("test_results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%I-%M-%S%p")
    results_file = f"test_results/concurrent_chat_results_{timestamp}.txt"

    async with aiohttp.ClientSession() as session:
        tasks = [run_session(sid, queries, session) for sid, queries in sessions.items()]
        all_sessions_results = await asyncio.gather(*tasks)

    with open(results_file, "w", encoding="utf-8") as f:
        for session_results in all_sessions_results:
            for payload, response, latency in session_results:
                f.write(f"Session ID: {payload['session_id']}\n")
                f.write(f"Query: {payload['query']}\n")
                if latency is not None:
                    f.write(f"Latency: {latency:.3f}s\n")
                else:
                    f.write("Latency: Error\n")
                f.write(f"Response: {response}\n")
                f.write("=" * 80 + "\n")

    print(f"Concurrent testing complete. Results saved in {results_file}")

if __name__ == "__main__":
    asyncio.run(main())
