
import datetime
import os
from retriever import UXChatAssistant


def run_batch_tests():
    assistant = UXChatAssistant()

    test_cases = [
        # ✅ Good Queries (Should reliably produce meaningful results)
        {"query": "Who is Nishad Patne?", "mode": "general"},
        {"query": "What is Nishad's UX design background?", "mode": "general"},
        {"query": "Summarize this case study.", "mode": "specific",
         "active_case_study": "iSTART Early Mobile App Case Study", "summarize": True},
        {"query": "Tell me more about the user psychology aspects in fan usability.", "mode": "specific",
         "active_case_study": "Pull-Chain Fan Usability Research", "section_context": "user psychology"},
        {"query": "Details about heuristic evaluations in Paani Foundation study?", "mode": "specific",
         "active_case_study": "Paani Foundation UX Research and Recommendations", "section_context": "heuristic evaluation"},
        {"query": "Summarize the resume of Nishad.", "mode": "specific",
         "active_case_study": "Nishad Patne Resume", "summarize": True},
        {"query": "Explain the UI design changes in the iSTART case study.", "mode": "specific",
         "active_case_study": "iSTART Early Mobile App Case Study", "section_context": "UI design"},
        {"query": "Describe Nishad's contributions in the LEI project.", "mode": "specific",
         "active_case_study": "Nishad LEI Contributions"},
        {"query": "What feedback was recorded in the WAT UI test?", "mode": "specific",
         "active_case_study": "WAT UI Test and Results"},
        {"query": "Give me an overview of Nishad’s projects.", "mode": "general"},
        {"query": "List all UX research case studies Nishad has conducted.",
            "mode": "general"},
        {"query": "What did Nishad recommend for improving the Paani Foundation website?", "mode": "specific",
         "active_case_study": "Paani Foundation UX Research and Recommendations"},

        # 🔄 Previously problematic (Retest specifically for accuracy)
        {"query": "Explain UI improvements in iSTART Early case study.", "mode": "specific",
         "active_case_study": "iSTART Early Mobile App Case Study", "section_context": "UI design"},
        {"query": "Describe user psychology findings from fan usability study.", "mode": "specific",
         "active_case_study": "Pull-Chain Fan Usability Research", "section_context": "user psychology"},

        # 🚨 Off-topic/Random Queries (Should trigger fallback reliably)
        {"query": "What's the weather on Mars?", "mode": "general"},
        {"query": "Tell me about black holes and frogs.", "mode": "general"},
        {"query": "Generate a rap battle between Kant and Naruto.", "mode": "general"},
        {"query": "What color is sadness?", "mode": "general"},
        {"query": "Explain string theory using potatoes.", "mode": "general"},
        {"query": "Why does toast land butter side down?", "mode": "general"},
        {"query": "Is Nishad secretly a chef?", "mode": "general"},
        {"query": "When was the great tomato revolution?", "mode": "general"},
        {"query": "What is the square root of design?", "mode": "general"},
        {"query": "If UX had a zodiac sign, what would it be?", "mode": "general"},
        {"query": "Decode this random text: sdklj2lkjLKJ", "mode": "general"},
        {"query": "Write me a pirate tale involving Nishad and wireframes.",
            "mode": "general"},
        {"query": "Are breadcrumbs edible in website navigation?", "mode": "general"},
        {"query": "Why are pizzas round but pizza boxes square?", "mode": "general"},
        {"query": "Explain quantum theory through tomato farming analogies.",
            "mode": "general"},
        {"query": "What sound does design make?", "mode": "general"},
        {"query": "Tell me the biography of an imaginary friend named Bob.",
            "mode": "general"}
    ]

    # Ensure output directory exists
    os.makedirs("test_results", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%I-%M-%S%p")
    result_file = f"test_results/test_results_{timestamp}.txt"
    log_file = f"test_results/test_logs_{timestamp}.log"

    with open(result_file, "w", encoding="utf-8") as result_f, open(log_file, "w", encoding="utf-8") as log_f:
        for i, case in enumerate(test_cases, 1):
            try:
                print(f"Running test case #{i}: {case['query']}")
                log_f.write(f"Running test case #{i}: {case['query']}\n")
                response = assistant.get_response(
                    query=case["query"],
                    session_id=f"test_session_{i}",
                    active_case_study=case.get("active_case_study"),
                    mode=case.get("mode", "general"),
                    section_context=case.get("section_context"),
                    summarize=case.get("summarize", False)
                )
                result_f.write(
                    f"Query {i}: {case['query']}\n"
                    f"Answer: {response['answer']}\n"
                    f"Sources: {response['sources']}\n"
                    f"Mode: {response['mode']}\n"
                    f"Fallback: {response['fallback']}\n"
                    f"Steps: {response.get('steps', [])}\n"
                    f"GIF: {response.get('gif_url', 'N/A')}\n\n{'='*80}\n"
                )
                log_f.write(f"Response received for Query {i}.\n\n")
            except Exception as e:
                error_msg = f"Error in Query {i}: {str(e)}\n"
                log_f.write(error_msg)
                result_f.write(
                    f"Query {i}: {case['query']}\nError: {str(e)}\n{'='*80}\n")


if __name__ == "__main__":
    run_batch_tests()
