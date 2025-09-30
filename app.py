import gradio as gr
import google.genai as genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
import re
import json
import asyncio
import os
from serpapi import GoogleSearch


SERPAPI_KEY = "ENTER YOUR API KEY HERE"

# ------------------------
# Configuration
# ------------------------
NUM_SEARCHES = 3

# ------------------------
# Pydantic Models
# ------------------------
class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")

class WebSearchPlan(BaseModel):
    searches: List[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query.")

class ReportData(BaseModel):
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")
    markdown_report: str = Field(description="The final report in markdown format")
    follow_up_questions: List[str] = Field(description="Suggested topics to research further")

# ------------------------
# Tool Functions
# ------------------------
def json_safe(text: str) -> str:
    """Escape characters that break JSON parsing."""
    if not text:
        return ""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
def extract_json(text: str):
    """
    Extracts the first JSON object from a text string.
    """
    try:
        # Look for {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in text.")
        json_text = match.group()
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {str(e)}\nPreview: {text[:500]}")


def search_web(query: str, reason: str) -> str:
    if not SERPAPI_KEY:
        raise ValueError("Missing SERPAPI_KEY environment variable")

    params = {
        "q": query,
        "engine": "google",
        "api_key": SERPAPI_KEY,
        "num": 5
    }

    search = GoogleSearch(params)
    data = search.get_dict()

    results = []
    for r in data.get("organic_results", []):
        title = r.get("title", "No title")
        url = r.get("link", "")
        snippet = r.get("snippet", "")
        results.append({"title": title, "url": url, "snippet": snippet})

    search_output = {
        "query": query,
        "reason": reason,
        "results": results,
        "summary": f"Top {len(results)} search results for '{query}'"
    }

    return json.dumps(search_output)
# ------------------------
# System Instructions
# ------------------------
SEARCH_PLANNER_INSTRUCTION = f"""You are a research planning assistant. 
Given a research query, create a strategic search plan with {NUM_SEARCHES} different searches. 
Each search should target different aspects to provide comprehensive coverage. 
Think about:
- Different angles and perspectives on the topic
- Various subtopics and related areas
- Recent developments and trends
- Comparisons with alternatives
- Technical details and implementation aspects
- Use cases and applications
- Expert opinions and reviews
- Challenges and limitations

Return a JSON object with the exact format: 
{{"searches": [{{"reason": "...", "query": "..."}}, ...]}}"""

REPORT_WRITER_INSTRUCTION = """You are a senior researcher tasked with writing a cohesive report for a research query. 
You will be provided with the original query and search results from a research assistant.

First, create an outline for the report that describes the structure and flow. 
Then, generate the full report based on that outline.

Requirements:
- The report must be in markdown format
- It should be detailed and comprehensive (aim for 1000+ words)
- Include proper sections with headers
- Provide a short 2-3 sentence summary
- Suggest 3-5 follow-up research questions

Return a JSON object with: {"short_summary": "...", "markdown_report": "...", "follow_up_questions": [...]}"""

# ------------------------
# Helper Functions
# ------------------------

async def plan_searches(query: str, progress=gr.Progress()) -> WebSearchPlan:
    progress(0.1, desc="🧠 Planning research strategy...")
    
    client = genai.Client(api_key="ENTER YOUR API KEY HERE")
    
    input_text = f"{SEARCH_PLANNER_INSTRUCTION}\n\nResearch Query: {query}\n\nPlease provide the search plan in JSON format."
    
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=input_text,
        config=types.GenerateContentConfig(
            temperature=0.7,
            response_mime_type="application/json"
        )
    )
    
    # JSON-safe parsing
    try:
        data = extract_json(response.text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from Gemini response: {str(e)}\nResponse preview: {response.text[:500]}")
    
    search_plan = WebSearchPlan(**data)
    progress(0.2, desc=f"✓ Planned {len(search_plan.searches)} searches")
    return search_plan
async def perform_searches(search_plan: WebSearchPlan, progress=gr.Progress()) -> List[dict]:
    """Execute web searches for each item in the search plan"""
    results = []
    total = len(search_plan.searches)
    
    for i, item in enumerate(search_plan.searches, 1):
        progress(0.2 + (0.5 * i / total), desc=f"🔍 Searching ({i}/{total}): {item.query[:50]}...")
        result_json = search_web(item.query, item.reason)
        results.append(json.loads(result_json))
        await asyncio.sleep(0.1)  # Small delay to show progress
    
    progress(0.7, desc="✓ Completed all searches")
    return results
def build_references(search_results: list) -> str:
    """
    Returns a clean Markdown references section with numbered items.
    Each reference includes title and URL.
    """
    seen = set()
    references_md = "## References\n\n"
    counter = 1
    
    for search in search_results:
        for res in search["results"]:
            title = res.get("title", "No title")
            url = res.get("url", "")
            key = (title, url)
            if key not in seen and url:
                seen.add(key)
                references_md += f"{counter}. [{title}]({url})\n"
                counter += 1
    
    return references_md

async def write_report(query: str, search_results: List[dict], progress=gr.Progress()) -> ReportData:
    progress(0.75, desc="📝 Writing comprehensive report...")

    client = genai.Client(api_key="ENTER YOUR API KEY HERE")

    # Build a JSON-safe summary of all search results
    results_summary = "\n\n".join([
        f"Search {i+1}: {json_safe(r['query'])}\nReason: {json_safe(r['reason'])}\nReferences:\n" +
        "\n".join([f"{j+1}. {json_safe(res.get('url',''))}" for j, res in enumerate(r['results'])])
        for i, r in enumerate(search_results)
    ])

    input_text = (
        f"{REPORT_WRITER_INSTRUCTION}\n\n"
        f"Original Research Query: {json_safe(query)}\n\n"
        f"Search Results:\n{results_summary}\n\n"
        "Please write a comprehensive report in JSON format with short_summary, markdown_report, and follow_up_questions."
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=input_text,
        config=types.GenerateContentConfig(
            temperature=0.7,
            response_mime_type="application/json"
        )
    )

    # Robust JSON extraction
    data = extract_json(response.text)
    report = ReportData(**data)

    # Append clean references
    references_md = build_references(search_results)
    report.markdown_report += "\n\n" + references_md

    progress(1.0, desc="✓ Report completed!")
    return report

async def run_research_pipeline(query: str, progress=gr.Progress()) -> tuple:
    """Main research pipeline"""
    try:
        # Validate input
        if not query or len(query.strip()) < 3:
            return "❌ Please enter a valid research query (at least 3 characters)", "", ""
        
        progress(0, desc="🚀 Starting research pipeline...")
        
        # Step 1: Plan searches
        search_plan = await plan_searches(query, progress)
        
        # Step 2: Perform searches
        search_results = await perform_searches(search_plan, progress)
        
        # Step 3: Write report
        report = await write_report(query, search_results, progress)
        # references_md = "## References\n\n"
        # counter = 1
        # for search in search_results:
        #     for r in search["results"]:
        #         references_md += f"{counter}. [{r['title']}]({r['url']})\n"
        #         counter += 1

        # # Append to the markdown report
        # report.markdown_report += "\n\n" + references_md
        # Format follow-up questions
        follow_up = "\n".join([f"{i}. {q}" for i, q in enumerate(report.follow_up_questions, 1)])
        
        return report.short_summary, report.markdown_report, follow_up
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, "", ""

# Wrapper for Gradio
def research_wrapper(query: str, progress=gr.Progress()):
    """Synchronous wrapper for Gradio"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(run_research_pipeline(query, progress))
    finally:
        loop.close()

# ------------------------
# Gradio Interface
# ------------------------
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
    ),
    css="""
    .gradio-container {
        max-width: 100% !important;
        width: 100% !important
    }
    .header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    """
) as demo:
    
    # Header
    gr.HTML("""
    <div class="header">
        <h1>🔬 AI Research Agent</h1>
        <p>Enter any topic and get a comprehensive research report with different perspectives</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            gr.Markdown("### 📝 Research Query")
            query_input = gr.Textbox(
                label="What would you like to research?",
                placeholder="e.g., Latest AI Agent frameworks in 2025, Quantum computing applications, Climate change solutions...",
                lines=3,
                max_lines=5
            )
            
            with gr.Row():
                submit_btn = gr.Button("🚀 Start Research", variant="primary", size="lg")
                clear_btn = gr.Button("🔄 Clear", variant="secondary", size="lg")
            
            gr.Markdown("### 💡 Example Topics")
            gr.Examples(
                examples=[
                    ["Latest AI Agent frameworks in 2025"],
                    ["Sustainable energy solutions for developing countries"],
                    ["Impact of quantum computing on cryptography"],
                    ["Future of work with AI automation"],
                    ["Advances in gene therapy and CRISPR technology"],
                ],
                inputs=query_input,
                label=""
            )
    
    # Output sections
    gr.Markdown("---")
    gr.Markdown("## 📊 Research Results")
    
    with gr.Row():
        with gr.Column():
            summary_output = gr.Textbox(
                label="📌 Executive Summary",
                lines=4,
                interactive=False
            )
    
    with gr.Row():
        with gr.Column(scale=2):
            report_output = gr.Markdown(
                label="📄 Full Report"
            )
        
        with gr.Column(scale=1):
            followup_output = gr.Textbox(
                label="🔮 Follow-up Questions",
                lines=10,
                interactive=False
            )
    
    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; opacity: 0.7; padding: 1rem;">
        <p>Powered by Google Gemini 2.0 Flash • Performs {num} diverse searches for comprehensive coverage</p>
    </div>
    """.format(num=NUM_SEARCHES))
    
    # Event handlers
    submit_btn.click(
        fn=research_wrapper,
        inputs=[query_input],
        outputs=[summary_output, report_output, followup_output]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", ""),
        inputs=[],
        outputs=[summary_output, report_output, followup_output]
    )
    
    query_input.submit(
        fn=research_wrapper,
        inputs=[query_input],
        outputs=[summary_output, report_output, followup_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
