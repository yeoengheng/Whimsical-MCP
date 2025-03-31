import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv

import openai
import anthropic

load_dotenv()

class AirlineResearchAgent:
    def __init__(self, anthropic_api_key: str, openai_api_key: str):
        self.claude_client = anthropic.Anthropic(api_key=anthropic_api_key)
        openai.api_key = openai_api_key
        self.openai_client = openai

        self.research_plan = {}
        self.documents = {}
        self.extracted_info = {}
        self.synthesis = {}
        self.insights = {}
        self.citations = []

    def analyze_airline(self, airline_name: str, focus_areas: List[str] = None) -> Dict[str, Any]:
        start_time = time.time()

        print("1. Creating research plan...")
        self.research_plan = self._create_research_plan(airline_name, focus_areas)

        print("2. Fetching documents using GPT-4 search...")
        self.documents = self._fetch_documents_openai(airline_name, self.research_plan["focus_areas"])

        print("3. Extracting key information...")
        self.extracted_info = self._extract_information(self.documents, self.research_plan)

        print("4. Synthesizing information...")
        self.synthesis = self._synthesize_information(self.extracted_info, self.research_plan)

        print("5. Generating insights...")
        self.insights = self._generate_insights(self.synthesis, airline_name)

        print("6. Formatting citations...")
        self.citations = self._format_citations(self.documents)

        print("7. Creating final report...")
        final_report = self._create_final_report()

        print(f"✅ Analysis completed in {time.time() - start_time:.2f} seconds")
        return final_report

    def _create_research_plan(self, airline_name: str, focus_areas: List[str] = None) -> Dict[str, Any]:
        if focus_areas is None:
            focus_areas = ["investor relations", "press releases", "leadership", "sustainability", "strategic initiatives"]

        planning_prompt = f"""
        Create a research plan for analyzing the airline {airline_name}.
        Focus areas:
        {', '.join(focus_areas)}

        Provide:
        - Key questions to answer
        - Types of documents to look for
        - Areas to prioritize
        """

        response = self.claude_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            system="You are an airline industry research planner. You are the most detailed planner. You make use of every finding and make every plans with evidence",
            messages=[{"role": "user", "content": planning_prompt}]
        )

        topics = [fa.title() for fa in focus_areas]
        return {
            "airline_name": airline_name,
            "focus_areas": focus_areas,
            "research_topics": topics,
            "creation_date": datetime.now().isoformat()
        }
    def _fetch_documents_openai(self, airline_name: str, focus_areas: List[str]) -> Dict[str, Any]:
        print("Fetching real documents via GPT-4 search-and-summarize...")
        documents = {}

        for i, topic in enumerate(focus_areas):
            query = f"{airline_name} {topic}"

            prompt = f"""
            Search the web for the latest public information on "{query}".
            Summarize the most relevant article, page, or press release in 6–7 paragraphs. Include as many specific details. 
            Focus on new developments, announcements, leadership statements, or ESG/sustainability actions.
            Return the summary and the source URL (if known).
            """

            try:
                response = openai.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system", "content": "You are a web-scraping assistant using live web access."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=800
                )

                content = response.choices[0].message.content.strip()

                documents[f"doc_{i+1}"] = {
                    "id": f"doc_{i+1}",
                    "topic": topic.title(),
                    "content": content,
                    "source": f"(source embedded in summary for {topic})",
                    "retrieval_date": datetime.now().isoformat()
                }

            except Exception as e:
                print(f"❌ OpenAI failed to fetch content for {topic}: {str(e)}")
                documents[f"doc_{i+1}"] = {
                    "id": f"doc_{i+1}",
                    "topic": topic.title(),
                    "content": f"(Error fetching content: {str(e)})",
                    "source": "(unknown)",
                    "retrieval_date": datetime.now().isoformat()
                }

        return documents

    def _extract_information(self, documents: Dict[str, Any], research_plan: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        extracted_info = {}
        for doc_id, document in documents.items():
            prompt = f"""
            Extract key information from the following document about {research_plan['airline_name']}.

            TOPIC: {document['topic']}

            CONTENT:
            {document['content']}

            Extract:
            - Key facts and data points
            - Strategic initiatives or statements
            - Notable leadership or ESG commentary
            """
            response = self.claude_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1000,
                system="You are an airline research assistant extracting key information from web documents.",
                messages=[{"role": "user", "content": prompt}]
            )

            bullets = [line.strip('- •* ') for line in response.content[0].text.splitlines() if line.strip().startswith(('-', '*', '•'))]
            extracted_info[doc_id] = [{"content": b, "topic": document['topic'], "source_document": doc_id} for b in bullets]
        return extracted_info

    def _synthesize_information(self, extracted_info: Dict[str, List[Dict[str, Any]]], research_plan: Dict[str, Any]) -> Dict[str, Any]:
        all_findings = []
        for findings in extracted_info.values():
            all_findings.extend([f["content"] for f in findings])

        joined_findings = "\n- " + "\n- ".join(all_findings)

        prompt = f"""
        Based on the following extracted findings, synthesize a report covering:
        1. Company Overview
        2. Strategic Focus Areas
        3. Notable Developments
        4. Risks or Challenges

        FINDINGS:
        {joined_findings}

        Write clearly, be as detailed as possible. reference as many of the specific details from the findings. Elaborate. Everything must be backed by evidence.
        """

        response = self.claude_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1500,
            system="You are a research analyst summarizing developments in the airline industry.",
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "summary": response.content[0].text.strip(),
            "creation_date": datetime.now().isoformat()
        }

    def _generate_insights(self, synthesis: Dict[str, Any], airline_name: str) -> Dict[str, Any]:
        prompt = f"""
        Based on this synthesis of {airline_name}, suggest 3–5 tailored strategic pitch angles
        a consulting or tech company could explore. Include the rationale for each.

        SYNTHESIS:
        {synthesis['summary']}
        """

        response = self.claude_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            system="You are a strategic advisor generating B2B pitch ideas.",
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "pitch_angles": response.content[0].text.strip(),
            "generation_date": datetime.now().isoformat()
        }

    def _format_citations(self, documents: Dict[str, Any]) -> List[str]:
        return [f"{doc['source']} ({doc['retrieval_date']}) - {doc['topic']}" for doc in documents.values()]

    def _create_final_report(self) -> Dict[str, Any]:
        return {
            "airline": self.research_plan.get("airline_name", "Unknown"),
            "executive_summary": self.synthesis.get("summary", ""),
            "pitch_recommendations": self.insights.get("pitch_angles", ""),
            "citations": self.citations,
            "metadata": {
                "research_date": datetime.now().isoformat(),
                "topics": self.research_plan.get("focus_areas", []),
                "document_count": len(self.documents)
            }
        }


def main():
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not anthropic_key or not openai_key:
        print("Missing API keys in .env")
        return

    agent = AirlineResearchAgent(anthropic_api_key=anthropic_key, openai_api_key=openai_key)

    airline_name = "Singapore Airlines"
    focus = ["investor relations", "press releases", "leadership", "sustainability", "strategic initiatives"]

    report = agent.analyze_airline(airline_name, focus)

    print("\n===== EXECUTIVE SUMMARY =====")
    print(report['executive_summary'][:1000] + "...")

    print("\n===== STRATEGIC PITCH ANGLES =====")
    print(report['pitch_recommendations'])

    with open(f"{airline_name.lower().replace(' ', '_')}_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Saved report to {airline_name.lower().replace(' ', '_')}_report.json")


if __name__ == "__main__":
    main()
