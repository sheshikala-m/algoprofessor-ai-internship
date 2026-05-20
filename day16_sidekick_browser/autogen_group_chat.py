"""
AutoGen Conversable Data Agents + Analytics Group Chat
Day 15 — Part of the D71–75 (May 17–21) multi-agent analytics week

Agents:
  ● PlannerAgent    — breaks research query into sub-tasks
  ● BrowserAgent    — calls SidekickBrowser to scrape data
  ● AnalystAgent    — runs basic analytics on extracted data
  ● ReporterAgent   — assembles final markdown report
  ● MetaAgent       — monitors quality, rebuilds strategy if needed

Uses: autogen-agentchat (open-source AutoGen v0.4 API)
"""

import json
import asyncio
from datetime import datetime
from typing import List, Dict
from pathlib import Path

# ─── Lightweight AutoGen-style base (works without API key for demo) ──────────
# Replace ConversableAgent with autogen_agentchat.agents.ConversableAgent
# when running with a real LLM backend.

class ConversableAgent:
    """Minimal stub — swap with `from autogen_agentchat.agents import ConversableAgent`."""
    def __init__(self, name: str, system_message: str = "", **kwargs):
        self.name = name
        self.system_message = system_message
        self._chat_log: List[Dict] = []

    def send(self, message: str, recipient: "ConversableAgent", request_reply: bool = True):
        self._chat_log.append({"from": self.name, "to": recipient.name, "msg": message})
        if request_reply:
            return recipient.receive(message, sender=self)

    def receive(self, message: str, sender: "ConversableAgent") -> str:
        reply = self._handle(message, sender.name)
        self._chat_log.append({"from": self.name, "to": sender.name, "msg": reply})
        return reply

    def _handle(self, message: str, from_agent: str) -> str:
        return f"[{self.name}] Received from {from_agent}: processed."


# ─── Specialised Agents ───────────────────────────────────────────────────────

class PlannerAgent(ConversableAgent):
    """Breaks a user query into ordered research tasks."""

    def __init__(self):
        super().__init__(
            name="PlannerAgent",
            system_message=(
                "You decompose data research queries into sequential tasks: "
                "1) URL list 2) scrape 3) extract 4) analyse 5) report."
            ),
        )

    def plan(self, query: str) -> List[str]:
        tasks = [
            f"TASK-1: Identify top URLs relevant to '{query}'",
            f"TASK-2: Scrape pages using SidekickBrowser for '{query}'",
            f"TASK-3: Extract structured data from scraped content",
            f"TASK-4: Run descriptive analytics on extracted records",
            f"TASK-5: Compile final markdown research report",
        ]
        print(f"[{self.name}] Planned {len(tasks)} tasks for: '{query}'")
        return tasks


class BrowserAgent(ConversableAgent):
    """Calls SidekickBrowser pipeline and returns extracted data."""

    def __init__(self):
        super().__init__(name="BrowserAgent",
                         system_message="You operate the SidekickBrowser scraper.")

    def scrape(self, query: str) -> List[Dict]:
        # Import and run SidekickBrowser graph
        try:
            from sidekick_browser import run as sb_run
            result = sb_run(query)
            return result.get("extracted_data", [])
        except Exception as exc:
            print(f"[{self.name}] SidekickBrowser error: {exc} — returning mock data")
            return self._mock_data(query)

    def _mock_data(self, query: str) -> List[Dict]:
        return [
            {
                "source_url": f"https://example.com/{query.replace(' ','_')}",
                "title": f"Research: {query}",
                "word_count": 1_200,
                "snippet": f"This article covers {query} in depth ...",
                "table_count": 2,
                "scraped_at": datetime.utcnow().isoformat(),
            }
        ]


class AnalystAgent(ConversableAgent):
    """Performs lightweight analytics over extracted records."""

    def __init__(self):
        super().__init__(name="AnalystAgent",
                         system_message="You compute descriptive stats on scraped data.")

    def analyse(self, records: List[Dict]) -> Dict:
        if not records:
            return {"error": "No records to analyse"}

        word_counts = [r.get("word_count", 0) for r in records]
        table_counts = [r.get("table_count", 0) for r in records]

        stats = {
            "total_pages":       len(records),
            "total_words":       sum(word_counts),
            "avg_words_per_page": round(sum(word_counts) / len(word_counts), 1),
            "max_words":         max(word_counts),
            "min_words":         min(word_counts),
            "total_tables":      sum(table_counts),
            "pages_with_tables": sum(1 for t in table_counts if t > 0),
            "sources":           [r.get("source_url", "") for r in records],
        }
        print(f"[{self.name}] Analytics - > {stats['total_pages']} pages, "
              f"{stats['total_words']} words, {stats['total_tables']} tables")
        return stats


class ReporterAgent(ConversableAgent):
    """Assembles the final markdown research report."""

    def __init__(self):
        super().__init__(name="ReporterAgent",
                         system_message="You write polished research reports.")

    def compile(self, query: str, records: List[Dict], stats: Dict) -> str:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            "# SidekickBrowser Multi-Agent Research Report",
            f"**Query**: {query}  |  **Generated**: {ts}",
            "",
            "## Analytics Summary",
            f"| Metric | Value |",
            f"|--------|-------|",
        ]
        for k, v in stats.items():
            if k != "sources":
                lines.append(f"| {k.replace('_',' ').title()} | {v} |")

        lines += ["", "## Sources", ""]
        for src in stats.get("sources", []):
            lines.append(f"- {src}")

        lines += ["", "## Page Details", ""]
        for i, rec in enumerate(records, 1):
            lines += [
                f"### {i}. {rec.get('title','—')}",
                f"- **URL**: {rec.get('source_url','')}",
                f"- **Words**: {rec.get('word_count',0)}  |  "
                f"**Tables**: {rec.get('table_count',0)}",
                f"- **Snippet**: {rec.get('snippet','')[:200]}",
                "",
            ]

        return "\n".join(lines)


class MetaAgent(ConversableAgent):
    """
    Monitors output quality; triggers replanning if report is too thin.
    Implements the 'adaptive analysis strategy' from D71-75 spec.
    """

    def __init__(self):
        super().__init__(name="MetaAgent",
                         system_message="You evaluate research quality and adapt strategy.")
        self.min_pages = 1
        self.min_words = 200

    def evaluate(self, stats: Dict) -> Dict:
        issues = []
        if stats.get("total_pages", 0) < self.min_pages:
            issues.append("Too few pages scraped")
        if stats.get("total_words", 0) < self.min_words:
            issues.append("Insufficient word count")

        verdict = {
            "passed":     len(issues) == 0,
            "issues":     issues,
            "action":     "DONE" if not issues else "REPLAN",
            "confidence": "HIGH" if not issues else "LOW",
        }
        print(f"[{self.name}] QA verdict: {verdict['action']} | "
              f"issues={verdict['issues']}")
        return verdict


# ─── Group Chat Orchestrator ──────────────────────────────────────────────────

class AnalyticsGroupChat:
    """
    Coordinates all agents in a sequential group-chat pattern.
    Supports one replanning loop via MetaAgent feedback.
    """

    def __init__(self):
        self.planner  = PlannerAgent()
        self.browser  = BrowserAgent()
        self.analyst  = AnalystAgent()
        self.reporter = ReporterAgent()
        self.meta     = MetaAgent()

    def run(self, query: str) -> str:
        print(f"\n{'='*62}")
        print(f" AutoGen Group Chat  |  Query: {query}")
        print(f"{'='*62}")

        # Round 1
        tasks   = self.planner.plan(query)
        records = self.browser.scrape(query)
        stats   = self.analyst.analyse(records)
        verdict = self.meta.evaluate(stats)

        # Adaptive replan if needed (MetaAgent strategy)
        if not verdict["passed"]:
            print(f"\n[GroupChat] MetaAgent triggered REPLAN — broadening query …")
            query   = query + " overview data"
            records = self.browser.scrape(query)
            stats   = self.analyst.analyse(records)

        report  = self.reporter.compile(query, records, stats)

        # Save report
        out_dir = Path(__file__).parent.parent / "outputs"
        out_dir.mkdir(exist_ok=True)
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"groupchat_report_{ts}.md"
        path.write_text(report, encoding="utf-8")

        print(f"\n[GroupChat]  Report saved - > {path}")
        print("\n--- REPORT PREVIEW (first 800 chars) ---")
        print(report[:800])
        return report


# ─── Entry-point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "climate change data"
    chat  = AnalyticsGroupChat()
    chat.run(query)
