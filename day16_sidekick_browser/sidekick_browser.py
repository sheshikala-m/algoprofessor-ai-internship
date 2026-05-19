"""
M11: SidekickBrowser — Automated Data Research Agent
Uses LangGraph for orchestration + Playwright for browser automation
Day 15 | Phase 3 | AutoGen + Multi-Agent Analytics Week
"""

import asyncio
import json
from datetime import datetime
from typing import TypedDict, Annotated, List
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from playwright.async_api import async_playwright, Page, Browser

# ── State Schema ──────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    """Shared state that flows through the LangGraph pipeline."""
    query: str
    urls: List[str]
    raw_pages: List[dict]          # {url, title, text, tables}
    extracted_data: List[dict]     # cleaned, structured records
    summary: str
    report_path: str
    messages: Annotated[list, add_messages]
    error: str


# ── Node 1: URL Planner ───────────────────────────────────────────────────────

def plan_urls(state: ResearchState) -> ResearchState:
    """
    Given a research query, return a list of URLs to scrape.
    In production, swap this with an LLM call or a search-engine tool.
    """
    query = state["query"].lower()

    # Demo routing — map keyword → curated URLs
    url_map = {
        "stock":    ["https://finance.yahoo.com/trending-tickers/",
                     "https://en.wikipedia.org/wiki/Stock_market"],
        "climate":  ["https://en.wikipedia.org/wiki/Climate_change",
                     "https://climate.nasa.gov/"],
        "ai":       ["https://en.wikipedia.org/wiki/Artificial_intelligence",
                     "https://huggingface.co/models"],
        "covid":    ["https://en.wikipedia.org/wiki/COVID-19_pandemic",
                     "https://www.who.int/emergencies/diseases/novel-coronavirus-2019"],
    }

    urls = ["https://en.wikipedia.org/wiki/Data_science"]   # fallback
    for kw, kw_urls in url_map.items():
        if kw in query:
            urls = kw_urls
            break

    print(f"[Planner] Query='{state['query']}' → {len(urls)} URL(s) planned")
    return {**state, "urls": urls, "raw_pages": [], "extracted_data": []}


# ── Node 2: Browser Scraper (async Playwright) ────────────────────────────────

async def _scrape_url(page: Page, url: str) -> dict:
    """Navigate to a URL and extract title, visible text, and table data."""
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        title = await page.title()

        # Grab visible paragraph text (limit to first 3 000 chars)
        paragraphs = await page.locator("p").all_text_contents()
        text = " ".join(p.strip() for p in paragraphs if p.strip())[:3_000]

        # Extract any HTML tables
        tables = []
        tbl_elements = await page.locator("table").all()
        for tbl in tbl_elements[:3]:          # cap at 3 tables per page
            rows = await tbl.locator("tr").all()
            table_data = []
            for row in rows[:10]:             # first 10 rows
                cells = await row.locator("td,th").all_text_contents()
                if cells:
                    table_data.append(cells)
            if table_data:
                tables.append(table_data)

        return {"url": url, "title": title, "text": text, "tables": tables, "ok": True}

    except Exception as exc:
        print(f"  [Scraper] ⚠ Failed {url}: {exc}")
        return {"url": url, "title": "", "text": "", "tables": [], "ok": False}


async def _run_scraper(urls: List[str]) -> List[dict]:
    async with async_playwright() as pw:
        browser: Browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        results = []
        for url in urls:
            print(f"  [Scraper] Visiting {url} …")
            data = await _scrape_url(page, url)
            results.append(data)
        await browser.close()
        return results


def scrape_pages(state: ResearchState) -> ResearchState:
    """LangGraph node — synchronous wrapper around async Playwright scraper."""
    raw = asyncio.run(_run_scraper(state["urls"]))
    print(f"[Scraper] Scraped {len(raw)} page(s), "
          f"{sum(1 for r in raw if r['ok'])} succeeded")
    return {**state, "raw_pages": raw}


# ── Node 3: Data Extractor ────────────────────────────────────────────────────

def extract_data(state: ResearchState) -> ResearchState:
    """
    Transform raw scraped pages into structured records.
    In production, feed page text to an LLM extraction prompt.
    """
    records = []
    for page in state["raw_pages"]:
        if not page["ok"]:
            continue
        words = page["text"].split()
        record = {
            "source_url":   page["url"],
            "title":        page["title"],
            "word_count":   len(words),
            "snippet":      " ".join(words[:60]) + ("…" if len(words) > 60 else ""),
            "table_count":  len(page["tables"]),
            "scraped_at":   datetime.utcnow().isoformat(),
        }
        # Flatten first table (if any) for demo
        if page["tables"]:
            record["first_table_preview"] = page["tables"][0][:3]
        records.append(record)

    print(f"[Extractor] {len(records)} structured record(s) extracted")
    return {**state, "extracted_data": records}


# ── Node 4: Summariser ────────────────────────────────────────────────────────

def summarise(state: ResearchState) -> ResearchState:
    """Build a plain-text research summary from extracted records."""
    lines = [
        f"# SidekickBrowser Research Report",
        f"Query    : {state['query']}",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"Pages    : {len(state['extracted_data'])}",
        "",
        "## Page Summaries",
    ]
    for i, rec in enumerate(state["extracted_data"], 1):
        lines += [
            f"\n### {i}. {rec['title']}",
            f"URL      : {rec['source_url']}",
            f"Words    : {rec['word_count']} | Tables: {rec['table_count']}",
            f"Snippet  : {rec['snippet']}",
        ]
        if "first_table_preview" in rec:
            lines.append("Table preview:")
            for row in rec["first_table_preview"]:
                lines.append("  | " + " | ".join(str(c) for c in row))

    summary = "\n".join(lines)
    print("[Summariser] Summary built")
    return {**state, "summary": summary}


# ── Node 5: Report Writer ─────────────────────────────────────────────────────

def write_report(state: ResearchState) -> ResearchState:
    """Persist the summary + raw JSON to the outputs/ directory."""
    out_dir = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_txt = out_dir / f"report_{ts}.md"
    report_json = out_dir / f"data_{ts}.json"

    report_txt.write_text(state["summary"], encoding="utf-8")
    report_json.write_text(
        json.dumps(state["extracted_data"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[Writer] Report → {report_txt}")
    print(f"[Writer] Data   → {report_json}")
    return {**state, "report_path": str(report_txt)}


# ── Graph Assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(ResearchState)

    g.add_node("plan_urls",    plan_urls)
    g.add_node("scrape_pages", scrape_pages)
    g.add_node("extract_data", extract_data)
    g.add_node("summarise",    summarise)
    g.add_node("write_report", write_report)

    g.set_entry_point("plan_urls")
    g.add_edge("plan_urls",    "scrape_pages")
    g.add_edge("scrape_pages", "extract_data")
    g.add_edge("extract_data", "summarise")
    g.add_edge("summarise",    "write_report")
    g.add_edge("write_report", END)

    return g.compile()


# ── CLI Entry-point ───────────────────────────────────────────────────────────

def run(query: str = "AI research trends"):
    print(f"\n{'='*60}")
    print(f" SidekickBrowser  |  Query: {query}")
    print(f"{'='*60}\n")

    graph = build_graph()
    init_state: ResearchState = {
        "query":          query,
        "urls":           [],
        "raw_pages":      [],
        "extracted_data": [],
        "summary":        "",
        "report_path":    "",
        "messages":       [],
        "error":          "",
    }

    final = graph.invoke(init_state)

    print(f"\n{'='*60}")
    print(" REPORT PREVIEW")
    print("="*60)
    print(final["summary"][:1_200])
    print(f"\n✅ Full report saved to: {final['report_path']}")
    return final


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "AI machine learning"
    run(q)
