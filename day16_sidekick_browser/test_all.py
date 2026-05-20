"""
Tests for Day 15 — SidekickBrowser Multi-Agent System
Run: python -m pytest tests/ -v
"""

import sys
import asyncio
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ─── SidekickBrowser graph ────────────────────────────────────────────────────

class TestSidekickBrowser:
    def test_plan_urls_known_keyword(self):
        from sidekick_browser import plan_urls
        state = {"query": "stock market trends", "urls": [], "raw_pages": [],
                 "extracted_data": [], "summary": "", "report_path": "",
                 "messages": [], "error": ""}
        result = plan_urls(state)
        assert len(result["urls"]) >= 1
        assert all(u.startswith("http") for u in result["urls"])

    def test_plan_urls_fallback(self):
        from sidekick_browser import plan_urls
        state = {"query": "random topic xyz", "urls": [], "raw_pages": [],
                 "extracted_data": [], "summary": "", "report_path": "",
                 "messages": [], "error": ""}
        result = plan_urls(state)
        assert len(result["urls"]) >= 1

    def test_extract_data_empty(self):
        from sidekick_browser import extract_data
        state = {"query": "test", "urls": [], "raw_pages": [], "extracted_data": [],
                 "summary": "", "report_path": "", "messages": [], "error": ""}
        result = extract_data(state)
        assert result["extracted_data"] == []

    def test_extract_data_failed_page(self):
        from sidekick_browser import extract_data
        state = {
            "query": "test", "urls": [], "extracted_data": [], "summary": "",
            "report_path": "", "messages": [], "error": "",
            "raw_pages": [{"url": "x", "title": "", "text": "",
                           "tables": [], "ok": False}],
        }
        result = extract_data(state)
        assert result["extracted_data"] == []

    def test_extract_data_good_page(self):
        from sidekick_browser import extract_data
        state = {
            "query": "test", "urls": [], "summary": "", "report_path": "",
            "messages": [], "error": "",
            "raw_pages": [{"url": "https://example.com", "title": "Example",
                           "text": "Hello " * 100, "tables": [], "ok": True}],
            "extracted_data": [],
        }
        result = extract_data(state)
        assert len(result["extracted_data"]) == 1
        assert result["extracted_data"][0]["word_count"] == 100

    def test_summarise(self):
        from sidekick_browser import summarise
        state = {
            "query": "AI trends", "urls": [], "raw_pages": [], "summary": "",
            "report_path": "", "messages": [], "error": "",
            "extracted_data": [{
                "source_url": "https://x.com", "title": "X",
                "word_count": 500, "snippet": "abc", "table_count": 0,
                "scraped_at": "2025-01-01",
            }],
        }
        result = summarise(state)
        assert "SidekickBrowser" in result["summary"]
        assert "AI trends" in result["summary"]


# ─── AutoGen Group Chat ───────────────────────────────────────────────────────

class TestAutoGenGroupChat:
    def test_planner_generates_tasks(self):
        from autogen_group_chat import PlannerAgent
        agent = PlannerAgent()
        tasks = agent.plan("climate change")
        assert len(tasks) == 5
        assert all("TASK-" in t for t in tasks)

    def test_analyst_empty_records(self):
        from autogen_group_chat import AnalystAgent
        agent  = AnalystAgent()
        result = agent.analyse([])
        assert "error" in result

    def test_analyst_with_records(self):
        from autogen_group_chat import AnalystAgent
        agent = AnalystAgent()
        recs  = [
            {"word_count": 1000, "table_count": 2, "source_url": "https://a.com"},
            {"word_count":  500, "table_count": 0, "source_url": "https://b.com"},
        ]
        stats = agent.analyse(recs)
        assert stats["total_pages"] == 2
        assert stats["total_words"] == 1500
        assert stats["avg_words_per_page"] == 750.0

    def test_meta_agent_pass(self):
        from autogen_group_chat import MetaAgent
        agent   = MetaAgent()
        verdict = agent.evaluate({"total_pages": 2, "total_words": 800})
        assert verdict["passed"] is True
        assert verdict["action"] == "DONE"

    def test_meta_agent_fail(self):
        from autogen_group_chat import MetaAgent
        agent   = MetaAgent()
        verdict = agent.evaluate({"total_pages": 0, "total_words": 10})
        assert verdict["passed"] is False
        assert verdict["action"] == "REPLAN"

    def test_reporter_compiles(self):
        from autogen_group_chat import ReporterAgent
        agent  = ReporterAgent()
        recs   = [{"title": "T", "source_url": "https://x.com",
                   "word_count": 100, "table_count": 0, "snippet": "abc"}]
        stats  = {"total_pages": 1, "total_words": 100,
                  "avg_words_per_page": 100, "max_words": 100,
                  "min_words": 100, "total_tables": 0,
                  "pages_with_tables": 0, "sources": ["https://x.com"]}
        report = agent.compile("test query", recs, stats)
        assert "SidekickBrowser" in report
        assert "test query" in report


# ─── A2A Messaging ────────────────────────────────────────────────────────────

class TestA2AMessaging:
    def test_agent_message_round_trip(self):
        from a2a_messaging import AgentMessage
        msg  = AgentMessage(topic="test", sender="AgentA",
                            payload={"key": "value"})
        raw  = msg.to_json()
        msg2 = AgentMessage.from_json(raw)
        assert msg2.topic   == "test"
        assert msg2.sender  == "AgentA"
        assert msg2.payload == {"key": "value"}

    def test_get_broker_inprocess(self):
        from a2a_messaging import get_broker, InProcessBroker
        b = get_broker("inprocess")
        assert isinstance(b, InProcessBroker)

    def test_get_broker_invalid(self):
        from a2a_messaging import get_broker
        with pytest.raises(ValueError):
            get_broker("unknown_backend")

    def test_inprocess_pub_sub(self):
        from a2a_messaging import get_broker

        async def _run():
            broker = get_broker("inprocess")
            await broker.publish("events", {"x": 1}, sender="A")
            async for msg in broker.subscribe("events"):
                assert msg.payload == {"x": 1}
                break
            await broker.close()

        asyncio.run(_run())


# ─── Time Series Agent ────────────────────────────────────────────────────────

class TestTimeSeriesAgent:
    def test_loader_generates_correct_length(self):
        from timeseries_agent import TimeSeriesLoaderAgent
        agent = TimeSeriesLoaderAgent()
        df    = agent.generate_synthetic(n_days=100)
        assert len(df) == 100
        assert list(df.columns) == ["ds", "y"]

    def test_preprocessor_split(self):
        from timeseries_agent import TimeSeriesLoaderAgent, PreprocessingAgent
        loader = TimeSeriesLoaderAgent()
        df     = loader.generate_synthetic(n_days=100)
        prep   = PreprocessingAgent()
        data   = prep.process(df)
        assert len(data["train"]) + len(data["test"]) == len(data["df"])
        assert "rolling_7" in data["df"].columns

    def test_arima_forecast_shape(self):
        from timeseries_agent import TimeSeriesLoaderAgent, PreprocessingAgent, ForecastingAgent
        loader    = TimeSeriesLoaderAgent()
        df        = loader.generate_synthetic(n_days=120)
        data      = PreprocessingAgent().process(df)
        forecaster = ForecastingAgent()
        fc        = forecaster._arima_forecast(data["train"], horizon=10)
        assert len(fc) == 10

    def test_evaluation_metrics(self):
        from timeseries_agent import EvaluationAgent
        import pandas as pd, numpy as np
        test = pd.DataFrame({
            "ds": pd.date_range("2024-01-01", periods=10),
            "y":  np.ones(10) * 100,
        })
        fc = {"ARIMA": np.ones(10) * 95}
        ev = EvaluationAgent()
        m  = ev.evaluate(test, fc)
        assert "ARIMA" in m
        assert m["ARIMA"]["RMSE"] == pytest.approx(5.0, rel=0.01)
        assert m["ARIMA"]["MAE"]  == pytest.approx(5.0, rel=0.01)
