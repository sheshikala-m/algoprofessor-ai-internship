# SidekickBrowser AI

SidekickBrowser AI is a multi-agent AI-powered research automation and forecasting system built using Python, Playwright, LangGraph, and AutoGen. The project combines intelligent web scraping, AI-agent collaboration, forecasting pipelines, and automated report generation.

---

# Project Overview

This project demonstrates:

- Multi-Agent AI Systems
- AI-powered Web Scraping
- Automated Research Pipelines
- Forecasting & Time Series Analysis
- Agent-to-Agent Communication
- AI-driven Report Generation

The system uses multiple AI-inspired agents that collaborate together to scrape websites, extract information, analyze content, generate reports, and create forecasting visualizations automatically.

---

# Core Features

## PlannerAgent
Handles workflow planning:

- accepts user queries
- selects websites
- manages research flow

---

## ScraperAgent
Runs automated scraping:

- visits websites using Playwright
- extracts webpage content
- collects structured text

---

## AnalystAgent
Handles analytics workflows:

- analyzes extracted content
- computes metrics
- generates summaries

---

## MetaAgent
Performs quality checks:

- validates outputs
- verifies reports
- checks workflow status

---

## ReportWriter
Generates final outputs:

- markdown reports
- JSON summaries
- forecasting outputs

---

# Forecasting Features

## Time Series Forecasting

Used for:

- trend prediction
- analytics visualization
- forecasting workflows

The system automatically:

- generates forecast charts
- exports PNG visualizations
- saves analytics metrics

---

# Technologies Used

## Languages

- Python

## AI & Automation

- LangGraph
- AutoGen
- Playwright

## Data & Visualization

- Pandas
- NumPy
- Matplotlib

---

# Project Structure

```text
sidekickbrowser_ai/
│
├── sidekick_browser.py
├── autogen_group_chat.py
├── a2a_messaging.py
├── timeseries_agent.py
├── test_all.py
├── requirements.txt
├── README.md
│
├── outputs/
│   ├── report_xxx.md
│   ├── data_xxx.json
│   ├── forecast_xxx.png
│   ├── metrics_xxx.json
│   └── groupchat_report_xxx.md
```

---

# Setup Instructions

## Install Dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

---

# Run Project

## SidekickBrowser Scraper

```bash
python sidekick_browser.py "AI trends"
```

## AutoGen Group Chat

```bash
python autogen_group_chat.py "climate data"
```

## A2A Messaging

```bash
python a2a_messaging.py
```

## Time Series Forecast

```bash
python timeseries_agent.py
```

---

# Outputs

The system generates:

- Markdown reports
- JSON structured data
- Forecast charts
- Analytics metrics

Generated outputs are stored inside:

```text
outputs/
```

---

# Key Learnings

This project helped in understanding:

- Multi-agent AI workflows
- AI automation systems
- Web scraping pipelines
- Forecasting workflows
- Report generation systems
- Agent orchestration

---

# Future Improvements

- Streamlit dashboard
- LLM integration
- Docker support
- Cloud deployment
- Real-time analytics

---

# Internship & Learning Context

Built as part of:

- AI & Data Science Internship
- Multi-Agent Systems Practice
- Agentic AI Workflow Learning
