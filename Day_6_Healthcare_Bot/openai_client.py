# ============================================================
# OPENAI CLIENT
# Day 6 Healthcare Bot: DataAssist Analytics Agent
# Author: Sheshikala
# Topic: OpenAI API client for data analytics tasks
# ============================================================

# OpenAI provides GPT models through a REST API. For data
# analytics this is useful because GPT-4 and GPT-3.5-turbo
# can read summarized dataframes and generate professional
# insight text, SQL queries, and structured reports. This
# file wraps the OpenAI API in a client class that falls back
# to deterministic mock responses when no API key is set so
# all downstream files run without any credentials. The
# OPENAI_API_KEY environment variable is read at startup.
# If you have a key, set it with:
#   Windows : set OPENAI_API_KEY=sk-...
#   Linux   : export OPENAI_API_KEY=sk-...

import os
import json
import urllib.request
import urllib.error

from data_loader import load_titanic, summarize_dataframe

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL   = "gpt-3.5-turbo"


# ============================================================
# SYSTEM PROMPTS
# ============================================================

ANALYST_SYSTEM_PROMPT = (
    "You are a senior data analyst specializing in passenger and "
    "transportation data. You provide clear, accurate, and concise "
    "analysis based only on the data provided. You always cite "
    "specific numbers when making claims."
)

SQL_SYSTEM_PROMPT = (
    "You are a SQL expert. When given a data analysis question and "
    "a table schema, you write clean, efficient SQL queries. "
    "Always include comments explaining each clause."
)

REPORT_SYSTEM_PROMPT = (
    "You are a professional report writer for data analytics teams. "
    "You write formal, well-structured report sections in 3-4 sentences. "
    "You always reference specific statistics from the provided data."
)


# ============================================================
# OPENAI CLIENT CLASS
# ============================================================

class OpenAIClient:
    """
    Thin wrapper around the OpenAI Chat Completions API.
    Automatically falls back to mock responses when the
    OPENAI_API_KEY environment variable is not set.

    Supported methods:
        chat()     - multi-turn conversation
        complete() - single-turn completion
    """

    def __init__(self, api_key=None, model=DEFAULT_MODEL):
        self.api_key = api_key or OPENAI_API_KEY
        self.model   = model

        if self.api_key:
            print("OpenAI client ready. Model: " + self.model)
        else:
            print("OPENAI_API_KEY not set. Using mock mode.")
            print("Set key: set OPENAI_API_KEY=sk-your-key-here")

    def chat(self, messages, temperature=0.3, max_tokens=500):
        """
        Sends a list of messages to the OpenAI chat completions
        endpoint and returns the assistant reply as a string.

        Parameters:
            messages    : list of dicts with role and content keys
            temperature : float 0.0-1.0, lower means more focused
            max_tokens  : int, maximum tokens in the response

        Returns:
            string response from the model
        """
        if not self.api_key:
            last = messages[-1]["content"] if messages else ""
            return self._mock_response(last)

        try:
            payload = json.dumps({
                "model"      : self.model,
                "messages"   : messages,
                "temperature": temperature,
                "max_tokens" : max_tokens
            }).encode("utf-8")

            request = urllib.request.Request(
                OPENAI_BASE_URL + "/chat/completions",
                data    = payload,
                headers = {
                    "Content-Type" : "application/json",
                    "Authorization": "Bearer " + self.api_key
                },
                method = "POST"
            )
            response = urllib.request.urlopen(request, timeout=30)
            data     = json.loads(response.read())
            return data["choices"][0]["message"]["content"].strip()

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            print("OpenAI HTTP error " + str(e.code) + ": " + error_body[:200])
            last = messages[-1]["content"] if messages else ""
            return self._mock_response(last)
        except Exception as e:
            print("OpenAI request failed: " + str(e))
            last = messages[-1]["content"] if messages else ""
            return self._mock_response(last)

    def complete(self, prompt, system_prompt=None, temperature=0.3, max_tokens=500):
        """
        Sends a single user prompt with an optional system prompt.
        Builds the messages list internally and calls chat().

        Parameters:
            prompt        : string, the user message
            system_prompt : string or None, sets the assistant role
            temperature   : float 0.0-1.0
            max_tokens    : int

        Returns:
            string response from the model
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    def _mock_response(self, prompt):
        """
        Returns a relevant mock response based on keywords in
        the prompt. Used when no API key is configured.
        """
        prompt_lower = prompt.lower()
        tag = "[GPT mock] "

        if any(w in prompt_lower for w in ["summarize", "summary", "overview"]):
            return (
                tag + "The Titanic dataset contains 891 passenger records "
                "from the 1912 disaster. The overall survival rate was 38.4 "
                "percent. Gender and passenger class were the two strongest "
                "predictors of survival with females surviving at 74.2 percent "
                "compared to 18.9 percent for males."
            )
        elif any(w in prompt_lower for w in ["sql", "query", "database"]):
            return (
                tag + "SELECT sex, pclass, "
                "ROUND(AVG(survived) * 100, 1) AS survival_rate, "
                "COUNT(*) AS passenger_count "
                "FROM titanic "
                "GROUP BY sex, pclass "
                "ORDER BY survival_rate DESC;"
            )
        elif any(w in prompt_lower for w in ["report", "findings", "section"]):
            return (
                tag + "Executive Summary: Analysis of 891 Titanic passengers "
                "reveals that survival was strongly influenced by gender and "
                "socioeconomic class. Female passengers had a 74.2 percent "
                "survival rate while males had only 18.9 percent. First class "
                "passengers survived at 63 percent versus 24 percent in third "
                "class, indicating that evacuation priority reflected the "
                "class structure of the era."
            )
        elif any(w in prompt_lower for w in ["recommend", "suggest", "improve"]):
            return (
                tag + "Three recommendations: First, implement equal evacuation "
                "procedures regardless of ticket class. Second, ensure lifeboat "
                "capacity equals total passenger count. Third, conduct mandatory "
                "safety briefings within 12 hours of departure for all passengers."
            )
        elif any(w in prompt_lower for w in ["compare", "difference", "versus"]):
            return (
                tag + "Female passengers survived at 74.2 percent compared to "
                "18.9 percent for males, a difference of 55.3 percentage points. "
                "First class passengers survived at 63.0 percent compared to "
                "24.2 percent in third class, a gap of 38.8 percentage points."
            )
        elif any(w in prompt_lower for w in ["predict", "would", "survive", "chances"]):
            return (
                tag + "Based on historical survival patterns, a first-class "
                "female passenger had approximately 96.5 percent survival "
                "probability. A third-class male passenger had approximately "
                "13.5 percent survival probability. Gender and class are the "
                "two strongest combined predictors."
            )
        else:
            return (
                tag + "Response for: " + prompt[:80] + "... "
                "Set OPENAI_API_KEY environment variable to get real GPT responses."
            )


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def analyze_data(client, data_summary, question):
    """
    Sends a data analysis question with dataset context to OpenAI.

    Parameters:
        client       : OpenAIClient instance
        data_summary : string, output of summarize_dataframe()
        question     : string, the analytical question

    Returns:
        string response
    """
    prompt = (
        "Dataset Summary:\n" + data_summary + "\n\n"
        "Question: " + question + "\n\n"
        "Provide a clear, data-driven answer."
    )
    return client.complete(prompt, system_prompt=ANALYST_SYSTEM_PROMPT)


def generate_sql(client, question, table_schema):
    """
    Asks OpenAI to generate a SQL query for the given question.

    Parameters:
        client       : OpenAIClient instance
        question     : string, what the query should answer
        table_schema : string, table name and column descriptions

    Returns:
        string containing the SQL query
    """
    prompt = (
        "Table schema:\n" + table_schema + "\n\n"
        "Write a SQL query to answer: " + question
    )
    return client.complete(prompt, system_prompt=SQL_SYSTEM_PROMPT)


def write_report_section(client, data_summary, section_title):
    """
    Generates a formal report section from data summary.

    Parameters:
        client        : OpenAIClient instance
        data_summary  : string, output of summarize_dataframe()
        section_title : string, e.g. "Executive Summary"

    Returns:
        string containing the report section text
    """
    prompt = (
        "Write a report section titled '" + section_title + "' "
        "using the following data:\n\n" + data_summary
    )
    return client.complete(prompt, system_prompt=REPORT_SYSTEM_PROMPT)


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("OPENAI CLIENT DEMO")
    print("=" * 55)

    client  = OpenAIClient()
    df      = load_titanic()
    summary = summarize_dataframe(df)

    print("\n-- Dataset Summary --")
    print(summary)

    print("\n-- Data Analysis Query --")
    response = analyze_data(
        client, summary,
        "Which passenger group had the highest survival rate?"
    )
    print(response)

    print("\n-- SQL Generation --")
    schema = (
        "Table: titanic\n"
        "Columns: survived INT, pclass INT, sex VARCHAR, "
        "age FLOAT, fare FLOAT, embarked VARCHAR"
    )
    sql = generate_sql(
        client,
        "Find survival rate by sex and passenger class ordered by survival rate",
        schema
    )
    print(sql)

    print("\n-- Report Section Generation --")
    for section in ["Executive Summary", "Key Findings", "Recommendations"]:
        print("\n" + section + ":")
        text = write_report_section(client, summary, section)
        print(text)

    print("\n-- Multi-Turn Chat --")
    messages = [
        {"role": "system",    "content": ANALYST_SYSTEM_PROMPT},
        {"role": "user",      "content": "What was the overall survival rate on the Titanic?"},
        {"role": "assistant", "content": "The overall survival rate was 38.4 percent based on 891 passenger records."},
        {"role": "user",      "content": "How did this differ between passenger classes?"}
    ]
    chat_response = client.chat(messages)
    print("Response: " + chat_response)

    print("\n-- OpenAI client demo complete --")


if __name__ == "__main__":
    run_demo()
