# ============================================================
# OLLAMA SETUP
# Day 5 NLP: DataAssist Analytics Agent
# Author: Sheshikala
# Topic: Setup Ollama client for Llama3 and Mistral local LLMs
# ============================================================

# Ollama is a tool that lets you run large language models
# locally on your own machine without any API costs or internet
# connection. You install Ollama, pull a model like llama3 or
# mistral, and then call it through a local HTTP server at
# http://localhost:11434. This file sets up the client class
# that all other files in this folder use to talk to Ollama.
# If Ollama is not installed or not running, the client
# automatically falls back to deterministic mock responses so
# every file still runs and produces useful output.

import json
import time
import urllib.request
import urllib.error

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "llama3"


# ============================================================
# OLLAMA CLIENT CLASS
# ============================================================

class OllamaClient:
    """
    HTTP client for the Ollama local LLM server.

    Supports:
      - Single-turn text generation via /api/generate
      - Multi-turn chat via /api/chat
      - Model listing via /api/tags
      - Automatic mock fallback when server is unavailable

    Setup instructions (run once):
      1. Download Ollama from https://ollama.ai
      2. Install and open Ollama
      3. Run in terminal: ollama pull llama3
      4. Run in terminal: ollama pull mistral
      5. Ollama server starts automatically on port 11434
    """

    def __init__(self, base_url=OLLAMA_BASE_URL):
        self.base_url  = base_url
        self.available = self._check_connection()

        if self.available:
            print("Ollama server is running at " + base_url)
            models = self.list_models()
            if models:
                print("Pulled models: " + ", ".join(models))
            else:
                print("No models pulled yet. Run: ollama pull llama3")
        else:
            print("Ollama server not detected at " + base_url)
            print("All responses will use mock mode.")
            print("To use real models:")
            print("  1. Install Ollama from https://ollama.ai")
            print("  2. Run: ollama pull llama3")
            print("  3. Run: ollama pull mistral")

    def _check_connection(self):
        """
        Sends a GET request to /api/tags to check if Ollama
        is running. Returns True if reachable, False otherwise.
        """
        try:
            response = urllib.request.urlopen(
                self.base_url + "/api/tags",
                timeout=3
            )
            return response.status == 200
        except Exception:
            return False

    def list_models(self):
        """
        Returns a list of model names that have been pulled
        to the local machine. Returns empty list on error.
        """
        if not self.available:
            return ["llama3 (mock)", "mistral (mock)"]
        try:
            response = urllib.request.urlopen(
                self.base_url + "/api/tags"
            )
            data = json.loads(response.read())
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            print("Error listing models: " + str(e))
            return []

    def generate(self, model, prompt):
        """
        Sends a single prompt to the specified model and returns
        the response text. Uses /api/generate endpoint with
        stream=False so the full response is returned at once.

        Parameters:
            model  : model name e.g. "llama3" or "mistral"
            prompt : the text prompt to send

        Returns:
            response string from the model
        """
        if not self.available:
            return self._mock_response(model, prompt)

        try:
            payload = json.dumps({
                "model" : model,
                "prompt": prompt,
                "stream": False
            }).encode("utf-8")

            request = urllib.request.Request(
                self.base_url + "/api/generate",
                data    = payload,
                headers = {"Content-Type": "application/json"},
                method  = "POST"
            )
            response = urllib.request.urlopen(request, timeout=120)
            data     = json.loads(response.read())
            return data.get("response", "No response returned by model.")

        except urllib.error.URLError as e:
            print("Connection error during generate: " + str(e))
            return self._mock_response(model, prompt)
        except Exception as e:
            print("Unexpected error during generate: " + str(e))
            return self._mock_response(model, prompt)

    def chat(self, model, messages):
        """
        Sends a list of messages to the model for multi-turn
        conversation. Uses /api/chat endpoint.

        Parameters:
            model    : model name e.g. "llama3" or "mistral"
            messages : list of dicts with "role" and "content" keys
                       e.g. [{"role": "user", "content": "Hello"}]

        Returns:
            response string from the model
        """
        if not self.available:
            last_message = messages[-1]["content"] if messages else ""
            return self._mock_response(model, last_message)

        try:
            payload = json.dumps({
                "model"   : model,
                "messages": messages,
                "stream"  : False
            }).encode("utf-8")

            request = urllib.request.Request(
                self.base_url + "/api/chat",
                data    = payload,
                headers = {"Content-Type": "application/json"},
                method  = "POST"
            )
            response = urllib.request.urlopen(request, timeout=120)
            data     = json.loads(response.read())
            return data.get("message", {}).get("content", "No response returned.")

        except urllib.error.URLError as e:
            print("Connection error during chat: " + str(e))
            last_message = messages[-1]["content"] if messages else ""
            return self._mock_response(model, last_message)
        except Exception as e:
            print("Unexpected error during chat: " + str(e))
            last_message = messages[-1]["content"] if messages else ""
            return self._mock_response(model, last_message)

    def _mock_response(self, model, prompt):
        """
        Returns a deterministic mock response when Ollama is
        not available. The response content is based on keywords
        in the prompt so output is always relevant and useful
        for testing the downstream pipeline logic.
        """
        prompt_lower = prompt.lower()
        tag = "[" + model + " mock] "

        if any(w in prompt_lower for w in ["summarize", "summary", "overview"]):
            return (
                tag + "The Titanic dataset contains 891 passenger records with "
                "features including passenger class, sex, age, fare, and survival "
                "status. The overall survival rate was 38.4 percent. Female "
                "passengers survived at a rate of 74.2 percent compared to 18.9 "
                "percent for male passengers, making gender the strongest predictor."
            )
        elif any(w in prompt_lower for w in ["analyze", "analysis", "insight", "pattern"]):
            return (
                tag + "Three key insights from the data: First, passenger class "
                "strongly predicted survival with first class at 63 percent, second "
                "class at 47 percent, and third class at 24 percent. Second, gender "
                "was the most significant factor with females four times more likely "
                "to survive than males. Third, children under 10 had higher survival "
                "rates than adult males regardless of passenger class."
            )
        elif any(w in prompt_lower for w in ["recommend", "recommendation", "action"]):
            return (
                tag + "Three recommendations based on the analysis: First, prioritize "
                "evacuation protocols that account for all passenger classes equally "
                "rather than socioeconomic status. Second, ensure lifeboat capacity "
                "matches total passenger count plus crew. Third, implement mandatory "
                "safety drills for all passengers within the first 24 hours of departure."
            )
        elif any(w in prompt_lower for w in ["predict", "survival", "classify"]):
            return (
                tag + "Based on historical Titanic survival patterns, the key factors "
                "in order of importance are: gender (female significantly higher), "
                "passenger class (first class significantly higher), age (children "
                "slightly favored), and fare paid (higher fare correlates with survival). "
                "A first-class female passenger had approximately 97 percent survival rate."
            )
        elif any(w in prompt_lower for w in ["explain", "what is", "describe", "define"]):
            return (
                tag + "The Titanic was a British ocean liner that sank on April 15, 1912 "
                "after colliding with an iceberg. The dataset records 891 of the 2224 "
                "passengers and crew. It is widely used in machine learning as a binary "
                "classification benchmark because it contains a mix of numeric and "
                "categorical features with real-world survival outcomes."
            )
        elif any(w in prompt_lower for w in ["compare", "difference", "versus", "vs"]):
            return (
                tag + "Comparison results: Male passengers had a survival rate of 18.9 "
                "percent while female passengers had 74.2 percent, a difference of 55.3 "
                "percentage points. First class passengers survived at 63 percent versus "
                "24 percent for third class, a gap of 39 percentage points. These "
                "differences are statistically significant and reflect the evacuation "
                "priorities of the era."
            )
        else:
            return (
                tag + "Response to query about Titanic passenger data analysis. "
                "The dataset provides valuable insights into survival factors "
                "including socioeconomic status, gender, and age. Statistical "
                "analysis confirms that gender and passenger class were the two "
                "strongest predictors of survival in the 1912 disaster."
            )


# ============================================================
# MODEL BENCHMARK UTILITY
# ============================================================

def benchmark_models(client, prompt, models=None):
    """
    Runs the same prompt on each model in the list and records
    response time and response length for comparison. Useful for
    deciding which model to use for a specific type of task.

    Parameters:
        client : OllamaClient instance
        prompt : the prompt to test
        models : list of model names to benchmark

    Returns:
        list of result dicts with model, time, length, response
    """
    if models is None:
        models = ["llama3", "mistral"]

    print("\nModel Benchmark")
    print("Prompt: " + prompt[:70] + "...")
    print("-" * 55)

    results = []
    for model in models:
        start    = time.time()
        response = client.generate(model, prompt)
        elapsed  = round(time.time() - start, 3)

        result = {
            "model"   : model,
            "response": response,
            "time_s"  : elapsed,
            "length"  : len(response)
        }
        results.append(result)

        print("\nModel    : " + model)
        print("Time     : " + str(elapsed) + "s")
        print("Length   : " + str(len(response)) + " characters")
        print("Response : " + response[:200])

    if len(results) == 2:
        faster = min(results, key=lambda x: x["time_s"])
        longer = max(results, key=lambda x: x["length"])
        print("\nFastest  : " + faster["model"] + " (" + str(faster["time_s"]) + "s)")
        print("Longest  : " + longer["model"] + " (" + str(longer["length"]) + " chars)")

    return results


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("OLLAMA SETUP DEMO")
    print("=" * 55)

    client = OllamaClient()

    print("\n-- Available Models --")
    models = client.list_models()
    for m in models:
        print("  " + m)

    print("\n-- Single Generation Test --")
    prompt   = "Summarize the Titanic dataset in two sentences for a data analytics report."
    response = client.generate(DEFAULT_MODEL, prompt)
    print("Prompt  : " + prompt)
    print("Response: " + response)

    print("\n-- Multi-Turn Chat Test --")
    messages = [
        {"role": "user",      "content": "What is the Titanic dataset?"},
        {"role": "assistant", "content": "The Titanic dataset contains records of 891 passengers from the 1912 Titanic voyage including survival outcome, class, age, sex, and fare."},
        {"role": "user",      "content": "What is the survival rate in the dataset?"}
    ]
    chat_response = client.chat(DEFAULT_MODEL, messages)
    print("Last user message  : " + messages[-1]["content"])
    print("Assistant response : " + chat_response)

    print("\n-- Model Benchmark --")
    benchmark_prompt = (
        "Analyze a dataset where 38 percent of passengers survived. "
        "What does this tell us about the event?"
    )
    benchmark_models(client, benchmark_prompt, models=["llama3", "mistral"])

    print("\n-- Ollama setup demo complete --")


if __name__ == "__main__":
    run_demo()
