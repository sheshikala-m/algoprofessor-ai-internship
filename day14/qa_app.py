# ============================================================
# QA APP
# Day 14: RAG Pipeline for ML Experiment Tracker
# Author: Sheshikala
# Topic: Interactive question-answering app using RAG pipeline
# ============================================================

# This is the final piece. I built a simple Q&A interface
# that uses the full RAG pipeline. The session history feature
# was interesting to add because it lets the app remember
# what was asked before in the same session.

import json
from datetime import datetime
from rag_pipeline import RAGPipeline


# ============================================================
# SESSION MANAGER
# ============================================================

class QASession:
    """
    Tracks conversation history within a session.
    Useful for multi-turn queries where later questions
    refer to earlier answers.
    """
    def __init__(self, session_id=None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history = []
        self.created_at = datetime.now().isoformat()

    def add_turn(self, question, answer, context=None):
        self.history.append({
            "turn": len(self.history) + 1,
            "question": question,
            "answer": answer,
            "context_count": len(context) if context else 0,
            "timestamp": datetime.now().isoformat()
        })

    def get_recent_context(self, n=3):
        """Returns last n questions as additional context."""
        recent = self.history[-n:]
        return " ".join(f"Previously asked: {t['question']}" for t in recent)

    def summary(self):
        return {
            "session_id": self.session_id,
            "total_turns": len(self.history),
            "created_at": self.created_at,
        }

    def export(self, filepath=None):
        data = {"session": self.summary(), "history": self.history}
        if filepath:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Session exported to {filepath}")
        return data


# ============================================================
# QUERY ENHANCER
# ============================================================

QUERY_EXPANSIONS = {
    "bert": ["BERT", "bert-base", "transformer", "NLP model"],
    "roberta": ["RoBERTa", "transformer", "NLP"],
    "resnet": ["ResNet50", "CNN", "image classification"],
    "lstm": ["LSTM", "recurrent", "time series"],
    "accuracy": ["accuracy", "performance", "f1 score", "precision", "recall"],
    "ananya": ["Ananya", "researcher Ananya"],
    "vikram": ["Vikram", "researcher Vikram"],
    "priya": ["Priya", "researcher Priya"],
    "rohan": ["Rohan", "researcher Rohan"],
}

def enhance_query(query):
    """
    Expands query with synonyms and related terms.
    Improves recall when user uses informal language.
    """
    query_lower = query.lower()
    expansions = []
    for keyword, terms in QUERY_EXPANSIONS.items():
        if keyword in query_lower:
            expansions.extend(terms)
    if expansions:
        enhanced = query + " " + " ".join(set(expansions))
        return enhanced
    return query


# ============================================================
# RESULT FORMATTER
# ============================================================

def format_result(result, show_context=False):
    """Formats a QA result for display."""
    lines = [
        f"Question: {result['question']}",
        f"Answer:   {result['answer']}",
    ]
    if show_context and result.get("context"):
        lines.append(f"Sources ({len(result['context'])} chunks):")
        for i, ctx in enumerate(result["context"][:2]):
            text = ctx.get("text", "")[:80]
            score = ctx.get("rrf_score", ctx.get("score", 0))
            lines.append(f"  [{i+1}] {text}... (score: {score:.4f})")
    return "\n".join(lines)


# ============================================================
# QA APP CLASS
# ============================================================

class MLExperimentQAApp:
    """
    Question-answering app for the ML Experiment Tracker.
    Uses RAG pipeline to answer questions about experiments,
    datasets, researchers, and metrics.
    """
    def __init__(self, retriever_type="hybrid", enhance_queries=True):
        print("Initializing ML Experiment QA App...")
        self.pipeline = RAGPipeline(retriever_type=retriever_type)
        self.enhance_queries = enhance_queries
        self.session = QASession()
        print("App ready. Type your questions about ML experiments.\n")

    def ask(self, question, top_k=3, show_context=False):
        """Answers a single question."""
        # optionally enhance query
        query = enhance_query(question) if self.enhance_queries else question

        result = self.pipeline.query(query, top_k=top_k)
        # store original question in result
        result["question"] = question

        # track in session
        self.session.add_turn(question, result["answer"], result.get("context"))

        return format_result(result, show_context=show_context)

    def ask_batch(self, questions, show_context=False):
        """Answers multiple questions."""
        outputs = []
        for q in questions:
            output = self.ask(q, show_context=show_context)
            outputs.append(output)
        return outputs

    def show_session_summary(self):
        """Prints session history summary."""
        summary = self.session.summary()
        print(f"\nSession Summary:")
        print(f"  Session ID:   {summary['session_id']}")
        print(f"  Total turns:  {summary['total_turns']}")
        print(f"  Started at:   {summary['created_at']}")
        if self.session.history:
            print(f"  Questions asked:")
            for turn in self.session.history:
                print(f"    [{turn['turn']}] {turn['question']}")

    def interactive_mode(self, max_turns=5):
        """
        Simulates an interactive session with predefined questions.
        In a real app, this would read from stdin.
        """
        print("=" * 55)
        print("INTERACTIVE QA SESSION (simulated)")
        print("=" * 55)

        demo_questions = [
            "Who ran experiments with BERT?",
            "What was the highest accuracy achieved?",
            "Which researcher worked on computer vision?",
            "What datasets were used in NLP research?",
            "How did EXP008 compare to EXP004?",
        ]

        for i, q in enumerate(demo_questions[:max_turns]):
            print(f"\nTurn {i+1}:")
            output = self.ask(q, show_context=True)
            print(output)
            print("-" * 40)


# ============================================================
# EVALUATION
# ============================================================

EVAL_SET = [
    {
        "question": "Which experiments did Ananya run?",
        "expected_keywords": ["EXP001", "EXP005", "Ananya"],
    },
    {
        "question": "What model did Vikram use?",
        "expected_keywords": ["RoBERTa", "DistilBERT", "Vikram"],
    },
    {
        "question": "What was the MSE for time series experiments?",
        "expected_keywords": ["MSE", "0.023", "0.018", "LSTM", "Transformer"],
    },
    {
        "question": "Which model gave top-1 accuracy of 0.87?",
        "expected_keywords": ["ResNet50", "0.87", "Priya"],
    },
]

def evaluate_app(app):
    """Evaluates QA app on predefined test cases."""
    print("\n-- Evaluating QA App --")
    hits = 0
    for case in EVAL_SET:
        result_text = app.ask(case["question"])
        answer = result_text.split("Answer:")[-1].lower()
        matched = [kw for kw in case["expected_keywords"] if kw.lower() in answer]
        is_hit = len(matched) > 0
        if is_hit:
            hits += 1
        status = "HIT" if is_hit else "MISS"
        print(f"[{status}] Q: {case['question'][:50]}")
        if not is_hit:
            print(f"       Expected: {case['expected_keywords']}")
    print(f"\nEval result: {hits}/{len(EVAL_SET)} hits ({hits/len(EVAL_SET)*100:.0f}%)")


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("QA APP DEMO")
    print("=" * 55)

    app = MLExperimentQAApp(retriever_type="hybrid", enhance_queries=True)

    # interactive session
    app.interactive_mode(max_turns=5)

    # evaluation
    evaluate_app(app)

    # session summary
    app.show_session_summary()

    # export session
    app.session.export("qa_session_export.json")

    import os
    if os.path.exists("qa_session_export.json"):
        os.remove("qa_session_export.json")

    print("\n-- QA App demo complete --")


if __name__ == "__main__":
    run_demo()
