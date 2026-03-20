# ============================================================
# FUNCTION CALLING
# Day 6 Healthcare Bot: DataAssist Analytics Agent
# Author: Sheshikala
# Topic: LLM function calling for structured data analysis
# ============================================================

# Function calling is a feature in modern LLMs where the model
# decides to call a predefined Python function instead of just
# generating text. You provide the model with function definitions
# including names, descriptions, and parameter schemas. The model
# returns a JSON object saying which function to call and with
# what arguments. Your code then runs the function and sends the
# result back to the model to generate a final natural language
# response. This is how tools work in AI agents. This file
# implements five data analysis functions and a dispatcher that
# routes questions to the correct function automatically based
# on keywords in the user question.

import json
import pandas as pd
import numpy as np

from data_loader import load_titanic, summarize_dataframe
from openai_client import OpenAIClient


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def get_survival_rate(df, group_by=None, group_value=None):
    """
    Returns survival rate as a percentage for all passengers
    or for a specific subgroup defined by group_by and group_value.

    Parameters:
        df          : pandas DataFrame with survived column
        group_by    : string column name to filter by, or None
        group_value : value to filter on, or None

    Returns:
        dict with survival rate, count, and group info
    """
    if "survived" not in df.columns:
        return {"error": "survived column not found"}

    if group_by and group_value is not None and group_by in df.columns:
        subset = df[df[group_by] == group_value]
        if len(subset) == 0:
            return {"error": "No records for " + str(group_by) + "=" + str(group_value)}
        rate  = round(subset["survived"].mean() * 100, 2)
        count = len(subset)
        return {
            "group"        : str(group_by) + "=" + str(group_value),
            "survival_rate": rate,
            "count"        : count
        }

    rate  = round(df["survived"].mean() * 100, 2)
    count = len(df)
    return {"group": "all", "survival_rate": rate, "count": count}


def get_column_statistics(df, column_name):
    """
    Returns mean, median, std, min, max, and missing count
    for a numeric column in the dataframe.

    Parameters:
        df          : pandas DataFrame
        column_name : string, name of the numeric column

    Returns:
        dict with descriptive statistics
    """
    if column_name not in df.columns:
        return {"error": "Column not found: " + column_name}

    col = df[column_name].dropna()
    if not pd.api.types.is_numeric_dtype(col):
        return {"error": column_name + " is not a numeric column"}

    return {
        "column" : column_name,
        "mean"   : round(col.mean(), 3),
        "median" : round(col.median(), 3),
        "std"    : round(col.std(), 3),
        "min"    : round(col.min(), 3),
        "max"    : round(col.max(), 3),
        "missing": int(df[column_name].isnull().sum())
    }


def get_group_comparison(df, group_column, target_column):
    """
    Computes mean and count of target_column for each unique
    value in group_column and returns a comparison table as dict.

    Parameters:
        df            : pandas DataFrame
        group_column  : string, column to group by
        target_column : string, column to aggregate

    Returns:
        dict with group names as keys and stats as values
    """
    for col in [group_column, target_column]:
        if col not in df.columns:
            return {"error": "Column not found: " + col}

    stats  = df.groupby(group_column)[target_column].agg(
        ["mean", "count"]
    ).round(3)
    result = {}
    for group_val, row in stats.iterrows():
        result[str(group_val)] = {
            "mean" : row["mean"],
            "count": int(row["count"])
        }
    return {
        "comparison" : result,
        "group_col"  : group_column,
        "target_col" : target_column
    }


def get_missing_values_report(df):
    """
    Returns a summary of all columns that have missing values
    including count and percentage for each column.

    Parameters:
        df : pandas DataFrame

    Returns:
        dict with total rows and per-column missing value info
    """
    report = {}
    total  = len(df)
    for col in df.columns:
        missing = int(df[col].isnull().sum())
        if missing > 0:
            report[col] = {
                "missing_count"  : missing,
                "missing_percent": round(missing / total * 100, 1)
            }
    return {"total_rows": total, "columns_with_missing": report}


def get_correlation(df, column_a, column_b):
    """
    Computes Pearson correlation coefficient between two numeric
    columns and returns value with plain English interpretation.

    Parameters:
        df       : pandas DataFrame
        column_a : string, first numeric column name
        column_b : string, second numeric column name

    Returns:
        dict with correlation value, strength, and direction
    """
    for col in [column_a, column_b]:
        if col not in df.columns:
            return {"error": "Column not found: " + col}

    pair = df[[column_a, column_b]].dropna()
    if len(pair) < 2:
        return {"error": "Not enough data to compute correlation"}

    corr      = round(pair.corr().iloc[0, 1], 4)
    strength  = "strong" if abs(corr) >= 0.7 else "moderate" if abs(corr) >= 0.4 else "weak"
    direction = "positive" if corr > 0 else "negative"

    return {
        "column_a"   : column_a,
        "column_b"   : column_b,
        "correlation": corr,
        "strength"   : strength,
        "direction"  : direction,
        "sample_size": len(pair)
    }


# ============================================================
# FUNCTION REGISTRY
# ============================================================

FUNCTION_REGISTRY = {
    "get_survival_rate"        : get_survival_rate,
    "get_column_statistics"    : get_column_statistics,
    "get_group_comparison"     : get_group_comparison,
    "get_missing_values_report": get_missing_values_report,
    "get_correlation"          : get_correlation,
}


# ============================================================
# FUNCTION CALL DISPATCHER
# ============================================================

class FunctionCallDispatcher:
    """
    Simulates the LLM function calling workflow by parsing the
    user question, deciding which function is most relevant,
    calling it with the dataframe, and returning the result.
    In a real OpenAI function calling setup the API returns a
    function_call JSON object. This class simulates that by
    using keyword matching on the question string.
    """

    def __init__(self, df):
        self.df = df

    def dispatch(self, function_name, arguments):
        """
        Calls the named function from the registry with the
        provided arguments and the stored dataframe.

        Parameters:
            function_name : string, must be a key in FUNCTION_REGISTRY
            arguments     : dict of keyword arguments for the function

        Returns:
            dict result from the function, or error dict
        """
        if function_name not in FUNCTION_REGISTRY:
            return {"error": "Function not found: " + function_name}

        func = FUNCTION_REGISTRY[function_name]
        try:
            return func(self.df, **arguments)
        except Exception as e:
            return {"error": "Function execution failed: " + str(e)}

    def auto_dispatch(self, question):
        """
        Automatically selects and calls the most relevant function
        based on keywords in the question. Returns a tuple of
        (function_name, result_dict).

        Parameters:
            question : string, the user question

        Returns:
            tuple of (function_name string, result dict)
        """
        q = question.lower()

        if "missing" in q or "null" in q or "incomplete" in q:
            return "get_missing_values_report", self.dispatch(
                "get_missing_values_report", {}
            )
        elif "correlation" in q or "relationship between" in q:
            if "age" in q and "fare" in q:
                return "get_correlation", self.dispatch(
                    "get_correlation", {"column_a": "age", "column_b": "fare"}
                )
            return "get_correlation", self.dispatch(
                "get_correlation", {"column_a": "age", "column_b": "survived"}
            )
        elif "compare" in q or "by sex" in q or "by class" in q or "group" in q:
            group_col  = "sex" if "sex" in q or "gender" in q else "pclass"
            target_col = "fare" if "fare" in q else "survived"
            return "get_group_comparison", self.dispatch(
                "get_group_comparison",
                {"group_column": group_col, "target_column": target_col}
            )
        elif "age" in q and any(w in q for w in ["stat", "average", "mean", "median"]):
            return "get_column_statistics", self.dispatch(
                "get_column_statistics", {"column_name": "age"}
            )
        elif "fare" in q and any(w in q for w in ["stat", "average", "mean", "median"]):
            return "get_column_statistics", self.dispatch(
                "get_column_statistics", {"column_name": "fare"}
            )
        elif "female" in q or "women" in q:
            return "get_survival_rate", self.dispatch(
                "get_survival_rate", {"group_by": "sex", "group_value": "female"}
            )
        elif "male" in q or "men" in q:
            return "get_survival_rate", self.dispatch(
                "get_survival_rate", {"group_by": "sex", "group_value": "male"}
            )
        elif "first class" in q or "class 1" in q or "pclass 1" in q:
            return "get_survival_rate", self.dispatch(
                "get_survival_rate", {"group_by": "pclass", "group_value": 1}
            )
        elif "third class" in q or "class 3" in q or "pclass 3" in q:
            return "get_survival_rate", self.dispatch(
                "get_survival_rate", {"group_by": "pclass", "group_value": 3}
            )
        else:
            return "get_survival_rate", self.dispatch(
                "get_survival_rate", {}
            )


# ============================================================
# FUNCTION CALLING AGENT
# ============================================================

class FunctionCallingAgent:
    """
    Uses function calling to answer data analysis questions.
    For each question the agent dispatches the appropriate
    analysis function, formats the result, and sends it to
    the LLM to generate a natural language answer.
    """

    def __init__(self):
        self.df         = load_titanic()
        self.summary    = summarize_dataframe(self.df)
        self.dispatcher = FunctionCallDispatcher(self.df)
        self.client     = OpenAIClient()
        print("Function Calling Agent initialized.")

    def answer(self, question):
        """
        Answers a question by calling the appropriate function
        and then generating a natural language response.

        Parameters:
            question : string, the analytical question

        Returns:
            string, the final answer
        """
        print("\nQuestion: " + question)

        func_name, func_result = self.dispatcher.auto_dispatch(question)
        result_str = json.dumps(func_result, indent=2)

        print("Function called: " + func_name)
        print("Result: " + result_str[:200])

        prompt = (
            "A user asked: " + question + "\n\n"
            "The following data was retrieved by calling " + func_name + ":\n"
            + result_str + "\n\n"
            "Write a clear, concise answer to the user question "
            "using only the data provided above."
        )
        answer_text = self.client.complete(prompt)
        print("Answer: " + answer_text)
        return answer_text


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("FUNCTION CALLING DEMO")
    print("=" * 55)

    df         = load_titanic()
    dispatcher = FunctionCallDispatcher(df)

    print("\n-- Direct Function Calls --")

    print("\nget_survival_rate (all passengers):")
    print(json.dumps(get_survival_rate(df), indent=2))

    print("\nget_survival_rate (female):")
    print(json.dumps(get_survival_rate(df, "sex", "female"), indent=2))

    print("\nget_survival_rate (male):")
    print(json.dumps(get_survival_rate(df, "sex", "male"), indent=2))

    print("\nget_column_statistics (age):")
    print(json.dumps(get_column_statistics(df, "age"), indent=2))

    print("\nget_column_statistics (fare):")
    print(json.dumps(get_column_statistics(df, "fare"), indent=2))

    print("\nget_group_comparison (pclass vs survived):")
    print(json.dumps(get_group_comparison(df, "pclass", "survived"), indent=2))

    print("\nget_group_comparison (sex vs survived):")
    print(json.dumps(get_group_comparison(df, "sex", "survived"), indent=2))

    print("\nget_missing_values_report:")
    print(json.dumps(get_missing_values_report(df), indent=2))

    print("\nget_correlation (age vs fare):")
    print(json.dumps(get_correlation(df, "age", "fare"), indent=2))

    print("\n-- Function Calling Agent --")
    agent = FunctionCallingAgent()

    questions = [
        "What is the survival rate for female passengers?",
        "What is the survival rate for male passengers?",
        "What are the age statistics in the dataset?",
        "How does survival rate compare across passenger classes?",
        "Are there any missing values in the dataset?",
        "Is there a correlation between age and fare paid?"
    ]

    for q in questions:
        agent.answer(q)

    print("\n-- Function Calling demo complete --")


if __name__ == "__main__":
    run_demo()
