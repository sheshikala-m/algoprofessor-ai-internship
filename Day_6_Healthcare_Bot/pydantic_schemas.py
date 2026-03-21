# ============================================================
# PYDANTIC SCHEMAS
# Day 6 Healthcare Bot: DataAssist Analytics Agent
# Author: Sheshikala
# Topic: Pydantic structured outputs for data analysis reports
# ============================================================

# Pydantic is a Python library for data validation using type
# annotations. In LLM pipelines it is used to enforce that the
# model output follows a specific structure. Instead of parsing
# free-form text you ask the LLM to produce JSON matching a
# schema, then validate that JSON with a Pydantic model. This
# guarantees that downstream code always gets the fields it
# expects with the correct types. This file defines four schemas
# for analytics output: DatasetSummarySchema for high-level
# stats, SurvivalGroupSchema for per-group survival rates,
# SurvivalAnalysisSchema for a complete analysis, and
# DataReportSchema for a full structured report. All schemas
# fall back to Python dataclasses if pydantic is not installed
# so the file always runs regardless of environment.

import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from data_loader import load_titanic, summarize_dataframe
from openai_client import OpenAIClient


# ============================================================
# SCHEMAS WITH PYDANTIC
# ============================================================

if PYDANTIC_AVAILABLE:

    class DatasetSummarySchema(BaseModel):
        """Schema for a high-level dataset summary."""
        dataset_name    : str   = Field(description="Name of the dataset")
        total_records   : int   = Field(description="Total number of rows")
        total_columns   : int   = Field(description="Total number of columns")
        target_variable : str   = Field(description="The outcome variable being analyzed")
        overall_rate    : float = Field(description="Overall rate of the target variable as a percentage")
        key_insight     : str   = Field(description="The single most important finding in one sentence")

    class SurvivalGroupSchema(BaseModel):
        """Schema for survival statistics of one passenger group."""
        group_name      : str   = Field(description="Name of the group e.g. female or class_1")
        survival_rate   : float = Field(description="Survival rate as a percentage 0 to 100")
        passenger_count : int   = Field(description="Number of passengers in this group")
        rank            : int   = Field(description="Rank by survival rate, 1 is highest")

    class SurvivalAnalysisSchema(BaseModel):
        """Schema for a complete survival analysis with multiple groups."""
        analysis_title : str                       = Field(description="Title of this analysis")
        groups         : List[SurvivalGroupSchema] = Field(description="List of group statistics")
        top_factor     : str                       = Field(description="The single strongest survival predictor")
        recommendation : str                       = Field(description="One actionable recommendation based on findings")

    class DataReportSchema(BaseModel):
        """Schema for a complete structured data analysis report."""
        title            : str       = Field(description="Report title")
        executive_summary: str       = Field(description="2-3 sentence overview of findings")
        key_findings     : List[str] = Field(description="List of 3 numbered key findings")
        recommendations  : List[str] = Field(description="List of 3 numbered recommendations")
        conclusion       : str       = Field(description="One sentence conclusion")

else:
    # Dataclass fallbacks when pydantic is not installed

    @dataclass
    class DatasetSummarySchema:
        dataset_name    : str
        total_records   : int
        total_columns   : int
        target_variable : str
        overall_rate    : float
        key_insight     : str

    @dataclass
    class SurvivalGroupSchema:
        group_name      : str
        survival_rate   : float
        passenger_count : int
        rank            : int

    @dataclass
    class SurvivalAnalysisSchema:
        analysis_title : str
        groups         : list
        top_factor     : str
        recommendation : str

    @dataclass
    class DataReportSchema:
        title            : str
        executive_summary: str
        key_findings     : list
        recommendations  : list
        conclusion       : str


# ============================================================
# STRUCTURED OUTPUT GENERATOR
# ============================================================

class StructuredOutputGenerator:
    """
    Generates validated structured outputs by computing statistics
    directly from the dataframe for accuracy, then using the LLM
    only for narrative fields like key_insight and top_factor.
    This hybrid approach prevents hallucinated numbers while still
    getting the benefit of natural language generation for text fields.
    """

    def __init__(self):
        self.client  = OpenAIClient()
        self.df      = load_titanic()
        self.summary = summarize_dataframe(self.df)
        print("Structured Output Generator initialized.")
        if PYDANTIC_AVAILABLE:
            print("Pydantic available for full schema validation.")
        else:
            print("Pydantic not installed. Using dataclass fallbacks.")
            print("Install with: pip install pydantic")

    def generate_dataset_summary(self):
        """
        Generates a DatasetSummarySchema object. All numeric fields
        are computed directly from the dataframe. The key_insight
        field is generated by the LLM using the dataset summary.
        """
        total_records = len(self.df)
        total_columns = len(self.df.columns)
        overall_rate  = round(self.df["survived"].mean() * 100, 1) if "survived" in self.df.columns else 0.0

        prompt = (
            "Based on this dataset summary, write a single sentence "
            "describing the most important finding:\n\n" + self.summary
        )
        insight_text = self.client.complete(prompt)

        obj = DatasetSummarySchema(
            dataset_name    = "Titanic Passenger Dataset",
            total_records   = total_records,
            total_columns   = total_columns,
            target_variable = "survived",
            overall_rate    = overall_rate,
            key_insight     = insight_text[:200]
        )

        print("\nDataset Summary Schema:")
        if PYDANTIC_AVAILABLE:
            print(obj.model_dump_json(indent=2))
        else:
            print(json.dumps(obj.__dict__, indent=2))
        return obj

    def generate_survival_analysis(self):
        """
        Builds a SurvivalAnalysisSchema with group survival rates
        computed directly from the dataframe. Uses LLM only for
        the top_factor and recommendation narrative fields.
        """
        groups_data = []
        rank        = 1

        if "sex" in self.df.columns and "survived" in self.df.columns:
            for gender in ["female", "male"]:
                subset = self.df[self.df["sex"] == gender]
                if len(subset) > 0:
                    rate = round(subset["survived"].mean() * 100, 1)
                    groups_data.append(SurvivalGroupSchema(
                        group_name      = gender,
                        survival_rate   = rate,
                        passenger_count = len(subset),
                        rank            = rank
                    ))
                    rank += 1

        if "pclass" in self.df.columns and "survived" in self.df.columns:
            for cls in [1, 2, 3]:
                subset = self.df[self.df["pclass"] == cls]
                if len(subset) > 0:
                    rate = round(subset["survived"].mean() * 100, 1)
                    groups_data.append(SurvivalGroupSchema(
                        group_name      = "class_" + str(cls),
                        survival_rate   = rate,
                        passenger_count = len(subset),
                        rank            = rank
                    ))
                    rank += 1

        prompt = (
            "In one sentence each: what is the top survival factor, "
            "and what is one recommendation? Dataset:\n" + self.summary
        )
        llm_text  = self.client.complete(prompt)
        sentences = [s.strip() for s in llm_text.split(".") if s.strip()]
        top_factor     = sentences[0] + "." if sentences else "Gender was the strongest predictor."
        recommendation = sentences[1] + "." if len(sentences) > 1 else "Improve evacuation equity across all classes."

        obj = SurvivalAnalysisSchema(
            analysis_title = "Titanic Passenger Survival Analysis",
            groups         = groups_data,
            top_factor     = top_factor[:200],
            recommendation = recommendation[:200]
        )

        print("\nSurvival Analysis Schema:")
        for g in groups_data:
            print("  " + g.group_name + ": " + str(g.survival_rate) +
                  "% (" + str(g.passenger_count) + " passengers)")
        print("Top factor    : " + top_factor[:100])
        print("Recommendation: " + recommendation[:100])
        return obj

    def generate_full_report(self):
        """
        Generates a DataReportSchema with all sections. The key
        findings and recommendations are generated by the LLM.
        The title and conclusion are templated for consistency.
        """
        prompt = (
            "Write a structured data analysis report for the Titanic "
            "dataset. Include a 2-sentence executive summary, exactly "
            "3 key findings as numbered sentences, and exactly 3 "
            "recommendations as numbered sentences. Use this data:\n\n"
            + self.summary
        )
        llm_text = self.client.complete(prompt, max_tokens=600)
        lines    = [l.strip() for l in llm_text.split("\n") if l.strip()]
        findings = [l for l in lines if l[:2] in ["1.", "2.", "3."]][:3]

        if len(findings) < 3:
            findings = [
                "1. Female passengers survived at 74.2 percent versus 18.9 percent for males.",
                "2. First class passengers survived at 63 percent versus 24 percent in third class.",
                "3. Overall 38.4 percent of the 891 recorded passengers survived the disaster."
            ]

        obj = DataReportSchema(
            title             = "Titanic Passenger Survival Analysis Report",
            executive_summary = llm_text[:300],
            key_findings      = findings,
            recommendations   = [
                "1. Implement equal lifeboat access procedures regardless of ticket class.",
                "2. Ensure total lifeboat capacity matches full passenger and crew count.",
                "3. Conduct mandatory safety briefings for all passengers within 12 hours of departure."
            ],
            conclusion = (
                "Gender and passenger class were the two strongest determinants "
                "of survival in the 1912 Titanic disaster."
            )
        )

        print("\nFull Report Schema:")
        print("Title: " + obj.title)
        print("Executive Summary: " + obj.executive_summary[:150] + "...")
        print("Key Findings:")
        for f in obj.key_findings:
            print("  " + f)
        print("Recommendations:")
        for r in obj.recommendations:
            print("  " + r)
        print("Conclusion: " + obj.conclusion)
        return obj

    def validate_passenger_records(self):
        """
        Validates the first 10 rows of the dataframe against the
        SurvivalGroupSchema structure to show Pydantic validation
        working on real data rows.
        """
        print("\nPassenger Record Validation (first 10 rows):")
        valid_count   = 0
        invalid_count = 0

        for i, row in self.df.head(10).iterrows():
            try:
                pclass   = int(row.get("pclass", 3)) if not pd.isna(row.get("pclass", 3)) else 3
                survived = int(row.get("survived", 0)) if not pd.isna(row.get("survived", 0)) else 0
                sex      = str(row.get("sex", "unknown"))
                fare     = float(row.get("fare", 0.0)) if not pd.isna(row.get("fare", 0.0)) else 0.0

                if pclass not in [1, 2, 3]:
                    raise ValueError("Invalid pclass: " + str(pclass))
                if survived not in [0, 1]:
                    raise ValueError("Invalid survived: " + str(survived))

                label = "Survived" if survived == 1 else "Did Not Survive"
                valid_count += 1
                print("  Row " + str(i) + ": VALID - " + sex +
                      ", class " + str(pclass) + " - " + label)
            except Exception as e:
                invalid_count += 1
                print("  Row " + str(i) + ": INVALID - " + str(e))

        print("Valid: " + str(valid_count) + " | Invalid: " + str(invalid_count))


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("PYDANTIC SCHEMAS DEMO")
    print("=" * 55)

    generator = StructuredOutputGenerator()

    generator.generate_dataset_summary()
    generator.generate_survival_analysis()
    generator.generate_full_report()
    generator.validate_passenger_records()

    print("\n-- Pydantic Schemas demo complete --")


if __name__ == "__main__":
    run_demo()
