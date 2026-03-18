Day 5 NLP: DataAssist Analytics Agent
Author: Sheshikala Mamidisetti
Week: 4 (Mar 15-24)
Topic: Local LLMs via Ollama and Prompt Engineering for Data Analytics
Dataset: Titanic passenger data loaded via seaborn

What I Built
This folder extends the Day 5 NLP work with Week 4 content covering local
large language models and advanced prompt engineering techniques applied to
data analytics tasks on the Titanic dataset.
I used Ollama to run Llama3 and Mistral locally on my machine without any
API costs or internet connection. Ollama works by running a local HTTP server
at localhost:11434 and exposing an API that works similarly to OpenAI. Once
a model is pulled it stays on the machine and can be called instantly.
For Llama3 I built a full analysis pipeline that reads the Titanic dataset,
computes a statistical summary, and sends it to the model to generate natural
language insights, survival predictions for individual passengers, and formatted
report sections. Llama3 gave detailed multi-paragraph responses which worked
well for report generation tasks.
For Mistral I built a separate pipeline focused on concise outputs. Mistral is
smaller and faster than Llama3 and produces shorter more direct answers which
is useful when you need quick summaries or bullet-point metrics. I also built
a side-by-side comparison function that runs the same prompt on both models
and shows the difference in response time and length.
For prompt engineering I implemented three techniques. Chain of Thought forces
the model to reason step by step before giving an answer by breaking the prompt
into numbered steps like identify relevant data, apply analysis, interpret
results, and state conclusion. This significantly improves accuracy on analytical
questions compared to asking directly. ReAct combines reasoning with tool use
where the model alternates between thinking about what to do next and calling
a real Python function on the dataframe to get an actual computed result. I
built five tools -- survival rate, descriptive stats, value counts, correlation,
and group comparison -- that the agent can call automatically based on the
question. DSPy replaces hand-written prompts with learnable modules where each
module has a defined Signature specifying input and output field names. I
implemented Predict and ChainOfThought modules from scratch so they work without
the dspy package installed and demonstrated how few-shot examples in the prompt
change the style and quality of the model output compared to zero-shot prompting.
All files include mock fallback responses so they run and produce output
immediately without Ollama installed. When Ollama is running the mock responses
are replaced by real model output automatically.

Files
FileTopicollama_setup.pyOllama client setup, connection check, model listing, benchmarkllama_pipeline.pyLlama3 data analysis -- insights, predictions, class comparison, reportmistral_pipeline.pyMistral pipeline -- quick summary, metrics, recommendations, model comparisoncot_prompting.pyChain of Thought -- step-by-step reasoning, prediction, group comparisonreact_prompting.pyReAct agent -- five analysis tools with keyword-based auto-dispatchdspy_pipeline.pyDSPy-style modular prompts -- Predict, ChainOfThought, few-shot demorequirements.txtPython dependenciesREADME.mdThis file

Key Concepts Learned
ConceptFileOllama local LLM server setup and clientollama_setup.pyLlama3 for multi-paragraph data report generationllama_pipeline.pyMistral for concise analytical outputsmistral_pipeline.pyChain of Thought step-by-step reasoningcot_prompting.pyReAct agent with real dataframe tool executionreact_prompting.pyDSPy Signature, Predict, ChainOfThought modulesdspy_pipeline.pyFew-shot vs zero-shot prompting comparisondspy_pipeline.pyModel benchmark comparing speed and response lengthmistral_pipeline.py

Dataset
All files use the Titanic passenger dataset loaded via seaborn. If seaborn
is not installed the files fall back to a 20-row inline sample automatically
so nothing breaks regardless of the environment.

