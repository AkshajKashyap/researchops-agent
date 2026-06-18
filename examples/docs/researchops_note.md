# ResearchOps Agent Pipeline Note

ResearchOps Agent is a local research workflow for documents and bounded experiments. The
pipeline ingests local PDF, Markdown, and text files, chunks them, retrieves citation-ready
evidence, and builds evidence packs.

The system can answer extractively from evidence, extract experiment-like claims, suggest
heuristic experiment configs, and run a bounded local time-series forecasting runner. Optional
LLM grounded answering can use retrieved evidence, but it must cite only evidence-pack
citations and abstain when evidence is insufficient.

The current limitations are intentional: no web search, no arbitrary code execution, no
database requirement, and only a small explicit runner task. These constraints make the system
auditable for local ResearchOps demos.
