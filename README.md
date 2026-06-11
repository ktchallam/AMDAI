# AMD MI300X Multi-Agent Earnings Call Script & Presentation Pipeline

This repository implements a production-grade multi-agent AI system optimized for AMD MI300X accelerators with ROCm. The system automates quarterly earnings call preparation for Investor Relations (IR) teams, predicting tough, adversarial analyst questions and preparing grounded suggested answers, alongside generating a beautiful, professional dark-navy executive PowerPoint deck and briefing document.

## Architecture Summary

The system orchestrates specialized agent nodes powered by a local vLLM server running Llama-3-70B-Instruct in float16 precision. When a quarterly analysis is triggered, the **Transcript Fetcher** retrieves public transcripts using a fallback chain across BSE, NSE, Trendlyne RSS, and local Whisper audio transcriptions. The raw transcript is passed to the **Extraction Agent**, which extracts speaker turns, tags sentiment per turn using ProsusAI/finbert, and ingests them into a dual-store ChromaDB configuration. The `hot_store` keeps the latest quarter chunks at high precision (256 tokens) to support direct QA retrieval, while the `cold_store` stores all historical quarters' analyst turns (512 tokens) to model concern patterns.

The **Predictive Analyst Agent** uses BERTopic (with LLM fallback) to extract concern topics from the historical stores, calculates guidance vs actual KPI deltas from the hot store, and generates adversarial questions. A **Relevance Gate** filters out questions that cannot be answered using the current quarter's disclosures, flagging them as data gaps. The remaining questions are answered by the **Answer Agent** in parallel batches. These answers are fact-checked by a **Self-Critique** module and approved by the **Validation Agent** on factual accuracy, professional tone, and completeness. Finally, the **Output Orchestrator** formats the final preparation package into a markdown briefing cheat sheet and a custom python-pptx presentation deck.

---

## File Structure Tree

```
c:/Users/ktcha/Documents/AMD/
├── 00_setup.ipynb              # Module 1 setup and server health checks
├── 01_fetch_transcripts.ipynb  # Module 2 transcript fetching validation
├── 02_extraction_agent.ipynb   # Module 3 ChromaDB ingestion validation
├── 03_predictive_analyst.ipynb # Module 4 topic mining & question gating
├── 04_answer_agent.ipynb       # Module 5 Q&A generation & critique
├── 05_validation_agent.ipynb   # Module 6 IR validation checks
├── 06_output_generator.ipynb   # Module 7 PowerPoint & markdown rendering
├── 07_full_pipeline.ipynb      # Module 8 end-to-end execution pipeline
├── config.py                   # AppConfig pydantic configuration settings
├── llm_client.py               # Async vLLM OpenAI wrapper with retry logic
├── transcript_fetcher.py       # Fallback scraper chain (BSE, NSE, RSS, Whisper)
├── extraction_agent.py         # Chunker, role extractor, embedder, tagger
├── predictive_analyst_agent.py # KPI computer, pattern miner, relevance gate
├── answer_agent.py             # CFO answering and critique fact-checker
├── validation_agent.py         # Fact, tone, and completeness validator
├── output_generator.py         # Markdown and python-pptx presentation compiler
├── pipeline.py                 # Full pipeline orchestrator class
└── setup_env.sh                # ROCm pip and server installation script
```

---

## How to Run from Scratch on AMD MI300X

Follow these numbered steps to run the pipeline from scratch on your AMD MI300X server with ROCm:

1. **Verify ROCm Environment**:
   Ensure you have ROCm properly installed and that your GPUs are visible. Run:
   ```bash
   rocm-smi
   ```

2. **Execute Environment Setup Script**:
   This script installs the non-negotiable libraries and dependencies using pip with system package override, and starts the vLLM server in the background:
   ```bash
   chmod +x setup_env.sh
   ./setup_env.sh
   ```
   *Note: This starts vLLM with meta-llama/Llama-3-70B-Instruct in FP16 with a tensor parallel size of 4, utilizing 85% GPU memory to leave VRAM for embeddings and sentiment tagging.*

3. **Verify vLLM Server Readiness**:
   Monitor the vLLM server startup using:
   ```bash
   tail -f vllm.log
   ```
   Or query the health endpoint:
   ```bash
   curl http://localhost:8000/v1/health
   ```

4. **Launch Jupyter Lab**:
   Start Jupyter in your workspace:
   ```bash
   jupyter lab --ip 0.0.0.0 --port 8888
   ```

5. **Run Verification Notebooks**:
   Open and execute the notebooks in sequence from `00_setup.ipynb` to `07_full_pipeline.ipynb` to run individual modules or the entire end-to-end multi-agent workflow.

---

## Known Limitations and Next Steps

- **Scraper Anti-Bot Protections**: BSE/NSE India frequently change headers, rate limit, or require dynamic cookie solving (which the scraper attempts via preliminary homepage requests). If external fetches block, the system automatically falls back to an LLM-simulated earnings call transcript to keep downstream agents functioning smoothly.
- **Model VRAM Allocation**: On systems with smaller GPUs, running embedding models (`bge-large-en-v1.5`), FinBERT classifier, and a 70B Llama-3 model simultaneously can trigger Out-Of-Memory (OOM) exceptions. The shared configuration uses memory-efficient batching of 64 for embedding and 16 for FinBERT to stay within the safe headroom of the MI300X.
- **Next Steps**:
  1. Integrate direct SEC EDGAR RSS scraping for US companies.
  2. Implement an active human-in-the-loop validation interface to edit proposed answers before presentation generation.
  3. Expand the feedback loop to save actual analyst questions to the cold store and re-train or fine-tune topic tags.
