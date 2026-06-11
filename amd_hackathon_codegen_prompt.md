# AMD AI Hackathon — Code Generation Prompt
## Autonomous Earning Call Script & Presentation — Multi-Agent Workflow

---

## HOW TO USE THIS DOCUMENT

This prompt is structured in **8 self-contained modules**. Feed each module
independently to an LLM (Claude, Llama 3 70B on vLLM, etc.) or feed the entire
document at once for a full codebase generation. Every module references shared
interfaces so the generated code is plug-compatible across modules.

**Target environment**: Jupyter Notebook + Linux terminal, AMD MI300X, ROCm,
vLLM, SGLang, FP16, Python 3.11+

---

## GLOBAL CONTEXT (include with every module prompt)

```
You are an expert Python engineer building a production-grade multi-agent AI system
on AMD MI300X with ROCm. The system automates quarterly earnings call preparation
for investor relations teams. It produces:
  (a) a CEO/CFO Q&A cheat sheet predicting tough analyst questions with grounded answers
  (b) an investor presentation deck (.pptx) with bullet points

Tech stack (non-negotiable):
- Inference: vLLM serving Llama-3-70B-Instruct in FP16 via ROCm
- Agent orchestration: SGLang for structured outputs and parallel forks
- Embeddings: BAAI/bge-large-en-v1.5 via sentence-transformers (ROCm backend)
- Vector stores: ChromaDB (two separate collections: hot_store, cold_store)
- Topic modelling: BERTopic
- Sentiment: FinBERT (ProsusAI/finbert)
- PDF parsing: pdfplumber
- Presentation output: python-pptx
- All code must run in Jupyter notebooks on ROCm; use --break-system-packages for pip

Write production-quality code: typed, documented, with error handling and logging.
Use async where appropriate. Never use OpenAI SDK — use vLLM's OpenAI-compatible
endpoint (http://localhost:8000/v1) via the openai Python client pointed at localhost.
```

---

## MODULE 1 — Environment Setup & Model Serving

### Prompt

```
Using the GLOBAL CONTEXT above, generate the complete environment setup for this
project. Produce the following:

1. A shell script `setup_env.sh` that:
   - Installs all Python dependencies via pip with --break-system-packages
   - Required packages: vllm, sglang, sentence-transformers, chromadb, bertopic,
     transformers, pdfplumber, python-pptx, pandas, numpy,
     httpx, openai, pydantic, loguru, tenacity, youtube-transcript-api, feedparser,
     faster-whisper, finbert-embedding
   - Verifies ROCm is available via rocm-smi
   - Starts a vLLM server in the background serving meta-llama/Llama-3-70B-Instruct
     with: --dtype float16 --tensor-parallel-size 4 --gpu-memory-utilization 0.85
     --max-model-len 8192 --port 8000
   - Waits for the server health endpoint to be ready before exiting

2. A Python module `config.py` with a Pydantic BaseSettings class `AppConfig`:
   - vllm_base_url: str = "http://localhost:8000/v1"
   - vllm_model: str = "meta-llama/Llama-3-70B-Instruct"
   - embedding_model: str = "BAAI/bge-large-en-v1.5"
   - hot_store_collection: str = "hot_store"
   - cold_store_collection: str = "cold_store"
   - hot_store_chunk_size: int = 256
   - cold_store_chunk_size: int = 512
   - relevance_gate_threshold: float = 0.75
   - top_n_questions: int = 20
   - chroma_persist_dir: str = "./chroma_db"
   - log_level: str = "INFO"
   - Plus a singleton get_config() function

3. A module `llm_client.py` with an async LLMClient class:
   - Wraps OpenAI client pointed at vllm_base_url
   - async generate(system, user, max_tokens=1000, response_format=None) -> str
   - async generate_json(system, user, schema: type[BaseModel], max_tokens=1000) -> BaseModel
   - Retry logic: tenacity, 3 attempts, exponential backoff
   - All calls logged with loguru including token counts

4. Jupyter notebook `00_setup.ipynb`:
   - Imports config, initialises LLMClient, runs health check against vLLM server
```

---

## MODULE 2 — Transcript Fetcher with Fallback Chain

### Prompt

```
Using the GLOBAL CONTEXT above, generate a complete `transcript_fetcher.py` module.

Implement a `TranscriptFetcher` class with a primary fetch method that tries
sources in priority order, moving to the next only on failure:

SOURCE 1 — BSE India
  - Endpoint: https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w
    params: pageno=1, strCat=Concall, strPrevDate=<90 days ago>,
    strScrip=<BSE code>, strSearch=P, strToDate=<today>, strType=C
  - Parse JSON to find PDF attachment URLs
  - Download PDF, parse with pdfplumber, extract full text
  - On HTTP != 200 or PDF not found → fall through to Source 2

SOURCE 2 — NSE India
  - Endpoint: https://www.nseindia.com/api/corp-announcements
    params: index=equities, symbol=<NSE symbol>
  - Set realistic browser headers (User-Agent, Referer, Accept)
  - Filter announcements by subject containing "concall" or "earnings call"
  - Download and parse PDF same as above
  - On failure → fall through to Source 3

SOURCE 3 — Trendlyne RSS
  - Feed: https://trendlyne.com/feeds/earning-calls-podcast-rss/
  - Parse with feedparser, fuzzy match company name (threshold 80%)
  - Extract YouTube URL, use youtube-transcript-api for text
  - On failure → fall through to Source 4

SOURCE 4 — Whisper local transcription
  - If audio URL found but transcript API failed
  - Download audio with httpx
  - Transcribe using faster-whisper (large-v3) on ROCm device
  - On total failure → raise TranscriptNotFoundError

The class must:
  - Accept: company_name, bse_code, nse_symbol, quarter (e.g. "Q4FY26")
  - Return: TranscriptResult(text, source, quarter, company, fetched_at)
  - Log each fallback attempt with loguru
  - Cache to ./transcript_cache/<company>_<quarter>.json
  - Be fully async using httpx.AsyncClient

Also generate Jupyter notebook `01_fetch_transcripts.ipynb` demonstrating
fetches for Reliance Industries and TCS showing source used and word count.
```

---

## MODULE 3 — Extraction Agent & Dual Store Ingestion

### Prompt

```
Using the GLOBAL CONTEXT above, generate a complete `extraction_agent.py` module.

Ingest raw transcript text into two separate ChromaDB collections:
- hot_store: latest quarter only, 256-token chunks, high precision
- cold_store: all historical quarters, 512-token chunks, semantic themes

Implement these classes:

CLASS TextChunker:
  - chunk(text, chunk_size, overlap=30) -> list[str]
  - Sliding window by token count (1 token ≈ 4 chars), respect sentence boundaries

CLASS SpeakerRoleExtractor:
  - extract(text) -> list[{speaker, role: "analyst"|"management"|"operator", text}]
  - Parse patterns: "Operator:", "Analyst:", "[Name]:", "Management:", "[Executive]:"

CLASS EmbeddingService:
  - Load BAAI/bge-large-en-v1.5 once at init on ROCm ("cuda")
  - embed(texts: list[str]) -> list[list[float]]
  - Batch in groups of 64 for MI300X memory safety

CLASS ExtractionAgent:
  - detect_quarter(text) -> str: extract "Q4FY26" pattern via regex
  - is_new_quarter(company, detected_quarter) -> bool: compare vs hot_store metadata
  - promote_hot_to_cold(company): copy hot chunks to cold, delete from hot
  - ingest_to_hot(result, sentiment_scores):
      chunk 256 tokens, embed, add with metadata:
      {company, quarter, chunk_index, page_estimate, speaker_role, sentiment_score}
  - ingest_to_cold(result, sentiment_scores):
      extract analyst turns only, chunk 512 tokens, embed, add with metadata:
      {company, quarter, speaker_role, topic_tag, sentiment}
  - run(result): detect quarter → check if new → promote if needed → ingest both stores

CLASS SentimentTagger:
  - Load ProsusAI/finbert at init on ROCm
  - tag(chunks) -> list[float]: score per chunk from -1.0 (negative) to +1.0 (positive)

Generate Jupyter notebook `02_extraction_agent.ipynb` showing chunk counts,
sample chunk with metadata for both stores.
```

---

## MODULE 4 — Predictive Analyst Agent

### Prompt

```
Using the GLOBAL CONTEXT above and the QUICK REFERENCE interfaces below,
generate a complete `predictive_analyst_agent.py` module.

This agent mines cold_store patterns, computes KPI delta from hot_store,
generates ranked tough questions via LLM, then gates each through hot_store.

CLASS PatternMiner:
  - mine(company, n_quarters=12) -> dict
    1. Query cold_store for all analyst-role chunks for this company
    2. Run BERTopic (min_topic_size=5) on chunk texts
    3. Per topic: count distinct quarters, compute avg sentiment
    4. Return {topics: [{topic_id, label, recurrence (0-1), avg_sentiment,
       sample_questions (top 3 chunks), quarters}], total_chunks}

CLASS KPIDeltaComputer:
  - compute(company) -> dict
    Query hot_store chunks for company, send to LLM with prompt:
    "Extract all financial KPIs with actual value and any prior guidance.
     Return JSON: [{metric, actual_value, guided_value, unit,
     is_miss: bool, delta_pct: float, quote: str}]"

CLASS QuestionGenerator:
  SYSTEM_PROMPT: "You are a senior institutional investor at a large asset
  management firm. You have followed this company for 8 quarters. You are
  skeptical and data-driven. You probe any gap between guidance and actual
  performance. Every question must reference a specific metric or statement.
  Do not ask questions management has already pre-empted."

  - generate(topics, kpi_delta, n_questions=20) -> list[AnalystQuestion]
    User prompt includes: top-15 topics with recurrence+sentiment+samples,
    KPI delta showing misses prominently.
    Returns JSON: [{question, topic, adversarial_score (0-1), why_tough,
    source_quarters}] sorted by score descending.

CLASS RelevanceGate:
  - check(questions, company, threshold=0.75)
      -> tuple[list[AnalystQuestion], list[AnalystQuestion]]
    For each question: embed → query hot_store n_results=3 → max similarity
    If >= threshold: set answerable=True, store hot_store_chunks
    Else: set answerable=False, set gap_reason

CLASS PredictiveAnalystAgent:
  - run(company) -> {answerable, data_gaps, kpi_delta, topics}

Generate Jupyter notebook `03_predictive_analyst.ipynb` showing top 10
questions with scores and the data gap list with reasons.
```

---

## MODULE 5 — Answer Agent with Self-Critique

### Prompt

```
Using the GLOBAL CONTEXT above and the QUICK REFERENCE interfaces below,
generate a complete `answer_agent.py` module.

CLASS AnswerAgent:

  SYSTEM_PROMPT_ANSWER: "You are a precise financial analyst preparing briefing
  materials for a CFO. Answer using ONLY the provided context. Be specific —
  cite exact numbers, quarters, and statements. If context is insufficient, say
  so explicitly. No speculation. 3-5 sentences maximum."

  SYSTEM_PROMPT_CRITIQUE: "You are a fact-checker reviewing financial Q&A.
  Verify every number, percentage, date, and claim in the answer is explicitly
  supported by the provided context. Flag any unsupported claim as a potential
  hallucination. Be strict."

  - async answer_single(question: AnalystQuestion) -> QAPair
    Use question.hot_store_chunks as context (pre-fetched by relevance gate)
    Generate answer via LLM

  - async self_critique(qa: QAPair) -> QAPair
    Send question + answer + supporting_chunks to LLM
    Prompt: check every claim traces to context
    Return JSON: {passes, issues, revised_answer, confidence (0-1)}
    Update qa.final_answer and qa.confidence

  - async run_batch(questions: list[AnalystQuestion]) -> list[QAPair]
    Use SGLang fork() for parallel answering in batches of 5
    Fall back to asyncio.gather if SGLang unavailable
    Run self_critique sequentially after answers are collected
    Sort by adversarial_score descending

Generate Jupyter notebook `04_answer_agent.ipynb` showing top 10 QAPairs
with critique pass/fail status and one example revised answer.
```

---

## MODULE 6 — Validation Agent

### Prompt

```
Using the GLOBAL CONTEXT above and the QUICK REFERENCE interfaces below,
generate a complete `validation_agent.py` module.

CLASS ValidationAgent:

  VALIDATION_PROMPT: "You are a senior investor relations advisor reviewing Q&A
  content before an earnings call. Evaluate on three dimensions:

  1. FACTUAL (0-1): Every number and claim must come from the quarterly filing.
     1 = all claims sourced, 0 = unsourced speculation present.

  2. TONE (0-1): Professional, calm, non-defensive. Avoid hedge words like
     'we believe', 'hopefully'. Prefer 'The company reported', 'Results showed'.

  3. COMPLETENESS (0-1): Does the answer actually address the question?
     'This is complex' = 0. Direct specific answer = 1.

  Return JSON: {factual_score, tone_score, completeness_score,
  issues: list[str], cleaned_answer: str (empty if no changes needed)}"

  - async validate_single(qa: QAPair, kpi_delta: dict) -> ValidationResult
    validation_passed = all scores >= 0.7
    If cleaned_answer non-empty: use as final_answer

  - async validate_batch(qa_pairs, kpi_delta, data_gaps) -> dict
    Returns: {validated_pairs, data_gaps, overall_quality_score, failed_count}

Generate Jupyter notebook `05_validation_agent.ipynb` showing overall quality
score, before/after for cleaned answers, and data gaps with reasons.
```

---

## MODULE 7 — Output Generation (PPTX + Cheat Sheet)

### Prompt

```
Using the GLOBAL CONTEXT above and the QUICK REFERENCE interfaces below,
generate a complete `output_generator.py` module.

CLASS CheatSheetGenerator:
  - generate(company, quarter, validated_pairs, data_gaps, kpi_delta) -> str (path)
  
  Markdown structure:
    # {Company} {Quarter} Earnings Call — CEO/CFO Preparation Brief
    ## KPI Summary vs Guidance
    Table: Metric | Guided | Actual | Delta % | Status (BEAT/MISS/IN LINE)
    ## Predicted Questions — Ranked by Difficulty
    For each pair: ### Q{n}: {question}
    Topic | Difficulty % | Sources | Suggested Answer (blockquote) | Why tough
    ## Data Gaps — Manual Preparation Required
    For each gap: question + gap_reason + "CFO must prepare manually"
    ## Quality Metrics: overall score, n answerable, n gaps

CLASS PresentationGenerator:
  - generate(company, quarter, validated_pairs, kpi_delta, topics) -> str (path)

  Slides (python-pptx, 16:9 widescreen 13.33x7.5 inches):
  SLIDE 1 — Title: "{Company} — {Quarter} Earnings Call", subtitle, date
  SLIDE 2 — KPI Dashboard: table with BEAT=green, MISS=red, IN LINE=gray rows
  SLIDE 3 — Analyst Sentiment: top 8 topics with recurrence + sentiment indicator
  SLIDES 4–N — Top 5 questions (one per slide):
    Title: "Anticipated: {topic}"
    Body: Q: {question} | Suggested talking point: {answer split to 3 bullets}
    Footer: difficulty score + source quarters
  LAST SLIDE — Appendix: data gaps bulleted list

  Styling: dark navy #1a1f36 background, white text, accent #4f9cf9,
  Calibri 36pt title / 18pt body

CLASS OutputOrchestrator:
  - run(company, quarter, validation_output, topics, kpi_delta)
      -> {cheat_sheet_path, pptx_path, summary}

Generate Jupyter notebook `06_output_generator.ipynb` running end-to-end
for one company, printing full cheat sheet and confirming PPTX slide count.
```

---

## MODULE 8 — Full Pipeline Orchestrator

### Prompt

```
Using the GLOBAL CONTEXT above and all modules 1–7, generate a complete
`pipeline.py` module that wires all agents into a single end-to-end run.

CLASS EarningsCallPipeline:
  - Init: load AppConfig, initialise all agents once (shared LLMClient and
    EmbeddingService — load models once, reuse across all agents)

  - async run(company, bse_code, nse_symbol, quarter) -> dict:
    Step 1: TranscriptFetcher.fetch → TranscriptResult (with fallback chain)
    Step 2: SentimentTagger.tag → sentiment_scores
    Step 3: ExtractionAgent.run → hot + cold store ingestion
    Step 4: PredictiveAnalystAgent.run → answerable, data_gaps, kpi_delta, topics
    Step 5: AnswerAgent.run_batch → qa_pairs
    Step 6: ValidationAgent.validate_batch → validation_output
    Step 7: OutputOrchestrator.run → cheat_sheet_path, pptx_path, summary
    Return full result dict with timing per step

  - run_sync(company, bse_code, nse_symbol, quarter) -> dict
    asyncio.run() wrapper for Jupyter use

Generate Jupyter notebook `07_full_pipeline.ipynb`:
  Cell 1: pip installs + imports
  Cell 2: Instantiate EarningsCallPipeline
  Cell 3: Run for Reliance Industries (BSE: 500325, NSE: RELIANCE, Q4FY26)
  Cell 4: Step-by-step timing table (%%time per step)
  Cell 5: Display top 5 QAPairs with IPython.display
  Cell 6: PPTX slide count + cheat sheet word count
  Cell 7: Post-call feedback stub — mark which predicted questions were
    actually asked → re-weight cold_store topic scores for next quarter

Also generate README.md with:
  - Architecture summary (2 paragraphs)
  - How to run from scratch on MI300X (numbered steps)
  - File structure tree
  - Known limitations and next steps
```

---

## QUICK REFERENCE — Shared Data Contracts

Paste this alongside any module prompt (especially 4–8) to ensure compatibility:

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class TranscriptResult:
    text: str
    source: str          # "BSE" | "NSE" | "Trendlyne" | "Whisper"
    quarter: str         # e.g. "Q4FY26"
    company: str
    fetched_at: datetime

@dataclass
class AnalystQuestion:
    question: str
    topic: str
    adversarial_score: float
    why_tough: str
    source_quarters: list[str]
    sentiment: float
    answerable: bool = True
    hot_store_chunks: list[str] = field(default_factory=list)
    gap_reason: str = ""

@dataclass
class QAPair:
    question: str
    topic: str
    adversarial_score: float
    answer: str
    supporting_chunks: list[str]
    critique_passed: bool
    critique_issues: list[str]
    revised_answer: str
    final_answer: str
    source_quarters: list[str]
    confidence: float

@dataclass
class ValidationResult:
    qa_pair: QAPair
    validation_passed: bool
    issues: list[str]
    tone_score: float
    factual_score: float
    completeness_score: float
    final_answer: str
```

---

## TIPS FOR BEST RESULTS

1. **Feed modules in order** — 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8.
   Each builds on the previous module's output types.

2. **Always prepend the GLOBAL CONTEXT block** to every prompt you send.
   It anchors the model to your exact stack.

3. **Include QUICK REFERENCE** with modules 4–8 so the LLM generates code
   using the correct dataclass shapes.

4. **One module per context window** for best output quality — large prompts
   get degraded attention on long functions.

5. **SGLang fork() on ROCm**: if unavailable, fall back to asyncio.gather
   with individual vLLM calls — same logical parallelism, simpler dependency.

6. **Benchmark**: add `%%time` magic to each major pipeline cell to measure
   MI300X throughput across embedding, inference, and vector search steps.

7. **FP16 memory**: at 70B FP16 you need ~140GB VRAM. MI300X has 192GB HBM3
   unified memory — headroom exists for concurrent embedding + inference.
   Use --gpu-memory-utilization 0.85 to leave room for ChromaDB and BERTopic.
