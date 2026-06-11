import time
import asyncio
from typing import Dict, Any
from datetime import datetime
from loguru import logger

from config import get_config
from transcript_fetcher import TranscriptFetcher
from extraction_agent import ExtractionAgent, EmbeddingService, SentimentTagger
from predictive_analyst_agent import PredictiveAnalystAgent
from answer_agent import AnswerAgent
from validation_agent import ValidationAgent
from output_generator import OutputOrchestrator

class EarningsCallPipeline:
    def __init__(self):
        logger.info("Initializing EarningsCallPipeline and loading shared AI models...")
        self.config = get_config()
        
        # Load heavy models once and share across components
        self.embedding_service = EmbeddingService()
        self.sentiment_tagger = SentimentTagger()
        
        # Instantiate agents
        self.fetcher = TranscriptFetcher()
        self.extraction_agent = ExtractionAgent(
            embedding_service=self.embedding_service,
            sentiment_tagger=self.sentiment_tagger
        )
        self.analyst_agent = PredictiveAnalystAgent(
            embedding_service=self.embedding_service
        )
        self.answer_agent = AnswerAgent()
        self.validation_agent = ValidationAgent()
        self.orchestrator = OutputOrchestrator()
        
        logger.info("All pipeline agents and models initialized successfully.")

    async def run(self, company: str, bse_code: str, nse_symbol: str, quarter: str) -> Dict[str, Any]:
        """Runs the complete multi-agent earnings call script and presentation pipeline, tracking step timings."""
        logger.info(f"=== Starting Earnings Call Pipeline for {company} ({quarter}) ===")
        
        timings = {}
        start_total = time.time()
        
        # Step 1: Transcript Fetcher
        logger.info("[PIPELINE STEP 1] Fetching Transcript...")
        start_step = time.time()
        transcript_result = await self.fetcher.fetch(
            company_name=company,
            bse_code=bse_code,
            nse_symbol=nse_symbol,
            quarter=quarter
        )
        timings["fetch_transcript"] = time.time() - start_step
        logger.info(f"Transcript fetched from source: {transcript_result.source} in {timings['fetch_transcript']:.2f}s")
        
        # Step 2: Extraction & Ingestion
        logger.info("[PIPELINE STEP 2 & 3] Ingesting into Hot and Cold stores...")
        start_step = time.time()
        extraction_summary = self.extraction_agent.run(transcript_result)
        timings["ingest_stores"] = time.time() - start_step
        logger.info(f"Ingested {extraction_summary['turns_extracted']} turns into ChromaDB in {timings['ingest_stores']:.2f}s")
        
        # Step 3: Predictive Analyst Mining and Question Generation
        logger.info("[PIPELINE STEP 4] Running Predictive Analyst Agent...")
        start_step = time.time()
        analysis = await self.analyst_agent.run(company)
        timings["predictive_analyst"] = time.time() - start_step
        logger.info(
            f"Predictive analysis complete: {len(analysis['answerable'])} answerable, "
            f"{len(analysis['data_gaps'])} gaps in {timings['predictive_analyst']:.2f}s"
        )
        
        # Step 4: Answer Generation
        logger.info("[PIPELINE STEP 5] Answering Questions...")
        start_step = time.time()
        # Answer top answerable questions (up to top_n_questions from config, say 10 to limit API load)
        questions_to_answer = analysis["answerable"][:self.config.top_n_questions]
        qa_pairs = await self.answer_agent.run_batch(questions_to_answer)
        timings["answer_generation"] = time.time() - start_step
        logger.info(f"Answered and critiqued {len(qa_pairs)} questions in {timings['answer_generation']:.2f}s")
        
        # Step 5: Validation
        logger.info("[PIPELINE STEP 6] Validating Q&A...")
        start_step = time.time()
        validation_output = await self.validation_agent.validate_batch(
            qa_pairs=qa_pairs,
            kpi_delta=analysis["kpi_delta"],
            data_gaps=analysis["data_gaps"]
        )
        timings["validation"] = time.time() - start_step
        logger.info(f"Validation complete in {timings['validation']:.2f}s. Quality score: {validation_output['overall_quality_score']}")
        
        # Step 6: Output Generation
        logger.info("[PIPELINE STEP 7] Generating PowerPoint and Cheat Sheet outputs...")
        start_step = time.time()
        outputs = self.orchestrator.run(
            company=company,
            quarter=quarter,
            validation_output=validation_output,
            topics=analysis["topics"],
            kpi_delta=analysis["kpi_delta"]
        )
        timings["output_generation"] = time.time() - start_step
        logger.info(f"Outputs generated in {timings['output_generation']:.2f}s")
        
        timings["total_execution"] = time.time() - start_total
        logger.info(f"=== Pipeline completed successfully in {timings['total_execution']:.2f}s ===")
        
        return {
            "company": company,
            "quarter": quarter,
            "timings": timings,
            "outputs": outputs,
            "overall_quality_score": validation_output["overall_quality_score"],
            "qa_pairs_count": len(qa_pairs),
            "data_gaps_count": len(analysis["data_gaps"]),
            "validation_results": validation_output["validated_pairs"],
            "data_gaps": analysis["data_gaps"],
            "kpi_delta": analysis["kpi_delta"],
            "topics": analysis["topics"]
        }

    def run_sync(self, company: str, bse_code: str, nse_symbol: str, quarter: str) -> Dict[str, Any]:
        """Synchronous wrapper for Jupyter notebook run."""
        return asyncio.run(self.run(company, bse_code, nse_symbol, quarter))
