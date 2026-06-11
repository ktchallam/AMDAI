import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from loguru import logger
from pydantic import BaseModel, Field

from config import get_config
from llm_client import LLMClient
from answer_agent import QAPair

# Shared Data Contract
@dataclass
class ValidationResult:
    qa_pair: QAPair
    validation_passed: bool
    issues: List[str]
    tone_score: float
    factual_score: float
    completeness_score: float
    final_answer: str

# Pydantic schema for validation response
class ValidationFeedbackSchema(BaseModel):
    factual_score: float = Field(description="Score between 0.0 and 1.0 indicating if all facts trace back to context")
    tone_score: float = Field(description="Score between 0.0 and 1.0 indicating if the tone is objective, professional, and free of defensive/hedged phrasing")
    completeness_score: float = Field(description="Score between 0.0 and 1.0 indicating if the answer directly and fully addresses the question")
    issues: List[str] = Field(description="List of issues, hedging language, or unsourced claims found")
    cleaned_answer: str = Field(description="Optimized answer with tone corrections, or empty if no adjustments are needed")

class ValidationAgent:
    def __init__(self):
        self.config = get_config()
        self.llm_client = LLMClient()
        self.validation_prompt = (
            "You are a senior investor relations advisor reviewing Q&A content before an earnings call. "
            "Evaluate on three dimensions:\n"
            "1. FACTUAL (0-1): Every number and claim must come from the quarterly filing. "
            "1 = all claims sourced, 0 = unsourced speculation present.\n"
            "2. TONE (0-1): Professional, calm, non-defensive. Avoid hedge words like "
            "'we believe', 'hopefully'. Prefer 'The company reported', 'Results showed'.\n"
            "3. COMPLETENESS (0-1): Does the answer actually address the question? "
            "'This is complex' = 0. Direct specific answer = 1.\n\n"
            "Evaluate the answer carefully and return a structured JSON conforming to the schema."
        )

    async def validate_single(self, qa: QAPair, kpi_delta: Dict[str, Any]) -> ValidationResult:
        """Evaluates a single QAPair on factuality, tone, and completeness. Cleans the answer if needed."""
        logger.info(f"Validating QAPair for question: '{qa.question[:50]}...'")
        
        context_str = "\n\n".join([f"Context:\n{chunk}" for chunk in qa.supporting_chunks])
        kpi_summary = json.dumps(kpi_delta)
        
        user_prompt = (
            f"Context: {context_str}\n\n"
            f"KPI Delta Reference: {kpi_summary}\n\n"
            f"Question: {qa.question}\n"
            f"Proposed Answer: {qa.final_answer}\n\n"
            f"Evaluate and score this QAPair. If tone can be made more professional or hedging can be removed, "
            f"provide the improved version in 'cleaned_answer'."
        )
        
        try:
            feedback = await self.llm_client.generate_json(
                system=self.validation_prompt,
                user=user_prompt,
                schema=ValidationFeedbackSchema,
                max_tokens=800
            )
            
            # validation passes if all scores are >= 0.7
            passed = (
                feedback.factual_score >= 0.7 and
                feedback.tone_score >= 0.7 and
                feedback.completeness_score >= 0.7
            )
            
            final_ans = qa.final_answer
            if feedback.cleaned_answer and feedback.cleaned_answer.strip():
                final_ans = feedback.cleaned_answer.strip()
                logger.info(f"Applying cleaned/improved answer from validation for: '{qa.question[:30]}...'")
                
            return ValidationResult(
                qa_pair=qa,
                validation_passed=passed,
                issues=feedback.issues,
                tone_score=feedback.tone_score,
                factual_score=feedback.factual_score,
                completeness_score=feedback.completeness_score,
                final_answer=final_ans
            )
        except Exception as e:
            logger.error(f"Validation failed for: {qa.question[:30]}. Error: {e}")
            # Safe fallback
            return ValidationResult(
                qa_pair=qa,
                validation_passed=True,
                issues=[f"Validation process failed: {str(e)}"],
                tone_score=1.0,
                factual_score=1.0,
                completeness_score=1.0,
                final_answer=qa.final_answer
            )

    async def validate_batch(self, qa_pairs: List[QAPair], kpi_delta: Dict[str, Any], data_gaps: List[Any]) -> Dict[str, Any]:
        """Evaluates a batch of QAPairs, computes overall metrics, and returns a summary dict."""
        logger.info(f"Validating batch of {len(qa_pairs)} QAPairs...")
        
        # Parallel validation checks
        tasks = [self.validate_single(qa, kpi_delta) for qa in qa_pairs]
        validated_results = await asyncio.gather(*tasks)
        
        failed_count = sum(1 for res in validated_results if not res.validation_passed)
        
        # Calculate overall quality score as the average of the three dimension scores
        total_score = 0.0
        for res in validated_results:
            total_score += (res.factual_score + res.tone_score + res.completeness_score) / 3.0
            
        overall_quality = total_score / len(validated_results) if validated_results else 1.0
        
        # In case cleaned_answer was set, update the original QAPair final_answer
        for res in validated_results:
            res.qa_pair.final_answer = res.final_answer
            
        return {
            "validated_pairs": validated_results,
            "data_gaps": data_gaps,
            "overall_quality_score": round(overall_quality, 2),
            "failed_count": failed_count
        }
