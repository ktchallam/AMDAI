import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass, field
from loguru import logger
from pydantic import BaseModel, Field

from config import get_config
from llm_client import LLMClient
from predictive_analyst_agent import AnalystQuestion

# Shared Data Contract
@dataclass
class QAPair:
    question: str
    topic: str
    adversarial_score: float
    answer: str
    supporting_chunks: List[str]
    critique_passed: bool
    critique_issues: List[str]
    revised_answer: str
    final_answer: str
    source_quarters: List[str]
    confidence: float

# Pydantic schema for critique validation
class CritiqueResultSchema(BaseModel):
    passes: bool = Field(description="True if all claims in the answer are strictly supported by the context")
    issues: List[str] = Field(description="List of unsupported claims, discrepancies, or potential hallucinations found")
    revised_answer: str = Field(description="A revised version of the answer correcting any flagged issues, or empty if the original is perfect")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 based on context alignment")

class AnswerAgent:
    def __init__(self):
        self.config = get_config()
        self.llm_client = LLMClient()
        self.system_prompt_answer = (
            "You are a precise financial analyst preparing briefing materials for a CFO. "
            "Answer using ONLY the provided context. Be specific — cite exact numbers, quarters, and statements. "
            "If context is insufficient, say so explicitly. No speculation. 3-5 sentences maximum."
        )
        self.system_prompt_critique = (
            "You are a fact-checker reviewing financial Q&A. "
            "Verify every number, percentage, date, and claim in the answer is explicitly supported by the provided context. "
            "Flag any unsupported claim as a potential hallucination. Be strict."
        )

    async def answer_single(self, question: AnalystQuestion) -> QAPair:
        """Generates an initial answer for a question based on its hot_store_chunks."""
        logger.info(f"Answering question: '{question.question[:50]}...'")
        
        context_str = "\n\n".join([f"Context chunk:\n{chunk}" for chunk in question.hot_store_chunks])
        user_prompt = (
            f"Context details:\n{context_str}\n\n"
            f"Question: {question.question}\n\n"
            f"Provide a precise, grounded answer."
        )
        
        answer = await self.llm_client.generate(
            system=self.system_prompt_answer,
            user=user_prompt,
            max_tokens=600
        )
        
        # Initialize default QAPair
        return QAPair(
            question=question.question,
            topic=question.topic,
            adversarial_score=question.adversarial_score,
            answer=answer,
            supporting_chunks=question.hot_store_chunks,
            critique_passed=False,
            critique_issues=[],
            revised_answer="",
            final_answer=answer,
            source_quarters=question.source_quarters,
            confidence=0.5
        )

    async def self_critique(self, qa: QAPair) -> QAPair:
        """Verifies the generated answer against context and revises if necessary."""
        logger.info(f"Running self-critique on answer for question: '{qa.question[:50]}...'")
        
        context_str = "\n\n".join([f"Context chunk:\n{chunk}" for chunk in qa.supporting_chunks])
        user_prompt = (
            f"Supporting Context:\n{context_str}\n\n"
            f"Question: {qa.question}\n\n"
            f"Proposed Answer: {qa.answer}\n\n"
            f"Analyze the answer word-by-word. Verify all facts and numbers. "
            f"Return a structured evaluation JSON containing 'passes', 'issues', 'revised_answer', and 'confidence'."
        )
        
        try:
            critique = await self.llm_client.generate_json(
                system=self.system_prompt_critique,
                user=user_prompt,
                schema=CritiqueResultSchema,
                max_tokens=800
            )
            
            qa.critique_passed = critique.passes
            qa.critique_issues = critique.issues
            qa.confidence = critique.confidence
            
            if critique.passes or not critique.revised_answer:
                qa.final_answer = qa.answer
                qa.revised_answer = ""
            else:
                qa.revised_answer = critique.revised_answer
                qa.final_answer = critique.revised_answer
                logger.info(f"Answer revised after critique. Confidence set to {critique.confidence}")
                
        except Exception as e:
            logger.error(f"Self-critique failed: {e}. Keeping original answer.")
            qa.critique_passed = False
            qa.critique_issues = [f"Critique process encountered an error: {str(e)}"]
            qa.final_answer = qa.answer
            
        return qa

    async def run_batch(self, questions: List[AnalystQuestion]) -> List[QAPair]:
        """Answers questions in parallel batches, runs self-critique sequentially, and returns sorted QAPairs."""
        if not questions:
            return []
            
        logger.info(f"Processing batch of {len(questions)} questions...")
        
        # Parallel answer generation in batches of 5
        qa_pairs = []
        batch_size = 5
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]
            tasks = [self.answer_single(q) for q in batch]
            # Parallel execution via asyncio.gather (fallback to SGLang orchestration)
            results = await asyncio.gather(*tasks)
            qa_pairs.extend(results)
            
        # Run self-critique sequentially
        critiqued_pairs = []
        for qa in qa_pairs:
            critiqued = await self.self_critique(qa)
            critiqued_pairs.append(critiqued)
            
        # Sort by adversarial_score descending
        sorted_pairs = sorted(critiqued_pairs, key=lambda qa: qa.adversarial_score, reverse=True)
        
        logger.info("Batch Q&A generation and validation finished.")
        return sorted_pairs
