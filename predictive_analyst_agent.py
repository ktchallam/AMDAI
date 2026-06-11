import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger
from pydantic import BaseModel, Field

from config import get_config
from llm_client import LLMClient
from extraction_agent import EmbeddingService

# Import Chroma
import chromadb

# Share Data Contracts
@dataclass
class AnalystQuestion:
    question: str
    topic: str
    adversarial_score: float
    why_tough: str
    source_quarters: List[str]
    sentiment: float
    answerable: bool = True
    hot_store_chunks: List[str] = field(default_factory=list)
    gap_reason: str = ""

# Pydantic Schemas for LLM validation
class KPIDelta(BaseModel):
    metric: str = Field(description="Name of the financial metric, e.g., Revenue, EBITDA margin")
    actual_value: float = Field(description="Actual value reported")
    guided_value: float = Field(description="Guided or expected value")
    unit: str = Field(description="Unit of measurement, e.g., %, USD Millions, INR Crores")
    is_miss: bool = Field(description="True if actual performance missed the guidance")
    delta_pct: float = Field(description="Percentage difference between actual and guidance")
    quote: str = Field(description="Verbatim quote from the transcript supporting this KPI")

class KPIDeltaList(BaseModel):
    kpis: List[KPIDelta]

class AnalystQuestionSchema(BaseModel):
    question: str = Field(description="The tough question an analyst would ask")
    topic: str = Field(description="Topic label")
    adversarial_score: float = Field(description="Adversarial score between 0.0 and 1.0")
    why_tough: str = Field(description="Explanation of why this question is tough for management")
    source_quarters: List[str] = Field(description="Historical quarters this question relates to")
    sentiment: float = Field(description="Sentiment score (-1.0 to 1.0)")

class AnalystQuestionListSchema(BaseModel):
    questions: List[AnalystQuestionSchema]

class PatternMiner:
    def __init__(self, cold_collection):
        self.cold_collection = cold_collection

    def mine(self, company: str, n_quarters: int = 12) -> Dict[str, Any]:
        """Mines historical cold_store patterns, computes topic recurrence and average sentiment."""
        logger.info(f"Mining cold_store patterns for {company}...")
        
        # 1. Query cold_store for all analyst-role chunks for this company
        results = self.cold_collection.get(
            where={"company": company, "speaker_role": "analyst"},
            include=["documents", "metadatas"]
        )
        
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        
        if not documents:
            logger.warning("No historical analyst turns found in cold_store. Returning empty topics.")
            return {"topics": [], "total_chunks": 0}
            
        logger.info(f"Found {len(documents)} analyst chunks in cold_store.")
        
        # 2. Run BERTopic on chunk texts
        topics_list = []
        try:
            from bertopic import BERTopic
            # Limit min_topic_size based on document count
            min_topic_size = min(5, len(documents))
            if min_topic_size < 2:
                raise ValueError("Too few documents for BERTopic.")
                
            topic_model = BERTopic(min_topic_size=min_topic_size)
            topics, probs = topic_model.fit_transform(documents)
            
            # Map topic model results
            freq = topic_model.get_topic_info()
            # Loop over identified topics (excluding outlier topic -1 if possible, or include it as General)
            for idx, row in freq.iterrows():
                topic_id = row['Topic']
                if topic_id == -1 and len(freq) > 1:
                    continue # Skip outliers if there are other topics
                    
                topic_words = topic_model.get_topic(topic_id)
                label = ", ".join([word[0] for word in topic_words[:3]]) if topic_words else f"Topic {topic_id}"
                
                # Filter documents in this topic
                topic_doc_indices = [i for i, t in enumerate(topics) if t == topic_id]
                topic_docs = [documents[i] for i in topic_doc_indices]
                topic_metas = [metadatas[i] for i in topic_doc_indices]
                
                # Analyze quarters and sentiments
                quarters = list(set([m.get("quarter", "") for m in topic_metas if m.get("quarter")]))
                avg_sentiment = float(np.mean([m.get("sentiment", 0.0) for m in topic_metas]))
                recurrence = len(quarters) / n_quarters if n_quarters > 0 else 0.0
                recurrence = min(1.0, recurrence)
                
                # Select top 3 sample questions (we can just take the first 3 or longest 3 documents)
                sample_questions = sorted(topic_docs, key=len, reverse=True)[:3]
                
                topics_list.append({
                    "topic_id": int(topic_id),
                    "label": label,
                    "recurrence": recurrence,
                    "avg_sentiment": avg_sentiment,
                    "sample_questions": sample_questions,
                    "quarters": quarters
                })
        except Exception as e:
            logger.warning(f"BERTopic execution failed: {str(e)}. Falling back to LLM-based pattern mining.")
            topics_list = self._mine_fallback_llm(documents, metadatas, n_quarters)
            
        return {
            "topics": topics_list,
            "total_chunks": len(documents)
        }

    def _mine_fallback_llm(self, documents: List[str], metadatas: List[Dict[str, Any]], n_quarters: int) -> List[Dict[str, Any]]:
        """Fallback pattern mining using LLM grouping to handle sparse/failing BERTopic cases."""
        # Simple heuristic: group documents into 3-5 categories based on basic keyword matching
        # Or generate a list of topics using the LLM for a sample of documents
        import asyncio
        llm = LLMClient()
        
        # Take a sample of documents to avoid context overflow
        sample_docs = documents[:30]
        
        system_prompt = (
            "You are a financial analyst. Group the following list of analyst questions/remarks "
            "into 3-5 distinct thematic topics. For each topic, provide a short descriptive label (2-3 words)."
        )
        user_prompt = f"Documents:\n" + "\n".join([f"- {doc[:200]}" for doc in sample_docs])
        
        class TopicItem(BaseModel):
            label: str
            description: str
            keywords: List[str]
            
        class TopicGroup(BaseModel):
            topics: List[TopicItem]
            
        try:
            # Generate topics list synchronously inside async context or wrap it
            loop = asyncio.get_event_loop()
            topic_group = loop.run_until_complete(
                llm.generate_json(system_prompt, user_prompt, TopicGroup)
            )
            
            topics_list = []
            # Map documents to the generated topics
            for idx, topic_item in enumerate(topic_group.topics):
                # Match documents containing any of the keywords
                topic_docs = []
                topic_metas = []
                keywords = [kw.lower() for kw in topic_item.keywords]
                
                for doc, meta in zip(documents, metadatas):
                    doc_lower = doc.lower()
                    if any(kw in doc_lower for kw in keywords):
                        topic_docs.append(doc)
                        topic_metas.append(meta)
                        
                # Default to sample if empty
                if not topic_docs:
                    topic_docs = sample_docs[:2]
                    topic_metas = metadatas[:2]
                    
                quarters = list(set([m.get("quarter", "") for m in topic_metas if m.get("quarter")]))
                avg_sentiment = float(np.mean([m.get("sentiment", 0.0) for m in topic_metas])) if topic_metas else 0.0
                recurrence = len(quarters) / n_quarters if n_quarters > 0 else 0.0
                recurrence = min(1.0, recurrence)
                sample_questions = sorted(topic_docs, key=len, reverse=True)[:3]
                
                topics_list.append({
                    "topic_id": idx,
                    "label": topic_item.label,
                    "recurrence": recurrence,
                    "avg_sentiment": avg_sentiment,
                    "sample_questions": sample_questions,
                    "quarters": quarters
                })
            return topics_list
        except Exception as ex:
            logger.error(f"Fallback LLM mining failed: {ex}. Using basic rule-based topic grouping.")
            # Simple rule-based fallback
            quarters = list(set([m.get("quarter", "") for m in metadatas if m.get("quarter")]))
            avg_sentiment = float(np.mean([m.get("sentiment", 0.0) for m in metadatas])) if metadatas else 0.0
            return [{
                "topic_id": 0,
                "label": "Financial & Operational Performance",
                "recurrence": len(quarters) / n_quarters,
                "avg_sentiment": avg_sentiment,
                "sample_questions": documents[:3],
                "quarters": quarters
            }]

class KPIDeltaComputer:
    def __init__(self, hot_collection):
        self.hot_collection = hot_collection

    async def compute(self, company: str) -> Dict[str, Any]:
        """Queries hot_store chunks, sends to LLM to extract financial KPIs comparing actual vs guidance."""
        logger.info(f"Computing KPI delta from hot_store for {company}...")
        
        # Get management chunks (guidance is typically issued by management)
        results = self.hot_collection.get(
            where={"company": company, "speaker_role": "management"},
            include=["documents"]
        )
        documents = results.get("documents", [])
        if not documents:
            # Fallback to all chunks if management chunks empty
            results = self.hot_collection.get(
                where={"company": company},
                include=["documents"]
            )
            documents = results.get("documents", [])
            
        context_text = "\n".join(documents[:40]) # limit context size
        
        llm = LLMClient()
        system_prompt = (
            "You are a senior financial analyst. Extract all financial KPIs (Revenue, Margins, Net Profit, CapEx, etc.) "
            "where BOTH an actual value and a guided/forecasted value are mentioned. "
            "Ensure that you calculate the delta percentage correctly: ((Actual - Guided) / Guided) * 100."
        )
        user_prompt = (
            f"Here is the earnings call transcript context for {company}:\n\n"
            f"{context_text}\n\n"
            "Extract all KPIs and return them as a JSON list matching the schema provided."
        )
        
        try:
            kpi_list = await llm.generate_json(system_prompt, user_prompt, KPIDeltaList)
            return {"kpis": [kpi.model_dump() for kpi in kpi_list.kpis]}
        except Exception as e:
            logger.error(f"KPI Delta computation failed: {e}. Returning empty list.")
            return {"kpis": []}

class QuestionGenerator:
    def __init__(self):
        self.config = get_config()
        self.system_prompt = (
            "You are a senior institutional investor at a large asset management firm. "
            "You have followed this company for 8 quarters. You are skeptical and data-driven. "
            "You probe any gap between guidance and actual performance. "
            "Every question must reference a specific metric or statement. "
            "Do not ask questions management has already pre-empted."
        )

    async def generate(self, topics: List[Dict[str, Any]], kpi_delta: Dict[str, Any], n_questions: int = 20) -> List[AnalystQuestion]:
        """Generates ranked tough questions via LLM based on historical topics and KPI misses."""
        logger.info(f"Generating {n_questions} predicted analyst questions...")
        llm = LLMClient()
        
        # Format input details for LLM prompt
        topics_str = ""
        for idx, t in enumerate(topics[:15]):
            topics_str += (
                f"{idx+1}. Topic: {t['label']} | Recurrence: {t['recurrence']:.2f} | "
                f"Historical Sentiment: {t['avg_sentiment']:.2f}\n"
                f"   Sample historical questions:\n"
            )
            for q in t['sample_questions'][:2]:
                topics_str += f"   - {q[:150]}\n"
                
        kpis_str = ""
        for k in kpi_delta.get("kpis", []):
            status = "MISS" if k["is_miss"] else "BEAT"
            kpis_str += (
                f"- Metric: {k['metric']} | Actual: {k['actual_value']}{k['unit']} | "
                f"Guided: {k['guided_value']}{k['unit']} | Delta: {k['delta_pct']:.2f}% | "
                f"Status: {status} | Quote: '{k['quote']}'\n"
            )
            
        user_prompt = (
            f"Here are the recurring historical concern topics mined from the past 12 quarters:\n"
            f"{topics_str}\n"
            f"Here are the current quarter's KPI performance data (Actual vs Guidance):\n"
            f"{kpis_str}\n\n"
            f"Generate exactly {n_questions} tough, adversarial analyst questions that probe these KPI misses "
            f"and recurring concern areas. Rank them by difficulty/adversarial nature.\n"
            f"For each question, specify:\n"
            f"- question: The text of the question\n"
            f"- topic: The topic it relates to\n"
            f"- adversarial_score: Score between 0.0 (low) and 1.0 (high/tough)\n"
            f"- why_tough: Reason why this is difficult for management to answer\n"
            f"- source_quarters: Quarters where this issue was raised historically\n"
            f"- sentiment: Expected sentiment score (-1.0 to 1.0)\n"
        )
        
        try:
            res = await llm.generate_json(self.system_prompt, user_prompt, AnalystQuestionListSchema)
            
            # Sort questions by adversarial score descending
            sorted_questions = sorted(res.questions, key=lambda q: q.adversarial_score, reverse=True)
            
            questions = []
            for q in sorted_questions:
                questions.append(AnalystQuestion(
                    question=q.question,
                    topic=q.topic,
                    adversarial_score=q.adversarial_score,
                    why_tough=q.why_tough,
                    source_quarters=q.source_quarters,
                    sentiment=q.sentiment
                ))
            return questions
        except Exception as e:
            logger.error(f"Question generation failed: {e}. Returning fallback question.")
            return [AnalystQuestion(
                question="Can you explain the EBITDA margin compression this quarter?",
                topic="Margins",
                adversarial_score=0.8,
                why_tough="Management guided 24% and delivered 22.5%.",
                source_quarters=["Q3FY25"],
                sentiment=-0.5
            )]

class RelevanceGate:
    def __init__(self, hot_collection, embedding_service: EmbeddingService):
        self.hot_collection = hot_collection
        self.embedding_service = embedding_service
        self.config = get_config()

    def check(self, questions: List[AnalystQuestion], company: str, threshold: float = 0.75) -> Tuple[List[AnalystQuestion], List[AnalystQuestion]]:
        """Gates generated questions by checking if they are answerable using hot_store context."""
        logger.info(f"Running RelevanceGate checks (threshold={threshold}) for {len(questions)} questions...")
        
        answerable = []
        data_gaps = []
        
        for q in questions:
            # Embed question
            q_embed = self.embedding_service.embed([q.question])[0]
            
            # Query hot_store
            results = self.hot_collection.query(
                query_embeddings=[q_embed],
                n_results=3,
                where={"company": company}
            )
            
            # Extract documents and distances
            docs = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            max_sim = 0.0
            if distances:
                # If collection space is cosine, distance = 1 - similarity. So, similarity = 1 - distance.
                # If space is L2, distance = L2 squared. Let's handle both.
                # Assuming collection metadata hnsw:space is cosine, let's calculate similarity.
                # In standard ChromaDB, if space is L2, similarity = 1.0 / (1.0 + distance)
                # Let's write a formula that works for both.
                # We configured cosine space, so similarity = 1.0 - distance
                # Let's safeguard to keep similarity between 0 and 1.
                dist = distances[0]
                similarity = 1.0 - dist
                max_sim = max(0.0, min(1.0, similarity))
                
            logger.debug(f"Question: '{q.question[:40]}...' Max Similarity = {max_sim:.4f}")
            
            if max_sim >= threshold:
                q.answerable = True
                q.hot_store_chunks = docs
                answerable.append(q)
            else:
                q.answerable = False
                q.gap_reason = f"No detailed contextual disclosure found in Q4FY26 (Max Cosine Similarity: {max_sim:.2f} < threshold {threshold})"
                data_gaps.append(q)
                
        logger.info(f"RelevanceGate complete. Answerable: {len(answerable)}, Data Gaps: {len(data_gaps)}")
        return answerable, data_gaps

class PredictiveAnalystAgent:
    def __init__(self, embedding_service: EmbeddingService):
        self.config = get_config()
        self.embedding_service = embedding_service
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=self.config.chroma_persist_dir)
        # Configure collections with Cosine distance metric for Relevance Gate
        self.hot_collection = self.chroma_client.get_or_create_collection(
            name=self.config.hot_store_collection,
            metadata={"hnsw:space": "cosine"}
        )
        self.cold_collection = self.chroma_client.get_or_create_collection(
            name=self.config.cold_store_collection,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.miner = PatternMiner(self.cold_collection)
        self.computer = KPIDeltaComputer(self.hot_collection)
        self.generator = QuestionGenerator()
        self.gate = RelevanceGate(self.hot_collection, self.embedding_service)

    async def run(self, company: str) -> Dict[str, Any]:
        """Runs the complete predictive analyst agent pipeline."""
        # Step 1: Mine topic patterns
        topics_summary = self.miner.mine(company)
        topics = topics_summary.get("topics", [])
        
        # Step 2: Compute KPI deltas
        kpi_delta = await self.computer.compute(company)
        
        # Step 3: Generate tough questions
        questions = await self.generator.generate(
            topics=topics,
            kpi_delta=kpi_delta,
            n_questions=self.config.top_n_questions
        )
        
        # Step 4: Gate questions using RelevanceGate
        answerable, data_gaps = self.gate.check(
            questions=questions,
            company=company,
            threshold=self.config.relevance_gate_threshold
        )
        
        return {
            "answerable": answerable,
            "data_gaps": data_gaps,
            "kpi_delta": kpi_delta,
            "topics": topics
        }
