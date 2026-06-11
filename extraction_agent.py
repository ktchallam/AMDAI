import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import torch
import chromadb
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

from config import get_config
from transcript_fetcher import TranscriptResult

class TextChunker:
    def chunk(self, text: str, chunk_size: int, overlap: int = 30) -> List[str]:
        """Chunks text into sliding window of chunk_size tokens (1 token ≈ 4 chars) respecting sentences."""
        if not text:
            return []
        
        # Split text into sentences using simple regex
        sentences = re.split(r'(?<=[.?!])\s+', text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Heuristic: 1 token ≈ 4 chars
            sentence_tokens = max(1, len(sentence) // 4)
            
            if current_tokens + sentence_tokens > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                # Slide window: preserve some sentences for overlap
                overlap_chunk = []
                overlap_tokens = 0
                for s in reversed(current_chunk):
                    s_tokens = max(1, len(s) // 4)
                    if overlap_tokens + s_tokens <= overlap:
                        overlap_chunk.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break
                current_chunk = overlap_chunk
                current_tokens = overlap_tokens
                
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

class SpeakerRoleExtractor:
    def extract(self, text: str) -> List[Dict[str, str]]:
        """Parses transcript text and separates it into speaker turns with roles."""
        lines = text.split("\n")
        turns = []
        current_speaker = "Operator"
        current_role = "operator"
        current_text = []
        
        # Matches: "Name:", "Name (Company):", "[Name]:", etc.
        speaker_regex = re.compile(r'^\[?([A-Z][a-zA-Z\s.\-]+(?:\s\([^)]+\))?)\]?\s*:(.*)$')
        
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
                
            match = speaker_regex.match(line_str)
            if match:
                # Save previous speaker turn
                if current_text:
                    turns.append({
                        "speaker": current_speaker,
                        "role": current_role,
                        "text": " ".join(current_text)
                    })
                    current_text = []
                
                current_speaker = match.group(1).strip()
                current_role = self._determine_role(current_speaker)
                content = match.group(2).strip()
                if content:
                    current_text.append(content)
            else:
                current_text.append(line_str)
                
        if current_text:
            turns.append({
                "speaker": current_speaker,
                "role": current_role,
                "text": " ".join(current_text)
            })
            
        return turns

    def _determine_role(self, speaker: str) -> str:
        name_lower = speaker.lower()
        if "operator" in name_lower or "facilitator" in name_lower or "moderator" in name_lower:
            return "operator"
        if "(" in name_lower or "analyst" in name_lower or "research" in name_lower:
            return "analyst"
        return "management"

class EmbeddingService:
    def __init__(self):
        config = get_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing EmbeddingService using {config.embedding_model} on device={self.device}")
        self.model = SentenceTransformer(config.embedding_model, device=self.device)
        
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings in batches of 64 for memory safety."""
        if not texts:
            return []
        
        all_embeddings = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = self.model.encode(batch, convert_to_numpy=True)
            all_embeddings.extend(embeddings.tolist())
        return all_embeddings

class SentimentTagger:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing SentimentTagger using ProsusAI/finbert on device={self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(self.device)
        
    def tag(self, chunks: List[str]) -> List[float]:
        """Tags text chunks with a sentiment score from -1.0 (negative) to +1.0 (positive)."""
        if not chunks:
            return []
            
        scores = []
        batch_size = 16
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                
            # FinBERT labels: 0 -> positive, 1 -> negative, 2 -> neutral
            for p in probs:
                p_pos = p[0].item()
                p_neg = p[1].item()
                scores.append(p_pos - p_neg)
        return scores

class ExtractionAgent:
    def __init__(self, embedding_service: EmbeddingService, sentiment_tagger: SentimentTagger):
        self.config = get_config()
        self.embedding_service = embedding_service
        self.sentiment_tagger = sentiment_tagger
        self.chunker = TextChunker()
        self.speaker_extractor = SpeakerRoleExtractor()
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=self.config.chroma_persist_dir)
        self.hot_collection = self.chroma_client.get_or_create_collection(name=self.config.hot_store_collection)
        self.cold_collection = self.chroma_client.get_or_create_collection(name=self.config.cold_store_collection)
        
    def detect_quarter(self, text: str) -> str:
        """Extracts quarter pattern (e.g. Q4FY26) via regex."""
        match = re.search(r'(Q[1-4]\s*FY\s*\d{2,4})|(Q[1-4]\s*\d{4})', text, re.IGNORECASE)
        if match:
            # Normalize to QXFYXX
            val = match.group(0).upper().replace(" ", "")
            return val
        return "Q4FY26"  # Default fallback

    def is_new_quarter(self, company: str, detected_quarter: str) -> bool:
        """Compares detected quarter against hot_store metadata."""
        results = self.hot_collection.get(
            where={"company": company},
            limit=1
        )
        if results and results.get("metadatas"):
            existing_quarter = results["metadatas"][0].get("quarter")
            if existing_quarter and existing_quarter != detected_quarter:
                logger.info(f"New quarter detected: {detected_quarter} vs existing {existing_quarter} for {company}")
                return True
        return False

    def promote_hot_to_cold(self, company: str):
        """Copies hot store chunks to cold store and deletes from hot store."""
        logger.info(f"Promoting hot_store chunks to cold_store for {company}...")
        results = self.hot_collection.get(
            where={"company": company},
            include=["documents", "embeddings", "metadatas"]
        )
        
        if results and results.get("ids"):
            ids = results["ids"]
            documents = results["documents"]
            embeddings = results["embeddings"]
            metadatas = results["metadatas"]
            
            # Write to cold store (all quarters are saved here)
            cold_ids = [f"{company}_cold_{idx}" for idx in ids]
            
            # Map hot metadata to cold metadata format
            cold_metadatas = []
            for meta in metadatas:
                cold_metadatas.append({
                    "company": meta.get("company"),
                    "quarter": meta.get("quarter"),
                    "speaker_role": meta.get("speaker_role"),
                    "topic_tag": "unassigned",
                    "sentiment": meta.get("sentiment_score", 0.0)
                })
                
            self.cold_collection.add(
                ids=cold_ids,
                embeddings=embeddings,
                metadatas=cold_metadatas,
                documents=documents
            )
            
            # Delete from hot store
            self.hot_collection.delete(ids=ids)
            logger.info(f"Successfully promoted {len(ids)} chunks to cold_store.")
        else:
            logger.info("No hot_store chunks found to promote.")

    def ingest_to_hot(self, result: TranscriptResult, detected_quarter: str, turns: List[Dict[str, str]]):
        """Ingests transcript turns into hot_store using 256-token chunks."""
        logger.info(f"Ingesting into hot_store for {result.company} ({detected_quarter})...")
        
        chunks = []
        metadatas = []
        chunk_index = 0
        
        for turn in turns:
            role = turn["role"]
            speaker = turn["speaker"]
            text = turn["text"]
            
            # Chunk each turn's text
            turn_chunks = self.chunker.chunk(text, chunk_size=self.config.hot_store_chunk_size)
            for chunk_text in turn_chunks:
                # Index/page calculations
                page_est = chunk_index // 4 + 1
                
                chunks.append(chunk_text)
                metadatas.append({
                    "company": result.company,
                    "quarter": detected_quarter,
                    "chunk_index": chunk_index,
                    "page_estimate": page_est,
                    "speaker_role": role,
                    "speaker_name": speaker,
                    # Placeholder for sentiment score; will update after tagging
                    "sentiment_score": 0.0
                })
                chunk_index += 1
                
        if not chunks:
            logger.warning("No chunks generated for hot_store.")
            return
            
        # Sentiment tagging
        sentiments = self.sentiment_tagger.tag(chunks)
        for idx, score in enumerate(sentiments):
            metadatas[idx]["sentiment_score"] = score
            
        # Embedding service
        embeddings = self.embedding_service.embed(chunks)
        
        # Save to hot collection
        ids = [f"{result.company}_hot_{idx}" for idx in range(len(chunks))]
        
        # Clean existing hot store data for this company before inserting new ones
        self.hot_collection.delete(where={"company": result.company})
        
        self.hot_collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=chunks
        )
        logger.info(f"Ingested {len(chunks)} chunks into hot_store.")

    def ingest_to_cold(self, result: TranscriptResult, detected_quarter: str, turns: List[Dict[str, str]]):
        """Ingests analyst turns only into cold_store using 512-token chunks."""
        logger.info(f"Ingesting analyst turns into cold_store for {result.company} ({detected_quarter})...")
        
        # Extract analyst turns
        analyst_turns = [turn for turn in turns if turn["role"] == "analyst"]
        
        chunks = []
        metadatas = []
        
        for idx, turn in enumerate(analyst_turns):
            text = turn["text"]
            speaker = turn["speaker"]
            
            turn_chunks = self.chunker.chunk(text, chunk_size=self.config.cold_store_chunk_size)
            for chunk_text in turn_chunks:
                chunks.append(chunk_text)
                metadatas.append({
                    "company": result.company,
                    "quarter": detected_quarter,
                    "speaker_role": "analyst",
                    "speaker_name": speaker,
                    "topic_tag": "unassigned",
                    "sentiment": 0.0
                })
                
        if not chunks:
            logger.info("No analyst chunks found for cold_store.")
            return
            
        # Sentiment tagging
        sentiments = self.sentiment_tagger.tag(chunks)
        for idx, score in enumerate(sentiments):
            metadatas[idx]["sentiment"] = score
            
        # Embedding
        embeddings = self.embedding_service.embed(chunks)
        
        # Save to cold collection
        ids = [f"{result.company}_cold_{detected_quarter}_{idx}" for idx in range(len(chunks))]
        self.cold_collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=chunks
        )
        logger.info(f"Ingested {len(chunks)} analyst chunks into cold_store.")

    def run(self, result: TranscriptResult) -> Dict[str, Any]:
        """Detects quarter, promotes old hot collection to cold, and ingests both collections."""
        detected_quarter = self.detect_quarter(result.text)
        logger.info(f"Quarter detected: {detected_quarter}")
        
        # Check if it's a new quarter
        if self.is_new_quarter(result.company, detected_quarter):
            logger.info(f"Promoting old quarter data to cold_store for {result.company}.")
            self.promote_hot_to_cold(result.company)
            
        # Parse turns
        turns = self.speaker_extractor.extract(result.text)
        
        # Ingest to stores
        self.ingest_to_hot(result, detected_quarter, turns)
        self.ingest_to_cold(result, detected_quarter, turns)
        
        return {
            "quarter": detected_quarter,
            "company": result.company,
            "turns_extracted": len(turns)
        }
