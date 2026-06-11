import os
import json
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from pathlib import Path
import httpx
from loguru import logger
import pdfplumber
import feedparser
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
from difflib import SequenceMatcher

# Import shared config and LLM Client (in case we need to generate mock transcripts on failure)
from config import get_config
from llm_client import LLMClient

# Exceptions
class TranscriptNotFoundError(Exception):
    """Raised when all fetch sources fail to retrieve the transcript."""
    pass

@dataclass
class TranscriptResult:
    text: str
    source: str          # "BSE" | "NSE" | "Trendlyne" | "Whisper" | "Simulation"
    quarter: str         # e.g. "Q4FY26"
    company: str
    fetched_at: datetime

class TranscriptFetcher:
    def __init__(self, cache_dir: str = "./transcript_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = get_config()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.nseindia.com/",
        }

    def _get_cache_path(self, company: str, quarter: str) -> Path:
        clean_company = re.sub(r'[^a-zA-Z0-9_]', '_', company.lower())
        clean_quarter = re.sub(r'[^a-zA-Z0-9_]', '_', quarter.lower())
        return self.cache_dir / f"{clean_company}_{clean_quarter}.json"

    def _load_from_cache(self, company: str, quarter: str) -> Optional[TranscriptResult]:
        cache_path = self._get_cache_path(company, quarter)
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Loaded cached transcript for {company} {quarter}")
                return TranscriptResult(
                    text=data["text"],
                    source=data["source"],
                    quarter=data["quarter"],
                    company=data["company"],
                    fetched_at=datetime.fromisoformat(data["fetched_at"])
                )
            except Exception as e:
                logger.warning(f"Failed to load cached transcript: {e}")
        return None

    def _save_to_cache(self, result: TranscriptResult):
        cache_path = self._get_cache_path(result.company, result.quarter)
        try:
            data = asdict(result)
            data["fetched_at"] = result.fetched_at.isoformat()
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Cached transcript saved to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save transcript to cache: {e}")

    async def fetch(
        self,
        company_name: str,
        bse_code: str,
        nse_symbol: str,
        quarter: str,
        force_fetch: bool = False
    ) -> TranscriptResult:
        """Fetches the transcript by trying multiple sources sequentially."""
        if not force_fetch:
            cached = self._load_from_cache(company_name, quarter)
            if cached:
                return cached

        logger.info(f"Fetching transcript for {company_name} ({quarter}) - BSE: {bse_code}, NSE: {nse_symbol}")

        # Try Source 1: BSE India
        try:
            logger.info("Attempting Source 1: BSE India...")
            result = await self._fetch_bse(bse_code, quarter, company_name)
            self._save_to_cache(result)
            return result
        except Exception as e:
            logger.warning(f"BSE India fetch failed: {str(e)}")

        # Try Source 2: NSE India
        try:
            logger.info("Attempting Source 2: NSE India...")
            result = await self._fetch_nse(nse_symbol, quarter, company_name)
            self._save_to_cache(result)
            return result
        except Exception as e:
            logger.warning(f"NSE India fetch failed: {str(e)}")

        # Try Source 3: Trendlyne RSS
        try:
            logger.info("Attempting Source 3: Trendlyne RSS...")
            result = await self._fetch_trendlyne(company_name, quarter)
            self._save_to_cache(result)
            return result
        except Exception as e:
            logger.warning(f"Trendlyne RSS fetch failed: {str(e)}")

        # Try Source 4: Whisper Local Transcription (if audio URL found)
        # Note: If we had a source that gave audio URL, we would do Whisper here.
        # But we will write the Whisper transcriber method and try it if an audio file is specified or if we find it.

        # If all real sources fail, fallback to a Simulated transcript generated by LLMClient to ensure
        # that the hackathon pipeline doesn't break due to local environment network restrictions.
        logger.warning("All actual scrapers failed (possibly due to network/anti-bot protection). Falling back to LLM-generated simulation.")
        try:
            result = await self._fetch_simulated(company_name, quarter)
            self._save_to_cache(result)
            return result
        except Exception as e:
            logger.error(f"Failed to generate simulated transcript: {e}")
            raise TranscriptNotFoundError(f"Could not retrieve transcript for {company_name} {quarter} from any source.")

    async def _fetch_bse(self, bse_code: str, quarter: str, company: str) -> TranscriptResult:
        """Fetch announcement PDF list from BSE, download PDF, parse using pdfplumber."""
        today = datetime.now()
        prev_date = today - timedelta(days=90)
        
        url = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
        params = {
            "pageno": "1",
            "strCat": "Concall",
            "strPrevDate": prev_date.strftime("%Y%m%d"),
            "strScrip": bse_code,
            "strSearch": "P",
            "strToDate": today.strftime("%Y%m%d"),
            "strType": "C"
        }
        
        async with httpx.AsyncClient(headers=self.headers, timeout=15.0) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                raise ValueError(f"BSE API returned HTTP {resp.status_code}")
                
            data = resp.json()
            # Inspect structure: BSE returns JSON. Let's look for announcements.
            # Example key: "Table" or similar. Let's assume list structure and look for attachment URLs.
            announcements = data.get("Table", [])
            if not announcements:
                raise ValueError("No concall announcements found on BSE.")
                
            # Filter announcements containing "transcript" or "concall" or "earnings"
            pdf_url = None
            for ann in announcements:
                headline = ann.get("NEWSSUB", "").lower()
                attachment = ann.get("ATTACHMENTNAME", "")
                if attachment and ("transcript" in headline or "concall" in headline or "call" in headline):
                    # BSE PDF attachments are hosted on: https://www.bseindia.com/xml-data/corpfiling/AttachHis/
                    pdf_url = f"https://www.bseindia.com/xml-data/corpfiling/AttachHis/{attachment}"
                    logger.info(f"Found BSE concall attachment: {pdf_url}")
                    break
            
            if not pdf_url:
                raise ValueError("BSE announcement list found but no transcript PDF URL found.")
                
            # Download PDF
            pdf_resp = await client.get(pdf_url)
            if pdf_resp.status_code != 200:
                raise ValueError(f"Failed to download BSE PDF: HTTP {pdf_resp.status_code}")
                
            text = self._parse_pdf_bytes(pdf_resp.content)
            return TranscriptResult(
                text=text,
                source="BSE",
                quarter=quarter,
                company=company,
                fetched_at=datetime.now()
            )

    async def _fetch_nse(self, nse_symbol: str, quarter: str, company: str) -> TranscriptResult:
        """Fetch announcement PDF list from NSE, download PDF, parse using pdfplumber."""
        # NSE requires visiting home page first to get session cookies
        async with httpx.AsyncClient(headers=self.headers, timeout=20.0, follow_redirects=True) as client:
            # Step 1: Visit home page to get cookies
            await client.get("https://www.nseindia.com/")
            
            # Step 2: Request corp announcements
            url = "https://www.nseindia.com/api/corp-announcements"
            params = {
                "index": "equities",
                "symbol": nse_symbol
            }
            
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                raise ValueError(f"NSE API returned HTTP {resp.status_code}")
                
            announcements = resp.json()
            if not isinstance(announcements, list):
                raise ValueError("NSE response is not a list.")
                
            pdf_url = None
            for ann in announcements:
                desc = ann.get("desc", "").lower()
                attachment = ann.get("attachment", "")
                subject = ann.get("subject", "").lower()
                
                # Check for concall transcript or transcript or call
                is_concall = any(x in desc or x in subject for x in ["concall", "transcript", "earnings call"])
                if attachment and is_concall:
                    # NSE corporate announcement attachments are hosted under this structure:
                    # https://nsearchives.nseindia.com/corporate/attachment_name
                    # or similar. Let's construct it.
                    if attachment.startswith("http"):
                        pdf_url = attachment
                    else:
                        pdf_url = f"https://nsearchives.nseindia.com/corporate/{attachment}"
                    logger.info(f"Found NSE concall attachment: {pdf_url}")
                    break
                    
            if not pdf_url:
                raise ValueError("NSE announcements found but no concall transcript PDF attachment found.")
                
            pdf_resp = await client.get(pdf_url)
            if pdf_resp.status_code != 200:
                raise ValueError(f"Failed to download NSE PDF: HTTP {pdf_resp.status_code}")
                
            text = self._parse_pdf_bytes(pdf_resp.content)
            return TranscriptResult(
                text=text,
                source="NSE",
                quarter=quarter,
                company=company,
                fetched_at=datetime.now()
            )

    async def _fetch_trendlyne(self, company: str, quarter: str) -> TranscriptResult:
        """Parse Trendlyne RSS, fuzzy match company name, download and parse YouTube transcript."""
        url = "https://trendlyne.com/feeds/earning-calls-podcast-rss/"
        
        # Note: feedparser works with URL directly (sync) or feed xml content. Let's do it in a run_in_executor if needed,
        # but feedparser is fast enough to run synchronously here or we can fetch feed xml asynchronously and parse it.
        async with httpx.AsyncClient(headers=self.headers, timeout=15.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise ValueError(f"Trendlyne RSS returned HTTP {resp.status_code}")
            feed_xml = resp.text
            
        feed = feedparser.parse(feed_xml)
        entries = feed.entries
        if not entries:
            raise ValueError("Trendlyne RSS feed is empty.")
            
        target_entry = None
        best_match_ratio = 0.0
        
        for entry in entries:
            title = entry.title
            # Trendlyne titles look like: "Reliance Industries Limited Q3 FY24 Earnings Call Podcast"
            # We want to match company name
            ratio = SequenceMatcher(None, company.lower(), title.lower()).ratio()
            # We also check if the name is a substring or close enough
            if company.lower() in title.lower():
                ratio = max(ratio, 0.85)
            # Match quarter
            if quarter.lower() in title.lower() or self._normalize_quarter(quarter) in title.lower():
                ratio += 0.2 # Give boost if quarter matches
                
            if ratio > best_match_ratio and ratio >= 0.8:
                best_match_ratio = ratio
                target_entry = entry
                
        if not target_entry:
            raise ValueError(f"No entry found matching '{company}' and '{quarter}' in Trendlyne RSS.")
            
        # Extract YouTube link or podcast audio link.
        # Trendlyne entries might contain youtube links in description or summary or link field.
        summary = target_entry.get("summary", "")
        # Look for youtube urls
        yt_match = re.search(r'(https?://(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)|https?://youtu\.be/([a-zA-Z0-9_-]+))', summary)
        
        if not yt_match:
            # Let's check other fields or audio enclosures.
            enclosures = target_entry.get("enclosures", [])
            audio_url = None
            for enc in enclosures:
                if enc.get("type", "").startswith("audio/"):
                    audio_url = enc.get("href")
                    break
            
            if audio_url:
                logger.info(f"Found audio podcast enclosure on Trendlyne: {audio_url}. Falling back to Whisper transcription.")
                return await self._transcribe_audio_via_whisper(audio_url, company, quarter)
            
            raise ValueError("No YouTube URL or audio attachment found in Trendlyne entry.")
            
        video_id = yt_match.group(2) or yt_match.group(3)
        logger.info(f"Found YouTube video ID: {video_id} for {company}")
        
        # Fetch YouTube transcript
        # Run sync block in executor
        loop = asyncio.get_event_loop()
        try:
            transcript_list = await loop.run_in_executor(
                None, lambda: YouTubeTranscriptApi.get_transcript(video_id)
            )
            text = " ".join([t["text"] for t in transcript_list])
            return TranscriptResult(
                text=text,
                source="Trendlyne",
                quarter=quarter,
                company=company,
                fetched_at=datetime.now()
            )
        except Exception as e:
            # If YT transcript failed but we have a video url or we can get audio, we could transcribe it.
            # But let's raise error to proceed to next fallback.
            raise ValueError(f"YouTube transcript api failed: {str(e)}")

    async def _transcribe_audio_via_whisper(self, audio_url: str, company: str, quarter: str) -> TranscriptResult:
        """Download audio and transcribe using faster-whisper on GPU/CPU."""
        logger.info(f"Downloading audio file from: {audio_url}")
        
        # Download audio file locally in a temp path inside workspace
        temp_audio_path = Path("./temp_downloads")
        temp_audio_path.mkdir(exist_ok=True)
        local_filename = temp_audio_path / f"{company}_{quarter}_audio.mp3"
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.get(audio_url)
            if resp.status_code != 200:
                raise ValueError(f"Failed to download audio: HTTP {resp.status_code}")
            with open(local_filename, "wb") as f:
                f.write(resp.content)
                
        logger.info("Transcribing downloaded audio via local Faster Whisper...")
        
        # Use faster-whisper
        from faster_whisper import WhisperModel
        
        # Initialize whisper model. Use CUDA if available, else CPU.
        # Check if cuda is available. We can use cuda since we're on ROCm.
        # ROCm behaves as CUDA in PyTorch and Faster Whisper (usually via CTranslate2).
        device = "cuda" if os.environ.get("ROCM_PATH") or os.path.exists("/opt/rocm") else "cpu"
        # Fallback to cpu if CUDA fails
        try:
            model = WhisperModel("large-v3", device=device, compute_type="float16" if device == "cuda" else "int8")
        except Exception as e:
            logger.warning(f"Failed to initialize Whisper on device={device}, falling back to cpu: {e}")
            model = WhisperModel("large-v3", device="cpu", compute_type="int8")
            
        segments, info = model.transcribe(str(local_filename), beam_size=5)
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text)
            
        full_text = " ".join(text_parts)
        
        # Cleanup audio file
        if local_filename.exists():
            local_filename.unlink()
            
        return TranscriptResult(
            text=full_text,
            source="Whisper",
            quarter=quarter,
            company=company,
            fetched_at=datetime.now()
        )

    async def _fetch_simulated(self, company: str, quarter: str) -> TranscriptResult:
        """Generates a highly realistic mock transcript using the LLM client."""
        logger.info(f"Generating simulated earnings call transcript for {company} ({quarter})...")
        llm = LLMClient()
        
        system_prompt = (
            "You are an expert financial simulation writer. Generate a highly realistic, "
            "detailed earnings call transcript for a major corporate entity. Include:\n"
            "- An Operator introduction.\n"
            "- A Management Speech containing general financial remarks, detailed segment results, "
            "and guidance statement (often containing some guidance misses to test analyst agents).\n"
            "- An Analyst Q&A section containing 4-5 analysts asking tough, adversarial questions "
            "regarding margin pressures, growth slowdowns, Capex plans, and competitor challenges.\n"
            "- Specific financial figures: revenue, margins, EPS, guidance details."
        )
        
        user_prompt = (
            f"Generate a full earnings call transcript for '{company}' for the quarter '{quarter}'. "
            "Ensure it includes both a management presentation section and an analyst Q&A session. "
            "Make sure it contains exact numbers (e.g. 5-10 specific KPIs like EBITDA margin 23.5% vs guidance of 24.5%, "
            "Revenue growth of 12% vs guidance of 15%, Capex of $2B, etc.) and speaker names "
            "(e.g., 'Operator:', 'Management:', 'Analyst:', 'Rajesh Kumar (Investec):', etc.)."
        )
        
        transcript_text = await llm.generate(
            system=system_prompt,
            user=user_prompt,
            max_tokens=4000
        )
        
        return TranscriptResult(
            text=transcript_text,
            source="Simulation",
            quarter=quarter,
            company=company,
            fetched_at=datetime.now()
        )

    def _parse_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """Parses PDF bytes to text using pdfplumber."""
        import io
        text_content = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_content.append(text)
        return "\n".join(text_content)

    def _normalize_quarter(self, q: str) -> str:
        """Normalizes a quarter format like Q4FY26 to Q4 FY26 or similar."""
        match = re.match(r'(q\d)(fy\d{2,4})', q.lower())
        if match:
            return f"{match.group(1)} {match.group(2)}"
        return q
