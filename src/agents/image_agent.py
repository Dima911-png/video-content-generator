"""Image agent for searching and downloading relevant images for video scripts.

This agent analyzes scripts to extract key visual concepts and searches
multiple image sources to find high-quality, relevant images for video
production.
"""

import asyncio
import logging
import re
import os
import hashlib
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup
from PIL import Image
import io

# Try to import icrawler for web image search
try:
    from icrawler.builtin import BingImageCrawler
    ICRAWLER_AVAILABLE = True
except ImportError:
    ICRAWLER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ImageResult:
    """Represents a found image."""
    url: str
    title: str
    source: str
    source_url: str
    width: Optional[int] = None
    height: Optional[int] = None
    license: Optional[str] = None
    photographer: Optional[str] = None
    local_path: Optional[str] = None  # For icrawler pre-downloaded images
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "source": self.source,
            "source_url": self.source_url,
            "width": self.width,
            "height": self.height,
            "license": self.license,
            "photographer": self.photographer,
            "local_path": self.local_path,
        }


@dataclass
class DownloadedImage:
    """Represents a downloaded image file."""
    filename: str
    filepath: str
    url: str
    title: str
    source: str
    section: str
    file_size: int
    content_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "filepath": self.filepath,
            "url": self.url,
            "title": self.title,
            "source": self.source,
            "section": self.section,
            "file_size": self.file_size,
            "content_type": self.content_type,
        }


@dataclass
class ScriptSection:
    """Represents a section of the script for image matching."""
    name: str
    content: str
    keywords: List[str] = field(default_factory=list)
    images: List[DownloadedImage] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "keywords": self.keywords,
            "images": [img.to_dict() for img in self.images],
        }


class ImageAgent:
    """Agent for searching and downloading images for video scripts."""
    
    # Free image API sources (no API key required)
    SOURCES = [
        "unsplash",      # Unsplash Source (direct URLs)
        "bing",          # Bing Image Search (no API key needed)
        "pexels",        # Pexels (scraping)
        "pixabay",       # Pixabay (requires free API key, fallback to scraping)
        "wikimedia",     # Wikimedia Commons
    ]
    
    def __init__(
        self,
        output_dir: str = "output",
        timeout: int = 15,
        images_per_section: int = 5,
        target_total_images: int = 30,
        min_width: int = 1280,
        min_height: int = 720,
        require_landscape: bool = True,
        api_keys: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize image agent.
        
        Args:
            output_dir: Directory for saving images
            timeout: Request timeout in seconds
            images_per_section: Number of images to find per script section (default 5)
            target_total_images: Target total images for the video (default 30, ~3 per minute)
            min_width: Minimum image width (default 1280 for HD)
            min_height: Minimum image height (default 720 for HD)
            require_landscape: Only accept landscape (horizontal) images
            api_keys: Optional API keys for sources (e.g., {"pexels": "key", "pixabay": "key"})
        """
        self.output_dir = output_dir
        self.timeout = timeout
        self.images_per_section = images_per_section
        self.target_total_images = target_total_images
        self.min_width = min_width
        self.min_height = min_height
        self.require_landscape = require_landscape
        self.api_keys = api_keys or {}
        
        # Async primitives for parallel downloads
        self._download_semaphore = asyncio.Semaphore(10)  # Limit concurrent downloads
        self._hash_lock = asyncio.Lock()  # Thread-safe hash checking
        
        # Create session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        
    async def find_images_for_script(
        self,
        script: str,
        topic: str,
        output_subdir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Find and download images for a complete script.
        
        Args:
            script: The full script text
            topic: Main topic for context
            output_subdir: Subdirectory for this script's images
            
        Returns:
            Dictionary with sections and their downloaded images
        """
        logger.info(f"[IMG] Finding images for script about: {topic}")
        
        # Parse script into sections
        sections = await self._parse_script_sections(script)
        logger.info(f"[NOTE] Found {len(sections)} script sections")
        
        # Extract keywords for each section
        for section in sections:
            section.keywords = await self._extract_keywords(section.content, topic)
            logger.debug(f"Section '{section.name}' keywords: {section.keywords}")
        
        # Create output directory
        if output_subdir:
            image_dir = Path(self.output_dir) / output_subdir / "images"
        else:
            safe_topic = "".join(c if c.isalnum() or c in " -_" else "_" for c in topic)
            safe_topic = safe_topic.replace(" ", "_")[:50]
            image_dir = Path(self.output_dir) / safe_topic / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Global hash set to track downloaded images across all sections
        global_downloaded_hashes = set()
        
        # Find and download images for each section
        total_downloaded = 0
        for section in sections:
            # Stop early if we've already reached the target
            if total_downloaded >= self.target_total_images:
                logger.info(f"  Reached target of {self.target_total_images} images, skipping remaining sections")
                break
            
            logger.info(f"[SEARCH] Searching images for section: {section.name}")
            
            # Search for images using keywords
            images_found = await self._search_images(section.keywords, topic)
            logger.info(f"  Found {len(images_found)} candidate images")
            
            # Download top images (pass global hash set to avoid duplicates)
            remaining_needed = self.target_total_images - total_downloaded
            download_limit = min(self.images_per_section, remaining_needed)
            downloaded = await self._download_images(
                images_found[:download_limit * 3],  # Get extra in case some fail/duplicate
                image_dir,
                section.name,
                global_downloaded_hashes,
            )
            section.images = downloaded[:download_limit]
            total_downloaded += len(section.images)
            
            logger.info(f"  [OK] Downloaded {len(section.images)} unique images for section")
        
        logger.info(f"[OK] Image search complete: {total_downloaded} images downloaded")
        
        # Create image manifest
        manifest = {
            "topic": topic,
            "generated_at": datetime.now().isoformat(),
            "total_images": total_downloaded,
            "image_directory": str(image_dir),
            "sections": [s.to_dict() for s in sections],
        }
        
        # Save manifest
        manifest_path = image_dir / "manifest.json"
        import json
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        
        return manifest
    
    async def _parse_script_sections(self, script: str) -> List[ScriptSection]:
        """
        Parse script into granular sections for image matching.
        
        Creates more sections to allow for ~100 images across a 10-min video.
        Splits by [Section] markers first, then by paragraphs within each.
        """
        sections = []
        
        # Pattern to match section headers like [Intro], [Content], etc.
        section_pattern = r'\[([\w\s]+)\]'
        
        # Find all section markers
        matches = list(re.finditer(section_pattern, script))
        
        if matches:
            for i, match in enumerate(matches):
                section_name = match.group(1).strip()
                start = match.end()
                
                # End is the start of next section or end of script
                if i + 1 < len(matches):
                    end = matches[i + 1].start()
                else:
                    end = len(script)
                
                content = script[start:end].strip()
                
                # Skip empty sections
                if content:
                    # Split large sections into sub-parts by paragraph or sentence groups
                    sub_sections = self._split_section_content(section_name, content)
                    sections.extend(sub_sections)
        else:
            # No sections found, split by paragraphs
            paragraphs = script.split('\n\n')
            for i, para in enumerate(paragraphs):
                if para.strip() and len(para.strip()) > 50:
                    sections.append(ScriptSection(
                        name=f"Part {i + 1}",
                        content=para.strip(),
                    ))
        
        # Ensure we have enough sections for target images (but don't over-expand)
        min_sections = max(3, self.target_total_images // self.images_per_section)
        if len(sections) < min_sections:
            sections = self._expand_sections(sections, min_sections)
        
        return sections
    
    def _split_section_content(self, section_name: str, content: str) -> List[ScriptSection]:
        """
        Split a section's content into smaller sub-sections.
        
        Aims for roughly 2-3 sentences per sub-section for diverse images.
        """
        sub_sections = []
        
        # Split by double newlines first (paragraphs)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if len(paragraphs) >= 3:
            # Use paragraphs as sub-sections
            for i, para in enumerate(paragraphs):
                if len(para) > 50:  # Skip very short paragraphs
                    sub_sections.append(ScriptSection(
                        name=f"{section_name} - Part {i + 1}",
                        content=para,
                    ))
        else:
            # Split by sentences if not enough paragraphs
            # Simple sentence splitting on . ! ?
            sentences = re.split(r'(?<=[.!?])\s+', content)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
            
            # Group sentences into chunks of 2-3
            chunk_size = 2
            for i in range(0, len(sentences), chunk_size):
                chunk = ' '.join(sentences[i:i + chunk_size])
                if len(chunk) > 50:
                    sub_sections.append(ScriptSection(
                        name=f"{section_name} - Part {i // chunk_size + 1}",
                        content=chunk,
                    ))
        
        # If we still have nothing, just use the whole section
        if not sub_sections and len(content) > 50:
            sub_sections.append(ScriptSection(
                name=section_name,
                content=content,
            ))
        
        return sub_sections
    
    def _expand_sections(self, sections: List[ScriptSection], target_count: int) -> List[ScriptSection]:
        """
        Expand sections to meet target count by creating keyword variations.
        """
        if not sections:
            return sections
        
        expanded = list(sections)
        section_idx = 0
        variation_num = 1
        
        while len(expanded) < target_count:
            # Create a variation of an existing section
            base_section = sections[section_idx % len(sections)]
            expanded.append(ScriptSection(
                name=f"{base_section.name} - Variation {variation_num}",
                content=base_section.content,
            ))
            section_idx += 1
            if section_idx % len(sections) == 0:
                variation_num += 1
        
        return expanded
    
    async def _extract_keywords(self, content: str, topic: str) -> List[str]:
        """
        Extract search keywords heavily weighted towards topic relevance.
        
        Prioritizes:
        1. The exact topic (e.g., "GitHub Copilot")
        2. Topic + tech/coding related terms
        3. Topic + content-specific terms only if they're tech-related
        """
        import random
        
        # Clean the content - remove SSML tags and special chars
        clean_content = re.sub(r'<[^>]+>', '', content)
        clean_content = re.sub(r'[^\w\s\-]', ' ', clean_content)
        words = clean_content.lower().split()
        
        # Tech-related terms that are good to combine with the topic
        tech_terms = {
            'ai', 'artificial', 'intelligence', 'machine', 'learning', 'coding',
            'programming', 'developer', 'software', 'code', 'github', 'copilot',
            'autocomplete', 'assistant', 'ide', 'vscode', 'visual', 'studio',
            'openai', 'gpt', 'model', 'api', 'automation', 'tool', 'productivity',
            'editor', 'syntax', 'debugging', 'testing', 'deployment', 'devops',
            'computer', 'laptop', 'keyboard', 'screen', 'terminal', 'command',
            'python', 'javascript', 'typescript', 'java', 'react', 'node',
            'function', 'algorithm', 'data', 'neural', 'network', 'transformer',
            'chatgpt', 'llm', 'prompt', 'completion', 'suggestion', 'intellisense',
        }
        
        # Find tech-related words mentioned in this section
        content_tech_words = [w for w in words if w in tech_terms and len(w) > 2]
        
        # Build search terms - ALWAYS topic-focused
        search_terms = []
        
        # 1. Core topic variations (50% of searches - most relevant!)
        core_searches = [
            f"{topic}",
            f"{topic} screenshot",
            f"{topic} coding",
            f"{topic} IDE",
            f"{topic} VS Code",
            f"{topic} autocomplete",
            f"{topic} AI assistant",
            f"{topic} programming",
        ]
        search_terms.extend(core_searches)
        
        # 2. Related tech terms (30% - still very relevant)
        related_searches = [
            "AI coding assistant",
            "AI pair programming",
            "code autocomplete AI",
            "GitHub AI tools",
            "developer productivity AI",
            "coding with AI",
            "machine learning code",
            "programming assistant",
        ]
        # Shuffle and pick a few
        random.shuffle(related_searches)
        search_terms.extend(related_searches[:4])
        
        # 3. Topic + any tech terms found in content (20% - contextual but relevant)
        for tech_word in content_tech_words[:3]:
            if tech_word.lower() not in topic.lower():
                search_terms.append(f"{topic} {tech_word}")
        
        # 4. Visual/screenshot oriented searches
        visual_searches = [
            f"{topic} interface demo",
            "code editor dark theme",
            "programming IDE screenshot",
            "AI code suggestion",
        ]
        search_terms.extend(random.sample(visual_searches, 2))
        
        # Deduplicate while preserving order
        seen = set()
        unique_terms = []
        for term in search_terms:
            term_lower = term.lower()
            if term_lower not in seen:
                seen.add(term_lower)
                unique_terms.append(term)
        
        logger.debug(f"Generated {len(unique_terms)} topic-focused search terms")
        
        return unique_terms[:3]  # Return up to 3 search terms per section for speed
    
    async def _search_images(self, keywords: List[str], topic: str) -> List[ImageResult]:
        """
        Search for images across multiple sources.
        Adds delays between requests to avoid rate limiting.
        Uses icrawler for Bing as the primary source (most reliable).
        """
        all_results = []
        
        for i, keyword in enumerate(keywords):
            # Try multiple sources with delays between each
            
            # Bing via icrawler only for FIRST keyword (slow but relevant)
            if ICRAWLER_AVAILABLE and i == 0:
                try:
                    icrawler_results = await self._search_bing_icrawler(keyword, max_images=10)
                    all_results.extend(icrawler_results)
                    logger.info(f"icrawler found {len(icrawler_results)} images for '{keyword}'")
                except Exception as e:
                    logger.debug(f"icrawler search failed: {e}")
            
            # Pexels (best quality if API key available)
            try:
                if self.api_keys.get("pexels"):
                    pexels_results = await self._search_pexels_api(keyword)
                    all_results.extend(pexels_results)
                    await asyncio.sleep(0.3)  # Rate limit
            except Exception as e:
                logger.debug(f"Pexels search failed: {e}")
            
            # Pixabay (if API key available)
            try:
                if self.api_keys.get("pixabay"):
                    pixabay_results = await self._search_pixabay_api(keyword)
                    all_results.extend(pixabay_results)
                    await asyncio.sleep(0.3)  # Rate limit
            except Exception as e:
                logger.debug(f"Pixabay search failed: {e}")
            
            # Wikimedia Commons (free, no API key - often has relevant images)
            try:
                wikimedia_results = await self._search_wikimedia(keyword)
                all_results.extend(wikimedia_results)
                await asyncio.sleep(0.3)  # Rate limit
            except Exception as e:
                logger.debug(f"Wikimedia search failed: {e}")
            
            # Skip DuckDuckGo and manual Bing scraping - they're blocked/rate limited
            # Lorem Picsum only as absolute last resort (random images)
            if len(all_results) < 10:
                try:
                    picsum_results = await self._get_picsum_images(keyword)
                    all_results.extend(picsum_results)
                except Exception as e:
                    logger.debug(f"Picsum failed: {e}")
        
        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        # Prioritize: icrawler (most relevant) > API sources > web search > placeholders
        def source_priority(img):
            if img.source == "Bing (icrawler)":
                return 0  # Best - topic-relevant web search
            elif img.source == "Pexels":
                return 1
            elif img.source == "Pixabay":
                return 2
            elif img.source == "Wikimedia Commons":
                return 3
            elif img.source == "DuckDuckGo":
                return 4
            elif img.source == "Bing Images":
                return 5
            else:  # Lorem Picsum / placeholders
                return 6
        
        unique_results.sort(key=source_priority)
        
        return unique_results
    
    async def _search_duckduckgo(self, query: str) -> List[ImageResult]:
        """
        Search DuckDuckGo Images - reliable and no API key needed.
        Returns actual relevant images based on search query.
        """
        results = []
        
        try:
            # Step 1: Get the vqd token from DuckDuckGo
            token_url = "https://duckduckgo.com/"
            token_params = {"q": query}
            token_response = self.session.get(token_url, params=token_params, timeout=self.timeout)
            
            # Extract vqd token
            vqd_match = re.search(r'vqd=([^&"\']+)', token_response.text)
            if not vqd_match:
                # Try alternative pattern
                vqd_match = re.search(r'vqd%3D([^&"\'%]+)', token_response.text)
            
            if vqd_match:
                vqd = vqd_match.group(1)
                
                # Step 2: Search images with the token
                img_url = "https://duckduckgo.com/i.js"
                img_params = {
                    "l": "us-en",
                    "o": "json",
                    "q": query,
                    "vqd": vqd,
                    "f": ",,,,,",
                    "p": "1",
                }
                
                img_response = self.session.get(img_url, params=img_params, timeout=self.timeout)
                
                if img_response.status_code == 200:
                    try:
                        data = img_response.json()
                        for item in data.get("results", [])[:15]:
                            img_url = item.get("image")
                            if img_url and not any(x in img_url.lower() for x in ['thumb', 'icon', 'logo']):
                                results.append(ImageResult(
                                    url=img_url,
                                    title=item.get("title", query),
                                    source="DuckDuckGo",
                                    source_url=item.get("url", ""),
                                    width=item.get("width"),
                                    height=item.get("height"),
                                    license="Check source",
                                ))
                    except:
                        pass
                        
        except Exception as e:
            logger.debug(f"DuckDuckGo search error: {e}")
        
        return results
    
    async def _search_bing(self, query: str) -> List[ImageResult]:
        """
        Search Bing Images - scrapes actual image URLs.
        """
        results = []
        
        try:
            url = "https://www.bing.com/images/search"
            params = {
                "q": query,
                "first": "1",
                "count": "35",
                "qft": "+filterui:imagesize-large",  # Large images only
            }
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                # Extract murl (media URL) from Bing's response
                pattern = r'"murl":"([^"]+)"'
                matches = re.findall(pattern, response.text)
                
                for img_url in matches[:15]:
                    # Unescape URL
                    img_url = img_url.replace("\\u0026", "&").replace("\\/", "/")
                    
                    # Skip thumbnails and icons
                    if any(x in img_url.lower() for x in ['thumb', 'icon', 'logo', 'avatar']):
                        continue
                    
                    results.append(ImageResult(
                        url=img_url,
                        title=f"{query}",
                        source="Bing Images",
                        source_url=f"https://www.bing.com/images/search?q={quote(query)}",
                        license="Check source",
                    ))
                        
        except Exception as e:
            logger.debug(f"Bing search error: {e}")
        
        return results
    
    async def _search_bing_icrawler(self, query: str, max_images: int = 20) -> List[ImageResult]:
        """
        Search Bing Images using icrawler library - most reliable method.
        Downloads images to a temp directory and returns ImageResult objects.
        Filters for large, landscape (horizontal) images suitable for video.
        """
        results = []
        
        if not ICRAWLER_AVAILABLE:
            logger.debug("icrawler not available, skipping Bing crawler")
            return results
        
        try:
            import tempfile
            import os
            
            # Create temp directory for this search
            temp_dir = tempfile.mkdtemp(prefix="icrawler_")
            
            # Run icrawler in thread pool (it's synchronous)
            loop = asyncio.get_event_loop()
            
            # Add "HD" or "wallpaper" to query for larger images
            enhanced_query = f"{query} HD wallpaper"
            
            def run_crawler():
                try:
                    crawler = BingImageCrawler(
                        storage={"root_dir": temp_dir},
                        downloader_threads=2,
                        feeder_threads=1,
                        parser_threads=1,
                    )
                    # Suppress icrawler's verbose output
                    import logging
                    icrawler_logger = logging.getLogger("icrawler")
                    icrawler_logger.setLevel(logging.WARNING)
                    
                    # Request more images to filter later (many will be vertical)
                    crawler.crawl(
                        keyword=enhanced_query,
                        max_num=max_images * 2,  # Get extra to filter
                        min_size=(self.min_width, self.min_height),  # HD minimum
                        file_idx_offset=0,
                    )
                except Exception as e:
                    logger.debug(f"Crawler error: {e}")
            
            # Run in executor to not block
            await loop.run_in_executor(None, run_crawler)
            
            # Collect downloaded files as ImageResults, filtering for landscape
            valid_count = 0
            rejected_count = 0
            if os.path.exists(temp_dir):
                for filename in os.listdir(temp_dir):
                    filepath = os.path.join(temp_dir, filename)
                    if os.path.isfile(filepath):
                        # Check image dimensions
                        try:
                            with Image.open(filepath) as img:
                                width, height = img.size
                                
                                # Skip vertical/portrait images
                                if self.require_landscape and height > width:
                                    logger.debug(f"Skipping vertical image: {filename} ({width}x{height})")
                                    os.remove(filepath)  # Clean up
                                    rejected_count += 1
                                    continue
                                
                                # Skip images that are too small
                                if width < self.min_width or height < self.min_height:
                                    logger.debug(f"Skipping small image: {filename} ({width}x{height})")
                                    os.remove(filepath)  # Clean up
                                    rejected_count += 1
                                    continue
                                
                                # Valid image!
                                results.append(ImageResult(
                                    url=f"file://{filepath}",
                                    title=query,
                                    source="Bing (icrawler)",
                                    source_url="https://www.bing.com/images",
                                    license="Check source",
                                    local_path=filepath,
                                    width=width,
                                    height=height,
                                ))
                                valid_count += 1
                                
                        except Exception as e:
                            logger.debug(f"Could not read image {filename}: {e}")
                            continue
            
            logger.info(f"icrawler: {valid_count} valid images for '{query}' ({rejected_count} rejected)")
            
        except Exception as e:
            logger.debug(f"icrawler search error: {e}")
        
        return results
    
    async def _search_pexels_api(self, query: str) -> List[ImageResult]:
        """
        Search Pexels using their API.
        Requires API key from https://www.pexels.com/api/
        """
        results = []
        api_key = self.api_keys.get("pexels")
        
        if not api_key:
            return results
        
        try:
            url = "https://api.pexels.com/v1/search"
            headers = {"Authorization": api_key}
            params = {
                "query": query,
                "per_page": 15,
                "orientation": "landscape" if self.prefer_landscape else "square",
            }
            
            response = self.session.get(url, headers=headers, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                for photo in data.get("photos", []):
                    results.append(ImageResult(
                        url=photo["src"]["large"],
                        title=photo.get("alt", query),
                        source="Pexels",
                        source_url=photo["url"],
                        width=photo.get("width"),
                        height=photo.get("height"),
                        license="Pexels License (free)",
                        photographer=photo.get("photographer"),
                    ))
        except Exception as e:
            logger.debug(f"Pexels API error: {e}")
        
        return results
    
    async def _search_pixabay_api(self, query: str) -> List[ImageResult]:
        """
        Search Pixabay using their API.
        Requires API key from https://pixabay.com/api/docs/
        """
        results = []
        api_key = self.api_keys.get("pixabay")
        
        if not api_key:
            return results
        
        try:
            url = "https://pixabay.com/api/"
            params = {
                "key": api_key,
                "q": query,
                "per_page": 15,
                "orientation": "horizontal" if self.prefer_landscape else "all",
                "min_width": self.min_width,
                "safesearch": "true",
            }
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                for hit in data.get("hits", []):
                    results.append(ImageResult(
                        url=hit["largeImageURL"],
                        title=hit.get("tags", query),
                        source="Pixabay",
                        source_url=hit["pageURL"],
                        width=hit.get("imageWidth"),
                        height=hit.get("imageHeight"),
                        license="Pixabay License (free)",
                        photographer=hit.get("user"),
                    ))
        except Exception as e:
            logger.debug(f"Pixabay API error: {e}")
        
        return results
    
    async def _search_wikimedia(self, query: str) -> List[ImageResult]:
        """
        Search Wikimedia Commons for images.
        No API key required.
        """
        results = []
        
        try:
            # Use Wikimedia Commons API
            url = "https://commons.wikimedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": f"filetype:bitmap {query}",
                "gsrlimit": 10,
                "prop": "imageinfo",
                "iiprop": "url|size|extmetadata",
            }
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                pages = data.get("query", {}).get("pages", {})
                
                for page_id, page in pages.items():
                    imageinfo = page.get("imageinfo", [{}])[0]
                    if imageinfo.get("url"):
                        # Skip small images
                        width = imageinfo.get("width", 0)
                        if width < self.min_width:
                            continue
                        
                        metadata = imageinfo.get("extmetadata", {})
                        results.append(ImageResult(
                            url=imageinfo["url"],
                            title=page.get("title", query).replace("File:", ""),
                            source="Wikimedia Commons",
                            source_url=f"https://commons.wikimedia.org/wiki/{page.get('title', '')}",
                            width=width,
                            height=imageinfo.get("height"),
                            license=metadata.get("LicenseShortName", {}).get("value", "CC"),
                            photographer=metadata.get("Artist", {}).get("value", ""),
                        ))
        except Exception as e:
            logger.debug(f"Wikimedia search error: {e}")
        
        return results
    
    async def _get_picsum_images(self, query: str) -> List[ImageResult]:
        """
        Get placeholder images from Lorem Picsum.
        Always works, good fallback.
        """
        results = []
        
        # Use random image IDs to ensure we get different images each time
        # Picsum has ~1000 images with IDs from 1-1084 (some gaps)
        import random
        import time
        
        # Use time + random to ensure truly different images each call
        random.seed(time.time_ns())
        used_ids = set()
        
        for i in range(8):
            # Get a unique random ID
            img_id = random.randint(1, 1000)
            while img_id in used_ids:
                img_id = random.randint(1, 1000)
            used_ids.add(img_id)
            
            # Use ID-based URL for unique images
            url = f"https://picsum.photos/id/{img_id}/1600/900"
            results.append(ImageResult(
                url=url,
                title=f"Image {img_id} for: {query}",
                source="Lorem Picsum",
                source_url=f"https://picsum.photos/id/{img_id}",
                width=1600,
                height=900,
                license="Unsplash License (free)",
            ))
        
        return results
    
    async def _download_images(
        self,
        images: List[ImageResult],
        output_dir: Path,
        section_name: str,
        global_downloaded_hashes: Optional[set] = None,
    ) -> List[DownloadedImage]:
        """
        Download images to disk in parallel, avoiding duplicates.
        
        Args:
            images: List of image results to download
            output_dir: Directory to save images
            section_name: Name of the script section
            global_downloaded_hashes: Set of already downloaded image hashes to avoid duplicates
        """
        if global_downloaded_hashes is None:
            global_downloaded_hashes = set()
        
        # Sanitize section name for filename
        safe_section = "".join(c if c.isalnum() or c in " -_" else "_" for c in section_name)
        safe_section = safe_section.replace(" ", "_").lower()
        
        # Process icrawler images first (they're already downloaded)
        downloaded = []
        remaining_images = []
        img_counter = 1
        
        for image in images:
            if hasattr(image, 'local_path') and image.local_path and Path(image.local_path).exists():
                result = self._process_icrawler_image(
                    image, output_dir, safe_section, img_counter, global_downloaded_hashes
                )
                if result:
                    downloaded.append(result)
                    img_counter += 1
            else:
                remaining_images.append(image)
        
        # Download remaining images in parallel
        if remaining_images:
            # Create download tasks
            tasks = []
            for image in remaining_images:
                task = self._download_single_image(
                    image, output_dir, safe_section, global_downloaded_hashes
                )
                tasks.append(task)
            
            # Run downloads in parallel with semaphore to limit concurrency
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, DownloadedImage):
                    # Renumber with correct counter
                    old_path = Path(result.filepath)
                    ext = old_path.suffix
                    new_filename = f"{safe_section}_{img_counter:02d}{ext}"
                    new_filepath = output_dir / new_filename
                    
                    if old_path != new_filepath:
                        old_path.rename(new_filepath)
                        result.filename = new_filename
                        result.filepath = str(new_filepath)
                    
                    downloaded.append(result)
                    img_counter += 1
                elif isinstance(result, Exception):
                    logger.debug(f"Download failed: {result}")
        
        return downloaded
    
    def _process_icrawler_image(
        self,
        image: ImageResult,
        output_dir: Path,
        safe_section: str,
        img_counter: int,
        global_downloaded_hashes: set,
    ) -> Optional[DownloadedImage]:
        """Process an image already downloaded by icrawler."""
        try:
            src_path = Path(image.local_path)
            image_data = src_path.read_bytes()
            
            # Check for duplicates
            content_hash = hashlib.md5(image_data).hexdigest()
            if content_hash in global_downloaded_hashes:
                return None
            global_downloaded_hashes.add(content_hash)
            
            # Determine extension
            ext = src_path.suffix.lstrip('.').lower() or 'jpg'
            if ext not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                ext = 'jpg'
            
            filename = f"{safe_section}_{img_counter:02d}.{ext}"
            filepath = output_dir / filename
            
            shutil.copy2(src_path, filepath)
            
            return DownloadedImage(
                filename=filename,
                filepath=str(filepath),
                url=image.url,
                title=image.title,
                source=image.source,
                section=safe_section,
                file_size=len(image_data),
                content_type=f"image/{ext}",
            )
        except Exception as e:
            logger.debug(f"Failed to process icrawler image: {e}")
            return None
    
    async def _download_single_image(
        self,
        image: ImageResult,
        output_dir: Path,
        safe_section: str,
        global_downloaded_hashes: set,
    ) -> Optional[DownloadedImage]:
        """Download a single image asynchronously."""
        try:
            # Use semaphore to limit concurrent downloads
            async with self._download_semaphore:
                # Run blocking request in thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.session.get(
                        image.url,
                        timeout=self.timeout,
                        allow_redirects=True,
                    )
                )
                
                if response.status_code != 200:
                    return None
                
                image_data = response.content
                
                # Validate image dimensions
                try:
                    img = Image.open(io.BytesIO(image_data))
                    width, height = img.size
                    
                    if self.require_landscape and height > width:
                        return None
                    
                    if width < self.min_width or height < self.min_height:
                        return None
                except Exception:
                    return None
                
                # Check for duplicates (thread-safe with lock)
                content_hash = hashlib.md5(image_data).hexdigest()
                async with self._hash_lock:
                    if content_hash in global_downloaded_hashes:
                        return None
                    global_downloaded_hashes.add(content_hash)
                
                # Determine extension
                content_type = response.headers.get("content-type", "image/jpeg")
                ext = "jpg"
                if "png" in content_type:
                    ext = "png"
                elif "gif" in content_type:
                    ext = "gif"
                elif "webp" in content_type:
                    ext = "webp"
                
                # Use temp filename (will be renamed later)
                import uuid
                filename = f"{safe_section}_temp_{uuid.uuid4().hex[:8]}.{ext}"
                filepath = output_dir / filename
                
                with open(filepath, "wb") as f:
                    f.write(image_data)
                
                return DownloadedImage(
                    filename=filename,
                    filepath=str(filepath),
                    url=image.url,
                    title=image.title,
                    source=image.source,
                    section=safe_section,
                    file_size=len(image_data),
                    content_type=content_type,
                )
                
        except Exception as e:
            logger.debug(f"Failed to download {image.url}: {e}")
            return None
    
    async def search_single_image(
        self,
        query: str,
        output_path: Optional[str] = None,
    ) -> Optional[DownloadedImage]:
        """
        Search and optionally download a single image.
        
        Args:
            query: Search query
            output_path: Optional path to save the image
            
        Returns:
            DownloadedImage if successful, None otherwise
        """
        images = await self._search_images([query], query)
        
        if not images:
            return None
        
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            downloaded = await self._download_images(images[:1], output_dir, "single")
            return downloaded[0] if downloaded else None
        
        return None


# Example usage
async def main():
    """Test the image agent."""
    agent = ImageAgent(
        output_dir="output/test_images",
        images_per_section=2,
    )
    
    test_script = """
    [Intro]
    Welcome to our video about Python programming! Today we'll explore the amazing world of coding.
    
    [Content]
    Python is a versatile programming language used in web development, data science, and AI.
    Let's look at some code examples and learn how to write clean, efficient Python code.
    
    [Conclusion]
    Thanks for watching! Don't forget to subscribe for more programming tutorials.
    """
    
    result = await agent.find_images_for_script(
        script=test_script,
        topic="Python programming tutorial",
    )
    
    print(f"\n[OK] Found {result['total_images']} images")
    for section in result['sections']:
        print(f"\n[DIR] Section: {section['name']}")
        print(f"   Keywords: {section['keywords']}")
        for img in section['images']:
            print(f"   - {img['filename']} ({img['source']})")


if __name__ == "__main__":
    asyncio.run(main())
