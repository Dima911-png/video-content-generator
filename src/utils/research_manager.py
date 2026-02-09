"""Research data management and citation handling for content generation."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ResearchSource:
    """Represents a research source with URL and credibility."""
    
    url: str
    title: str
    source_type: str  # 'academic', 'industry', 'news', 'official', etc.
    date: Optional[str] = None
    author: Optional[str] = None
    description: Optional[str] = None
    
    def to_citation(self) -> str:
        """Format as a citation."""
        author_part = f"{self.author}. " if self.author else ""
        date_part = f"({self.date}). " if self.date else ""
        return f"{author_part}{date_part}{self.title} - {self.source_type.upper()}"


@dataclass
class ResearchPoint:
    """A single research finding with supporting details."""
    
    topic: str
    finding: str
    details: List[str] = field(default_factory=list)
    statistics: List[str] = field(default_factory=list)
    sources: List[ResearchSource] = field(default_factory=list)
    confidence: str = "high"  # 'high', 'medium', 'low'
    
    def to_formatted_text(self, citation_number: int = 1) -> str:
        """Format as readable text with in-line citation."""
        text = f"{self.finding}[{citation_number}]\n"
        
        if self.details:
            for detail in self.details:
                text += f"  - {detail}\n"
        
        if self.statistics:
            text += "\n  Statistics:\n"
            for stat in self.statistics:
                text += f"  - {stat}\n"
        
        return text


@dataclass
class ResearchData:
    """Complete research data for a topic."""
    
    topic: str
    research_points: List[ResearchPoint] = field(default_factory=list)
    summary: str = ""
    key_takeaways: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_all_sources(self) -> List[ResearchSource]:
        """Get all unique sources across all research points."""
        sources_dict = {}
        for point in self.research_points:
            for source in point.sources:
                sources_dict[source.url] = source
        return list(sources_dict.values())
    
    def to_formatted_text(self) -> str:
        """Format complete research as readable text."""
        text = f"RESEARCH: {self.topic.upper()}\n"
        text += "=" * 60 + "\n\n"
        
        if self.summary:
            text += "SUMMARY:\n"
            text += self.summary + "\n\n"
        
        if self.research_points:
            text += "KEY FINDINGS:\n"
            text += "-" * 60 + "\n"
            for i, point in enumerate(self.research_points, 1):
                text += point.to_formatted_text(citation_number=i)
                text += "\n"
        
        if self.key_takeaways:
            text += "\nKEY TAKEAWAYS:\n"
            for takeaway in self.key_takeaways:
                text += f"  - {takeaway}\n"
        
        # Add citations section
        sources = self.get_all_sources()
        if sources:
            text += "\n" + "=" * 60 + "\n"
            text += "CITATIONS:\n"
            text += "-" * 60 + "\n"
            for i, source in enumerate(sources, 1):
                text += f"[{i}] {source.to_citation()}\n"
                if source.description:
                    text += f"    {source.description}\n"
                if source.url:
                    text += f"    URL: {source.url}\n"
                text += "\n"
        
        return text
    
    def to_script_context(self) -> str:
        """Format research for inclusion in script generation prompt."""
        context = f"RESEARCH CONTEXT FOR: {self.topic}\n\n"
        
        if self.summary:
            context += f"Overview: {self.summary}\n\n"
        
        context += "Key Points:\n"
        for i, point in enumerate(self.research_points, 1):
            context += f"{i}. {point.finding}\n"
            for detail in point.details[:2]:  # Limit details
                context += f"   - {detail}\n"
            if point.statistics:
                context += f"   - Statistics: {', '.join(point.statistics[:1])}\n"
        
        if self.key_takeaways:
            context += "\nKey Takeaways:\n"
            for takeaway in self.key_takeaways:
                context += f"- {takeaway}\n"
        
        context += "\nNote: Include relevant findings and cite sources appropriately in the script."
        
        return context


class ResearchManager:
    """Manages research data for content generation."""
    
    def __init__(self):
        """Initialize research manager."""
        self.research_cache: Dict[str, ResearchData] = {}
    
    def create_research_data(self, topic: str) -> ResearchData:
        """Create a new research data object."""
        return ResearchData(topic=topic)
    
    def add_research_point(
        self,
        topic: str,
        finding: str,
        details: List[str] = None,
        statistics: List[str] = None,
        sources: List[ResearchSource] = None,
        confidence: str = "high"
    ) -> ResearchData:
        """Add a research point to topic research."""
        if topic not in self.research_cache:
            self.research_cache[topic] = self.create_research_data(topic)
        
        research = self.research_cache[topic]
        point = ResearchPoint(
            topic=topic,
            finding=finding,
            details=details or [],
            statistics=statistics or [],
            sources=sources or [],
            confidence=confidence
        )
        research.research_points.append(point)
        return research
    
    def set_research_summary(self, topic: str, summary: str) -> None:
        """Set research summary for a topic."""
        if topic not in self.research_cache:
            self.research_cache[topic] = self.create_research_data(topic)
        self.research_cache[topic].summary = summary
    
    def add_key_takeaway(self, topic: str, takeaway: str) -> None:
        """Add a key takeaway."""
        if topic not in self.research_cache:
            self.research_cache[topic] = self.create_research_data(topic)
        self.research_cache[topic].key_takeaways.append(takeaway)
    
    def get_research(self, topic: str) -> Optional[ResearchData]:
        """Get research data for a topic."""
        return self.research_cache.get(topic)
    
    def format_research_output(self, topic: str) -> str:
        """Get formatted research output."""
        research = self.get_research(topic)
        if research:
            return research.to_formatted_text()
        return f"No research data available for: {topic}"
    
    def get_research_for_script(self, topic: str) -> str:
        """Get research formatted for script context."""
        research = self.get_research(topic)
        if research:
            return research.to_script_context()
        return ""
