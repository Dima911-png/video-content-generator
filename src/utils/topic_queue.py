"""
Topic Queue Manager for batch video generation.

Reads topics from a CSV file and tracks completion status.
"""

import csv
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TopicStatus(Enum):
    """Status of a topic in the queue."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TopicEntry:
    """A single topic entry from the CSV queue."""
    
    topic: str
    target_duration_minutes: float = 10.0  # Target video length in minutes
    status: TopicStatus = TopicStatus.PENDING
    priority: int = 0  # Lower = higher priority
    tone: str = "informative"
    style: str = "conversational"
    notes: str = ""
    
    # Tracking fields (updated after processing)
    actual_duration_minutes: Optional[float] = None
    word_count: Optional[int] = None
    output_path: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV writing."""
        return {
            "topic": self.topic,
            "target_duration_minutes": self.target_duration_minutes,
            "status": self.status.value,
            "priority": self.priority,
            "tone": self.tone,
            "style": self.style,
            "notes": self.notes,
            "actual_duration_minutes": self.actual_duration_minutes or "",
            "word_count": self.word_count or "",
            "output_path": self.output_path or "",
            "completed_at": self.completed_at or "",
            "error_message": self.error_message or "",
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TopicEntry":
        """Create from dictionary (CSV row)."""
        # Handle status conversion
        status_str = str(data.get("status", "pending")).lower().strip()
        try:
            status = TopicStatus(status_str)
        except ValueError:
            # Handle legacy/simple values
            if status_str in ("done", "completed", "yes", "true", "1"):
                status = TopicStatus.COMPLETED
            elif status_str in ("failed", "error"):
                status = TopicStatus.FAILED
            elif status_str in ("skip", "skipped"):
                status = TopicStatus.SKIPPED
            elif status_str in ("running", "in_progress", "in-progress"):
                status = TopicStatus.IN_PROGRESS
            else:
                status = TopicStatus.PENDING
        
        # Parse duration
        duration = data.get("target_duration_minutes", 10)
        try:
            duration = float(duration) if duration else 10.0
        except (ValueError, TypeError):
            duration = 10.0
        
        # Parse priority
        priority = data.get("priority", 0)
        try:
            priority = int(priority) if priority else 0
        except (ValueError, TypeError):
            priority = 0
        
        # Parse actual duration if present
        actual_duration = data.get("actual_duration_minutes", "")
        try:
            actual_duration = float(actual_duration) if actual_duration else None
        except (ValueError, TypeError):
            actual_duration = None
        
        # Parse word count if present
        word_count = data.get("word_count", "")
        try:
            word_count = int(word_count) if word_count else None
        except (ValueError, TypeError):
            word_count = None
        
        return cls(
            topic=str(data.get("topic", "")).strip(),
            target_duration_minutes=duration,
            status=status,
            priority=priority,
            tone=str(data.get("tone", "informative")).strip() or "informative",
            style=str(data.get("style", "conversational")).strip() or "conversational",
            notes=str(data.get("notes", "")).strip(),
            actual_duration_minutes=actual_duration,
            word_count=word_count,
            output_path=str(data.get("output_path", "")).strip() or None,
            completed_at=str(data.get("completed_at", "")).strip() or None,
            error_message=str(data.get("error_message", "")).strip() or None,
        )
    
    def get_length_setting(self) -> str:
        """
        Convert target duration to script length setting.
        
        Returns 'short', 'medium', or 'long' based on target duration.
        """
        if self.target_duration_minutes <= 5:
            return "short"
        elif self.target_duration_minutes <= 8:
            return "short"  # ~8 min
        elif self.target_duration_minutes <= 12:
            return "medium"  # ~10-12 min
        else:
            return "long"  # ~15-18 min
    
    def get_target_word_count(self) -> int:
        """
        Calculate target word count based on duration.
        
        Assumes ~150 words per minute of spoken content.
        """
        words_per_minute = 150
        return int(self.target_duration_minutes * words_per_minute)


class TopicQueue:
    """
    Manages a queue of topics from a CSV file.
    
    CSV Format:
        topic,target_duration_minutes,status,priority,tone,style,notes,actual_duration_minutes,word_count,output_path,completed_at,error_message
    
    Example:
        "How AI is Changing Healthcare",10,pending,1,informative,conversational,Focus on recent breakthroughs,,,,,
    """
    
    CSV_FIELDS = [
        "topic",
        "target_duration_minutes", 
        "status",
        "priority",
        "tone",
        "style",
        "notes",
        "actual_duration_minutes",
        "word_count",
        "output_path",
        "completed_at",
        "error_message",
    ]
    
    def __init__(self, csv_path: str = "topics.csv"):
        """
        Initialize topic queue.
        
        Args:
            csv_path: Path to the CSV file containing topics.
        """
        self.csv_path = Path(csv_path)
        self.topics: List[TopicEntry] = []
        
        # Load existing topics if file exists
        if self.csv_path.exists():
            self.load()
        else:
            logger.info(f"Topic queue file not found: {csv_path}. Creating empty queue.")
    
    def load(self) -> None:
        """Load topics from CSV file."""
        self.topics = []
        
        try:
            with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("topic", "").strip():  # Skip empty rows
                        entry = TopicEntry.from_dict(row)
                        self.topics.append(entry)
            
            logger.info(f"Loaded {len(self.topics)} topics from {self.csv_path}")
        
        except Exception as e:
            logger.error(f"Error loading topics from {self.csv_path}: {e}")
            raise
    
    def save(self) -> None:
        """Save topics to CSV file."""
        try:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
                writer.writeheader()
                for entry in self.topics:
                    writer.writerow(entry.to_dict())
            
            logger.info(f"Saved {len(self.topics)} topics to {self.csv_path}")
        
        except Exception as e:
            logger.error(f"Error saving topics to {self.csv_path}: {e}")
            raise
    
    def add_topic(
        self,
        topic: str,
        target_duration_minutes: float = 10.0,
        priority: int = 0,
        tone: str = "informative",
        style: str = "conversational",
        notes: str = "",
    ) -> TopicEntry:
        """
        Add a new topic to the queue.
        
        Args:
            topic: The video topic/title.
            target_duration_minutes: Target video length in minutes.
            priority: Priority (lower = higher priority).
            tone: Content tone.
            style: Content style.
            notes: Additional notes.
        
        Returns:
            The created TopicEntry.
        """
        entry = TopicEntry(
            topic=topic,
            target_duration_minutes=target_duration_minutes,
            priority=priority,
            tone=tone,
            style=style,
            notes=notes,
        )
        self.topics.append(entry)
        self.save()
        return entry
    
    def get_next_pending(self) -> Optional[TopicEntry]:
        """
        Get the next pending topic (highest priority first).
        
        Returns:
            Next pending TopicEntry or None if queue is empty.
        """
        pending = [t for t in self.topics if t.status == TopicStatus.PENDING]
        if not pending:
            return None
        
        # Sort by priority (lower = higher priority)
        pending.sort(key=lambda t: t.priority)
        return pending[0]
    
    def get_all_pending(self) -> List[TopicEntry]:
        """Get all pending topics sorted by priority."""
        pending = [t for t in self.topics if t.status == TopicStatus.PENDING]
        pending.sort(key=lambda t: t.priority)
        return pending
    
    def mark_in_progress(self, topic: str) -> bool:
        """Mark a topic as in progress."""
        for entry in self.topics:
            if entry.topic == topic:
                entry.status = TopicStatus.IN_PROGRESS
                self.save()
                return True
        return False
    
    def mark_completed(
        self,
        topic: str,
        actual_duration_minutes: Optional[float] = None,
        word_count: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> bool:
        """
        Mark a topic as completed with results.
        
        Args:
            topic: The topic to mark.
            actual_duration_minutes: Actual video duration.
            word_count: Script word count.
            output_path: Path to output files.
        
        Returns:
            True if topic was found and updated.
        """
        for entry in self.topics:
            if entry.topic == topic:
                entry.status = TopicStatus.COMPLETED
                entry.actual_duration_minutes = actual_duration_minutes
                entry.word_count = word_count
                entry.output_path = output_path
                entry.completed_at = datetime.now().isoformat()
                entry.error_message = None
                self.save()
                return True
        return False
    
    def mark_failed(self, topic: str, error_message: str) -> bool:
        """
        Mark a topic as failed.
        
        Args:
            topic: The topic to mark.
            error_message: Error description.
        
        Returns:
            True if topic was found and updated.
        """
        for entry in self.topics:
            if entry.topic == topic:
                entry.status = TopicStatus.FAILED
                entry.error_message = error_message
                entry.completed_at = datetime.now().isoformat()
                self.save()
                return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        stats = {
            "total": len(self.topics),
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "failed": 0,
            "skipped": 0,
            "total_duration_minutes": 0.0,
            "avg_duration_minutes": 0.0,
        }
        
        completed_durations = []
        
        for entry in self.topics:
            if entry.status == TopicStatus.PENDING:
                stats["pending"] += 1
            elif entry.status == TopicStatus.IN_PROGRESS:
                stats["in_progress"] += 1
            elif entry.status == TopicStatus.COMPLETED:
                stats["completed"] += 1
                if entry.actual_duration_minutes:
                    completed_durations.append(entry.actual_duration_minutes)
            elif entry.status == TopicStatus.FAILED:
                stats["failed"] += 1
            elif entry.status == TopicStatus.SKIPPED:
                stats["skipped"] += 1
        
        if completed_durations:
            stats["total_duration_minutes"] = sum(completed_durations)
            stats["avg_duration_minutes"] = stats["total_duration_minutes"] / len(completed_durations)
        
        return stats
    
    def reset_failed(self) -> int:
        """Reset all failed topics to pending. Returns count of reset topics."""
        count = 0
        for entry in self.topics:
            if entry.status == TopicStatus.FAILED:
                entry.status = TopicStatus.PENDING
                entry.error_message = None
                count += 1
        
        if count > 0:
            self.save()
        return count
    
    def create_template(self, num_entries: int = 5) -> None:
        """
        Create a template CSV file with example entries.
        
        Args:
            num_entries: Number of example entries to create.
        """
        if self.csv_path.exists():
            raise FileExistsError(f"File already exists: {self.csv_path}")
        
        examples = [
            TopicEntry(
                topic="The History of Artificial Intelligence",
                target_duration_minutes=10.0,
                priority=1,
                tone="informative",
                style="documentary",
                notes="Cover from 1950s to present day",
            ),
            TopicEntry(
                topic="How Electric Vehicles Work",
                target_duration_minutes=8.0,
                priority=2,
                tone="educational",
                style="conversational",
                notes="Focus on battery technology",
            ),
            TopicEntry(
                topic="The Science of Sleep",
                target_duration_minutes=12.0,
                priority=3,
                tone="informative",
                style="narrative",
                notes="Include practical tips",
            ),
            TopicEntry(
                topic="Cryptocurrency Explained Simply",
                target_duration_minutes=10.0,
                priority=4,
                tone="casual",
                style="educational",
                notes="Beginner-friendly explanation",
            ),
            TopicEntry(
                topic="Climate Change: What You Need to Know",
                target_duration_minutes=15.0,
                priority=5,
                tone="informative",
                style="documentary",
                notes="Include latest research",
            ),
        ]
        
        self.topics = examples[:num_entries]
        self.save()
        logger.info(f"Created template CSV with {len(self.topics)} example topics")
    
    def __len__(self) -> int:
        return len(self.topics)
    
    def __iter__(self):
        return iter(self.topics)
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"TopicQueue({stats['pending']} pending, {stats['completed']} completed, {stats['failed']} failed)"
