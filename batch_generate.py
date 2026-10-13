#!/usr/bin/env python
"""
Batch Video Generation Script

Reads topics from a CSV file and generates videos for each pending topic.
Tracks completion status and allows for incremental processing.

Usage:
    python batch_generate.py                    # Process next pending topic
    python batch_generate.py --all              # Process all pending topics
    python batch_generate.py --topic "Topic"    # Process specific topic
    python batch_generate.py --create-template  # Create example topics.csv
    python batch_generate.py --status           # Show queue status
    python batch_generate.py --reset-failed     # Reset failed topics to pending
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logging_config import setup_logging
from src.utils.topic_queue import TopicQueue, TopicStatus
from src.agents.orchestrator import ContentOrchestrator
from src.services.script_service import ScriptService
from src.services.voice_service import VoiceService
from src.services.video_service import VideoConfig
from src.providers.scripts.openai_provider import OpenAIScriptProvider
from src.providers.voice.openai_tts_provider import OpenAITTSVoiceProvider
from src.models.schemas import ProviderConfig
from src.utils.output_manager import OutputManager

# Configure logging once at startup
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_api_keys() -> dict:
    """Load API keys from environment or .env file."""
    from dotenv import load_dotenv
    load_dotenv()
    
    return {
        "openai": os.getenv("OPENAI_API_KEY"),
        "elevenlabs": os.getenv("ELEVENLABS_API_KEY"),
        "pexels": os.getenv("PEXELS_API_KEY"),
        "pixabay": os.getenv("PIXABAY_API_KEY"),
    }


def create_orchestrator(api_keys: dict) -> ContentOrchestrator:
    """Create and configure the content orchestrator."""
    # Initialize providers with ProviderConfig
    script_config = ProviderConfig(
        provider_type="openai",
        api_key=api_keys["openai"],
        model_name="gpt-4o",  # or gpt-4-turbo for better quality
    )
    script_provider = OpenAIScriptProvider(config=script_config)
    
    # Prefer Russian TTS Studio VoxCPM2 when its server URL is configured.
    # The URL must point to a running, reachable Russian TTS Studio server.
    tts_studio_url = os.getenv("RUSSIAN_TTS_STUDIO_URL", "").strip()
    if tts_studio_url:
        from src.providers.voice.russian_tts_studio_provider import RussianTTSStudioVoiceProvider
        voice_config = ProviderConfig(
            provider_type="russian_tts_studio",
            api_url=tts_studio_url,
            model_name=os.getenv("RUSSIAN_TTS_ENGINE", "voxcpm"),
            timeout=int(os.getenv("RUSSIAN_TTS_TIMEOUT", "600")),
            extra_params={
                "instruct": (
                    "Russian educational YouTube narration. Natural conversational delivery, "
                    "warm confident adult voice, expressive but restrained intonation, clear "
                    "pronunciation, medium pace, and natural pauses. Avoid robotic delivery."
                ),
            },
        )
        voice_provider = RussianTTSStudioVoiceProvider(config=voice_config)
        logger.info("Voice engine: Russian TTS Studio (%s)", os.getenv("RUSSIAN_TTS_ENGINE", "voxcpm"))
    else:
        voice_config = ProviderConfig(
            provider_type="openai_tts",
            api_key=api_keys["openai"],
            model_name="gpt-4o-mini-tts",
            timeout=180,
            extra_params={
                "voice": os.getenv("OPENAI_TTS_VOICE", "marin"),
                "instructions": (
                    "Speak in natural, fluent Russian like a professional educational YouTube narrator. "
                    "Use clear pronunciation, warm confident delivery, natural intonation and moderate pacing. "
                    "Avoid robotic monotony, exaggerated acting, and unnatural pauses."
                ),
            },
        )
        voice_provider = OpenAITTSVoiceProvider(config=voice_config)
    
    # Initialize services
    script_service = ScriptService(provider=script_provider)
    voice_service = VoiceService(provider=voice_provider)
    output_manager = OutputManager()
    
    # Video config with smooth Ken Burns effects
    video_config = VideoConfig()
    video_config.output_resolution = (1920, 1080)
    video_config.fps = 24
    video_config.subtitle_font_size = 28
    video_config.subtitle_bg_opacity = 0.6
    
    # Image API keys
    image_api_keys = {
        "pexels": api_keys.get("pexels"),
        "pixabay": api_keys.get("pixabay"),
    }
    
    # Create orchestrator
    orchestrator = ContentOrchestrator(
        script_service=script_service,
        voice_service=voice_service,
        output_manager=output_manager,
        image_api_keys=image_api_keys,
        video_config=video_config,
    )
    
    return orchestrator


async def process_topic(
    orchestrator: ContentOrchestrator, 
    topic_entry, 
    queue: TopicQueue,
    skip_voice: bool = False,
    skip_images: bool = False,
    skip_video: bool = False,
    language: str = "ru",
) -> bool:
    """
    Process a single topic from the queue.
    
    Returns True if successful, False otherwise.
    """
    topic = topic_entry.topic
    
    logger.info(f"=" * 60)
    logger.info(f"Processing: {topic}")
    logger.info(f"Target duration: {topic_entry.target_duration_minutes} minutes")
    logger.info(f"Tone: {topic_entry.tone}, Style: {topic_entry.style}")
    if skip_voice:
        logger.info("[MODE] Voice generation DISABLED (--skip-voice)")
    if skip_images:
        logger.info("[MODE] Image search DISABLED (--skip-images)")
    if skip_video:
        logger.info("[MODE] Video assembly DISABLED (--skip-video)")
    logger.info(f"=" * 60)
    
    # Mark as in progress
    queue.mark_in_progress(topic)
    
    try:
        # Calculate target images based on duration (~3 per minute, each shown ~20s with Ken Burns)
        target_images = max(10, int(topic_entry.target_duration_minutes * 3))
        
        # Generate content
        result = await orchestrator.generate_content(
            topic=topic,
            tone=topic_entry.tone,
            style=topic_entry.style,
            language=language,
            length=topic_entry.get_length_setting(),
            # Generate with a 20% buffer because model word counts and speaking pace vary;
            # this helps the final narration meet the requested minimum duration.
            target_duration_minutes=round(topic_entry.target_duration_minutes * 1.2, 2),
            include_ssml=True,
            generate_voice=not skip_voice,
            find_images=not skip_images,
            target_total_images=target_images,
            create_video=not skip_video and not skip_voice,  # Can't make video without voice
        )
        
        if result.get("success"):
            # Get actual duration from voice output (may be None if --skip-voice)
            voice_output = result.get("workflow_state", {}).get("data", {}).get("voice_output")
            actual_duration_seconds = voice_output.get("duration_seconds", 0) if voice_output else 0
            actual_duration_minutes = actual_duration_seconds / 60 if actual_duration_seconds else None
            
            # Mark as completed
            queue.mark_completed(
                topic=topic,
                actual_duration_minutes=actual_duration_minutes,
                word_count=result.get("script_word_count"),
                output_path=result.get("video_path") or result.get("script_path"),
            )
            
            logger.info(f"[OK] SUCCESS: {topic}")
            logger.info(f"   Word count: {result.get('script_word_count')}")
            if actual_duration_minutes:
                logger.info(f"   Actual duration: {actual_duration_minutes:.1f} min")
            elif skip_voice:
                logger.info(f"   Duration: N/A (voice skipped)")
            else:
                logger.info(f"   Duration: N/A")
            if result.get('video_path'):
                logger.info(f"   Video: {result.get('video_path')}")
            elif result.get('script_path'):
                logger.info(f"   Script: {result.get('script_path')}")
            return True
        else:
            error_msg = result.get("error", "Unknown error")
            queue.mark_failed(topic, error_msg)
            logger.error(f"[FAIL] FAILED: {topic} - {error_msg}")
            return False
    
    except Exception as e:
        error_msg = str(e)
        queue.mark_failed(topic, error_msg)
        logger.error(f"[FAIL] FAILED: {topic} - {error_msg}")
        return False


async def main():
    parser = argparse.ArgumentParser(description="Batch video generation from CSV topic queue")
    parser.add_argument("--csv", default="topics.csv", help="Path to topics CSV file")
    parser.add_argument("--all", action="store_true", help="Process all pending topics")
    parser.add_argument("--topic", type=str, help="Process a specific topic by name")
    parser.add_argument("--language", choices=["ru", "en"], default="ru", help="Script and narration language (default: Russian)")
    parser.add_argument("--create-template", action="store_true", help="Create example topics.csv")
    parser.add_argument("--status", action="store_true", help="Show queue status")
    parser.add_argument("--reset-failed", action="store_true", help="Reset failed topics to pending")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without running")
    parser.add_argument("--skip-voice", action="store_true", help="Skip voice generation (saves ElevenLabs credits)")
    parser.add_argument("--skip-images", action="store_true", help="Skip image search and download")
    parser.add_argument("--skip-video", action="store_true", help="Skip video assembly")
    
    args = parser.parse_args()
    
    # Initialize queue
    queue = TopicQueue(csv_path=args.csv)
    
    # Handle special commands
    if args.create_template:
        if Path(args.csv).exists():
            print(f"Error: {args.csv} already exists. Delete it first or use a different name.")
            return
        queue.create_template()
        print(f"[OK] Created template at {args.csv}")
        print("Edit this file to add your topics, then run: python batch_generate.py")
        return
    
    if args.status:
        stats = queue.get_stats()
        print("\n== Topic Queue Status")
        print("=" * 40)
        print(f"Total topics:     {stats['total']}")
        print(f"Pending:          {stats['pending']}")
        print(f"In Progress:      {stats['in_progress']}")
        print(f"Completed:        {stats['completed']}")
        print(f"Failed:           {stats['failed']}")
        print(f"Skipped:          {stats['skipped']}")
        if stats['completed'] > 0:
            print(f"\nCompleted Stats:")
            print(f"  Total duration: {stats['total_duration_minutes']:.1f} min")
            print(f"  Avg duration:   {stats['avg_duration_minutes']:.1f} min")
        print()
        
        # Show pending topics
        pending = queue.get_all_pending()
        if pending:
            print("Pending Topics:")
            for i, entry in enumerate(pending, 1):
                print(f"  {i}. [{entry.priority}] {entry.topic} ({entry.target_duration_minutes} min)")
        return
    
    if args.reset_failed:
        count = queue.reset_failed()
        print(f"[OK] Reset {count} failed topics to pending")
        return
    
    # Load API keys
    api_keys = load_api_keys()
    
    # Validate API keys
    if not api_keys.get("openai"):
        print("Error: OPENAI_API_KEY not found in environment")
        return
    if not api_keys.get("elevenlabs"):
        print("Error: ELEVENLABS_API_KEY not found in environment")
        return
    
    # Create orchestrator
    orchestrator = create_orchestrator(api_keys)
    
    # Determine which topics to process
    topics_to_process = []
    
    if args.topic:
        # Find specific topic
        for entry in queue.topics:
            if entry.topic.lower() == args.topic.lower():
                topics_to_process.append(entry)
                break
        if not topics_to_process:
            print(f"Error: Topic '{args.topic}' not found in queue")
            return
    elif args.all:
        topics_to_process = queue.get_all_pending()
    else:
        # Get next pending topic
        next_topic = queue.get_next_pending()
        if next_topic:
            topics_to_process.append(next_topic)
    
    if not topics_to_process:
        print("[OK] No pending topics in queue!")
        return
    
    if args.dry_run:
        print(f"\n[DRY RUN] Would process {len(topics_to_process)} topic(s):")
        for entry in topics_to_process:
            print(f"  • {entry.topic} ({entry.target_duration_minutes} min, {entry.tone})")
        return
    
    # Process topics
    print(f"\n>> Starting batch processing of {len(topics_to_process)} topic(s)...")
    
    successful = 0
    failed = 0
    
    for entry in topics_to_process:
        try:
            success = await process_topic(
                orchestrator, entry, queue,
                skip_voice=args.skip_voice,
                skip_images=args.skip_images,
                skip_video=args.skip_video,
                language=args.language,
            )
            if success:
                successful += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            print("\n[!] Interrupted by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error processing {entry.topic}: {e}")
            failed += 1
    
    # Final summary
    print("\n" + "=" * 60)
    print("== Batch Processing Complete")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    
    stats = queue.get_stats()
    print(f"\nQueue Status: {stats['pending']} pending, {stats['completed']} completed, {stats['failed']} failed")


if __name__ == "__main__":
    asyncio.run(main())
