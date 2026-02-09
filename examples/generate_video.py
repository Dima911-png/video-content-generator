"""
Full Video Generation Pipeline Example.

This demonstrates the complete end-to-end workflow for generating a YouTube video:
1. Generate script using OpenAI
2. Generate voice using ElevenLabs
3. Download relevant images using Bing Image Search
4. Assemble final video with images as visuals and voice as VO

Usage:
    python examples/generate_video.py "Your Topic Here"
    python examples/generate_video.py  # Uses default topic
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.providers_config import ProvidersConfig
from src.providers.scripts import OpenAIScriptProvider
from src.providers.voice import ElevenLabsVoiceProvider
from src.services import ScriptService, VoiceService
from src.services.video_service import check_moviepy_available, VideoConfig
from src.agents import ContentOrchestrator
from src.utils.output_manager import OutputManager


async def generate_video(
    topic: str,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel
    images_per_section: int = 10,
    target_total_images: int = 100,
) -> None:
    """
    Generate a complete video for the given topic.
    
    Args:
        topic: The topic to create content about.
        voice_id: ElevenLabs voice ID to use.
        images_per_section: Number of images per script section.
        target_total_images: Target total number of images.
    """
    print("=" * 70)
    print("🎬 YOUTUBE VIDEO GENERATION PIPELINE")
    print("=" * 70)
    print(f"📋 Topic: {topic}")
    print()
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_moviepy_available():
        print("❌ moviepy is not installed!")
        print("   Install it with: pip install moviepy")
        print("   Note: You may also need to install ffmpeg on your system.")
        return
    print("   ✓ moviepy available")
    
    # Load configuration
    print("📋 Loading configuration...")
    config = ProvidersConfig()
    
    # Initialize providers
    print("🔧 Initializing providers...")
    script_config = config.get_script_provider_config()
    voice_config = config.get_voice_provider_config()
    
    script_provider = OpenAIScriptProvider(script_config)
    voice_provider = ElevenLabsVoiceProvider(voice_config)
    
    # Initialize services
    script_service = ScriptService(script_provider)
    voice_service = VoiceService(voice_provider)
    
    # Initialize output manager
    output_manager = OutputManager()
    
    # Get image API keys
    image_api_keys = config.get_image_api_keys()
    
    # Configure video settings
    video_config = VideoConfig(
        output_resolution=(1920, 1080),  # Full HD
        fps=24,
        min_image_duration=3.0,
        max_image_duration=8.0,
    )
    
    # Initialize orchestrator with video config
    print("🎼 Initializing orchestrator...")
    orchestrator = ContentOrchestrator(
        script_service=script_service,
        voice_service=voice_service,
        output_manager=output_manager,
        image_api_keys=image_api_keys,
        video_config=video_config,
    )
    
    # Check services
    print("🔍 Checking service status...")
    script_status = await script_service.get_provider_status()
    voice_status = await voice_service.get_provider_status()
    
    print(f"   Script Service: {'✓' if script_status.is_available else '✗'}")
    print(f"   Voice Service: {'✓' if voice_status.is_available else '✗'}")
    
    if not script_status.is_available:
        print(f"❌ Script service error: {script_status.message}")
        return
    
    if not voice_status.is_available:
        print(f"❌ Voice service error: {voice_status.message}")
        return
    
    print()
    print("🚀 Starting video generation pipeline...")
    print("-" * 70)
    
    # Generate content with video
    result = await orchestrator.generate_content(
        topic=topic,
        tone="informative",
        length="medium",
        style="conversational",
        language="en",
        include_ssml=True,
        voice_id=voice_id,
        voice_speed=1.0,
        voice_pitch=1.0,
        auto_save=True,
        find_images=True,
        images_per_section=images_per_section,
        target_total_images=target_total_images,
        create_video=True,
    )
    
    # Display results
    print()
    print("=" * 70)
    print("📊 PIPELINE RESULTS")
    print("=" * 70)
    
    if result["success"]:
        print("✅ Status: SUCCESS")
        print(f"⏱️  Duration: {result.get('duration_seconds', 0):.2f} seconds")
        print()
        
        print("📄 SCRIPT:")
        print(f"   Word Count: {result.get('script_word_count', 0)}")
        
        print()
        print("🎵 VOICE:")
        print(f"   Generated: {'Yes' if result.get('voice_generated') else 'No'}")
        print(f"   Format: {result.get('voice_format', 'N/A')}")
        
        print()
        print("🖼️  IMAGES:")
        print(f"   Downloaded: {result.get('images_found', 0)}")
        
        print()
        print("🎬 VIDEO:")
        print(f"   Created: {'Yes' if result.get('video_created') else 'No'}")
        if result.get('video_path'):
            print(f"   Path: {result.get('video_path')}")
        
        # Show saved files
        workflow_state = result.get("workflow_state", {})
        data = workflow_state.get("data", {})
        saved_files = data.get("saved_files", {})
        
        if saved_files:
            print()
            print("💾 SAVED FILES:")
            for file_type, path in saved_files.items():
                if file_type != "directory":
                    print(f"   {file_type}: {path}")
        
        # Show video details if available
        video_data = data.get("video", {})
        if video_data and video_data.get("success"):
            print()
            print("🎬 VIDEO DETAILS:")
            print(f"   Duration: {video_data.get('duration_seconds', 0):.2f} seconds")
            print(f"   Images Used: {video_data.get('num_images_used', 0)}")
            print(f"   Resolution: {video_data.get('resolution', 'N/A')}")
            file_size = video_data.get('file_size_bytes', 0)
            if file_size > 0:
                size_mb = file_size / (1024 * 1024)
                print(f"   File Size: {size_mb:.2f} MB")
        
        print()
        print("=" * 70)
        print("🎉 Video generation complete!")
        print("=" * 70)
    else:
        print("❌ Status: FAILED")
        print(f"   Error: {result.get('error', 'Unknown error')}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate a YouTube video from a topic"
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default="How to use Python for automation",
        help="Topic for the video (default: 'How to use Python for automation')"
    )
    parser.add_argument(
        "--voice-id",
        default="21m00Tcm4TlvDq8ikWAM",
        help="ElevenLabs voice ID (default: Rachel)"
    )
    parser.add_argument(
        "--images-per-section",
        type=int,
        default=10,
        help="Number of images per script section (default: 10)"
    )
    parser.add_argument(
        "--target-images",
        type=int,
        default=100,
        help="Target total number of images (default: 100)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(generate_video(
        topic=args.topic,
        voice_id=args.voice_id,
        images_per_section=args.images_per_section,
        target_total_images=args.target_images,
    ))


if __name__ == "__main__":
    main()
