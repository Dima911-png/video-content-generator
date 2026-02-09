"""
Example: Generate YouTube script and voice in one pipeline.

This demonstrates the complete workflow:
1. Generate a script from a topic using OpenAI
2. Enhance the script with SSML tags
3. Generate voice from the script using ElevenLabs
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.providers_config import ProvidersConfig
from src.providers.scripts import OpenAIScriptProvider
from src.providers.voice import ElevenLabsVoiceProvider
from src.services import ScriptService, VoiceService, ContentGenerator
from src.models.schemas import (
    ScriptRequest,
    VoiceRequest,
    ContentRequest,
)
from src.utils.output_manager import OutputManager


async def main():
    """Main example function."""
    
    # Load configuration from .env and settings.json
    config = ProvidersConfig()
    
    # Initialize providers
    script_config = config.get_script_provider_config()
    voice_config = config.get_voice_provider_config()
    
    script_provider = OpenAIScriptProvider(script_config)
    voice_provider = ElevenLabsVoiceProvider(voice_config)
    
    # Initialize services
    script_service = ScriptService(script_provider)
    voice_service = VoiceService(voice_provider)
    
    # Create content generator pipeline
    generator = ContentGenerator(script_service, voice_service)
    
    # Check service status
    print("Checking service status...")
    status = await generator.get_service_status()
    print(f"Script Service: {status['script_service']['available']}")
    print(f"Voice Service: {status['voice_service']['available']}")
    
    if not (status['script_service']['available'] and status['voice_service']['available']):
        print("ERROR: Services are not available. Check API keys in .env")
        return
    
    # Create a content request
    content_request = ContentRequest(
        topic="How to Start a YouTube Channel in 2026",
        script_request=ScriptRequest(
            topic="How to Start a YouTube Channel in 2026",
            tone="informative",
            length="medium",
            style="conversational",
            language="en",
            include_ssml_tags=False,  # Set to False to test without SSML enhancement
        ),
        voice_request=VoiceRequest(
            text="",  # Will be filled by generator
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel (ElevenLabs)
            speed=1.0,
            pitch=1.0,
            language="en",
            format="mp3",
        ),
    )
    
    # Generate script first
    print("\nGenerating script...")
    print("(This calls OpenAI - may take 10-30 seconds)")
    try:
        script_response = await asyncio.wait_for(
            generator.generate_script_only(content_request),
            timeout=120  # 2 minute timeout for OpenAI
        )
        
        print("\n=== GENERATED SCRIPT ===")
        script_preview = script_response.script[:300] + "..." if len(script_response.script) > 300 else script_response.script
        print(script_preview)
        print(f"\nWord Count: {script_response.word_count}")
        print(f"Has SSML Tags: {script_response.has_ssml_tags}")
        
    except asyncio.TimeoutError:
        print("ERROR: Script generation timed out (>120 seconds)")
        return
    except Exception as e:
        print(f"ERROR generating script: {str(e)}")
        return
    
    # Generate voice from the script
    print("\n\nGenerating voice...")
    print("(This calls ElevenLabs - may take 10-30 seconds for SSML processing)")
    try:
        voice_response = await asyncio.wait_for(
            generator.generate_voice_only(script_response.script, content_request.voice_request),
            timeout=180  # 3 minute timeout for ElevenLabs (SSML can be slower)
        )
        
        print("\n=== GENERATED VOICE ===")
        print(f"Format: {voice_response.format}")
        audio_size_kb = len(voice_response.audio_base64) / 1024
        print(f"Audio Size: {audio_size_kb:.1f} KB (base64 encoded)")
        print(f"Voice ID: {voice_response.voice_id}")
        
        # Save output to files
        print("\n\nSaving files to disk...")
        output_manager = OutputManager()
        saved_files = output_manager.save_content(
            topic=content_request.topic,
            script_response=script_response,
            voice_response=voice_response,
        )
        
        print(f"✅ Script saved: {saved_files['script']}")
        print(f"✅ Audio saved: {saved_files['audio']}")
        print(f"✅ Metadata saved: {saved_files['metadata']}")
        print(f"\n📁 Output directory: {saved_files['directory']}")
        print("\n✅ Content generation completed successfully!")
        
    except asyncio.TimeoutError:
        print("ERROR: Voice generation timed out (>60 seconds)")
        return
    except Exception as e:
        print(f"ERROR generating voice: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
