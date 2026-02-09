"""
Multi-agent content generation pipeline example.

This demonstrates the complete end-to-end workflow using the ContentOrchestrator
to generate YouTube content from topic to final voice output.

Example topic: "Using GitHub Copilot"
"""

import asyncio
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.providers_config import ProvidersConfig
from src.providers.scripts import OpenAIScriptProvider
from src.providers.voice import ElevenLabsVoiceProvider
from src.services import ScriptService, VoiceService
from src.agents import ContentOrchestrator
from src.utils.output_manager import OutputManager


class MultiAgentPipeline:
    """
    Complete multi-agent pipeline for YouTube content generation.
    
    Orchestrates:
    - Topic research agent
    - Outline generation agent
    - Content writing agent
    - SSML formatting agent
    - Voice generation agent
    """
    
    def __init__(self):
        """Initialize the pipeline with all necessary services."""
        self.config = None
        self.script_service = None
        self.voice_service = None
        self.orchestrator = None
        self.output_manager = None
    
    async def initialize(self) -> bool:
        """
        Initialize all pipeline components.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            print("🚀 Initializing Multi-Agent Pipeline...")
            print("-" * 50)
            
            # Load configuration
            print("📋 Loading configuration...")
            self.config = ProvidersConfig()
            
            # Initialize providers
            print("🔧 Initializing providers...")
            script_config = self.config.get_script_provider_config()
            voice_config = self.config.get_voice_provider_config()
            
            script_provider = OpenAIScriptProvider(script_config)
            voice_provider = ElevenLabsVoiceProvider(voice_config)
            
            # Initialize services
            print("⚙️  Initializing services...")
            self.script_service = ScriptService(script_provider)
            self.voice_service = VoiceService(voice_provider)
            
            # Initialize output manager
            self.output_manager = OutputManager()
            
            # Get image API keys
            image_api_keys = self.config.get_image_api_keys()
            if image_api_keys:
                print(f"🖼️  Image APIs configured: {', '.join(image_api_keys.keys())}")
            else:
                print("🖼️  No image API keys found (using free sources only)")
            
            # Initialize orchestrator
            print("🎼 Initializing orchestrator...")
            self.orchestrator = ContentOrchestrator(
                self.script_service,
                self.voice_service,
                self.output_manager,
                image_api_keys=image_api_keys,
            )
            
            # Check service status
            print("🔍 Checking service status...")
            status = await self._check_services()
            
            if not status["script_available"] or not status["voice_available"]:
                print("❌ ERROR: Required services are not available")
                print(f"   Script Service: {'✓' if status['script_available'] else '✗'}")
                print(f"   Voice Service: {'✓' if status['voice_available'] else '✗'}")
                return False
            
            print("✅ All services initialized and ready!")
            print("-" * 50)
            return True
        
        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            return False
    
    async def _check_services(self) -> Dict[str, Any]:
        """
        Check the status of all services.
        
        Returns:
            Dictionary with service status information.
        """
        script_status = await self.script_service.get_provider_status()
        voice_status = await self.voice_service.get_provider_status()
        
        return {
            "script_available": script_status.is_available,
            "script_message": script_status.message,
            "voice_available": voice_status.is_available,
            "voice_message": voice_status.message,
        }
    
    async def generate_content(
        self,
        topic: str,
        tone: str = "informative",
        length: str = "medium",
        style: str = "conversational",
        language: str = "en",
        include_ssml: bool = True,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel (ElevenLabs)
        auto_save: bool = True,
        find_images: bool = True,
        images_per_section: int = 10,
        target_total_images: int = 100,
        create_video: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate complete YouTube content through the multi-agent pipeline.
        
        Args:
            topic: Topic for content generation.
            tone: Tone of the content.
            length: Length of the script.
            style: Style of the content.
            language: Language code.
            include_ssml: Whether to include SSML formatting.
            voice_id: Voice ID for voice generation.
            auto_save: Whether to save outputs to disk.
            find_images: Whether to search and download images.
            images_per_section: Number of images per script section.
            target_total_images: Target total images (~10 per minute of video).
            create_video: Whether to assemble final video from images and audio.
        
        Returns:
            Dictionary with workflow results.
        """
        print(f"\n📝 Generating content for: '{topic}'")
        print("=" * 60)
        
        start_time = time.time()
        
        result = await self.orchestrator.generate_content(
            topic=topic,
            tone=tone,
            length=length,
            style=style,
            language=language,
            include_ssml=include_ssml,
            voice_id=voice_id,
            voice_speed=1.0,
            voice_pitch=1.0,
            auto_save=auto_save,
            find_images=find_images,
            images_per_section=images_per_section,
            target_total_images=target_total_images,
            create_video=create_video,
        )
        
        elapsed_time = time.time() - start_time
        
        # Display results
        self._display_results(result, elapsed_time)
        
        return result
    
    def _display_results(self, result: Dict[str, Any], elapsed_time: float) -> None:
        """
        Display the workflow results in a formatted way.
        
        Args:
            result: The workflow result dictionary.
            elapsed_time: Total elapsed time in seconds.
        """
        print("\n📊 PIPELINE EXECUTION RESULTS")
        print("=" * 60)
        
        if result["success"]:
            print("✅ Status: SUCCESS")
            print(f"⏱️  Total Duration: {elapsed_time:.2f} seconds")
            print(f"📄 Script Word Count: {result.get('script_word_count', 'N/A')}")
            print(f"🎵 Voice Format: {result.get('voice_format', 'N/A')}")
            print(f"🎵 Voice Generated: {'Yes' if result.get('voice_generated') else 'No'}")
            print(f"🖼️  Images Found: {result.get('images_found', 0)}")
            print(f"🎬 Video Created: {'Yes' if result.get('video_created') else 'No'}")
            if result.get('video_path'):
                print(f"🎬 Video Path: {result.get('video_path')}")
            
            # Display workflow stages
            workflow_state = result.get("workflow_state", {})
            data = workflow_state.get("data", {})
            
            # Display research section
            if data.get("research_formatted"):
                print("\n" + "=" * 60)
                print("📚 RESEARCH SUMMARY")
                print("=" * 60)
                print(data["research_formatted"])
            
            if data.get("outline"):
                print(f"\n📋 Outline ({len(data['outline'])} points):")
                for i, point in enumerate(data["outline"], 1):
                    truncated = point[:60] + "..." if len(point) > 60 else point
                    print(f"   {i}. {truncated}")
            
            # Display script preview
            script = result.get("script", "")
            if script:
                print("\n📝 Script Preview (first 400 characters):")
                print("-" * 60)
                preview = script[:400] + "..." if len(script) > 400 else script
                print(preview)
                print("-" * 60)
            
            # Display saved files info
            if data.get("saved_files"):
                print("\n💾 Saved Files:")
                saved_files = data["saved_files"]
                for file_type, path in saved_files.items():
                    if file_type != "directory":
                        print(f"   {file_type}: {path}")
                print(f"   Directory: {saved_files.get('directory', 'N/A')}")
            
            # Display image info
            if data.get("images"):
                images_data = data["images"]
                print(f"\n🖼️  Images ({images_data.get('total_images', 0)} downloaded):")
                print(f"   Directory: {images_data.get('image_directory', 'N/A')}")
                for section in images_data.get("sections", []):
                    if section.get("images"):
                        print(f"   [{section['name']}]: {len(section['images'])} images")
                        for img in section["images"][:2]:  # Show first 2
                            print(f"      - {img['filename']} ({img['source']})")
        
        else:
            print("❌ Status: FAILED")
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            print(f"⏱️  Duration before failure: {elapsed_time:.2f} seconds")
            
            # Display errors from workflow
            workflow_state = result.get("workflow_state", {})
            if workflow_state.get("data", {}).get("errors"):
                print("\n📋 Errors encountered:")
                for error_info in workflow_state["data"]["errors"]:
                    print(f"   - Stage: {error_info['stage']}")
                    print(f"     Message: {error_info['error']}")
        
        print("=" * 60)
    
    async def generate_with_sections(
        self,
        topic: str,
        sections: list[str],
    ) -> Dict[str, Any]:
        """
        Generate content with parallel section writing.
        
        This demonstrates the orchestrator's ability to write multiple
        sections in parallel before combining them.
        
        Args:
            topic: Main topic for the content.
            sections: List of section titles to write.
        
        Returns:
            Dictionary with section content and metadata.
        """
        print(f"\n📝 Generating content with parallel sections for: '{topic}'")
        print("=" * 60)
        print(f"📋 Sections to write ({len(sections)}):")
        for i, section in enumerate(sections, 1):
            print(f"   {i}. {section}")
        print()
        
        start_time = time.time()
        
        try:
            section_content = await self.orchestrator.parallel_section_writing(
                topic=topic,
                sections=sections,
            )
            
            elapsed_time = time.time() - start_time
            
            print(f"\n✅ Parallel section writing completed in {elapsed_time:.2f} seconds")
            print("=" * 60)
            
            # Display results
            for section, content in section_content.items():
                word_count = len(content.split())
                preview = content[:150] + "..." if len(content) > 150 else content
                print(f"\n📄 Section: {section}")
                print(f"   Words: {word_count}")
                print(f"   Preview: {preview}")
            
            return {
                "success": True,
                "topic": topic,
                "sections": section_content,
                "total_duration": elapsed_time,
            }
        
        except Exception as e:
            print(f"❌ Parallel section writing failed: {str(e)}")
            return {
                "success": False,
                "topic": topic,
                "error": str(e),
            }


async def main():
    """Main entry point for the multi-agent pipeline example."""
    
    print("\n" + "=" * 60)
    print("🎬 MULTI-AGENT YOUTUBE CONTENT GENERATION PIPELINE")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = MultiAgentPipeline()
    
    if not await pipeline.initialize():
        print("\n❌ Failed to initialize pipeline. Exiting.")
        return
    
    # ===== TEST CASE 1: Standard content generation =====
    print("\n" + "=" * 60)
    print("📌 TEST CASE 1: Using GitHub Copilot")
    print("=" * 60)
    
    result1 = await pipeline.generate_content(
        topic="Using GitHub Copilot",
        tone="informative",
        length="medium",
        style="conversational",
        language="en",
        include_ssml=True,
        auto_save=True,
    )
    
    # ===== TEST CASE 2: Parallel section writing =====
    print("\n" + "=" * 60)
    print("📌 TEST CASE 2: Parallel Section Writing Example")
    print("=" * 60)
    
    sections = [
        "Getting Started with GitHub Copilot",
        "Advanced Features and Tips",
        "Best Practices for AI-Assisted Development",
    ]
    
    result2 = await pipeline.generate_with_sections(
        topic="GitHub Copilot for Developers",
        sections=sections,
    )
    
    # ===== SUMMARY =====
    print("\n" + "=" * 60)
    print("📊 PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    
    test1_status = "✅ PASSED" if result1.get("success") else "❌ FAILED"
    test2_status = "✅ PASSED" if result2.get("success") else "❌ FAILED"
    
    print(f"Test Case 1 (Standard Generation): {test1_status}")
    if result1.get("success"):
        print(f"  - Duration: {result1.get('duration_seconds', 0):.2f}s")
        print(f"  - Script Words: {result1.get('script_word_count', 0)}")
    
    print(f"Test Case 2 (Parallel Sections): {test2_status}")
    if result2.get("success"):
        print(f"  - Duration: {result2.get('total_duration', 0):.2f}s")
        print(f"  - Sections Generated: {len(result2.get('sections', {}))}")
    
    print("=" * 60)
    print("\n✨ Multi-Agent Pipeline Example Completed!")


if __name__ == "__main__":
    asyncio.run(main())
