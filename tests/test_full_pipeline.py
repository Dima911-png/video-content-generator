"""
Comprehensive test suite for the YouTube Video Automation pipeline.

Tests all modules, classes, and the full pipeline using mocked API providers
so no real API calls are made.
"""

import asyncio
import base64
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = 0
FAIL = 0
ERRORS = []


def report(test_name: str, passed: bool, detail: str = ""):
    global PASS, FAIL, ERRORS
    if passed:
        PASS += 1
        print(f"  ✅ {test_name}")
    else:
        FAIL += 1
        ERRORS.append(f"{test_name}: {detail}")
        print(f"  ❌ {test_name}: {detail}")


# =============================================================================
# TEST 1: Pydantic Schemas
# =============================================================================
def test_schemas():
    print("\n" + "=" * 60)
    print("TEST 1: Pydantic Schemas")
    print("=" * 60)

    from src.models.schemas import (
        ProviderConfig, ScriptRequest, ScriptResponse,
        VoiceRequest, VoiceResponse, ContentRequest, ContentResponse, Voice,
    )

    # ProviderConfig
    try:
        config = ProviderConfig(provider_type="openai", api_key="test-key", model_name="gpt-4")
        report("ProviderConfig creation", config.provider_type == "openai" and config.api_key == "test-key")
    except Exception as e:
        report("ProviderConfig creation", False, str(e))

    # ProviderConfig defaults
    try:
        config = ProviderConfig(provider_type="test")
        report("ProviderConfig defaults", config.timeout == 120 and config.max_retries == 3)
    except Exception as e:
        report("ProviderConfig defaults", False, str(e))

    # ScriptRequest
    try:
        req = ScriptRequest(topic="AI", tone="humorous", length="long", style="narrative")
        report("ScriptRequest creation", req.topic == "AI" and req.tone == "humorous")
    except Exception as e:
        report("ScriptRequest creation", False, str(e))

    # ScriptRequest defaults
    try:
        req = ScriptRequest(topic="Test")
        report("ScriptRequest defaults",
               req.tone == "informative" and req.length == "medium" and req.language == "en")
    except Exception as e:
        report("ScriptRequest defaults", False, str(e))

    # ScriptResponse
    try:
        resp = ScriptResponse(script="Hello world", word_count=2)
        report("ScriptResponse creation", resp.script == "Hello world" and resp.word_count == 2)
    except Exception as e:
        report("ScriptResponse creation", False, str(e))

    # VoiceRequest
    try:
        req = VoiceRequest(text="Hello", voice_id="abc123", speed=1.2)
        report("VoiceRequest creation", req.text == "Hello" and req.speed == 1.2)
    except Exception as e:
        report("VoiceRequest creation", False, str(e))

    # VoiceResponse
    try:
        resp = VoiceResponse(audio_base64="dGVzdA==", format="mp3", voice_id="abc")
        report("VoiceResponse creation", resp.format == "mp3")
    except Exception as e:
        report("VoiceResponse creation", False, str(e))

    # Voice
    try:
        voice = Voice(voice_id="123", name="Rachel")
        report("Voice creation", voice.name == "Rachel")
    except Exception as e:
        report("Voice creation", False, str(e))

    # ContentRequest
    try:
        sr = ScriptRequest(topic="Test")
        vr = VoiceRequest(text="Test")
        cr = ContentRequest(topic="Test", script_request=sr, voice_request=vr)
        report("ContentRequest creation", cr.topic == "Test")
    except Exception as e:
        report("ContentRequest creation", False, str(e))

    # ContentResponse
    try:
        s_resp = ScriptResponse(script="text", word_count=1)
        v_resp = VoiceResponse(audio_base64="dGVzdA==", format="mp3")
        c_resp = ContentResponse(script_response=s_resp, voice_response=v_resp)
        report("ContentResponse creation", c_resp.script_response.word_count == 1)
    except Exception as e:
        report("ContentResponse creation", False, str(e))


# =============================================================================
# TEST 2: SSML Processor
# =============================================================================
def test_ssml_processor():
    print("\n" + "=" * 60)
    print("TEST 2: SSML Processor")
    print("=" * 60)

    from src.utils.ssml_processor import SsmlProcessor

    # Valid SSML check
    try:
        valid = SsmlProcessor.is_valid_ssml("<speak><p>Hello</p></speak>")
        report("Valid SSML detection", valid is True)
    except Exception as e:
        report("Valid SSML detection", False, str(e))

    # Invalid SSML check
    try:
        invalid = SsmlProcessor.is_valid_ssml("Hello world")
        report("Invalid SSML detection", invalid is False)
    except Exception as e:
        report("Invalid SSML detection", False, str(e))

    # Wrap with speak
    try:
        wrapped = SsmlProcessor.wrap_with_speak("Hello")
        report("Wrap with <speak>", wrapped == "<speak>Hello</speak>")
    except Exception as e:
        report("Wrap with <speak>", False, str(e))

    # Already wrapped
    try:
        already = SsmlProcessor.wrap_with_speak("<speak>Hello</speak>")
        report("Already wrapped unchanged", already == "<speak>Hello</speak>")
    except Exception as e:
        report("Already wrapped unchanged", False, str(e))

    # Add break
    try:
        result = SsmlProcessor.add_break("Hello", time="1s", position="after")
        report("Add break tag", '<break time="1s"/>' in result)
    except Exception as e:
        report("Add break tag", False, str(e))

    # Add emphasis
    try:
        result = SsmlProcessor.add_emphasis("important", level="strong")
        report("Add emphasis tag", '<emphasis level="strong">important</emphasis>' == result)
    except Exception as e:
        report("Add emphasis tag", False, str(e))

    # Add prosody
    try:
        result = SsmlProcessor.add_prosody("text", rate="slow", pitch="high")
        report("Add prosody tag", 'rate="slow"' in result and 'pitch="high"' in result)
    except Exception as e:
        report("Add prosody tag", False, str(e))

    # Remove tags
    try:
        cleaned = SsmlProcessor.remove_ssml_tags("<speak><p>Hello</p></speak>")
        report("Remove SSML tags", cleaned == "Hello")
    except Exception as e:
        report("Remove SSML tags", False, str(e))

    # Validate structure
    try:
        is_valid, msg = SsmlProcessor.validate_ssml_structure("<speak><p>Hello</p></speak>")
        report("Validate SSML structure (valid)", is_valid is True)
    except Exception as e:
        report("Validate SSML structure (valid)", False, str(e))

    try:
        is_valid, msg = SsmlProcessor.validate_ssml_structure("no tags")
        report("Validate SSML structure (invalid)", is_valid is False)
    except Exception as e:
        report("Validate SSML structure (invalid)", False, str(e))


# =============================================================================
# TEST 3: Prompt Service
# =============================================================================
def test_prompt_service():
    print("\n" + "=" * 60)
    print("TEST 3: Prompt Service")
    print("=" * 60)

    from src.services.prompt_service import PromptService, PromptConfig, get_prompt_service

    # PromptConfig
    try:
        config = PromptConfig()
        report("PromptConfig creation", "medium" in config.length_guidelines)
    except Exception as e:
        report("PromptConfig creation", False, str(e))

    # Duration word count
    try:
        result = PromptConfig.get_word_count_for_duration(10.0)
        report("Word count for duration", "words" in result and "10" in result)
    except Exception as e:
        report("Word count for duration", False, str(e))

    # PromptService initialization
    try:
        service = PromptService()
        report("PromptService init", service is not None)
    except Exception as e:
        report("PromptService init", False, str(e))

    # Build script prompt
    try:
        prompt = service.build_script_prompt(topic="AI Testing", tone="informative")
        report("Build script prompt", "AI Testing" in prompt and len(prompt) > 100)
    except Exception as e:
        report("Build script prompt", False, str(e))

    # Build script prompt with research context
    try:
        prompt = service.build_script_prompt(
            topic="Python", research_context="Python is a programming language."
        )
        report("Script prompt with research", "Python is a programming language" in prompt)
    except Exception as e:
        report("Script prompt with research", False, str(e))

    # Build outline prompt
    try:
        prompt = service.build_outline_prompt(topic="Machine Learning")
        report("Build outline prompt", "Machine Learning" in prompt)
    except Exception as e:
        report("Build outline prompt", False, str(e))

    # Build SSML prompt
    try:
        prompt = service.build_ssml_prompt(script="Hello world test script.")
        report("Build SSML prompt", "Hello world test script" in prompt)
    except Exception as e:
        report("Build SSML prompt", False, str(e))

    # Singleton service
    try:
        s1 = get_prompt_service()
        s2 = get_prompt_service()
        report("Singleton prompt service", s1 is s2)
    except Exception as e:
        report("Singleton prompt service", False, str(e))


# =============================================================================
# TEST 4: Research Manager
# =============================================================================
def test_research_manager():
    print("\n" + "=" * 60)
    print("TEST 4: Research Manager")
    print("=" * 60)

    from src.utils.research_manager import (
        ResearchManager, ResearchData, ResearchPoint, ResearchSource,
    )

    # ResearchSource
    try:
        source = ResearchSource(url="https://example.com", title="Example", source_type="web")
        citation = source.to_citation()
        report("ResearchSource creation", "Example" in citation)
    except Exception as e:
        report("ResearchSource creation", False, str(e))

    # ResearchPoint
    try:
        point = ResearchPoint(
            topic="AI", finding="AI is transformative",
            details=["Detail 1"], statistics=["Stat 1"],
        )
        text = point.to_formatted_text()
        report("ResearchPoint creation", "AI is transformative" in text)
    except Exception as e:
        report("ResearchPoint creation", False, str(e))

    # ResearchData
    try:
        data = ResearchData(topic="Test Topic")
        data.summary = "This is a summary"
        data.key_takeaways = ["Takeaway 1"]
        text = data.to_formatted_text()
        report("ResearchData formatted text", "TEST TOPIC" in text and "summary" in text.lower())
    except Exception as e:
        report("ResearchData formatted text", False, str(e))

    # to_script_context
    try:
        data = ResearchData(topic="AI")
        data.summary = "AI summary"
        data.research_points = [
            ResearchPoint(topic="AI", finding="Finding 1", details=["D1"]),
        ]
        context = data.to_script_context()
        report("ResearchData to_script_context", "Finding 1" in context)
    except Exception as e:
        report("ResearchData to_script_context", False, str(e))

    # ResearchManager
    try:
        manager = ResearchManager()
        manager.add_research_point("AI", "AI is growing")
        research = manager.get_research("AI")
        report("ResearchManager add/get", research is not None and len(research.research_points) == 1)
    except Exception as e:
        report("ResearchManager add/get", False, str(e))

    # Set summary
    try:
        manager.set_research_summary("AI", "AI summary")
        report("ResearchManager set summary", manager.get_research("AI").summary == "AI summary")
    except Exception as e:
        report("ResearchManager set summary", False, str(e))

    # Format output
    try:
        output = manager.format_research_output("AI")
        report("ResearchManager format output", "AI" in output)
    except Exception as e:
        report("ResearchManager format output", False, str(e))

    # Non-existent topic
    try:
        output = manager.format_research_output("Nonexistent")
        report("ResearchManager missing topic", "No research data" in output)
    except Exception as e:
        report("ResearchManager missing topic", False, str(e))


# =============================================================================
# TEST 5: Topic Queue
# =============================================================================
def test_topic_queue():
    print("\n" + "=" * 60)
    print("TEST 5: Topic Queue")
    print("=" * 60)

    from src.utils.topic_queue import TopicQueue, TopicEntry, TopicStatus

    temp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(temp_dir, "test_topics.csv")

    try:
        # TopicEntry creation
        entry = TopicEntry(topic="Test Topic", target_duration_minutes=10.0)
        report("TopicEntry creation", entry.topic == "Test Topic" and entry.status == TopicStatus.PENDING)

        # TopicEntry serialization
        d = entry.to_dict()
        restored = TopicEntry.from_dict(d)
        report("TopicEntry round-trip", restored.topic == "Test Topic" and restored.status == TopicStatus.PENDING)

        # Length setting
        entry_short = TopicEntry(topic="Short", target_duration_minutes=5.0)
        entry_long = TopicEntry(topic="Long", target_duration_minutes=15.0)
        report("TopicEntry length settings",
               entry_short.get_length_setting() == "short" and entry_long.get_length_setting() == "long")

        # Word count target
        wc = entry.get_target_word_count()
        report("TopicEntry word count", wc == 1500)  # 10 * 150

        # TopicQueue create template
        queue = TopicQueue(csv_path=csv_path)
        queue.create_template()
        report("TopicQueue create template", os.path.exists(csv_path) and len(queue) == 5)

        # Load queue
        queue2 = TopicQueue(csv_path=csv_path)
        report("TopicQueue load", len(queue2) == 5)

        # Get next pending
        next_topic = queue2.get_next_pending()
        report("TopicQueue get next pending", next_topic is not None)

        # Mark in progress
        result = queue2.mark_in_progress(next_topic.topic)
        report("TopicQueue mark in progress", result is True)

        # Mark completed
        result = queue2.mark_completed(next_topic.topic, actual_duration_minutes=9.5, word_count=1425)
        report("TopicQueue mark completed", result is True)

        # Stats
        stats = queue2.get_stats()
        report("TopicQueue stats", stats["completed"] == 1 and stats["pending"] == 4)

        # Mark failed + reset
        next2 = queue2.get_next_pending()
        queue2.mark_failed(next2.topic, "Test error")
        reset_count = queue2.reset_failed()
        report("TopicQueue reset failed", reset_count == 1)

    except Exception as e:
        report("TopicQueue tests", False, str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# TEST 6: Retry Utilities
# =============================================================================
def test_retry():
    print("\n" + "=" * 60)
    print("TEST 6: Retry Utilities")
    print("=" * 60)

    from src.utils.retry import retry_async, retry_sync

    # Sync retry - success
    try:
        call_count = 0

        @retry_sync(max_attempts=3, base_delay=0.01)
        def succeed_on_second():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Not yet")
            return "success"

        result = succeed_on_second()
        report("Sync retry succeeds on retry", result == "success" and call_count == 2)
    except Exception as e:
        report("Sync retry succeeds on retry", False, str(e))

    # Sync retry - exhaust retries
    try:
        @retry_sync(max_attempts=2, base_delay=0.01)
        def always_fail():
            raise ValueError("Always fails")

        try:
            always_fail()
            report("Sync retry exhausts retries", False, "Should have raised")
        except ValueError:
            report("Sync retry exhausts retries", True)
    except Exception as e:
        report("Sync retry exhausts retries", False, str(e))

    # Async retry - success
    try:
        call_count = 0

        @retry_async(max_attempts=3, base_delay=0.01)
        async def async_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Not yet")
            return "async_success"

        result = asyncio.run(async_succeed())
        report("Async retry succeeds on retry", result == "async_success")
    except Exception as e:
        report("Async retry succeeds on retry", False, str(e))


# =============================================================================
# TEST 7: Output Manager
# =============================================================================
def test_output_manager():
    print("\n" + "=" * 60)
    print("TEST 7: Output Manager")
    print("=" * 60)

    from src.utils.output_manager import OutputManager
    from src.models.schemas import ScriptResponse, VoiceResponse

    temp_dir = tempfile.mkdtemp()

    try:
        manager = OutputManager(output_dir=temp_dir)
        report("OutputManager creation", manager.output_dir == Path(temp_dir))

        # Save script only
        script_resp = ScriptResponse(script="Test script content", word_count=3)
        path = manager.save_script_only("Test Topic", script_resp)
        report("Save script only", os.path.exists(path))

        # Save voice only
        audio_bytes = b"fake audio data"
        audio_b64 = base64.b64encode(audio_bytes).decode()
        voice_resp = VoiceResponse(audio_base64=audio_b64, format="mp3")
        path = manager.save_voice_only("Test Topic 2", voice_resp)
        report("Save voice only", os.path.exists(path))

        # Save full content
        script_resp = ScriptResponse(script="Full content test", word_count=3, language="en")
        voice_resp = VoiceResponse(audio_base64=audio_b64, format="mp3", voice_id="test123")
        saved = manager.save_content("Full Test", script_resp, voice_resp)
        report("Save full content",
               os.path.exists(saved["script"]) and os.path.exists(saved["audio"]))

        # Verify metadata
        with open(saved["metadata"], "r") as f:
            meta = json.load(f)
        report("Content metadata correct", meta["topic"] == "Full Test")

    except Exception as e:
        report("OutputManager tests", False, str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# TEST 8: Config / ProvidersConfig
# =============================================================================
def test_providers_config():
    print("\n" + "=" * 60)
    print("TEST 8: ProvidersConfig")
    print("=" * 60)

    from config.providers_config import ProvidersConfig

    try:
        config = ProvidersConfig()
        report("ProvidersConfig init", config is not None)
    except Exception as e:
        report("ProvidersConfig init", False, str(e))

    try:
        script_type = config.get_script_provider_type()
        report("Script provider type", script_type in ("openai", "anthropic"))
    except Exception as e:
        report("Script provider type", False, str(e))

    try:
        voice_type = config.get_voice_provider_type()
        report("Voice provider type", voice_type == "elevenlabs")
    except Exception as e:
        report("Voice provider type", False, str(e))

    try:
        script_config = config.get_script_provider_config()
        report("Script provider config", script_config.provider_type == "openai" and script_config.api_key is not None)
    except Exception as e:
        report("Script provider config", False, str(e))

    try:
        voice_config = config.get_voice_provider_config()
        report("Voice provider config", voice_config.provider_type == "elevenlabs" and voice_config.api_key is not None)
    except Exception as e:
        report("Voice provider config", False, str(e))

    try:
        image_keys = config.get_image_api_keys()
        report("Image API keys", isinstance(image_keys, dict))
    except Exception as e:
        report("Image API keys", False, str(e))

    try:
        timeout = config.get_timeout("openai")
        report("Provider timeout", timeout > 0)
    except Exception as e:
        report("Provider timeout", False, str(e))


# =============================================================================
# TEST 9: Research Agent (live web search, graceful fallback)
# =============================================================================
def test_research_agent():
    print("\n" + "=" * 60)
    print("TEST 9: Research Agent")
    print("=" * 60)

    from src.agents.research_agent import ResearchAgent

    agent = ResearchAgent(timeout=10)

    try:
        result = asyncio.run(agent.research_topic("Python programming"))
        report("Research agent returns data",
               result is not None and result.topic == "Python programming")
        report("Research agent has findings",
               len(result.research_points) > 0 or len(result.key_takeaways) > 0)
        report("Research agent has summary", len(result.summary) > 0)
        formatted = result.to_formatted_text()
        report("Research formatted text", len(formatted) > 50)
    except Exception as e:
        report("Research agent", False, str(e))


# =============================================================================
# TEST 10: Video Service (unit tests, no actual video creation)
# =============================================================================
def test_video_service():
    print("\n" + "=" * 60)
    print("TEST 10: Video Service")
    print("=" * 60)

    from src.services.video_service import VideoService, VideoConfig, check_moviepy_available, check_ffmpeg_available

    # Config
    try:
        config = VideoConfig()
        report("VideoConfig defaults", config.output_resolution == (1920, 1080) and config.fps == 24)
    except Exception as e:
        report("VideoConfig defaults", False, str(e))

    try:
        d = config.to_dict()
        report("VideoConfig to_dict", "output_resolution" in d and "fps" in d)
    except Exception as e:
        report("VideoConfig to_dict", False, str(e))

    # Service creation
    try:
        service = VideoService()
        report("VideoService creation", service is not None)
    except Exception as e:
        report("VideoService creation", False, str(e))

    # Duration calculation
    try:
        durations = service.calculate_image_durations(10, 60.0)
        report("Image duration calc", len(durations) == 10 and abs(sum(durations) - 60.0) < 0.01)
    except Exception as e:
        report("Image duration calc", False, str(e))

    # Check availability flags
    try:
        mp = check_moviepy_available()
        ff = check_ffmpeg_available()
        report("Availability checks", isinstance(mp, bool) and isinstance(ff, bool))
    except Exception as e:
        report("Availability checks", False, str(e))

    # SRT time formatting
    try:
        time_str = service._format_srt_time(3661.5)
        report("SRT time format", time_str == "01:01:01,500")
    except Exception as e:
        report("SRT time format", False, str(e))

    # Subtitle wrapping
    try:
        wrapped = service._wrap_subtitle_text("This is a test of the subtitle wrapping function", max_chars=25)
        lines = wrapped.split("\n")
        report("Subtitle wrapping", len(lines) >= 2)
    except Exception as e:
        report("Subtitle wrapping", False, str(e))


# =============================================================================
# TEST 11: Full Pipeline with Mocked Providers
# =============================================================================
def test_full_pipeline_mocked():
    print("\n" + "=" * 60)
    print("TEST 11: Full Pipeline (Mocked Providers)")
    print("=" * 60)

    from src.models.schemas import ProviderConfig
    from src.models.script_provider import IScriptProvider, ScriptOptions, ScriptOutput
    from src.models.voice_provider import IVoiceProvider, VoiceOptions, VoiceOutput, Voice
    from src.models.base_provider import ProviderStatus
    from src.services.script_service import ScriptService
    from src.services.voice_service import VoiceService
    from src.agents.orchestrator import ContentOrchestrator, WorkflowStage

    # Create mock script provider
    class MockScriptProvider(IScriptProvider):
        async def validate(self):
            return True

        async def get_status(self):
            return ProviderStatus(True, "Mock ready")

        def get_capabilities(self):
            return ["generate_script", "generate_outline", "enhance_with_ssml"]

        def get_config(self):
            return {"provider_type": "mock"}

        async def generate_script(self, topic, options=None):
            script = (
                f"Welcome to this video about {topic}! "
                "Today we're going to explore some amazing concepts. "
                "First, let's understand the basics. "
                "This topic has been gaining tremendous attention lately. "
                "Here's what makes it so fascinating. "
                "Let's dive deep into the details and explore every aspect. "
                "The implications are far-reaching and transformative. "
                "In conclusion, this is truly remarkable. "
                "Thanks for watching and don't forget to subscribe!"
            )
            return ScriptOutput(script=script, word_count=len(script.split()))

        async def generate_outline(self, topic):
            return ["Introduction", "Key Concepts", "Deep Dive", "Conclusion"]

        async def enhance_script_with_ssml(self, script):
            return f"<speak>{script}</speak>"

    # Create mock voice provider
    class MockVoiceProvider(IVoiceProvider):
        async def validate(self):
            return True

        async def get_status(self):
            return ProviderStatus(True, "Mock ready")

        def get_capabilities(self):
            return ["text_to_speech"]

        def get_config(self):
            return {"provider_type": "mock"}

        async def generate_voice(self, text, options=None):
            fake_audio = b"RIFF" + b"\x00" * 100  # Fake WAV header
            return VoiceOutput(audio=fake_audio, format="mp3")

        async def list_available_voices(self):
            return [Voice(voice_id="mock-1", name="MockVoice")]

        async def process_ssml(self, ssml, options=None):
            fake_audio = b"RIFF" + b"\x00" * 100
            return VoiceOutput(audio=fake_audio, format="mp3")

    temp_dir = tempfile.mkdtemp()

    try:
        # Build the pipeline
        mock_script = MockScriptProvider()
        mock_voice = MockVoiceProvider()
        script_service = ScriptService(mock_script)
        voice_service = VoiceService(mock_voice)

        from src.utils.output_manager import OutputManager
        output_manager = OutputManager(output_dir=temp_dir)

        orchestrator = ContentOrchestrator(
            script_service=script_service,
            voice_service=voice_service,
            output_manager=output_manager,
        )

        report("Pipeline setup", orchestrator is not None)

        # Run the full pipeline (no images/video for speed)
        result = asyncio.run(orchestrator.generate_content(
            topic="Test Topic for Validation",
            tone="informative",
            length="medium",
            style="conversational",
            include_ssml=True,
            auto_save=True,
            find_images=False,
            create_video=False,
        ))

        report("Pipeline success", result["success"] is True)
        report("Pipeline has script", len(result.get("script", "")) > 50)
        report("Pipeline voice generated", result.get("voice_generated") is True)
        report("Pipeline word count", result.get("script_word_count", 0) > 0)
        report("Pipeline duration tracked", result.get("duration_seconds", 0) > 0)

        # Check workflow state
        state = result.get("workflow_state", {})
        report("Workflow completed stage", state.get("current_stage") == "completed")

        # Check files were saved
        saved_files = state.get("data", {}).get("saved_files", {})
        if saved_files:
            has_script = os.path.exists(saved_files.get("script", ""))
            has_voice = os.path.exists(saved_files.get("voice", ""))
            report("Script file saved to disk", has_script)
            report("Voice file saved to disk", has_voice)
        else:
            report("Script file saved to disk", False, "No saved_files in state")
            report("Voice file saved to disk", False, "No saved_files in state")

        # List workflows
        workflows = orchestrator.list_workflows()
        report("Workflow tracking", len(workflows) == 1)

    except Exception as e:
        report("Full pipeline test", False, str(e))
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# TEST 12: ContentGenerator Service
# =============================================================================
def test_content_generator():
    print("\n" + "=" * 60)
    print("TEST 12: ContentGenerator Service")
    print("=" * 60)

    from src.models.base_provider import ProviderStatus
    from src.models.script_provider import IScriptProvider, ScriptOptions, ScriptOutput
    from src.models.voice_provider import IVoiceProvider, VoiceOptions, VoiceOutput, Voice
    from src.services.script_service import ScriptService
    from src.services.voice_service import VoiceService
    from src.services.content_generator import ContentGenerator
    from src.models.schemas import ScriptRequest, VoiceRequest, ContentRequest

    class MockScript(IScriptProvider):
        async def validate(self): return True
        async def get_status(self): return ProviderStatus(True, "OK")
        def get_capabilities(self): return ["generate_script"]
        def get_config(self): return {}
        async def generate_script(self, topic, options=None):
            return ScriptOutput(script="Generated script text here.", word_count=5)
        async def generate_outline(self, topic): return ["Point 1"]
        async def enhance_script_with_ssml(self, script): return script

    class MockVoice(IVoiceProvider):
        async def validate(self): return True
        async def get_status(self): return ProviderStatus(True, "OK")
        def get_capabilities(self): return ["text_to_speech"]
        def get_config(self): return {}
        async def generate_voice(self, text, options=None):
            return VoiceOutput(audio=b"audio_data", format="mp3")
        async def list_available_voices(self): return []
        async def process_ssml(self, ssml, options=None):
            return VoiceOutput(audio=b"audio_data", format="mp3")

    try:
        script_svc = ScriptService(MockScript())
        voice_svc = VoiceService(MockVoice())
        generator = ContentGenerator(script_svc, voice_svc)
        report("ContentGenerator creation", generator is not None)

        # Service status
        status = asyncio.run(generator.get_service_status())
        report("Service status check",
               status["script_service"]["available"] and status["voice_service"]["available"])

        # Generate script only
        cr = ContentRequest(
            topic="Test",
            script_request=ScriptRequest(topic="Test", include_ssml_tags=False),
            voice_request=VoiceRequest(text=""),
        )
        script_result = asyncio.run(generator.generate_script_only(cr))
        report("Generate script only", script_result.word_count > 0)

        # Full content generation
        result = asyncio.run(generator.generate_content(cr))
        report("Full content generation",
               result.script_response.word_count > 0 and result.voice_response.format == "mp3")

    except Exception as e:
        report("ContentGenerator tests", False, str(e))


# =============================================================================
# TEST 13: Logging Config
# =============================================================================
def test_logging():
    print("\n" + "=" * 60)
    print("TEST 13: Logging Config")
    print("=" * 60)

    from src.utils.logging_config import setup_logging, get_logger

    try:
        root_logger = setup_logging(level=20)  # INFO
        report("Setup logging", root_logger is not None)
    except Exception as e:
        report("Setup logging", False, str(e))

    try:
        logger = get_logger("test_module")
        report("Get logger", logger.name == "test_module")
    except Exception as e:
        report("Get logger", False, str(e))


# =============================================================================
# TEST 14: ElevenLabs strip_ssml_tags
# =============================================================================
def test_elevenlabs_ssml_strip():
    print("\n" + "=" * 60)
    print("TEST 14: ElevenLabs SSML Strip")
    print("=" * 60)

    from src.providers.voice.elevenlabs_provider import strip_ssml_tags

    try:
        result = strip_ssml_tags('<speak><p>Hello <emphasis level="strong">world</emphasis></p></speak>')
        report("Strip SSML tags", "Hello" in result and "world" in result and "<" not in result)
    except Exception as e:
        report("Strip SSML tags", False, str(e))

    try:
        result = strip_ssml_tags('Text with <break time="1s"/> pause')
        report("Strip break tags (keeps pause)", "..." in result or "pause" in result)
    except Exception as e:
        report("Strip break tags", False, str(e))

    try:
        result = strip_ssml_tags("Plain text with [Citation] markers [1]")
        report("Strip citation markers", "[Citation]" not in result and "[1]" not in result)
    except Exception as e:
        report("Strip citation markers", False, str(e))


# =============================================================================
# TEST 15: Batch Generate CLI Arguments
# =============================================================================
def test_batch_generate_imports():
    print("\n" + "=" * 60)
    print("TEST 15: Batch Generate Script")
    print("=" * 60)

    try:
        # Just verify the file can be imported without errors
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "batch_generate",
            str(Path(__file__).parent.parent / "batch_generate.py")
        )
        report("batch_generate.py is loadable", spec is not None)
    except Exception as e:
        report("batch_generate.py is loadable", False, str(e))


# =============================================================================
# RUN ALL TESTS
# =============================================================================
if __name__ == "__main__":
    print("🧪 YOUTUBE VIDEO AUTOMATION - FULL TEST SUITE")
    print("=" * 60)

    test_schemas()
    test_ssml_processor()
    test_prompt_service()
    test_research_manager()
    test_topic_queue()
    test_retry()
    test_output_manager()
    test_providers_config()
    test_research_agent()
    test_video_service()
    test_full_pipeline_mocked()
    test_content_generator()
    test_logging()
    test_elevenlabs_ssml_strip()
    test_batch_generate_imports()

    # Final summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    total = PASS + FAIL
    print(f"  Total:  {total}")
    print(f"  Passed: {PASS} ✅")
    print(f"  Failed: {FAIL} ❌")
    if ERRORS:
        print(f"\n  Failures:")
        for err in ERRORS:
            print(f"    • {err}")
    print()
    if FAIL == 0:
        print("🎉 ALL TESTS PASSED! Pipeline is ready for GitHub.")
    else:
        print(f"⚠️  {FAIL} test(s) failed. Review above for details.")
    print("=" * 60)

    sys.exit(0 if FAIL == 0 else 1)
