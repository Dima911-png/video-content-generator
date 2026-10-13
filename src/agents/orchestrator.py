"""
Multi-agent orchestrator for coordinating YouTube content generation workflow.

This module manages the complete workflow from topic research to final SSML-formatted
script generation, coordinating multiple agents and handling data flow between stages.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from src.services.script_service import ScriptService
from src.services.voice_service import VoiceService
from src.services.video_service import VideoService, VideoConfig, check_moviepy_available
from src.models.schemas import ScriptRequest, VoiceRequest, ScriptResponse, VoiceResponse
from src.utils.output_manager import OutputManager
from src.utils.research_manager import ResearchManager, ResearchSource, ResearchPoint
from src.agents.research_agent import ResearchAgent
from src.agents.image_agent import ImageAgent

logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """Stages in the content generation workflow."""
    
    TOPIC_RESEARCH = "topic_research"
    OUTLINE_GENERATION = "outline_generation"
    CONTENT_WRITING = "content_writing"
    SSML_FORMATTING = "ssml_formatting"
    VOICE_GENERATION = "voice_generation"
    IMAGE_SEARCH = "image_search"
    VIDEO_ASSEMBLY = "video_assembly"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowState:
    """Manages the state of a content generation workflow."""
    
    def __init__(self, topic: str):
        """
        Initialize workflow state.
        
        Args:
            topic: The topic being processed.
        """
        self.topic = topic
        self.current_stage = WorkflowStage.TOPIC_RESEARCH
        self.started_at = datetime.now()
        self.completed_at: Optional[datetime] = None
        self.data: Dict[str, Any] = {
            "topic": topic,
            "research_data": None,
            "research_formatted": None,
            "outline": None,
            "raw_script": None,
            "formatted_script": None,
            "voice_output": None,
            "errors": [],
        }
        self.retry_count = 0
        self.max_retries = 3
    
    def update_stage(self, stage: WorkflowStage) -> None:
        """Update the current stage."""
        logger.info(f"Workflow stage transition: {self.current_stage.value} -> {stage.value}")
        self.current_stage = stage
    
    def add_error(self, stage: str, error: Exception) -> None:
        """Add an error to the workflow state."""
        self.data["errors"].append({
            "stage": stage,
            "error": str(error),
            "timestamp": datetime.now().isoformat(),
        })
    
    def get_duration(self) -> float:
        """Get duration in seconds."""
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "topic": self.topic,
            "current_stage": self.current_stage.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.get_duration(),
            "retry_count": self.retry_count,
            "data": self.data,
        }


class ContentOrchestrator:
    """
    Master orchestrator coordinating all agents through the content generation workflow.
    
    Manages:
    - Workflow state and progression
    - Agent coordination and sequencing
    - Data flow between agents
    - Error handling and retries
    - Progress logging
    - Parallel section writing
    """
    
    def __init__(
        self,
        script_service: ScriptService,
        voice_service: VoiceService,
        output_manager: Optional[OutputManager] = None,
        research_manager: Optional[ResearchManager] = None,
        research_agent: Optional[ResearchAgent] = None,
        image_agent: Optional[ImageAgent] = None,
        image_api_keys: Optional[Dict[str, str]] = None,
        video_config: Optional[VideoConfig] = None,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            script_service: Service for script generation.
            voice_service: Service for voice generation.
            output_manager: Optional manager for saving outputs.
            research_manager: Optional manager for research data.
            research_agent: Optional agent for web research.
            image_agent: Optional agent for finding/downloading images.
            image_api_keys: Optional API keys for image services (pexels, pixabay).
            video_config: Optional configuration for video assembly.
        """
        self.script_service = script_service
        self.voice_service = voice_service
        self.output_manager = output_manager or OutputManager()
        self.research_manager = research_manager or ResearchManager()
        self.research_agent = research_agent or ResearchAgent()
        self.image_agent = image_agent or ImageAgent(
            output_dir=self.output_manager.output_dir,
            api_keys=image_api_keys or {},
        )
        self.video_service = VideoService(config=video_config) if check_moviepy_available() else None
        self.workflows: Dict[str, WorkflowState] = {}
    
    async def generate_content(
        self,
        topic: str,
        tone: str = "informative",
        length: str = "medium",
        style: str = "conversational",
        language: str = "en",
        include_ssml: bool = True,
        generate_voice: bool = True,
        voice_id: Optional[str] = None,
        voice_speed: float = 1.0,
        voice_pitch: float = 1.0,
        auto_save: bool = True,
        find_images: bool = True,
        images_per_section: int = 10,
        target_total_images: int = 100,
        create_video: bool = True,
        target_duration_minutes: Optional[float] = None,
        custom_instructions: str = "",
    ) -> Dict[str, Any]:
        """
        Generate complete content through the full workflow.
        
        Args:
            topic: Topic for content generation.
            tone: Tone of the content.
            length: Length of the script (ignored if target_duration_minutes is set).
            style: Style of the content.
            language: Language code.
            include_ssml: Whether to include SSML formatting.
            generate_voice: Whether to generate voice audio (set False to save API credits).
            voice_id: Voice ID for voice generation.
            voice_speed: Voice generation speed multiplier.
            voice_pitch: Voice generation pitch multiplier.
            auto_save: Whether to automatically save outputs.
            find_images: Whether to search and download images.
            images_per_section: Number of images per script section.
            target_total_images: Target total images for video (~10 per minute).
            create_video: Whether to assemble final video from images and audio.
            target_duration_minutes: Explicit target video duration in minutes.
        
        Returns:
            Dictionary with complete workflow results.
        
        Raises:
            RuntimeError: If critical stages fail after retries.
        """
        # Store target duration in workflow for later use
        workflow = WorkflowState(topic)
        workflow.data["target_duration_minutes"] = target_duration_minutes
        self.workflows[topic] = workflow
        
        logger.info(f"Starting content generation workflow for topic: {topic}" + 
                    (f" (target: {target_duration_minutes} min)" if target_duration_minutes else ""))
        
        try:
            # Stage 1: Topic Research (gathering context)
            workflow.update_stage(WorkflowStage.TOPIC_RESEARCH)
            logger.info("Stage 1/5: Researching topic...")
            await self._research_topic(workflow)
            
            # Stage 2: Outline Generation
            workflow.update_stage(WorkflowStage.OUTLINE_GENERATION)
            logger.info("Stage 2/5: Generating outline...")
            await self._generate_outline(workflow)
            
            # Stage 3: Content Writing
            workflow.update_stage(WorkflowStage.CONTENT_WRITING)
            logger.info("Stage 3/5: Writing content script...")
            script_response = await self._write_content(
                workflow, tone, length, style, language, target_duration_minutes, custom_instructions
            )
            workflow.data["raw_script"] = script_response.script
            
            # Stage 4: SSML Formatting
            workflow.update_stage(WorkflowStage.SSML_FORMATTING)
            logger.info("Stage 4/5: Formatting with SSML tags...")
            formatted_script = await self._format_with_ssml(
                workflow, script_response, include_ssml
            )
            workflow.data["formatted_script"] = formatted_script
            
            # Save script immediately after formatting (before voice which might fail)
            if auto_save:
                script_saved = await self._save_script_only(
                    workflow, script_response, formatted_script
                )
                logger.info(f"[FILE] Script saved: {script_saved.get('script', 'N/A')}")
            
            # Stage 5: Voice Generation (optional - failures won't lose script)
            workflow.update_stage(WorkflowStage.VOICE_GENERATION)
            voice_response = None
            if generate_voice:
                logger.info("Stage 5/5: Generating voice...")
                try:
                    voice_response = await self._generate_voice(
                        workflow,
                        formatted_script,
                        language,
                        voice_id,
                        voice_speed,
                        voice_pitch,
                    )
                    workflow.data["voice_output"] = {
                        "format": voice_response.format,
                        "duration_seconds": voice_response.duration_seconds,
                        "voice_id": voice_response.voice_id,
                    }
                    
                    # Save voice output
                    if auto_save and voice_response:
                        await self._save_voice_only(workflow, voice_response)
                except Exception as voice_error:
                    logger.warning(f"[WARN] Voice generation failed (script still saved): {str(voice_error)}")
                    workflow.add_error("voice_generation", voice_error)
            else:
                logger.info("Stage 5/5: Voice generation SKIPPED (--skip-voice)")
            
            # Stage 6: Image Search (optional - failures won't affect other outputs)
            images_result = None
            if find_images:
                workflow.update_stage(WorkflowStage.IMAGE_SEARCH)
                logger.info("Stage 6: Searching and downloading images...")
                try:
                    # Get output subdirectory from saved files
                    saved_files = workflow.data.get("saved_files", {})
                    script_path = saved_files.get("script", "")
                    if script_path:
                        from pathlib import Path
                        output_subdir = Path(script_path).parent.name
                    else:
                        output_subdir = None
                    
                    # Configure image agent for target image count
                    self.image_agent.images_per_section = images_per_section
                    self.image_agent.target_total_images = target_total_images
                    
                    images_result = await self.image_agent.find_images_for_script(
                        script=formatted_script,
                        topic=topic,
                        output_subdir=output_subdir,
                    )
                    workflow.data["images"] = images_result
                    logger.info(f"[IMG] Downloaded {images_result.get('total_images', 0)} images")
                except Exception as image_error:
                    logger.warning(f"[WARN] Image search failed: {str(image_error)}")
                    workflow.add_error("image_search", image_error)
            
            # Stage 7: Video Assembly (optional - requires images and voice)
            video_result = None
            if create_video and find_images and voice_response and self.video_service:
                workflow.update_stage(WorkflowStage.VIDEO_ASSEMBLY)
                logger.info("Stage 7: Assembling video from images and audio...")
                try:
                    video_result = await self._assemble_video(
                        workflow=workflow,
                        images_result=images_result,
                    )
                    workflow.data["video"] = video_result
                    if video_result and video_result.get("success"):
                        logger.info(f"[VIDEO] Video created: {video_result.get('video_path')}")
                except Exception as video_error:
                    logger.warning(f"[WARN] Video assembly failed: {str(video_error)}")
                    workflow.add_error("video_assembly", video_error)
            elif create_video:
                if not self.video_service:
                    logger.warning("[WARN] Video assembly skipped: moviepy not installed")
                elif not voice_response:
                    logger.warning("[WARN] Video assembly skipped: no voice audio available")
                elif not find_images:
                    logger.warning("[WARN] Video assembly skipped: images not requested")
            
            # Finalize workflow
            workflow.update_stage(WorkflowStage.COMPLETED)
            workflow.completed_at = datetime.now()
            
            logger.info(f"[OK] Workflow completed in {workflow.get_duration():.2f}s")
            
            return {
                "success": True,
                "topic": topic,
                "workflow_state": workflow.to_dict(),
                "script": formatted_script,
                "script_word_count": script_response.word_count,
                "voice_format": voice_response.format if voice_response else None,
                "voice_generated": voice_response is not None,
                "images_found": images_result.get("total_images", 0) if images_result else 0,
                "video_created": video_result.get("success", False) if video_result else False,
                "video_path": video_result.get("video_path") if video_result else None,
                "duration_seconds": workflow.get_duration(),
            }
        
        except Exception as e:
            workflow.update_stage(WorkflowStage.FAILED)
            workflow.completed_at = datetime.now()
            workflow.add_error(workflow.current_stage.value, e)
            
            logger.error(
                f"[FAIL] Workflow failed at stage {workflow.current_stage.value}: {str(e)}"
            )
            
            return {
                "success": False,
                "topic": topic,
                "error": str(e),
                "workflow_state": workflow.to_dict(),
                "duration_seconds": workflow.get_duration(),
            }
    
    async def _research_topic(self, workflow: WorkflowState) -> None:
        """
        Research the topic using web search.
        
        Args:
            workflow: The workflow state being processed.
        """
        try:
            logger.info(f"Researching topic: {workflow.topic}")
            
            # Use ResearchAgent to conduct real web research
            research_data = await self.research_agent.research_topic(workflow.topic)
            
            # Store research data
            self.research_manager.research_cache[workflow.topic] = research_data
            workflow.data["research_data"] = research_data
            workflow.data["research_formatted"] = research_data.to_formatted_text()
            
            logger.info(f"Research completed: {len(research_data.research_points)} findings")
        
        except Exception as e:
            workflow.add_error("research_topic", e)
            logger.error(f"Error during topic research: {str(e)}")
            # Continue with empty research rather than failing
            research_data = self.research_manager.create_research_data(workflow.topic)
            research_data.summary = f"Research unavailable: {str(e)}"
            workflow.data["research_data"] = research_data
            workflow.data["research_formatted"] = research_data.to_formatted_text()
    
    async def _generate_outline(self, workflow: WorkflowState) -> None:
        """
        Generate an outline for the topic.
        
        Args:
            workflow: The workflow state being processed.
        """
        try:
            logger.debug(f"Generating outline for: {workflow.topic}")
            
            outline = await self.script_service.generate_outline(workflow.topic)
            workflow.data["outline"] = outline
            
            logger.debug(f"Generated outline with {len(outline)} points")
        
        except Exception as e:
            workflow.add_error("generate_outline", e)
            logger.error(f"Error generating outline: {str(e)}")
            raise
    
    async def _write_content(
        self,
        workflow: WorkflowState,
        tone: str,
        length: str,
        style: str,
        language: str,
        target_duration_minutes: Optional[float] = None,
        custom_instructions: str = "",
    ) -> ScriptResponse:
        """
        Write the main content script.
        
        Args:
            workflow: The workflow state being processed.
            tone: Tone of the content.
            length: Length of the script (ignored if target_duration_minutes is set).
            style: Style of the content.
            language: Language code.
            target_duration_minutes: Explicit target video duration in minutes.
        
        Returns:
            Generated script response.
        """
        try:
            logger.debug("Initiating content writing agent")
            
            # Get research context for the script
            research_context = ""
            if workflow.data.get("research_data"):
                research_context = workflow.data["research_data"].to_script_context()
            
            # Build extra params including target duration
            extra_params = {"research_context": research_context}
            if target_duration_minutes:
                extra_params["target_duration_minutes"] = target_duration_minutes
            if custom_instructions.strip():
                extra_params["custom_instructions"] = custom_instructions.strip()
            
            script_request = ScriptRequest(
                topic=workflow.topic,
                tone=tone,
                length=length,
                style=style,
                language=language,
                include_ssml_tags=False,  # Will be done in next stage
                extra_params=extra_params,
            )
            
            # Retry logic for content writing
            for attempt in range(workflow.max_retries):
                try:
                    script_response = await asyncio.wait_for(
                        self.script_service.generate_script(script_request),
                        timeout=180,  # 3 minute timeout
                    )
                    logger.debug(
                        f"Content writing completed: {script_response.word_count} words"
                    )
                    return script_response
                
                except asyncio.TimeoutError:
                    logger.warning(f"Content writing timed out (attempt {attempt + 1}/{workflow.max_retries})")
                    if attempt < workflow.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise RuntimeError("Content writing failed after max retries")
        
        except Exception as e:
            workflow.add_error("write_content", e)
            logger.error(f"Error writing content: {str(e)}")
            raise
    
    async def _format_with_ssml(
        self,
        workflow: WorkflowState,
        script_response: ScriptResponse,
        include_ssml: bool,
    ) -> str:
        """
        Format the script with SSML tags.
        
        Args:
            workflow: The workflow state being processed.
            script_response: The generated script response.
            include_ssml: Whether to include SSML formatting.
        
        Returns:
            Formatted script with SSML tags.
        """
        try:
            if not include_ssml:
                logger.debug("SSML formatting skipped (disabled)")
                return script_response.script
            
            logger.debug("Applying SSML formatting")
            
            # Note: In a real implementation, this would be a separate agent
            # For now, we'll use the provider's enhancement capability
            formatted_script = await self.script_service.provider.enhance_script_with_ssml(
                script_response.script
            )
            
            logger.debug("SSML formatting completed")
            return formatted_script
        
        except Exception as e:
            logger.warning(f"SSML formatting failed, using raw script: {str(e)}")
            # Fall back to raw script if SSML formatting fails
            return script_response.script
    
    async def _generate_voice(
        self,
        workflow: WorkflowState,
        script: str,
        language: str,
        voice_id: Optional[str],
        voice_speed: float,
        voice_pitch: float,
    ) -> VoiceResponse:
        """
        Generate voice from the formatted script.
        
        Args:
            workflow: The workflow state being processed.
            script: The formatted script text.
            language: Language code.
            voice_id: Voice ID to use.
            voice_speed: Voice speed multiplier.
            voice_pitch: Voice pitch multiplier.
        
        Returns:
            Generated voice response.
        """
        try:
            logger.debug("Initiating voice generation agent")
            
            # Default voice if not specified
            if not voice_id:
                voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel (ElevenLabs default)
            
            voice_request = VoiceRequest(
                text=script,
                voice_id=voice_id,
                speed=voice_speed,
                pitch=voice_pitch,
                language=language,
                format="mp3",
            )
            
            # Retry logic for voice generation
            for attempt in range(workflow.max_retries):
                try:
                    voice_response = await asyncio.wait_for(
                        self.voice_service.generate_voice(voice_request),
                        timeout=300,  # 5 minute timeout (SSML can be slower)
                    )
                    logger.debug(
                        f"Voice generation completed: {voice_response.format} format"
                    )
                    return voice_response
                
                except asyncio.TimeoutError:
                    logger.warning(f"Voice generation timed out (attempt {attempt + 1}/{workflow.max_retries})")
                    if attempt < workflow.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise RuntimeError("Voice generation failed after max retries")
        
        except Exception as e:
            workflow.add_error("generate_voice", e)
            logger.error(f"Error generating voice: {str(e)}")
            raise
    
    async def _save_script_only(
        self,
        workflow: WorkflowState,
        script_response: ScriptResponse,
        formatted_script: str,
    ) -> Dict[str, str]:
        """
        Save just the script to disk (without voice).
        
        This ensures scripts are preserved even if voice generation fails.
        
        Args:
            workflow: The workflow state being processed.
            script_response: The generated script response.
            formatted_script: The SSML-formatted script.
            
        Returns:
            Dictionary with saved file paths.
        """
        try:
            logger.debug("Saving script to disk (before voice generation)")
            
            # Create output directory based on topic
            import os
            from pathlib import Path
            
            # Sanitize topic for filename
            safe_topic = "".join(c if c.isalnum() or c in " -_" else "_" for c in workflow.topic)
            safe_topic = safe_topic.replace(" ", "_")[:50]
            
            # Use output manager's directory structure
            output_dir = Path(self.output_manager.output_dir) / safe_topic
            output_dir.mkdir(parents=True, exist_ok=True)
            
            saved_files = {}
            
            # Save raw script
            raw_script_path = output_dir / "script_raw.txt"
            raw_script_path.write_text(script_response.script, encoding="utf-8")
            saved_files["script_raw"] = str(raw_script_path)
            
            # Save formatted (SSML) script
            formatted_script_path = output_dir / "script_formatted.txt"
            formatted_script_path.write_text(formatted_script, encoding="utf-8")
            saved_files["script"] = str(formatted_script_path)
            
            # Save metadata
            metadata = {
                "topic": workflow.topic,
                "word_count": script_response.word_count,
                "language": script_response.language,
                "has_ssml_tags": True,
                "generated_at": workflow.started_at.isoformat() if workflow.started_at else None,
                "workflow_data": workflow.data.get("research_summary", {}),
            }
            metadata_path = output_dir / "metadata.json"
            import json
            metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
            saved_files["metadata"] = str(metadata_path)
            
            workflow.data["saved_files"] = saved_files
            logger.info(f"[FILE] Script saved to: {output_dir}")
            
            return saved_files
            
        except Exception as e:
            logger.warning(f"Failed to save script: {str(e)}")
            workflow.add_error("save_script", e)
            return {}
    
    async def _save_voice_only(
        self,
        workflow: WorkflowState,
        voice_response: VoiceResponse,
    ) -> Dict[str, str]:
        """
        Save the voice file to disk (after script is already saved).
        
        Args:
            workflow: The workflow state being processed.
            voice_response: The generated voice response.
            
        Returns:
            Dictionary with saved file path.
        """
        try:
            logger.debug("Saving voice output to disk")
            
            from pathlib import Path
            
            # Sanitize topic for filename
            safe_topic = "".join(c if c.isalnum() or c in " -_" else "_" for c in workflow.topic)
            safe_topic = safe_topic.replace(" ", "_")[:50]
            
            output_dir = Path(self.output_manager.output_dir) / safe_topic
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save voice file (decode from base64)
            import base64
            voice_ext = voice_response.format or "mp3"
            voice_path = output_dir / f"voice.{voice_ext}"
            audio_data = base64.b64decode(voice_response.audio_base64)
            voice_path.write_bytes(audio_data)
            
            # Update saved files
            if "saved_files" not in workflow.data:
                workflow.data["saved_files"] = {}
            workflow.data["saved_files"]["voice"] = str(voice_path)
            
            logger.info(f"[AUDIO] Voice saved to: {voice_path}")
            
            return {"voice": str(voice_path)}
            
        except Exception as e:
            logger.warning(f"Failed to save voice: {str(e)}")
            workflow.add_error("save_voice", e)
            return {}

    async def _assemble_video(
        self,
        workflow: WorkflowState,
        images_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Assemble the final video from images and audio.
        
        Args:
            workflow: The workflow state being processed.
            images_result: Result from image search containing downloaded images.
            
        Returns:
            Dictionary with video assembly results.
        """
        try:
            from pathlib import Path
            
            if not self.video_service:
                return {"success": False, "error": "Video service not available"}
            
            # Get voice file path
            saved_files = workflow.data.get("saved_files", {})
            voice_path = saved_files.get("voice", "")
            
            if not voice_path or not Path(voice_path).exists():
                return {"success": False, "error": "Voice audio file not found"}
            
            # Get images directory from the image search result
            images_dir = None
            image_list = []
            
            if images_result:
                # Collect all downloaded image paths
                sections = images_result.get("sections", [])
                for section in sections:
                    section_images = section.get("images", [])
                    for img in section_images:
                        filepath = img.get("filepath", "")
                        if filepath and Path(filepath).exists():
                            image_list.append(filepath)
                
                # Get the images output directory
                if image_list:
                    images_dir = str(Path(image_list[0]).parent)
            
            if not image_list:
                return {"success": False, "error": "No valid images found for video"}
            
            # Determine output video path
            output_dir = Path(voice_path).parent
            video_path = output_dir / "video.mp4"
            
            logger.info(f"Assembling video with {len(image_list)} images and audio from {voice_path}")
            
            # Get script text for subtitle generation
            script_text = workflow.data.get("formatted_script") or ""
            
            # Create the video
            result = await self.video_service.create_video_async(
                images_dir=images_dir or "",
                audio_path=voice_path,
                output_path=str(video_path),
                image_list=image_list,
                script_text=script_text if script_text else None,
            )
            
            # Update saved files
            if result.success:
                workflow.data["saved_files"]["video"] = result.video_path
            
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"Video assembly failed: {str(e)}")
            workflow.add_error("video_assembly", e)
            return {"success": False, "error": str(e)}

    async def _save_outputs(
        self,
        workflow: WorkflowState,
        script_response: ScriptResponse,
        formatted_script: str,
        voice_response: VoiceResponse,
    ) -> None:
        """
        Save all outputs to disk.
        
        Args:
            workflow: The workflow state being processed.
            script_response: The generated script response.
            formatted_script: The SSML-formatted script.
            voice_response: The generated voice response.
        """
        try:
            logger.debug("Saving outputs to disk")
            
            # Create a modified script response with formatted content
            output_response = ScriptResponse(
                script=formatted_script,
                word_count=script_response.word_count,
                language=script_response.language,
                has_ssml_tags=True,
                metadata=script_response.metadata,
            )
            
            saved_files = self.output_manager.save_content(
                topic=workflow.topic,
                script_response=output_response,
                voice_response=voice_response,
            )
            
            workflow.data["saved_files"] = saved_files
            logger.debug(f"Outputs saved to {saved_files.get('directory', 'output')}")
        
        except Exception as e:
            logger.warning(f"Failed to save outputs: {str(e)}")
            workflow.add_error("save_outputs", e)
    
    def get_workflow_state(self, topic: str) -> Optional[WorkflowState]:
        """
        Get the workflow state for a topic.
        
        Args:
            topic: Topic identifier.
        
        Returns:
            WorkflowState if it exists, None otherwise.
        """
        return self.workflows.get(topic)
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """
        List all workflows with their states.
        
        Returns:
            List of workflow state dictionaries.
        """
        return [workflow.to_dict() for workflow in self.workflows.values()]
    
    async def parallel_section_writing(
        self,
        topic: str,
        sections: List[str],
        tone: str = "informative",
        style: str = "conversational",
        language: str = "en",
    ) -> Dict[str, str]:
        """
        Write multiple script sections in parallel.
        
        This is useful for longer scripts where sections can be written independently
        and then combined.
        
        Args:
            topic: Main topic for context.
            sections: List of section titles/prompts to write.
            tone: Tone for all sections.
            style: Style for all sections.
            language: Language code for all sections.
        
        Returns:
            Dictionary mapping section title to generated content.
        """
        logger.info(f"Starting parallel section writing for {len(sections)} sections")
        
        tasks = []
        for section in sections:
            request = ScriptRequest(
                topic=f"{topic} - {section}",
                tone=tone,
                length="short",  # Each section is shorter
                style=style,
                language=language,
                include_ssml_tags=False,
            )
            tasks.append(self._write_section(section, request))
        
        # Execute all section writing tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        section_content = {}
        for section, result in zip(sections, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to write section '{section}': {str(result)}")
                section_content[section] = ""
            else:
                section_content[section] = result.script
        
        logger.info(f"Parallel section writing completed for {len(section_content)} sections")
        return section_content
    
    async def _write_section(self, section_title: str, request: ScriptRequest) -> ScriptResponse:
        """
        Write a single section (helper for parallel writing).
        
        Args:
            section_title: Title of the section being written.
            request: Script generation request.
        
        Returns:
            Generated script response for the section.
        """
        logger.debug(f"Writing section: {section_title}")
        try:
            response = await asyncio.wait_for(
                self.script_service.generate_script(request),
                timeout=120,
            )
            logger.debug(f"Section '{section_title}' completed: {response.word_count} words")
            return response
        except Exception as e:
            logger.error(f"Error writing section '{section_title}': {str(e)}")
            raise
