"""Video assembly service for creating videos from images and audio.

This module provides functionality to combine images and audio into videos
with Ken Burns effects (zoom/pan animations), subtitles, and various
output formats using FFmpeg.
"""

import os
import sys
import signal
import logging
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Windows-specific flags for subprocess to avoid Ctrl+C interrupts
SUBPROCESS_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0


@contextmanager
def ignore_sigint():
    """Context manager to temporarily ignore SIGINT (Ctrl+C) during subprocess calls."""
    if sys.platform == 'win32':
        original_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, original_handler)
    else:
        yield

try:
    from moviepy import (
        ImageClip,
        AudioFileClip,
        concatenate_videoclips,
        CompositeVideoClip,
    )
    MOVIEPY_AVAILABLE = True
except ImportError:
    # Fall back to old import style for moviepy 1.x
    try:
        from moviepy.editor import (
            ImageClip,
            AudioFileClip,
            concatenate_videoclips,
            CompositeVideoClip,
        )
        MOVIEPY_AVAILABLE = True
    except ImportError:
        MOVIEPY_AVAILABLE = False

# Check for ffmpeg availability
FFMPEG_AVAILABLE = shutil.which('ffmpeg') is not None

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    """Configuration for video generation."""
    # Output settings
    output_resolution: Tuple[int, int] = (1920, 1080)  # Full HD
    fps: int = 24
    codec: str = "libx264"
    audio_codec: str = "aac"
    bitrate: str = "8000k"
    
    # Transition settings
    transition_duration: float = 0.5  # seconds for crossfade
    
    # Image display settings
    min_image_duration: float = 3.0  # minimum seconds per image
    max_image_duration: float = 8.0  # maximum seconds per image
    
    # Ken Burns effect (subtle zoom/pan)
    enable_ken_burns: bool = True  # Enable by default
    ken_burns_zoom_ratio: float = 1.08  # Subtle 8% zoom
    
    # Text overlay settings
    enable_subtitles: bool = True
    subtitle_font_size: int = 28
    subtitle_font: str = "Arial"
    subtitle_color: str = "white"
    subtitle_outline_color: str = "black"
    subtitle_outline_width: int = 2
    subtitle_position: str = "bottom"  # "bottom", "center", "top"
    subtitle_margin: int = 40  # Margin from edge
    subtitle_bg_opacity: float = 0.6  # Background opacity (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_resolution": self.output_resolution,
            "fps": self.fps,
            "codec": self.codec,
            "audio_codec": self.audio_codec,
            "bitrate": self.bitrate,
            "transition_duration": self.transition_duration,
            "min_image_duration": self.min_image_duration,
            "max_image_duration": self.max_image_duration,
            "enable_ken_burns": self.enable_ken_burns,
            "ken_burns_zoom_ratio": self.ken_burns_zoom_ratio,
            "enable_subtitles": self.enable_subtitles,
            "subtitle_font_size": self.subtitle_font_size,
        }


@dataclass
class VideoResult:
    """Result of video generation."""
    success: bool
    video_path: Optional[str] = None
    duration_seconds: float = 0.0
    num_images_used: int = 0
    resolution: Tuple[int, int] = (1920, 1080)
    file_size_bytes: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "video_path": self.video_path,
            "duration_seconds": self.duration_seconds,
            "num_images_used": self.num_images_used,
            "resolution": self.resolution,
            "file_size_bytes": self.file_size_bytes,
            "error": self.error,
        }


class VideoService:
    """Service for assembling videos from images and audio."""
    
    def __init__(self, config: Optional[VideoConfig] = None):
        """
        Initialize the video service.
        
        Args:
            config: Video configuration. Uses defaults if not provided.
        """
        # MoviePy is optional since we use ffmpeg for most operations
        self.config = config or VideoConfig()
        logger.info(f"VideoService initialized with resolution {self.config.output_resolution}")
    
    def get_image_files(self, images_dir: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.webp')) -> List[str]:
        """
        Get all image files from a directory.
        
        Args:
            images_dir: Path to the images directory.
            extensions: Tuple of valid image file extensions.
        
        Returns:
            List of image file paths, sorted alphabetically.
        """
        images_path = Path(images_dir)
        if not images_path.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            return []
        
        image_files = []
        for ext in extensions:
            image_files.extend(images_path.glob(f"*{ext}"))
            image_files.extend(images_path.glob(f"*{ext.upper()}"))
        
        # Sort by filename for consistent ordering
        image_files = sorted(set(image_files), key=lambda x: x.name)
        
        logger.info(f"Found {len(image_files)} images in {images_dir}")
        return [str(f) for f in image_files]
    
    def validate_image(self, image_path: str) -> bool:
        """
        Validate that an image file can be opened and read.
        
        Args:
            image_path: Path to the image file.
        
        Returns:
            True if image is valid, False otherwise.
        """
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception as e:
            logger.warning(f"Invalid image {image_path}: {e}")
            return False
    
    def calculate_image_durations(
        self,
        num_images: int,
        total_duration: float,
    ) -> List[float]:
        """
        Calculate how long each image should be displayed.
        
        Args:
            num_images: Number of images to display.
            total_duration: Total video duration in seconds.
        
        Returns:
            List of durations for each image.
        """
        if num_images == 0:
            return []
        
        # Calculate ideal duration per image
        ideal_duration = total_duration / num_images
        
        # Clamp to min/max bounds
        duration = max(
            self.config.min_image_duration,
            min(self.config.max_image_duration, ideal_duration)
        )
        
        # If we have too few images, just distribute evenly
        if ideal_duration > self.config.max_image_duration:
            # Repeat images to fill the time
            passes = int(total_duration / (num_images * self.config.max_image_duration)) + 1
            duration = total_duration / (num_images * passes)
            num_images = num_images * passes
        
        durations = [duration] * num_images
        
        # Adjust last duration to exactly match total
        total_assigned = sum(durations[:-1]) if len(durations) > 1 else 0
        if durations:
            durations[-1] = total_duration - total_assigned
        
        return durations
    
    def prepare_image_clip(
        self,
        image_path: str,
        duration: float,
        target_size: Tuple[int, int],
    ) -> Optional[Any]:
        """
        Create an image clip with proper sizing.
        
        Args:
            image_path: Path to the image file.
            duration: Duration to display the image.
            target_size: Target resolution (width, height).
        
        Returns:
            ImageClip object or None if failed.
        """
        try:
            # Open and resize image to fit target resolution
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Calculate scaling to cover the target size (crop excess)
                img_ratio = img.width / img.height
                target_ratio = target_size[0] / target_size[1]
                
                if img_ratio > target_ratio:
                    # Image is wider - scale by height, crop width
                    new_height = target_size[1]
                    new_width = int(new_height * img_ratio)
                else:
                    # Image is taller - scale by width, crop height
                    new_width = target_size[0]
                    new_height = int(new_width / img_ratio)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Center crop to target size
                left = (new_width - target_size[0]) // 2
                top = (new_height - target_size[1]) // 2
                img = img.crop((left, top, left + target_size[0], top + target_size[1]))
                
                # Save to temp file for moviepy
                import tempfile
                temp_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
                img.save(temp_path, 'JPEG', quality=95)
            
            # Create the clip - moviepy 2.x uses duration in constructor or with_duration()
            clip = ImageClip(temp_path, duration=duration)
            
            # Clean up temp file reference (moviepy will handle it)
            clip._temp_path = temp_path
            
            return clip
            
        except Exception as e:
            logger.error(f"Failed to prepare image clip for {image_path}: {e}")
            return None
    
    def create_video(
        self,
        images_dir: str,
        audio_path: str,
        output_path: str,
        image_list: Optional[List[str]] = None,
        use_ffmpeg: bool = True,
        script_text: Optional[str] = None,
    ) -> VideoResult:
        """
        Create a video from images and audio.
        
        Args:
            images_dir: Directory containing images (ignored if image_list provided).
            audio_path: Path to the audio file (MP3, WAV, etc.).
            output_path: Path for the output video file.
            image_list: Optional explicit list of image paths to use.
            use_ffmpeg: Use ffmpeg directly for much faster rendering (default True).
            script_text: Optional script text for subtitle generation.
        
        Returns:
            VideoResult with details about the generated video.
        """
        logger.info(f"Starting video creation: {output_path}")
        
        try:
            # Validate audio file
            if not os.path.exists(audio_path):
                return VideoResult(
                    success=False,
                    error=f"Audio file not found: {audio_path}"
                )
            
            # Get audio duration using ffprobe (faster than loading with moviepy)
            total_duration = self._get_audio_duration(audio_path)
            if total_duration <= 0:
                return VideoResult(
                    success=False,
                    error="Could not determine audio duration"
                )
            logger.info(f"Audio duration: {total_duration:.2f} seconds")
            
            # Get image files
            if image_list:
                image_files = [f for f in image_list if os.path.exists(f)]
            else:
                image_files = self.get_image_files(images_dir)
            
            # Validate images
            valid_images = [f for f in image_files if self.validate_image(f)]
            
            if not valid_images:
                return VideoResult(
                    success=False,
                    error="No valid images found for video creation"
                )
            
            logger.info(f"Using {len(valid_images)} valid images")
            
            # Calculate durations for each image
            durations = self.calculate_image_durations(len(valid_images), total_duration)
            
            # Repeat images if we need more to cover the audio
            while len(durations) > len(valid_images):
                valid_images = valid_images * 2
            valid_images = valid_images[:len(durations)]
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            
            # Use ffmpeg for much faster rendering
            if use_ffmpeg:
                return self._create_video_ffmpeg(
                    valid_images, durations, audio_path, output_path, total_duration, script_text
                )
            
            # Fallback to moviepy (slower but more features)
            return self._create_video_moviepy(
                valid_images, durations, audio_path, output_path, total_duration
            )
            
        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            return VideoResult(
                success=False,
                error=str(e)
            )
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration using ffprobe (fast)."""
        import subprocess
        try:
            result = subprocess.run(
                [
                    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
                ],
                capture_output=True, text=True, timeout=10
            )
            return float(result.stdout.strip())
        except Exception:
            # Fallback to moviepy
            try:
                audio = AudioFileClip(audio_path)
                duration = audio.duration
                audio.close()
                return duration
            except:
                return 0.0
    
    def _create_video_ffmpeg(
        self,
        images: List[str],
        durations: List[float],
        audio_path: str,
        output_path: str,
        total_duration: float,
        script_text: Optional[str] = None,
    ) -> VideoResult:
        """
        Create video using ffmpeg with Ken Burns effects and optional subtitles.
        
        Uses zoompan filter for smooth camera movements on each image.
        """
        import subprocess
        import tempfile
        import random
        
        logger.info("Using ffmpeg for video rendering with Ken Burns effects...")
        
        temp_dir = tempfile.mkdtemp(prefix="ffmpeg_video_")
        
        try:
            width, height = self.config.output_resolution
            fps = self.config.fps
            
            # Prepare images with extra size for Ken Burns movement
            logger.info(f"Preparing {len(images)} images with Ken Burns padding...")
            prepared_images = []
            ken_burns_effects = []
            
            # Zoom ratio for Ken Burns (e.g., 1.08 means 8% zoom range)
            zoom_ratio = self.config.ken_burns_zoom_ratio if self.config.enable_ken_burns else 1.0
            
            for i, (image_path, duration) in enumerate(zip(images, durations)):
                prepared_path = os.path.join(temp_dir, f"img_{i:04d}.jpg")
                
                # Prepare image slightly larger for Ken Burns movement room
                self._prepare_image_for_ken_burns(image_path, prepared_path, zoom_ratio)
                prepared_images.append((prepared_path, duration))
                
                # Generate random Ken Burns effect for this image
                if self.config.enable_ken_burns:
                    effect = self._generate_ken_burns_effect(duration, fps, zoom_ratio)
                else:
                    effect = {'zoom': 'zoom', 'x': f'(iw-{width})/2', 'y': f'(ih-{height})/2'}
                ken_burns_effects.append(effect)
            
            # Generate subtitle file if script provided
            subtitle_file = None
            if script_text and self.config.enable_subtitles:
                subtitle_file = os.path.join(temp_dir, "subtitles.srt")
                self._generate_subtitles(script_text, total_duration, subtitle_file)
                logger.info("Generated subtitle file")
            
            # Create video segments with smooth Ken Burns effect
            segment_files = []
            
            # Define Ken Burns effect types for variety
            # Each effect uses 3x upscale and smooth cumulative expressions
            effect_types = ['zoom_in', 'zoom_out', 'pan_left', 'pan_right', 'pan_up', 'pan_down']
            
            logger.info("Creating video segments with Ken Burns effects...")
            for i, (img_path, duration) in enumerate(prepared_images):
                segment_path = os.path.join(temp_dir, f"segment_{i:04d}.mp4")
                frames = int(duration * fps)
                
                # Cycle through effect types for variety
                effect_type = effect_types[i % len(effect_types)]
                zoom_amount = 0.08
                
                # Build zoompan expressions based on effect type
                if effect_type == 'zoom_in':
                    # Zoom in from 1.0 to 1.08, centered
                    z_expr = f"min(zoom+{zoom_amount}/{frames},1+{zoom_amount})"
                    x_expr = "iw/2-(iw/zoom/2)"
                    y_expr = "ih/2-(ih/zoom/2)"
                elif effect_type == 'zoom_out':
                    # Zoom out from 1.08 to 1.0, centered
                    z_expr = f"if(eq(on,0),1+{zoom_amount},max(zoom-{zoom_amount}/{frames},1.0))"
                    x_expr = "iw/2-(iw/zoom/2)"
                    y_expr = "ih/2-(ih/zoom/2)"
                elif effect_type == 'pan_right':
                    # Pan from left to right, constant zoom at 1.08
                    z_expr = f"1+{zoom_amount}"
                    x_expr = f"(iw-iw/zoom)*on/{frames}"
                    y_expr = "ih/2-(ih/zoom/2)"
                elif effect_type == 'pan_left':
                    # Pan from right to left, constant zoom at 1.08
                    z_expr = f"1+{zoom_amount}"
                    x_expr = f"(iw-iw/zoom)*(1-on/{frames})"
                    y_expr = "ih/2-(ih/zoom/2)"
                elif effect_type == 'pan_down':
                    # Pan from top to bottom, constant zoom at 1.08
                    z_expr = f"1+{zoom_amount}"
                    x_expr = "iw/2-(iw/zoom/2)"
                    y_expr = f"(ih-ih/zoom)*on/{frames}"
                elif effect_type == 'pan_up':
                    # Pan from bottom to top, constant zoom at 1.08
                    z_expr = f"1+{zoom_amount}"
                    x_expr = "iw/2-(iw/zoom/2)"
                    y_expr = f"(ih-ih/zoom)*(1-on/{frames})"
                
                # Upscale to 3x for smooth interpolation - THIS IS REQUIRED for smooth motion
                vf = (
                    f"scale=3*iw:3*ih,"
                    f"zoompan=z='{z_expr}':"
                    f"x='{x_expr}':"
                    f"y='{y_expr}':"
                    f"d={frames}:s={width}x{height}:fps={fps}"
                )
                
                logger.debug(f"Segment {i}: {effect_type}")
                
                cmd = [
                    'ffmpeg', '-y',
                    '-i', img_path,
                    '-vf', vf,
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    segment_path
                ]
                
                # Long timeout - 3x upscale is slow but smooth
                # Use DEVNULL for stdout/stderr and CREATE_NEW_PROCESS_GROUP to avoid Windows Ctrl+C interrupts
                result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800, creationflags=SUBPROCESS_FLAGS)
                if result.returncode != 0:
                    logger.warning(f"Segment {i} failed (exit code: {result.returncode})")
                
                segment_files.append(segment_path)
            
            # Simple concat (no xfade needed - Ken Burns provides the motion)
            logger.info("Concatenating segments...")
            video_no_audio = os.path.join(temp_dir, "video_no_audio.mp4")
            
            concat_file = os.path.join(temp_dir, "segments.txt")
            with open(concat_file, 'w') as f:
                for seg_path in segment_files:
                    escaped = seg_path.replace('\\', '/').replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")
            
            cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_file, '-c', 'copy', video_no_audio]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120, creationflags=SUBPROCESS_FLAGS)
            
            # Add audio and optionally subtitles
            logger.info("Adding audio" + (" and subtitles..." if subtitle_file else "..."))
            
            if subtitle_file and os.path.exists(subtitle_file):
                # Use subtitles filter with styling
                # BorderStyle=4 = transparent box behind text, BackColour sets box color with alpha
                sub_escaped = subtitle_file.replace('\\', '/').replace(':', '\\:')
                bg_alpha = int((1 - self.config.subtitle_bg_opacity) * 255)  # Convert to ASS alpha (0=opaque, 255=transparent)
                bg_color = f"&H{bg_alpha:02X}000000"  # Semi-transparent black
                subtitle_filter = (
                    f"subtitles='{sub_escaped}':"
                    f"force_style='FontSize={self.config.subtitle_font_size},"
                    f"FontName={self.config.subtitle_font},"
                    f"PrimaryColour=&HFFFFFF,"
                    f"OutlineColour=&H000000,"
                    f"BackColour={bg_color},"
                    f"BorderStyle=4,"
                    f"Outline={self.config.subtitle_outline_width},"
                    f"Shadow=0,"
                    f"MarginV={self.config.subtitle_margin}'"
                )
                
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_no_audio,
                    '-i', audio_path,
                    '-vf', subtitle_filter,
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-shortest',
                    '-movflags', '+faststart',
                    output_path
                ]
            else:
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_no_audio,
                    '-i', audio_path,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-shortest',
                    '-movflags', '+faststart',
                    output_path
                ]
            
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300, creationflags=SUBPROCESS_FLAGS)
            
            if result.returncode != 0:
                logger.error(f"Final ffmpeg failed with exit code: {result.returncode}")
                return VideoResult(
                    success=False,
                    error=f"ffmpeg failed with exit code: {result.returncode}"
                )
            
            file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            logger.info(f"[OK] Video created successfully: {output_path}")
            
            return VideoResult(
                success=True,
                video_path=output_path,
                duration_seconds=total_duration,
                num_images_used=len(images),
                resolution=self.config.output_resolution,
                file_size_bytes=file_size,
            )
            
        finally:
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
    
    def _prepare_image_for_ken_burns(self, input_path: str, output_path: str, zoom_ratio: float) -> None:
        """Prepare image with extra padding for Ken Burns movement."""
        with Image.open(input_path) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            target_w, target_h = self.config.output_resolution
            
            # Add extra size for Ken Burns movement room (e.g., 1.08 = 8% extra)
            padded_w = int(target_w * zoom_ratio)
            padded_h = int(target_h * zoom_ratio)
            
            # Scale to cover padded size
            img_ratio = img.width / img.height
            target_ratio = padded_w / padded_h
            
            if img_ratio > target_ratio:
                new_height = padded_h
                new_width = int(new_height * img_ratio)
            else:
                new_width = padded_w
                new_height = int(new_width / img_ratio)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Center crop to padded size
            left = (new_width - padded_w) // 2
            top = (new_height - padded_h) // 2
            img = img.crop((left, top, left + padded_w, top + padded_h))
            
            img.save(output_path, 'JPEG', quality=92)
    
    def _generate_ken_burns_effect(self, duration: float, fps: int, zoom_ratio: float) -> Dict[str, str]:
        """
        Generate smooth Ken Burns effect with NO jitter.
        
        Uses only a simple, stable linear zoom centered on the image.
        No pan movements or random variations to avoid micro-jitter.
        """
        frames = int(duration * fps)
        
        # Very gentle zoom - just 5% over the entire clip duration
        # This is smooth and visible but not dramatic
        zoom_start = 1.0
        zoom_end = 1.05  # Only 5% zoom, very subtle
        
        # Linear zoom expression - simple and smooth
        # At frame 0: zoom=1.0, at frame N: zoom=1.05
        zoom_expr = f"1+0.05*on/{frames}"
        
        # Keep the image perfectly centered - use FIXED center point
        # iw/2 = center X, ih/2 = center Y (these don't change)
        # The zoom happens around this center point
        x_expr = "(iw-iw/zoom)/2"
        y_expr = "(ih-ih/zoom)/2"
        
        return {
            'zoom': zoom_expr,
            'x': x_expr,
            'y': y_expr,
        }
    
    def _generate_subtitles(self, script_text: str, total_duration: float, output_path: str) -> None:
        """
        Generate SRT subtitle file from script text.
        
        Distributes time proportionally based on sentence length (word count).
        Longer sentences get more screen time to match speech pacing.
        """
        import re
        
        # Clean script - remove SSML tags and extra whitespace
        clean_text = re.sub(r'<[^>]+>', '', script_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        
        if not sentences:
            return
        
        # Calculate word counts for each sentence
        word_counts = [len(s.split()) for s in sentences]
        total_words = sum(word_counts)
        
        # Distribute time proportionally: sentences with more words get more time
        durations = [(count / total_words) * total_duration for count in word_counts]
        
        # Generate SRT content
        srt_lines = []
        current_time = 0.0
        
        for i, (sentence, duration) in enumerate(zip(sentences, durations)):
            start_time = current_time
            end_time = current_time + duration
            current_time = end_time
            
            # Cap at total duration
            end_time = min(end_time, total_duration)
            
            # Format times as SRT format: HH:MM:SS,mmm
            start_str = self._format_srt_time(start_time)
            end_str = self._format_srt_time(end_time)
            
            # Word wrap long sentences (max ~80 chars per line for 2-3 lines)
            wrapped = self._wrap_subtitle_text(sentence, max_chars=80)
            
            srt_lines.append(f"{i+1}")
            srt_lines.append(f"{start_str} --> {end_str}")
            srt_lines.append(wrapped)
            srt_lines.append("")
            
            if end_time >= total_duration:
                break
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_lines))
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _wrap_subtitle_text(self, text: str, max_chars: int = 60) -> str:
        """Wrap subtitle text to fit on screen."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_chars:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Return all lines (no truncation) - let ffmpeg handle display
        return '\n'.join(lines)
    
    def _prepare_image_for_ffmpeg(self, input_path: str, output_path: str) -> None:
        """Resize and prepare an image for ffmpeg."""
        with Image.open(input_path) as img:
            # Convert to RGB
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            target_size = self.config.output_resolution
            
            # Calculate scaling to cover the target size
            img_ratio = img.width / img.height
            target_ratio = target_size[0] / target_size[1]
            
            if img_ratio > target_ratio:
                new_height = target_size[1]
                new_width = int(new_height * img_ratio)
            else:
                new_width = target_size[0]
                new_height = int(new_width / img_ratio)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Center crop
            left = (new_width - target_size[0]) // 2
            top = (new_height - target_size[1]) // 2
            img = img.crop((left, top, left + target_size[0], top + target_size[1]))
            
            img.save(output_path, 'JPEG', quality=90)
    
    def _create_video_moviepy(
        self,
        images: List[str],
        durations: List[float],
        audio_path: str,
        output_path: str,
        total_duration: float,
    ) -> VideoResult:
        """Create video using moviepy (slower but more features)."""
        logger.info("Using moviepy for video rendering...")
        
        audio_clip = AudioFileClip(audio_path)
        clips = []
        temp_files = []
        
        try:
            for i, (image_path, duration) in enumerate(zip(images, durations)):
                logger.debug(f"Processing image {i+1}/{len(images)}: {Path(image_path).name}")
                
                clip = self.prepare_image_clip(
                    image_path,
                    duration,
                    self.config.output_resolution
                )
                
                if clip:
                    clips.append(clip)
                    if hasattr(clip, '_temp_path'):
                        temp_files.append(clip._temp_path)
            
            if not clips:
                audio_clip.close()
                return VideoResult(
                    success=False,
                    error="Failed to create any image clips"
                )
            
            logger.info(f"Concatenating {len(clips)} clips...")
            video = concatenate_videoclips(clips, method="compose")
            video = video.with_audio(audio_clip)
            
            logger.info(f"Writing video to {output_path}...")
            video.write_videofile(
                output_path,
                fps=self.config.fps,
                codec=self.config.codec,
                audio_codec=self.config.audio_codec,
                bitrate=self.config.bitrate,
                logger=None,
            )
            
            file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            
            video.close()
            audio_clip.close()
            for clip in clips:
                clip.close()
            
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
            logger.info(f"[OK] Video created successfully: {output_path}")
            
            return VideoResult(
                success=True,
                video_path=output_path,
                duration_seconds=total_duration,
                num_images_used=len(images),
                resolution=self.config.output_resolution,
                file_size_bytes=file_size,
            )
            
        except Exception as e:
            audio_clip.close()
            for clip in clips:
                try:
                    clip.close()
                except:
                    pass
            raise
    
    async def create_video_async(
        self,
        images_dir: str,
        audio_path: str,
        output_path: str,
        image_list: Optional[List[str]] = None,
        script_text: Optional[str] = None,
    ) -> VideoResult:
        """
        Async wrapper for create_video.
        
        Args:
            images_dir: Directory containing images.
            audio_path: Path to the audio file.
            output_path: Path for the output video file.
            image_list: Optional explicit list of image paths.
            script_text: Optional script text for subtitle generation.
        
        Returns:
            VideoResult with details about the generated video.
        """
        import asyncio
        
        # Run in executor since moviepy is blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.create_video(images_dir, audio_path, output_path, image_list, script_text=script_text)
        )


def check_moviepy_available() -> bool:
    """Check if moviepy or ffmpeg is available for video creation."""
    return MOVIEPY_AVAILABLE or FFMPEG_AVAILABLE


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available for video creation."""
    return FFMPEG_AVAILABLE


if __name__ == "__main__":
    # Quick test
    if not MOVIEPY_AVAILABLE:
        print("[FAIL] moviepy is not installed. Run: pip install moviepy")
    else:
        print("[OK] moviepy is available")
        
        service = VideoService()
        print(f"Default config: {service.config.to_dict()}")
