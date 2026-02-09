# 🎬 YouTube Video Automation

An AI-powered pipeline for automated YouTube video generation. This system researches topics, writes scripts, generates voiceovers, finds relevant images, and assembles complete videos with Ken Burns effects and subtitles.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- **🔍 Automated Research** - Gathers information from Wikipedia, Reddit, Hacker News, arXiv, GitHub, and more
- **✍️ AI Script Writing** - Generates engaging scripts using GPT-4 or Claude
- **🎙️ Voice Generation** - Creates natural voiceovers with ElevenLabs
- **🖼️ Image Search** - Finds relevant, high-quality images from multiple sources
- **🎬 Video Assembly** - Combines everything with smooth Ken Burns effects
- **📝 Subtitles** - Automatically generates styled subtitles
- **📊 Batch Processing** - Process multiple topics from a CSV queue

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Research Agent │────▶│ Script Service  │────▶│  Voice Service  │
│  (Web Search)   │     │ (GPT-4/Claude)  │     │  (ElevenLabs)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐              │
│   Image Agent   │────▶│  Video Service  │◀─────────────┘
│  (Bing/Pexels)  │     │    (FFmpeg)     │
└─────────────────┘     └─────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- FFmpeg (for video processing)
- API keys for OpenAI and ElevenLabs

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/youtube-video-automation.git
   cd youtube-video-automation
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install FFmpeg**
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - **Mac**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`

5. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your API keys:
   ```env
   OPENAI_API_KEY=sk-your-openai-key
   ELEVENLABS_API_KEY=your-elevenlabs-key
   PEXELS_API_KEY=your-pexels-key  # Optional
   PIXABAY_API_KEY=your-pixabay-key  # Optional
   ```

## 🚀 Usage

### Single Video Generation

```python
import asyncio
from src.agents.orchestrator import ContentOrchestrator

async def main():
    orchestrator = ContentOrchestrator(...)
    result = await orchestrator.generate_content(
        topic="How Quantum Computing Works",
        tone="informative",
        style="conversational",
        target_duration_minutes=10,
        create_video=True,
    )
    print(f"Video created: {result['video_path']}")

asyncio.run(main())
```

### Batch Processing

1. **Create a topics file**
   ```bash
   cp topics.csv.example topics.csv
   ```

2. **Edit topics.csv** with your desired topics

3. **Check queue status**
   ```bash
   python batch_generate.py --status
   ```

4. **Process all pending topics**
   ```bash
   python batch_generate.py --all
   ```

5. **Process a specific topic**
   ```bash
   python batch_generate.py --topic "How Quantum Computing Works"
   ```

### CLI Options

```
Usage: python batch_generate.py [OPTIONS]

Options:
  --csv PATH          Path to topics CSV file (default: topics.csv)
  --all               Process all pending topics
  --topic TEXT        Process a specific topic by name
  --create-template   Create example topics.csv file
  --status            Show queue status
  --reset-failed      Reset failed topics to pending
  --dry-run           Show what would be processed without running
```

## 📁 Project Structure

```
youtube-video-automation/
├── src/
│   ├── agents/
│   │   ├── orchestrator.py      # Main workflow coordinator
│   │   ├── research_agent.py    # Web research agent
│   │   └── image_agent.py       # Image search agent
│   ├── services/
│   │   ├── video_service.py     # Video assembly with FFmpeg
│   │   ├── voice_service.py     # Voice generation orchestration
│   │   ├── script_service.py    # Script generation orchestration
│   │   └── prompt_service.py    # Prompt template management
│   ├── providers/
│   │   ├── scripts/
│   │   │   ├── openai_provider.py     # GPT-4 integration
│   │   │   └── anthropic_provider.py  # Claude integration
│   │   └── voice/
│   │       └── elevenlabs_provider.py # ElevenLabs integration
│   ├── models/
│   │   ├── schemas.py           # Data models and types
│   │   └── base_provider.py     # Provider interfaces
│   └── utils/
│       ├── logging_config.py    # Centralized logging
│       ├── topic_queue.py       # CSV queue management
│       ├── output_manager.py    # Output directory handling
│       └── retry.py             # Retry decorators
├── prompts/
│   ├── outline.txt              # Script outline template
│   ├── script_generation.txt    # Full script template
│   └── ssml_enhancement.txt     # SSML markup template
├── config/
│   └── settings.json.example    # Configuration template
├── batch_generate.py            # Main CLI script
├── requirements.txt             # Python dependencies
└── README.md
```

## ⚙️ Configuration

### Video Settings

```python
from src.services.video_service import VideoConfig

config = VideoConfig()
config.output_resolution = (1920, 1080)  # Full HD
config.fps = 24                           # Frame rate
config.subtitle_font_size = 28            # Subtitle size
config.subtitle_bg_opacity = 0.6          # Background opacity
```

### Topics CSV Format

| Column | Description | Default |
|--------|-------------|---------|
| `topic` | Video topic/title | Required |
| `target_duration_minutes` | Target video length | 10.0 |
| `status` | pending/completed/failed | pending |
| `priority` | Processing order (lower = first) | 0 |
| `tone` | informative/educational/entertaining | informative |
| `style` | conversational/formal/casual | conversational |
| `notes` | Additional instructions | empty |

## 🎨 Ken Burns Effects

The video service applies smooth Ken Burns effects (zoom/pan animations) to static images:

- **Zoom In** - Gradual zoom from wide to close
- **Zoom Out** - Start close, zoom out to reveal
- **Pan Left/Right** - Horizontal movement across image
- **Pan Up/Down** - Vertical movement across image

Effects are randomly distributed for visual variety, with 3x upscaling for smooth motion.

## 🔧 Extending

### Adding a New Script Provider

```python
from src.models.script_provider import IScriptProvider

class MyProvider(IScriptProvider):
    async def generate_script(self, topic, options):
        # Your implementation
        pass
```

### Adding a New Voice Provider

```python
from src.models.voice_provider import IVoiceProvider

class MyVoiceProvider(IVoiceProvider):
    async def generate_voice(self, text, options):
        # Your implementation
        pass
```

## 📊 Output

Generated videos are saved to `output/<topic_name>/`:

```
output/
└── How_Quantum_Computing_Works/
    ├── video.mp4          # Final video with audio and subtitles
    ├── voice.mp3          # Generated voiceover
    ├── script_raw.txt     # Plain text script
    ├── script_formatted.txt # Formatted script
    ├── metadata.json      # Generation metadata
    └── images/            # Downloaded images
        ├── part_1_01.jpg
        ├── part_1_02.jpg
        └── ...
```

## 🐛 Troubleshooting

### Common Issues

**FFmpeg not found**
- Ensure FFmpeg is installed and in your PATH
- Test with: `ffmpeg -version`

**API rate limits**
- The system includes automatic retry with exponential backoff
- Consider adding delays between batch items for large queues

**Memory issues with long videos**
- Videos are processed in segments to manage memory
- For very long videos (30+ min), consider splitting into parts

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [OpenAI](https://openai.com/) for GPT-4
- [ElevenLabs](https://elevenlabs.io/) for voice synthesis
- [FFmpeg](https://ffmpeg.org/) for video processing
