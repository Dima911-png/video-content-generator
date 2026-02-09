"""Run batch pipeline for first N pending topics (detached from terminal)."""
import asyncio
import signal
import sys
import os
import logging
from pathlib import Path

# On Windows, ignore CTRL+C so the process doesn't get interrupted
# by terminal operations in other VS Code terminal tabs
if sys.platform == 'win32':
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# Force UTF-8 for stdout/stderr to avoid cp1252 encoding errors on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Set up file logging so we can monitor progress
LOG_FILE = Path(__file__).parent / "output" / "pipeline.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(str(LOG_FILE), mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.root.addHandler(file_handler)
logging.root.setLevel(logging.INFO)

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logging_config import setup_logging

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_TOPICS = 3  # Only process first 3 pending topics


async def main():
    from dotenv import load_dotenv
    load_dotenv()

    from src.utils.topic_queue import TopicQueue
    from batch_generate import create_orchestrator, load_api_keys, process_topic

    api_keys = load_api_keys()
    orchestrator = create_orchestrator(api_keys)
    queue = TopicQueue(csv_path="topics.csv")

    pending = queue.get_all_pending()[:MAX_TOPICS]
    print(f"Processing {len(pending)} topics: {[t.topic for t in pending]}")

    successful, failed = 0, 0
    for entry in pending:
        try:
            ok = await process_topic(orchestrator, entry, queue)
            if ok:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Error: {entry.topic}: {e}")
            failed += 1

    print(f"\nDone! Successful: {successful}, Failed: {failed}")


if __name__ == "__main__":
    asyncio.run(main())
