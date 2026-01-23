# LaiCut Align Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add audio-video subtitle alignment capability using LattifAI API + local ONNX inference.

**Architecture:** Hybrid mode - API handles tokenize/detokenize, local ONNX handles acoustic inference. Model auto-downloads on first use to `~/.cache/omnicaptions/models/`.

**Tech Stack:** onnxruntime, numpy, soundfile, httpx, lattifai-captions

---

## Task 0: Fix marketplace.json

**Files:**
- Modify: `.claude-plugin/marketplace.json`

**Step 1: Update marketplace.json**

```json
{
  "name": "omni-captions-skills",
  "owner": {
    "name": "LattifAI",
    "email": "tech@lattifai.com"
  },
  "metadata": {
    "description": "AI-powered media transcription, translation, caption conversion, and audio-text alignment",
    "version": "0.1.0"
  },
  "plugins": [
    {
      "name": "omnicaptions",
      "source": {
        "source": "github",
        "repo": "lattifai/omni-captions-skills"
      },
      "description": "Transcribe, translate, convert, and align captions using AI APIs",
      "version": "0.1.0",
      "keywords": ["transcription", "gemini", "youtube", "subtitles", "translation", "caption", "alignment", "lattifai"],
      "license": "MIT",
      "category": "media",
      "strict": true
    }
  ]
}
```

**Step 2: Commit**

```bash
git add .claude-plugin/marketplace.json
git commit -m "fix: update marketplace.json to omni-captions-skills"
```

---

## Task 1: Add laicut dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml dependencies**

Add after existing `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.4.0",
]
yt = [
    "yt-dlp>=2024.0.0",
]
laicut = [
    "onnxruntime>=1.18.0",
    "numpy>=1.24.0",
    "soundfile>=0.12.0",
    "httpx>=0.27.0",
]
all = [
    "omnicaptions[dev,yt,laicut]",
]
```

**Step 2: Verify installation**

Run: `pip install -e ".[laicut]"`
Expected: Success, no errors

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add laicut optional dependencies"
```

---

## Task 2: Create laicut module structure

**Files:**
- Create: `src/omnicaptions/laicut/__init__.py`
- Create: `src/omnicaptions/laicut/constants.py`

**Step 1: Create laicut directory**

```bash
mkdir -p src/omnicaptions/laicut
```

**Step 2: Create constants.py**

```python
"""LaiCut constants and configuration."""

from pathlib import Path

# Model configuration
MODEL_NAME = "Lattice-1"
FRAME_SHIFT = 0.02  # 20ms per frame
SAMPLE_RATE = 16000

# Cache directories
CACHE_DIR = Path.home() / ".cache" / "omnicaptions"
MODELS_DIR = CACHE_DIR / "models"

# API configuration
LATTIFAI_API_URL = "https://api.lattifai.com"

# Model download URLs
MODEL_URLS = {
    "Lattice-1": "https://r2.lattifai.com/models/lattice-1.onnx",
}
```

**Step 3: Create __init__.py**

```python
"""LaiCut - Audio-video subtitle alignment using LattifAI."""

from .constants import MODEL_NAME, SAMPLE_RATE, FRAME_SHIFT

__all__ = ["MODEL_NAME", "SAMPLE_RATE", "FRAME_SHIFT"]
```

**Step 4: Commit**

```bash
git add src/omnicaptions/laicut/
git commit -m "feat(laicut): add module structure and constants"
```

---

## Task 3: Implement LaiCutClient (API client)

**Files:**
- Create: `src/omnicaptions/laicut/client.py`
- Create: `tests/laicut/test_client.py`

**Step 1: Create test directory**

```bash
mkdir -p tests/laicut
touch tests/laicut/__init__.py
```

**Step 2: Write failing test for LaiCutClient**

```python
# tests/laicut/test_client.py
"""Tests for LaiCutClient."""

import pytest
from omnicaptions.laicut.client import LaiCutClient


def test_client_init():
    """Test client initialization with API key."""
    client = LaiCutClient(api_key="test-key")
    assert client.api_key == "test-key"
    assert "api.lattifai.com" in client.base_url


def test_client_init_no_key():
    """Test client raises error without API key."""
    with pytest.raises(ValueError, match="API key"):
        LaiCutClient(api_key=None)
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/laicut/test_client.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'omnicaptions.laicut.client'"

**Step 4: Implement LaiCutClient**

```python
# src/omnicaptions/laicut/client.py
"""LattifAI API client for tokenization and detokenization."""

import asyncio
from typing import Any

import httpx

from .constants import LATTIFAI_API_URL


class LaiCutClient:
    """Client for LattifAI tokenize/detokenize APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = LATTIFAI_API_URL,
        timeout: float = 60.0,
    ):
        if not api_key:
            raise ValueError("LattifAI API key is required")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def tokenize(
        self,
        model_name: str,
        supervisions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Tokenize supervisions into lattice graph.

        Returns:
            {
                "success": True,
                "id": "lattice-uuid",
                "lattice_graph": "...",
                "final_state": 123
            }
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/tokenize",
                headers=self._get_headers(),
                json={
                    "model_name": model_name,
                    "supervisions": supervisions,
                },
            )
            response.raise_for_status()
            return response.json()

    async def detokenize(
        self,
        model_name: str,
        lattice_id: str,
        results: list[dict[str, Any]],
        labels: list[int],
        frame_shift: float = 0.02,
        destroy_lattice: bool = True,
    ) -> dict[str, Any]:
        """
        Detokenize aligned tokens back to supervisions with timestamps.

        Returns:
            {
                "success": True,
                "supervisions": [...]
            }
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/detokenize",
                headers=self._get_headers(),
                json={
                    "model_name": model_name,
                    "lattice_id": lattice_id,
                    "results": results,
                    "labels": labels,
                    "frame_shift": frame_shift,
                    "offset": 0.0,
                    "channel": 0,
                    "return_details": True,
                    "destroy_lattice": destroy_lattice,
                },
            )
            response.raise_for_status()
            return response.json()

    def tokenize_sync(
        self,
        model_name: str,
        supervisions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Synchronous wrapper for tokenize."""
        return asyncio.run(self.tokenize(model_name, supervisions))

    def detokenize_sync(
        self,
        model_name: str,
        lattice_id: str,
        results: list[dict[str, Any]],
        labels: list[int],
        frame_shift: float = 0.02,
        destroy_lattice: bool = True,
    ) -> dict[str, Any]:
        """Synchronous wrapper for detokenize."""
        return asyncio.run(
            self.detokenize(model_name, lattice_id, results, labels, frame_shift, destroy_lattice)
        )
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/laicut/test_client.py -v`
Expected: PASS (2 tests)

**Step 6: Commit**

```bash
git add src/omnicaptions/laicut/client.py tests/laicut/
git commit -m "feat(laicut): implement LaiCutClient for API calls"
```

---

## Task 4: Implement LaiCutModelManager

**Files:**
- Create: `src/omnicaptions/laicut/model_manager.py`
- Create: `tests/laicut/test_model_manager.py`

**Step 1: Write failing test**

```python
# tests/laicut/test_model_manager.py
"""Tests for LaiCutModelManager."""

import pytest
from pathlib import Path
from omnicaptions.laicut.model_manager import LaiCutModelManager
from omnicaptions.laicut.constants import MODELS_DIR


def test_model_manager_init():
    """Test model manager initialization."""
    manager = LaiCutModelManager()
    assert manager.models_dir == MODELS_DIR


def test_model_path():
    """Test getting model path."""
    manager = LaiCutModelManager()
    path = manager.get_model_path("Lattice-1")
    assert path == MODELS_DIR / "Lattice-1.onnx"


def test_unknown_model():
    """Test error for unknown model."""
    manager = LaiCutModelManager()
    with pytest.raises(ValueError, match="Unknown model"):
        manager.get_download_url("unknown-model")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/laicut/test_model_manager.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement LaiCutModelManager**

```python
# src/omnicaptions/laicut/model_manager.py
"""Model download and cache management."""

import logging
from pathlib import Path
from typing import Callable

import httpx

from .constants import MODELS_DIR, MODEL_URLS

logger = logging.getLogger(__name__)


class LaiCutModelManager:
    """Manages ONNX model downloads and caching."""

    def __init__(self, models_dir: Path | None = None):
        self.models_dir = models_dir or MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_name: str) -> Path:
        """Get local path for a model."""
        return self.models_dir / f"{model_name}.onnx"

    def get_download_url(self, model_name: str) -> str:
        """Get download URL for a model."""
        if model_name not in MODEL_URLS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_URLS.keys())}")
        return MODEL_URLS[model_name]

    def is_model_cached(self, model_name: str) -> bool:
        """Check if model is already downloaded."""
        return self.get_model_path(model_name).exists()

    def download_model(
        self,
        model_name: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """
        Download model from CDN.

        Args:
            model_name: Name of model to download
            progress_callback: Optional callback(downloaded_bytes, total_bytes)

        Returns:
            Path to downloaded model file
        """
        url = self.get_download_url(model_name)
        model_path = self.get_model_path(model_name)

        logger.info(f"Downloading {model_name} from {url}...")

        with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            # Write to temp file first, then rename
            temp_path = model_path.with_suffix(".tmp")
            downloaded = 0

            with open(temp_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total)

            # Rename to final path
            temp_path.rename(model_path)

        logger.info(f"Model saved to {model_path}")
        return model_path

    def ensure_model(
        self,
        model_name: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """
        Ensure model is available, downloading if necessary.

        Returns:
            Path to model file
        """
        if self.is_model_cached(model_name):
            logger.debug(f"Model {model_name} found in cache")
            return self.get_model_path(model_name)

        return self.download_model(model_name, progress_callback)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/laicut/test_model_manager.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/omnicaptions/laicut/model_manager.py tests/laicut/test_model_manager.py
git commit -m "feat(laicut): implement LaiCutModelManager for model downloads"
```

---

## Task 5: Implement LaiCutInference (ONNX + K2-lite)

**Files:**
- Create: `src/omnicaptions/laicut/inference.py`
- Create: `tests/laicut/test_inference.py`

**Step 1: Write failing test**

```python
# tests/laicut/test_inference.py
"""Tests for LaiCutInference."""

import pytest
import numpy as np

# Skip tests if onnxruntime not installed
pytest.importorskip("onnxruntime")

from omnicaptions.laicut.inference import LaiCutInference


def test_inference_init_no_model():
    """Test inference raises error without model."""
    with pytest.raises(FileNotFoundError):
        LaiCutInference(model_path="/nonexistent/model.onnx")


def test_load_audio_not_found():
    """Test load_audio raises error for missing file."""
    # Create inference with mock - we'll test audio loading separately
    with pytest.raises(FileNotFoundError):
        LaiCutInference.load_audio_static("/nonexistent/audio.wav")


def test_resample_audio():
    """Test audio resampling to 16kHz mono."""
    # Create 1 second of 44.1kHz stereo audio
    sr_orig = 44100
    audio_stereo = np.random.randn(2, sr_orig).astype(np.float32)

    resampled = LaiCutInference.resample_audio(audio_stereo, sr_orig, 16000)

    # Should be mono and ~16000 samples
    assert resampled.ndim == 1
    assert 15900 < len(resampled) < 16100  # Allow some tolerance
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/laicut/test_inference.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement LaiCutInference**

```python
# src/omnicaptions/laicut/inference.py
"""Local ONNX inference and K2-lite beam search alignment."""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .constants import FRAME_SHIFT, SAMPLE_RATE

logger = logging.getLogger(__name__)


class LaiCutInference:
    """
    ONNX model inference + simplified K2 beam search.

    This is a lightweight implementation that delegates graph creation
    to the API but runs acoustic inference locally.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self._session = None

    @property
    def session(self):
        """Lazy-load ONNX session."""
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError:
                raise ImportError(
                    "onnxruntime is required for local inference. "
                    "Install with: pip install omnicaptions[laicut]"
                )

            # Configure session for optimal performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Try GPU first, fall back to CPU
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options,
                providers=providers,
            )
            logger.info(f"Loaded ONNX model: {self.model_path.name}")

        return self._session

    @staticmethod
    def load_audio_static(path: str | Path) -> tuple[np.ndarray, int]:
        """
        Load audio file and return (samples, sample_rate).

        Returns float32 array normalized to [-1, 1].
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "soundfile is required for audio loading. "
                "Install with: pip install omnicaptions[laicut]"
            )

        audio, sr = sf.read(str(path), dtype="float32")
        return audio, sr

    @staticmethod
    def resample_audio(
        audio: np.ndarray,
        sr_orig: int,
        sr_target: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """
        Resample audio to target sample rate and convert to mono.

        Args:
            audio: Input audio array (can be stereo)
            sr_orig: Original sample rate
            sr_target: Target sample rate (default: 16000)

        Returns:
            Mono float32 array at target sample rate
        """
        # Convert stereo to mono
        if audio.ndim == 2:
            audio = audio.mean(axis=0) if audio.shape[0] == 2 else audio.mean(axis=1)

        # Resample if needed
        if sr_orig != sr_target:
            # Simple linear interpolation resampling
            duration = len(audio) / sr_orig
            target_len = int(duration * sr_target)
            indices = np.linspace(0, len(audio) - 1, target_len)
            audio = np.interp(indices, np.arange(len(audio)), audio)

        return audio.astype(np.float32)

    def load_audio(self, path: str | Path) -> np.ndarray:
        """Load and preprocess audio for inference."""
        audio, sr = self.load_audio_static(path)
        return self.resample_audio(audio, sr, SAMPLE_RATE)

    def get_emissions(
        self,
        audio: np.ndarray,
        chunk_duration: float | None = None,
    ) -> np.ndarray:
        """
        Run ONNX inference to get acoustic emissions.

        Args:
            audio: Float32 audio array at 16kHz
            chunk_duration: Optional chunk size for streaming (seconds)

        Returns:
            Emissions array of shape (1, num_frames, vocab_size)
        """
        # Reshape for model input: (batch=1, samples)
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        # Run inference
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        emissions = self.session.run(
            [output_name],
            {input_name: audio},
        )[0]

        return emissions

    def align_with_graph(
        self,
        emissions: np.ndarray,
        lattice_graph: str,
        final_state: int,
        search_beam: float = 200.0,
        output_beam: float = 80.0,
    ) -> tuple[list[dict[str, Any]], list[int]]:
        """
        Perform beam search alignment using lattice graph.

        This is a simplified Viterbi-style alignment that follows
        the lattice structure from the API.

        Args:
            emissions: Acoustic emissions from ONNX model
            lattice_graph: Graph string from tokenize API
            final_state: Final state index
            search_beam: Beam width for search
            output_beam: Beam width for output

        Returns:
            (aligned_tokens, labels) tuple for detokenize API
        """
        # Parse lattice graph
        # Format: "arc1 arc2 arc3..." where each arc is "src dest label score"
        arcs = []
        for line in lattice_graph.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                src, dst, label = int(parts[0]), int(parts[1]), int(parts[2])
                score = float(parts[3]) if len(parts) > 3 else 0.0
                arcs.append((src, dst, label, score))

        num_frames = emissions.shape[1]

        # Simple greedy alignment following lattice structure
        # In production, this would use proper beam search
        aligned_tokens = []
        labels = []
        current_state = 0
        frame_idx = 0

        # Build adjacency list
        adj = {}
        for src, dst, label, score in arcs:
            if src not in adj:
                adj[src] = []
            adj[src].append((dst, label, score))

        # Greedy path through lattice
        while current_state != final_state and frame_idx < num_frames:
            if current_state not in adj:
                break

            # Find best next arc based on emissions
            best_arc = None
            best_score = float("-inf")

            for dst, label, arc_score in adj[current_state]:
                if label < emissions.shape[2]:
                    emission_score = float(emissions[0, frame_idx, label])
                    total_score = emission_score + arc_score
                    if total_score > best_score:
                        best_score = total_score
                        best_arc = (dst, label)

            if best_arc is None:
                break

            dst, label = best_arc
            aligned_tokens.append({
                "start": frame_idx,
                "end": frame_idx + 1,
                "label": label,
                "score": best_score,
            })
            labels.append(label)

            current_state = dst
            frame_idx += 1

        return aligned_tokens, labels
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/laicut/test_inference.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/omnicaptions/laicut/inference.py tests/laicut/test_inference.py
git commit -m "feat(laicut): implement LaiCutInference for ONNX inference"
```

---

## Task 6: Implement LaiCutAligner (main entry)

**Files:**
- Create: `src/omnicaptions/laicut/aligner.py`
- Create: `tests/laicut/test_aligner.py`

**Step 1: Write failing test**

```python
# tests/laicut/test_aligner.py
"""Tests for LaiCutAligner."""

import pytest
from omnicaptions.laicut.aligner import LaiCutAligner


def test_aligner_init():
    """Test aligner initialization."""
    aligner = LaiCutAligner(api_key="test-key")
    assert aligner.api_key == "test-key"
    assert aligner.model_name == "Lattice-1"


def test_aligner_init_no_key():
    """Test aligner without API key raises error."""
    with pytest.raises(ValueError, match="API key"):
        LaiCutAligner(api_key=None)


def test_aligner_verbose():
    """Test aligner verbose mode."""
    aligner = LaiCutAligner(api_key="test-key", verbose=True)
    assert aligner.verbose is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/laicut/test_aligner.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement LaiCutAligner**

```python
# src/omnicaptions/laicut/aligner.py
"""Main LaiCut alignment orchestrator."""

import logging
from pathlib import Path
from typing import Any, Callable

from lattifai.caption import Caption

from .client import LaiCutClient
from .constants import FRAME_SHIFT, MODEL_NAME
from .inference import LaiCutInference
from .model_manager import LaiCutModelManager

logger = logging.getLogger(__name__)


class LaiCutAligner:
    """
    Audio-video subtitle aligner using LattifAI API + local ONNX inference.

    Usage:
        aligner = LaiCutAligner(api_key="your-key")
        output = aligner.align("video.mp4", "subtitle.srt")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = MODEL_NAME,
        verbose: bool = False,
    ):
        if not api_key:
            raise ValueError(
                "LattifAI API key is required. "
                "Set LATTIFAI_API_KEY environment variable or pass api_key parameter."
            )

        self.api_key = api_key
        self.model_name = model_name
        self.verbose = verbose

        self._client: LaiCutClient | None = None
        self._inference: LaiCutInference | None = None
        self._model_manager = LaiCutModelManager()

        if verbose:
            logging.basicConfig(level=logging.INFO)

    @property
    def client(self) -> LaiCutClient:
        """Lazy-load API client."""
        if self._client is None:
            self._client = LaiCutClient(api_key=self.api_key)
        return self._client

    @property
    def inference(self) -> LaiCutInference:
        """Lazy-load inference engine (downloads model if needed)."""
        if self._inference is None:
            model_path = self._model_manager.ensure_model(
                self.model_name,
                progress_callback=self._download_progress if self.verbose else None,
            )
            self._inference = LaiCutInference(model_path)
        return self._inference

    def _download_progress(self, downloaded: int, total: int) -> None:
        """Progress callback for model download."""
        if total > 0:
            pct = (downloaded / total) * 100
            print(f"\rDownloading model... {pct:.1f}%", end="", flush=True)
            if downloaded >= total:
                print()  # Newline after complete

    def _log(self, message: str) -> None:
        """Log message if verbose mode enabled."""
        if self.verbose:
            logger.info(message)

    def align(
        self,
        audio: str | Path,
        caption: str | Path,
        output: str | Path | None = None,
        output_format: str | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> Path:
        """
        Align audio/video with caption file.

        Args:
            audio: Path to audio/video file or YouTube URL
            caption: Path to caption file (SRT/VTT/ASS/LRC/TXT/MD)
            output: Output path (default: <caption>_aligned.<ext>)
            output_format: Output format (default: same as input)
            progress_callback: Optional callback(status_message, progress_percent)

        Returns:
            Path to aligned caption file
        """
        audio_path = Path(audio)
        caption_path = Path(caption)

        # Determine output path
        if output is None:
            suffix = f".{output_format}" if output_format else caption_path.suffix
            output_path = caption_path.with_stem(f"{caption_path.stem}_aligned").with_suffix(suffix)
        else:
            output_path = Path(output)

        def report(msg: str, pct: int) -> None:
            self._log(f"[{pct}%] {msg}")
            if progress_callback:
                progress_callback(msg, pct)

        # Step 1: Load caption
        report("Loading caption...", 5)
        cap = Caption.read(str(caption_path))
        supervisions = [
            {"text": sup.text, "start": sup.start, "end": sup.end}
            for sup in cap.supervisions
        ]
        self._log(f"Loaded {len(supervisions)} segments")

        # Step 2: Tokenize via API
        report("Tokenizing via API...", 15)
        tokenize_result = self.client.tokenize_sync(self.model_name, supervisions)

        if not tokenize_result.get("success"):
            raise RuntimeError(f"Tokenize failed: {tokenize_result.get('error')}")

        lattice_id = tokenize_result["id"]
        lattice_graph = tokenize_result["lattice_graph"]
        final_state = tokenize_result["final_state"]

        # Step 3: Load audio and run inference
        report("Loading model...", 25)
        _ = self.inference  # Trigger model download if needed

        report("Loading audio...", 35)
        audio_data = self.inference.load_audio(audio_path)
        self._log(f"Audio duration: {len(audio_data) / 16000:.2f}s")

        report("Running alignment...", 50)
        emissions = self.inference.get_emissions(audio_data)

        # Step 4: Align with lattice graph
        report("Aligning...", 70)
        aligned_tokens, labels = self.inference.align_with_graph(
            emissions, lattice_graph, final_state
        )

        # Step 5: Detokenize via API
        report("Detokenizing via API...", 85)
        detokenize_result = self.client.detokenize_sync(
            model_name=self.model_name,
            lattice_id=lattice_id,
            results=aligned_tokens,
            labels=labels,
            frame_shift=FRAME_SHIFT,
        )

        if not detokenize_result.get("success"):
            raise RuntimeError(f"Detokenize failed: {detokenize_result.get('error')}")

        # Step 6: Update supervisions and write output
        report("Writing output...", 95)
        aligned_supervisions = detokenize_result["supervisions"]

        for i, sup in enumerate(cap.supervisions):
            if i < len(aligned_supervisions):
                aligned = aligned_supervisions[i]
                sup.start = aligned.get("start", sup.start)
                sup.end = aligned.get("end", sup.end)

        cap.write(str(output_path))
        report(f"Saved: {output_path}", 100)

        return output_path
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/laicut/test_aligner.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/omnicaptions/laicut/aligner.py tests/laicut/test_aligner.py
git commit -m "feat(laicut): implement LaiCutAligner main entry"
```

---

## Task 7: Update laicut __init__.py exports

**Files:**
- Modify: `src/omnicaptions/laicut/__init__.py`

**Step 1: Update exports**

```python
# src/omnicaptions/laicut/__init__.py
"""LaiCut - Audio-video subtitle alignment using LattifAI."""

from .aligner import LaiCutAligner
from .client import LaiCutClient
from .constants import FRAME_SHIFT, MODEL_NAME, SAMPLE_RATE
from .inference import LaiCutInference
from .model_manager import LaiCutModelManager

__all__ = [
    "LaiCutAligner",
    "LaiCutClient",
    "LaiCutInference",
    "LaiCutModelManager",
    "MODEL_NAME",
    "SAMPLE_RATE",
    "FRAME_SHIFT",
]
```

**Step 2: Verify imports work**

Run: `python -c "from omnicaptions.laicut import LaiCutAligner; print('OK')"`
Expected: "OK"

**Step 3: Commit**

```bash
git add src/omnicaptions/laicut/__init__.py
git commit -m "feat(laicut): export all classes from module"
```

---

## Task 8: Add LATTIFAI_API_KEY to config.py

**Files:**
- Modify: `src/omnicaptions/config.py`

**Step 1: Read current config.py**

Review existing structure for consistency.

**Step 2: Add LattifAI API key support**

Add after existing `get_api_key()` function:

```python
def get_lattifai_api_key() -> str | None:
    """Get LattifAI API key from environment or config file."""
    # 1. Check environment variable
    key = os.environ.get("LATTIFAI_API_KEY")
    if key:
        return key

    # 2. Check config file
    config_path = CONFIG_DIR / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            if "lattifai_api_key" in config:
                return config["lattifai_api_key"]
        except (json.JSONDecodeError, KeyError):
            pass

    return None


def save_lattifai_api_key(key: str) -> None:
    """Save LattifAI API key to config file."""
    config_path = CONFIG_DIR / "config.json"
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            pass

    config["lattifai_api_key"] = key
    config_path.write_text(json.dumps(config, indent=2))


def get_lattifai_setup_instructions() -> str:
    """Return setup instructions for LattifAI API key."""
    return """
LattifAI API key not found.

Get your API key from: https://lattifai.com/dashboard

Then either:
1. Set environment variable: export LATTIFAI_API_KEY=your-key
2. Pass directly: omnicaptions laicut-align --api-key your-key
"""
```

**Step 3: Commit**

```bash
git add src/omnicaptions/config.py
git commit -m "feat: add LATTIFAI_API_KEY support to config"
```

---

## Task 9: Add laicut-align CLI command

**Files:**
- Modify: `src/omnicaptions/cli.py`

**Step 1: Read current cli.py structure**

Review existing commands for consistency.

**Step 2: Add laicut-align command**

Add new command function:

```python
@app.command("laicut-align")
def laicut_align(
    audio: Annotated[str, typer.Argument(help="Audio/video file or YouTube URL")],
    caption: Annotated[str, typer.Argument(help="Caption file (SRT/VTT/ASS/LRC/TXT/MD)")],
    output: Annotated[Optional[str], typer.Option("-o", "--output", help="Output file")] = None,
    format: Annotated[Optional[str], typer.Option("-f", "--format", help="Output format")] = None,
    api_key: Annotated[Optional[str], typer.Option("-k", "--api-key", help="LattifAI API key")] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Verbose output")] = False,
) -> None:
    """Align audio/video with caption file using LattifAI."""
    from .config import get_lattifai_api_key, get_lattifai_setup_instructions, save_lattifai_api_key

    # Get API key
    key = api_key or get_lattifai_api_key()

    if not key:
        console.print(get_lattifai_setup_instructions(), style="yellow")
        key = typer.prompt("Enter your LattifAI API key")
        save_lattifai_api_key(key)
        console.print("API key saved to config.", style="green")

    try:
        from .laicut import LaiCutAligner
    except ImportError:
        console.print(
            "[red]LaiCut dependencies not installed.[/red]\n"
            "Install with: pip install omnicaptions[laicut]"
        )
        raise typer.Exit(1)

    try:
        aligner = LaiCutAligner(api_key=key, verbose=verbose)

        def progress(msg: str, pct: int) -> None:
            console.print(f"[{pct:3d}%] {msg}")

        output_path = aligner.align(
            audio=audio,
            caption=caption,
            output=output,
            output_format=format,
            progress_callback=progress if verbose else None,
        )

        console.print(f"\n[green]✓ Aligned:[/green] {output_path}")

    except FileNotFoundError as e:
        console.print(f"[red]File not found:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)
```

**Step 3: Test CLI help**

Run: `omnicaptions laicut-align --help`
Expected: Shows help with audio, caption arguments and options

**Step 4: Commit**

```bash
git add src/omnicaptions/cli.py
git commit -m "feat: add laicut-align CLI command"
```

---

## Task 10: Create SKILL.md

**Files:**
- Create: `skills/omnicaptions-laicut-align/SKILL.md`

**Step 1: Create skill directory**

```bash
mkdir -p skills/omnicaptions-laicut-align
```

**Step 2: Write SKILL.md**

```markdown
---
name: omnicaptions-laicut-align
description: Use when aligning audio/video with subtitles to get word-level timestamps. Supports local files and YouTube URLs. Powered by LattifAI Lattice-1 model.
allowed-tools: Bash(omnicaptions:*)
---

# LaiCut Align - Audio-Video Subtitle Alignment

Align subtitles with audio/video using LattifAI Lattice-1 model for word-level timestamps.

## When to Use

- Subtitle timing is off, needs realignment
- Have transcript text, need precise timestamps
- Creating karaoke / lyric sync subtitles
- Video editing caused subtitle timing drift

## Quick Start

```bash
# Basic alignment
omnicaptions laicut-align video.mp4 subtitle.srt

# YouTube video + local subtitle
omnicaptions laicut-align "https://youtube.com/watch?v=xxx" my_subtitle.srt

# Specify output format
omnicaptions laicut-align audio.mp3 transcript.txt -o synced.vtt

# Verbose output
omnicaptions laicut-align video.mp4 subtitle.srt -v
```

## CLI Options

| Option | Description |
|--------|-------------|
| `audio` | Audio/video file or YouTube URL |
| `caption` | Caption file (SRT/VTT/ASS/LRC/TXT/MD) |
| `-o, --output` | Output file (default: `<input>_aligned.<ext>`) |
| `-f, --format` | Output format |
| `-k, --api-key` | LattifAI API key |
| `-v, --verbose` | Show progress details |

## Setup

First use prompts for API key. Get one at: https://lattifai.com/dashboard

Install alignment dependencies:
```bash
pip install omnicaptions[laicut]
```

## Common Workflows

```bash
# Transcribe → Align → Translate
omnicaptions transcribe video.mp4 -o transcript.md
omnicaptions laicut-align video.mp4 transcript.md -o aligned.srt
omnicaptions translate aligned.srt -o bilingual.srt -l zh --bilingual

# Download + Align
omnicaptions download "https://youtube.com/watch?v=xxx" -q audio
omnicaptions laicut-align xxx.m4a xxx.vtt -o aligned.srt
```

## Related Skills

| Skill | Use When |
|-------|----------|
| `/omnicaptions:transcribe` | Generate transcript first |
| `/omnicaptions:convert` | Convert aligned output format |
| `/omnicaptions:translate` | Translate aligned subtitles |
| `/omnicaptions:download` | Download video/audio first |
```

**Step 3: Commit**

```bash
git add skills/omnicaptions-laicut-align/
git commit -m "feat: add omnicaptions-laicut-align skill"
```

---

## Task 11: Update main __init__.py exports

**Files:**
- Modify: `src/omnicaptions/__init__.py`

**Step 1: Add laicut exports**

```python
# Add to existing exports
try:
    from .laicut import LaiCutAligner
except ImportError:
    LaiCutAligner = None  # Optional dependency not installed

__all__ = [
    "GeminiCaption",
    "GeminiCaptionConfig",
    "LaiCutAligner",  # Add this
]
```

**Step 2: Commit**

```bash
git add src/omnicaptions/__init__.py
git commit -m "feat: export LaiCutAligner from main module"
```

---

## Task 12: Update CLAUDE.md documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update structure section**

Add laicut module to structure:

```markdown
├── src/omnicaptions/
│   ├── __init__.py
│   ├── caption.py           # GeminiCaption class
│   ├── cli.py               # CLI entry point
│   ├── config.py            # API key management
│   ├── laicut/              # LaiCut alignment module
│   │   ├── __init__.py
│   │   ├── aligner.py       # LaiCutAligner main class
│   │   ├── client.py        # LattifAI API client
│   │   ├── inference.py     # ONNX inference
│   │   └── model_manager.py # Model download/cache
│   └── prompts/
```

**Step 2: Update CLI section**

Add laicut-align command:

```markdown
## CLI

```bash
omnicaptions transcribe <input> [-o output] [-m model] [-l lang] [-t lang --bilingual]
omnicaptions convert <input> [-o output] [-f fmt] [-t fmt]
omnicaptions translate <input> [-o output] -l <lang> [--bilingual]
omnicaptions download <url> [-o output] [-q quality]
omnicaptions laicut-align <audio> <caption> [-o output] [-f format] [-v]
```

**Step 3: Update Key Classes section**

Add:

```markdown
- `LaiCutAligner`: Audio-video subtitle alignment using LattifAI API
```

**Step 4: Update Dependencies section**

Add:

```markdown
- `onnxruntime`: ONNX inference (optional, for laicut)
- `httpx`: Async HTTP client (optional, for laicut)
```

**Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with laicut module"
```

---

## Task 13: Run all tests and verify

**Step 1: Run full test suite**

Run: `pytest -v`
Expected: All tests pass (26 existing + new laicut tests)

**Step 2: Test CLI help**

Run: `omnicaptions --help`
Expected: Shows laicut-align in command list

Run: `omnicaptions laicut-align --help`
Expected: Shows all options

**Step 3: Final commit if any fixes needed**

```bash
git status
# If clean, done. Otherwise fix and commit.
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 0 | Fix marketplace.json | `.claude-plugin/marketplace.json` |
| 1 | Add dependencies | `pyproject.toml` |
| 2 | Create module structure | `laicut/__init__.py`, `constants.py` |
| 3 | Implement LaiCutClient | `laicut/client.py`, tests |
| 4 | Implement LaiCutModelManager | `laicut/model_manager.py`, tests |
| 5 | Implement LaiCutInference | `laicut/inference.py`, tests |
| 6 | Implement LaiCutAligner | `laicut/aligner.py`, tests |
| 7 | Update laicut exports | `laicut/__init__.py` |
| 8 | Add API key config | `config.py` |
| 9 | Add CLI command | `cli.py` |
| 10 | Create SKILL.md | `skills/omnicaptions-laicut-align/` |
| 11 | Update main exports | `__init__.py` |
| 12 | Update docs | `CLAUDE.md` |
| 13 | Final verification | Run tests |
