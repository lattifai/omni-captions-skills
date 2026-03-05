"""Translate service -- caption translation via Gemini API.

Provides standalone functions (no class dependency) for translating
caption files using the Gemini generative API.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, Union

from omnicaptions.config import API_KEY_URL, get_gemini_api_key
from omnicaptions.services.base import ErrorCode, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_BATCH_SIZE = 30
DEFAULT_CONTEXT_LINES = 5

LANGUAGE_NAMES = {
    "zh": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
}


def _build_translate_prompt(
    texts: list[str],
    target_lang: str,
    bilingual: bool,
    context_before: list[str] | None = None,
    context_after: list[str] | None = None,
) -> str:
    """Build the translation prompt for a batch of texts.

    Args:
        texts: Lines to translate.
        target_lang: Target language code or name.
        bilingual: If True, return both original and translated text.
        context_before: Preceding lines for context (not translated).
        context_after: Following lines for context (not translated).

    Returns:
        Fully assembled prompt string.
    """
    lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)

    if bilingual:
        instruction = (
            f"Translate captions to {lang_name}. Return JSON array with "
            f'"original" and "translated" keys.\n\n'
            f"Rules:\n"
            f'1. ONLY translate the lines in "to_translate" - context is for reference only\n'
            f"2. Maintain speaker voice and tone consistency\n"
            f"3. Handle split sentences naturally (context helps understand full meaning)\n"
            f"4. Adapt idioms and cultural references appropriately\n"
            f"5. Keep the exact same order and count as input"
        )
    else:
        instruction = (
            f"Translate captions to {lang_name}. Return JSON array of translated strings.\n\n"
            f"Rules:\n"
            f'1. ONLY translate the lines in "to_translate" - context is for reference only\n'
            f"2. Maintain speaker voice and tone consistency\n"
            f"3. Handle split sentences naturally (context helps understand full meaning)\n"
            f"4. Adapt idioms and cultural references appropriately\n"
            f"5. Keep the exact same order and count as input"
        )

    # Build input with context
    input_data: dict = {"to_translate": texts}
    if context_before:
        input_data["context_before"] = context_before
    if context_after:
        input_data["context_after"] = context_after

    return f"{instruction}\n\nInput:\n{json.dumps(input_data, ensure_ascii=False)}"


async def _translate_batch_async(
    texts: list[str],
    target_lang: str,
    bilingual: bool,
    api_key: str,
    model: str = DEFAULT_MODEL,
    context_before: list[str] | None = None,
    context_after: list[str] | None = None,
) -> list:
    """Translate a batch of texts with surrounding context.

    Args:
        texts: Lines to translate.
        target_lang: Target language code.
        bilingual: Whether to include originals in output.
        api_key: Gemini API key.
        model: Model name.
        context_before: Preceding context lines.
        context_after: Following context lines.

    Returns:
        List of translated strings (monolingual) or dicts (bilingual).
    """
    from google import genai
    from google.genai.types import GenerateContentConfig

    client = genai.Client(api_key=api_key)
    prompt = _build_translate_prompt(texts, target_lang, bilingual, context_before, context_after)

    config = GenerateContentConfig(response_mime_type="application/json")

    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        ),
    )

    if not response.text:
        raise RuntimeError("Empty response from Gemini API")

    return json.loads(response.text)


async def _translate_file_async(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    target_lang: str,
    bilingual: bool = False,
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    context_lines: int = DEFAULT_CONTEXT_LINES,
    verbose: bool = False,
) -> Path:
    """Translate a caption file (async).

    Args:
        input_file: Path to the input caption file.
        output_file: Path for the translated output.
        target_lang: Target language code.
        bilingual: Include original text alongside translation.
        api_key: Gemini API key.
        model: Model name.
        batch_size: Number of segments per API call.
        context_lines: Number of surrounding lines for context.
        verbose: Enable verbose logging.

    Returns:
        Path to the written output file.
    """
    from lattifai.caption import Caption, GeminiReader, GeminiWriter

    input_path = Path(input_file)
    output_path = Path(output_file)

    if verbose:
        logger.info("Loading: %s", input_path)

    # Handle Gemini markdown format
    if input_path.suffix.lower() == ".md":
        supervisions = GeminiReader.extract_for_alignment(str(input_path))
        cap = Caption.from_supervisions(supervisions)
    else:
        cap = Caption.read(str(input_path))

    texts = [sup.text for sup in cap.supervisions]

    if verbose:
        logger.info("Translating %d segments to %s...", len(texts), target_lang)

    # Translate in batches with context
    results: list = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # Extract context before and after current batch
        ctx_before = texts[max(0, i - context_lines) : i] if i > 0 else None
        batch_end = i + batch_size
        ctx_after = texts[batch_end : batch_end + context_lines] if batch_end < len(texts) else None

        if verbose:
            logger.info(
                "Batch %d/%d",
                i // batch_size + 1,
                (len(texts) - 1) // batch_size + 1,
            )

        batch_result = await _translate_batch_async(
            batch, target_lang, bilingual, api_key, model, ctx_before, ctx_after
        )
        results.extend(batch_result)

    # Update supervisions
    for idx, sup in enumerate(cap.supervisions):
        if bilingual:
            original = results[idx].get("original", sup.text)
            translated = results[idx].get("translated", "")
            sup.text = f"{original}\n{translated}"
        else:
            sup.text = results[idx] if isinstance(results[idx], str) else results[idx]

    # Write output -- use GeminiWriter for .md files
    if output_path.suffix.lower() == ".md":
        GeminiWriter.write(cap.supervisions, str(output_path))
    else:
        cap.write(str(output_path))

    if verbose:
        logger.info("Saved: %s", output_path)

    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def translate_caption(
    input_file: str,
    target_lang: str,
    output: Optional[str] = None,
    bilingual: bool = False,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    context_lines: int = DEFAULT_CONTEXT_LINES,
    verbose: bool = False,
) -> ToolResult:
    """Translate a caption file and save the result.

    Args:
        input_file: Path to the input caption file.
        target_lang: Target language code (e.g. "zh", "ja", "en").
        output: Output file path. Auto-generated if None.
        bilingual: Include original text alongside translation.
        api_key: Gemini API key. Falls back to config if None.
        model: Gemini model name.
        batch_size: Number of segments per API call.
        context_lines: Number of surrounding lines for context.
        verbose: Enable verbose logging.

    Returns:
        ToolResult with data containing output_path, segment_count,
        target_lang, bilingual.
    """
    # Resolve API key
    resolved_key = api_key or get_gemini_api_key()
    if not resolved_key:
        return ToolResult(
            status="error",
            message=f"Gemini API key not found. Get one at {API_KEY_URL}",
            error_code=ErrorCode.MISSING_API_KEY,
        )

    # Validate input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        return ToolResult(
            status="error",
            message=f"File not found: {input_file}",
            error_code=ErrorCode.INVALID_INPUT,
        )

    # Auto-generate output path if not provided
    if output:
        output_path = Path(output)
    else:
        lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)
        output_path = input_path.with_name(
            f"{input_path.stem}_Gemini_{lang_name}{input_path.suffix}"
        )

    # Run translation
    try:
        from lattifai.caption import Caption, GeminiReader

        # Pre-count segments for the result
        if input_path.suffix.lower() == ".md":
            supervisions = GeminiReader.extract_for_alignment(str(input_path))
            segment_count = len(supervisions)
        else:
            cap = Caption.read(str(input_path))
            segment_count = len(cap.supervisions)

        asyncio.run(
            _translate_file_async(
                input_file=input_file,
                output_file=str(output_path),
                target_lang=target_lang,
                bilingual=bilingual,
                api_key=resolved_key,
                model=model,
                batch_size=batch_size,
                context_lines=context_lines,
                verbose=verbose,
            )
        )
    except Exception as e:
        return ToolResult(
            status="error",
            message=f"Translation failed: {e}",
            error_code=ErrorCode.TRANSLATION_FAILED,
        )

    return ToolResult(
        status="success",
        message=f"Translation saved to {output_path}",
        data={
            "output_path": str(output_path),
            "segment_count": segment_count,
            "target_lang": target_lang,
            "bilingual": bilingual,
        },
    )
