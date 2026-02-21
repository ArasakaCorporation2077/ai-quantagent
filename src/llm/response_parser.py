"""Parse and validate LLM responses for alpha generation."""

import json
import logging
import re

from src.data.schema import VALID_FREQUENCIES

logger = logging.getLogger(__name__)


class ResponseParseError(Exception):
    pass


def extract_json_array(text: str) -> list[dict]:
    """Extract a JSON array from LLM response text.

    Handles: code fences, surrounding commentary, minor formatting issues.
    """
    # Strategy 1: Direct parse
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from ```json ... ``` code fence
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find [...] in text (try finding balanced brackets)
    # First try to find the outermost balanced array
    start_idx = text.find('[')
    if start_idx >= 0:
        depth = 0
        end_idx = start_idx
        in_string = False
        escape_next = False
        for i in range(start_idx, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == '\\':
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break

        if depth == 0:
            candidate = text[start_idx:end_idx + 1]
            try:
                result = json.loads(candidate)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

            # Strategy 4: Fix common issues (trailing commas, single quotes)
            fixed = candidate
            # Remove trailing commas before ] or }
            fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
            # Replace single quotes with double quotes
            fixed = fixed.replace("'", '"')
            try:
                result = json.loads(fixed)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

    # Strategy 5: Try line-by-line object extraction
    objects = []
    for match in re.finditer(r'\{[^{}]+\}', text):
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            pass
    if objects:
        return objects

    raise ResponseParseError(f'Could not extract JSON array from response: {text[:200]}...')


def validate_alpha_response(parsed: list[dict]) -> list[dict]:
    """Validate and filter alpha response items."""
    valid = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if 'frequency' not in item or 'alpha' not in item:
            logger.debug(f'Skipping item missing keys: {item}')
            continue
        if item['frequency'] not in VALID_FREQUENCIES:
            logger.debug(f'Skipping invalid frequency: {item["frequency"]}')
            continue
        alpha = item['alpha'].strip()
        if not alpha:
            continue
        # Basic sanity: balanced parentheses
        if alpha.count('(') != alpha.count(')'):
            logger.debug(f'Skipping unbalanced parens: {alpha[:60]}...')
            continue
        item['alpha'] = alpha
        valid.append(item)
    return valid


def parse_categories(text: str) -> list[str]:
    """Parse category list from LLM response."""
    # Try JSON first (model might return JSON array of strings)
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(c).strip() for c in result if isinstance(c, str) and c.strip()]
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from code fence or brackets
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, list):
                return [str(c).strip() for c in result if isinstance(c, str) and c.strip()]
        except json.JSONDecodeError:
            pass

    bracket_match = re.search(r'\[.*\]', text, re.DOTALL)
    if bracket_match:
        try:
            result = json.loads(bracket_match.group(0))
            if isinstance(result, list):
                return [str(c).strip() for c in result if isinstance(c, str) and c.strip()]
        except json.JSONDecodeError:
            pass

    # Fallback: line-based parsing
    lines = text.split('\n')
    categories = []
    for line in lines:
        line = line.strip()
        # Remove markdown bold
        line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
        # Remove bullet markers
        line = line.strip('-').strip('*').strip('•').strip()
        # Remove numbering like "1. " or "1) "
        line = re.sub(r'^\d+[.)]\s*', '', line)
        # Remove trailing descriptions after colon or dash
        if ':' in line:
            line = line.split(':')[0].strip()
        if ' - ' in line:
            line = line.split(' - ')[0].strip()
        if line and 3 < len(line) < 80:
            # Skip intro/outro lines
            lower = line.lower()
            skip_words = ['here are', 'following', 'below', 'these are', 'sure', 'certainly', 'of course']
            if any(w in lower for w in skip_words):
                continue
            categories.append(line)
    return categories
