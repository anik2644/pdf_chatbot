"""
Utility for cleaning and formatting LLM answers.
"""

import re
from typing import Optional


def clean_answer(raw_answer: str) -> str:
    """
    Clean and format the raw LLM answer.

    Removes context, instructions, and formatting artifacts.
    Extracts only the relevant answer portion.

    Args:
        raw_answer: The raw answer from the LLM

    Returns:
        Cleaned and formatted answer string
    """
    if not raw_answer:
        return "No answer generated"

    answer = raw_answer

    # Step 1: Extract answer after "Answer:" marker if present
    answer = _extract_answer_section(answer)

    # Step 2: Remove common unwanted sections
    answer = _remove_unwanted_sections(answer)

    # Step 3: Clean up whitespace and formatting
    answer = _clean_formatting(answer)

    # Step 4: Remove duplicate lines
    answer = _remove_duplicate_lines(answer)

    # Step 5: Final trim and validation
    answer = answer.strip()

    if not answer:
        return "No answer could be extracted"

    return answer


def _extract_answer_section(text: str) -> str:
    """Extract the answer section from the text."""

    # Common markers that indicate the start of the actual answer
    answer_markers = [
        r"Answer:\s*",
        r"Response:\s*",
        r"Here's the answer:\s*",
        r"Here is the answer:\s*",
        r"Based on the context[,:]?\s*",
    ]

    for marker in answer_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            text = text[match.end():]
            break

    return text


def _remove_unwanted_sections(text: str) -> str:
    """Remove unwanted sections like context, instructions, etc."""

    # Patterns to remove (everything before these markers)
    start_markers = [
        r"^.*?(?=To archive|To create|To update|To delete|The following|Here are|Step 1|1\.)",
        r"^.*?(?=Based on the provided context,?\s*)",
    ]

    # Patterns to remove (these sections and everything after)
    end_markers = [
        r"\n\s*Question:.*$",
        r"\n\s*Instructions:.*$",
        r"\n\s*Context:.*$",
        r"\n\s*Note:.*?(?=\n\n|\Z)",
        r"\n\s*---.*$",
        r"\n\s*Sources?:.*$",
    ]

    # Remove content after end markers
    for pattern in end_markers:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    return text


def _clean_formatting(text: str) -> str:
    """Clean up formatting issues."""

    # Replace multiple spaces with single space
    text = re.sub(r'[ \t]+', ' ', text)

    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Clean up bullet points and numbering
    text = re.sub(r'•\s+', '• ', text)
    text = re.sub(r'(\d+)\.\s+', r'\1. ', text)

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Remove box drawing characters and artifacts
    text = re.sub(r'[│─┌┐└┘├┤┬┴┼]', '', text)

    # Remove excessive punctuation
    text = re.sub(r'\.{2,}', '.', text)

    return text


def _remove_duplicate_lines(text: str) -> str:
    """Remove duplicate consecutive lines."""
    lines = text.split('\n')
    cleaned_lines = []
    prev_line = None

    for line in lines:
        stripped = line.strip()
        if stripped != prev_line:
            cleaned_lines.append(line)
            prev_line = stripped

    return '\n'.join(cleaned_lines)


def format_answer_for_display(answer: str, max_width: int = 80) -> str:
    """
    Format the answer for terminal display.

    Args:
        answer: The cleaned answer
        max_width: Maximum line width

    Returns:
        Formatted answer string
    """
    import textwrap

    paragraphs = answer.split('\n\n')
    formatted_paragraphs = []

    for para in paragraphs:
        # Check if it's a list item
        if para.strip().startswith(('•', '-', '*')) or re.match(r'^\d+\.', para.strip()):
            # Keep list formatting
            lines = para.split('\n')
            formatted_lines = []
            for line in lines:
                if len(line) > max_width:
                    # Wrap long list items
                    wrapped = textwrap.fill(line, width=max_width, subsequent_indent='  ')
                    formatted_lines.append(wrapped)
                else:
                    formatted_lines.append(line)
            formatted_paragraphs.append('\n'.join(formatted_lines))
        else:
            # Wrap regular paragraphs
            wrapped = textwrap.fill(para, width=max_width)
            formatted_paragraphs.append(wrapped)

    return '\n\n'.join(formatted_paragraphs)


def extract_steps(answer: str) -> list[str]:
    """
    Extract numbered steps from an answer.

    Args:
        answer: The answer text

    Returns:
        List of step strings
    """
    # Match numbered steps (1., 2., etc.) or bullet points
    step_pattern = r'(?:^|\n)\s*(?:(\d+)[.):]\s*|[•\-\*]\s*)(.+?)(?=\n\s*(?:\d+[.):]\s*|[•\-\*]\s*)|\Z)'
    matches = re.findall(step_pattern, answer, re.DOTALL)

    steps = []
    for num, content in matches:
        step_text = content.strip()
        if step_text:
            if num:
                steps.append(f"{num}. {step_text}")
            else:
                steps.append(f"• {step_text}")

    return steps


# Convenience function for quick cleaning
def quick_clean(raw_answer: str) -> str:
    """Quick cleaning for simple cases."""
    answer = clean_answer(raw_answer)
    return format_answer_for_display(answer)