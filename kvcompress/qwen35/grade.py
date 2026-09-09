"""Explicit final-answer and paper-compatible diagnostics using the repo grader.

The main final-boxed metric never scores a box left inside unfinished thinking.
The paper-compatible metric uses the existing extract_answer on the entire
completion, including an unfinished trace. Both are persisted, not conflated.
Symbolic comparison inherits the original grader's normalizations/tolerances.
"""
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1] / "harness"
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))
from Utils.grader import check_is_correct
from Utils.parser import extract_answer


def last_boxed(text):
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    start += len(marker)
    depth = 1
    for end in range(start, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return text[start:end]
    return None


def grade_completion(completion, gold, termination):
    final = completion.rsplit("</think>", 1)[1] if "</think>" in completion else ""
    boxed = last_boxed(final)
    errors = []
    try:
        final_correct = bool(check_is_correct(boxed, gold)) if boxed is not None else False
    except Exception as exc:
        errors.append(f"final: {type(exc).__name__}: {exc}")
        final_correct = False
    try:
        paper_prediction = extract_answer(completion, "math")
        paper_correct = bool(check_is_correct(paper_prediction, gold))
    except Exception as exc:
        errors.append(f"paper: {type(exc).__name__}: {exc}")
        paper_prediction, paper_correct = None, False
    return dict(final_text=final, final_boxed=boxed, final_correct=final_correct,
                final_correct_terminated=final_correct and termination == "eos",
                paper_prediction=paper_prediction, paper_correct=paper_correct,
                thinking_closed="</think>" in completion, grading_errors=errors)


if __name__ == "__main__":
    # Run outside the CUDA inference process: the inherited symbolic grader
    # forks timeout workers and should not fork a loaded multi-threaded GPU job.
    import json
    payload = json.load(sys.stdin)
    print(json.dumps(grade_completion(**payload), ensure_ascii=False))
