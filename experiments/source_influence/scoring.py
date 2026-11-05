"""GSM8K scoring as in lm-eval gsm8k_cot (strict-match / flexible-extract) plus format diagnostics."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baton"))
import re

STRICT = re.compile(r"The answer is (\-?[0-9\.\,]+).")
FLEX = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
IGNORE = [re.compile(r","), re.compile(r"\$"), re.compile(r"(?s).*#### "), re.compile(r"\.$")]
STOPS = ("Q:",)


def norm(s):
    s = s.strip()
    for r in IGNORE:
        s = r.sub("", s)
    return s.strip().lower()


def strict_extract(text):
    m = STRICT.findall(text)
    return m[0] if m else "[invalid]"


def flex_extract(text):
    m = FLEX.findall(text)
    if not m:
        return "[invalid]"
    a, b = m[-1]
    return a or b


def score(text, gold):
    s, f = strict_extract(text), flex_extract(text)
    g = norm(gold)
    answers = [norm(x) for x in STRICT.findall(text)]
    first_ans = text.find("The answer is")
    first_eq = text.find("=")
    return {
        "strict": s, "flex": f,
        "strict_ok": s != "[invalid]" and norm(s) == g,
        "flex_ok": f != "[invalid]" and norm(f) == g,
        "n_answer_stmts": len(answers),
        "conflicting": len(set(answers)) > 1,
        "answer_first": first_ans >= 0 and (first_eq < 0 or first_ans < first_eq),
        "no_answer": f == "[invalid]",
    }


def cut_at_stop(text):
    idx = [text.find(s) for s in STOPS if s in text]
    return text[: min(idx)] if idx else text
