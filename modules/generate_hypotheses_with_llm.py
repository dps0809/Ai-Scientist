import os
import json
import requests
# pyrefly: ignore [missing-import]
from .config import OLLAMA_MODEL, OLLAMA_URL, OUTPUT_DIR


def generate_hypotheses_with_llm(context):
    print("\n" + "=" * 70)
    print("9. LOCAL LLM HYPOTHESIS GENERATION")
    print("=" * 70)

    evidence = context.get("discovered_evidence", [])
    evidence_text = json.dumps(evidence, indent=2)

    prompt = f"""
You are a scientific hypothesis generation assistant.

A Python analysis system has already analyzed a
diabetes dataset.

The system discovered the following evidence:

{evidence_text}

Generate exactly 5 testable hypotheses.

IMPORTANT RULES:

1. Use ONLY the evidence provided above.
2. Do NOT invent relationships.
3. Do NOT claim causation.
4. Each hypothesis must be specific.
5. Each hypothesis must be testable using the dataset.
6. Keep each hypothesis concise.
7. Explain the evidence supporting each hypothesis.
8. Suggest one simple experiment to test each hypothesis.
9. Return ONLY valid JSON.
10. Do NOT use markdown.
11. Do NOT write anything before or after the JSON.

Return exactly this structure:

[
  {{
    "hypothesis_id": "H1",
    "hypothesis": "One specific testable hypothesis.",
    "evidence_from_data": "Evidence supporting the hypothesis.",
    "proposed_experiment": "A simple experiment that can test it."
  }},
  {{
    "hypothesis_id": "H2",
    "hypothesis": "One specific testable hypothesis.",
    "evidence_from_data": "Evidence supporting the hypothesis.",
    "proposed_experiment": "A simple experiment that can test it."
  }},
  {{
    "hypothesis_id": "H3",
    "hypothesis": "One specific testable hypothesis.",
    "evidence_from_data": "Evidence supporting the hypothesis.",
    "proposed_experiment": "A simple experiment that can test it."
  }}
]
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 8192
        }
    }

    text = ""
    try:
        print("\nSending discovered evidence to local LLM...")
        print("Waiting for LLM response...")

        response = requests.post(OLLAMA_URL, json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()
        text = result.get("response", "").strip()

        if not text:
            print("\nLLM returned an empty response.")
            return []

        print("\nRaw LLM response:")
        print(text)

        # ----------------------------------------------------------------
        # Extract the JSON array robustly.
        # The model may prepend chain-of-thought reasoning before the JSON.
        # _extract_json_array() finds the LAST complete '[...]' in the text.
        # ----------------------------------------------------------------
        hypotheses = _extract_json_array(text)

        if hypotheses is None:
            print("\n[ERROR] Could not extract a valid JSON array from LLM output.")
            return []

        if not isinstance(hypotheses, list):
            print("\nLLM response was not a JSON list.")
            return []

        print("\n" + "=" * 70)
        print("GENERATED HYPOTHESES")
        print("=" * 70)

        for hypothesis in hypotheses:
            print(f"\n{hypothesis.get('hypothesis_id', 'H?')}")
            print("-" * 70)
            print("Hypothesis:")
            print(hypothesis.get("hypothesis", "N/A"))
            print("\nEvidence:")
            print(hypothesis.get("evidence_from_data", "N/A"))
            print("\nProposed Experiment:")
            print(hypothesis.get("proposed_experiment", "N/A"))

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "generated_hypotheses.json")
        with open(output_path, "w") as f:
            json.dump(hypotheses, f, indent=4)

        print("\nHypotheses saved successfully:")
        print(output_path)
        return hypotheses

    except requests.exceptions.Timeout:
        print("\nLLM generation timed out.")
        print("Ollama is running, but the model did not respond within the timeout.")
        return []
    except requests.exceptions.ConnectionError:
        print("\nCould not connect to Ollama.")
        print(f"Make sure Ollama is running and '{OLLAMA_MODEL}' is installed.")
        return []
    except Exception as e:
        print("\nLLM generation failed:")
        print(e)
        return []


def _extract_json_array(text: str):
    """
    Find the LAST complete JSON array in *text* and return the parsed list.
    Returns None if no valid array is found.

    This handles models that prepend chain-of-thought reasoning before the
    actual JSON output — we scan backwards for the final '[...]' block.
    """
    # Strip markdown code fences if present
    text = text.replace("```json", "").replace("```", "")

    # Work backwards: find the last ']' then walk left to its matching '['
    last_close = text.rfind("]")
    if last_close == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    i = last_close
    while i >= 0:
        ch = text[i]
        # Handle escape sequences inside strings (scanning backwards)
        if i > 0 and text[i - 1] == "\\" and in_string:
            i -= 1
            continue
        if ch == '"':
            in_string = not in_string
        if not in_string:
            if ch == "]":
                depth += 1
            elif ch == "[":
                depth -= 1
                if depth == 0:
                    candidate = text[i:last_close + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        return None
        i -= 1
    return None
