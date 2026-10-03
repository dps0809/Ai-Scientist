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

Generate exactly 3 testable hypotheses.

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
            "num_predict": 1500
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

        text = text.replace("```json", "").replace("```", "").strip()
        hypotheses = json.loads(text)

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
    except json.JSONDecodeError:
        print("\nLLM returned invalid JSON.")
        print("\nRaw response:")
        print(text)
        return []
    except Exception as e:
        print("\nLLM generation failed:")
        print(e)
        return []
