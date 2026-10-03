import os
import json
# pyrefly: ignore [missing-import]
from .config import OUTPUT_DIR

def save_patterns(patterns):
    print("\n" + "=" * 70)
    print("8. DISCOVERED PATTERNS")
    print("=" * 70)

    path = os.path.join(OUTPUT_DIR, "discovered_patterns.json")
    with open(path, "w") as f:
        json.dump(patterns, f, indent=4)

    print(f"\nTotal discovered patterns: {len(patterns)}")

    for i, pattern in enumerate(patterns, 1):
        print(f"\nPattern {i}:")
        print(json.dumps(pattern, indent=2))

    return patterns
