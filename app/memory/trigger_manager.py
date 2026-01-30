"""
TriggerManager: Data-driven classification system for Phase 6.4
Replaces hardcoded keyword logic with CSV-based trigger word matching.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Optional
import re


class TriggerManager:
    """
    Dynamic trigger-based classification system.
    Loads trigger words from CSV files and provides semantic matching.

    Key Features:
    - Word-level matching (not phrase-level)
    - Data-driven classification
    - Supports memory classification and adaptation detection
    """

    def __init__(self, data_dir: str = "app/memory/data"):
        """
        Initialize TriggerManager with trigger word datasets.

        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self.triggers = self._load_triggers()

    def _load_triggers(self) -> Dict:
        """
        Load triggers from CSV files.

        Returns:
            Dictionary structure:
            {
                "long_term": {
                    "identity": {"am", "is", "are", ...},
                    "relationship": {...},
                    ...
                },
                "short_term": {
                    "emotion": {"happy", "sad", "scared", ...},
                    "activity": {...},
                    ...
                },
                "noise": {"hi", "hello", "ok", ...},
                "adaptation": {
                    "vulnerability": {"scared", "worried", "anxious", ...},
                    "gratitude": {"thank", "grateful", ...},
                    ...
                }
            }
        """
        triggers = {
            "long_term": {},
            "short_term": {},
            "noise": set(),
            "adaptation": {}
        }

        # Load full dataset
        try:
            full_dataset_path = self.data_dir / "full_dataset.csv"
            if full_dataset_path.exists():
                df = pd.read_csv(full_dataset_path)

                # Process each row
                for _, row in df.iterrows():
                    category = row["Category"].lower().replace("_", "")
                    subcategory = row["Subcategory"].lower()
                    word = str(row["Word/Phrase"]).lower().strip()
                    notes = str(row.get("Notes", "")).lower()

                    # Handle LONG_TERM triggers
                    if category == "longterm":
                        if subcategory not in triggers["long_term"]:
                            triggers["long_term"][subcategory] = set()
                        triggers["long_term"][subcategory].add(word)

                    # Handle SHORT_TERM triggers
                    elif category == "shortterm":
                        if subcategory not in triggers["short_term"]:
                            triggers["short_term"][subcategory] = set()
                        triggers["short_term"][subcategory].add(word)

                        # Build adaptation triggers from emotion subcategory
                        if subcategory == "emotion":
                            # Vulnerability: emotions with "fear" notes
                            if "fear" in notes:
                                if "vulnerability" not in triggers["adaptation"]:
                                    triggers["adaptation"]["vulnerability"] = set()
                                triggers["adaptation"]["vulnerability"].add(word)

                            # Gratitude: emotions with "gratitude" notes
                            elif "gratitude" in notes or "thank" in notes:
                                if "gratitude" not in triggers["adaptation"]:
                                    triggers["adaptation"]["gratitude"] = set()
                                triggers["adaptation"]["gratitude"].add(word)

                    # Handle NOISE triggers
                    elif category == "noise":
                        triggers["noise"].add(word)

                        # Check for adaptation triggers in noise category
                        # (e.g., "thanks", "thank you" have gratitude notes)
                        if "gratitude" in notes or "thank" in notes:
                            if "gratitude" not in triggers["adaptation"]:
                                triggers["adaptation"]["gratitude"] = set()
                            triggers["adaptation"]["gratitude"].add(word)

        except Exception as e:
            print(f"Warning: Error loading triggers from {self.data_dir}: {e}")
            # Fallback to minimal triggers to prevent complete failure
            triggers["adaptation"]["vulnerability"] = {
                "scared", "worried", "anxious", "afraid", "nervous", "fear", "panic"
            }
            triggers["adaptation"]["gratitude"] = {
                "thank", "thanks", "grateful", "appreciate"
            }

        return triggers

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        CRITICAL: Word-level matching (not phrase-level).
        "I feel scared" → ["i", "feel", "scared"]

        Args:
            text: Input text

        Returns:
            List of lowercase words
        """
        words = re.findall(r'\b\w+\b', text.lower())
        return words

    def _semantic_match(self, text: str, trigger_words: Set[str]) -> List[str]:
        """
        Match words and phrases semantically.

        CRITICAL FIX: "I feel scared" should match "scared" trigger,
        not require exact phrase "i'm scared" or "i am scared".

        Supports both:
        - Single word triggers: "scared" matches in "I feel scared"
        - Multi-word triggers: "thank you" matches in "Thank you so much"

        Args:
            text: Input text
            trigger_words: Set of trigger words/phrases to match

        Returns:
            List of matched words/phrases
        """
        text_lower = text.lower()
        tokens = self._tokenize(text)
        found = []

        # Check each trigger (can be single word or multi-word phrase)
        for trigger in trigger_words:
            # Multi-word trigger: check if phrase exists in text
            if ' ' in trigger:
                if trigger in text_lower:
                    found.append(trigger)
            # Single-word trigger: check if word in tokens
            else:
                if trigger in tokens:
                    found.append(trigger)

        return found

    def classify(self, text: str) -> Dict:
        """
        Classify text using trigger word matching.

        Args:
            text: Input text to classify

        Returns:
            {
                "category": "long_term" | "short_term" | "noise",
                "subcategory": "identity" | "emotion" | etc.,
                "triggers_found": ["word1", "word2"]
            }
        """
        # Check noise first (lowest priority)
        noise_matches = self._semantic_match(text, self.triggers["noise"])

        # Check long-term triggers
        long_term_matches = {}
        for subcategory, words in self.triggers["long_term"].items():
            matches = self._semantic_match(text, words)
            if matches:
                long_term_matches[subcategory] = matches

        # Check short-term triggers
        short_term_matches = {}
        for subcategory, words in self.triggers["short_term"].items():
            matches = self._semantic_match(text, words)
            if matches:
                short_term_matches[subcategory] = matches

        # Determine category (priority: long_term > short_term > noise)
        if long_term_matches:
            # Return first matching subcategory
            subcategory = list(long_term_matches.keys())[0]
            return {
                "category": "long_term",
                "subcategory": subcategory,
                "triggers_found": long_term_matches[subcategory]
            }
        elif short_term_matches:
            # Return first matching subcategory
            subcategory = list(short_term_matches.keys())[0]
            return {
                "category": "short_term",
                "subcategory": subcategory,
                "triggers_found": short_term_matches[subcategory]
            }
        elif noise_matches:
            return {
                "category": "noise",
                "subcategory": None,
                "triggers_found": noise_matches
            }
        else:
            # Default to short_term if no matches
            return {
                "category": "short_term",
                "subcategory": None,
                "triggers_found": []
            }

    def check_adaptation_trigger(self, text: str) -> Dict:
        """
        Check for emotional/behavioral triggers for adaptation.

        CRITICAL: Uses word-level matching.
        "I feel scared" → tokens: ["i", "feel", "scared"]
        Check: "scared" in vulnerability triggers → True

        Args:
            text: User input text

        Returns:
            {
                "dimension": "warmth" | "formality" | "initiative" | None,
                "delta": float (0.05, 0.10, etc.),
                "triggers_found": ["word1", "word2"]
            }
        """
        # Check vulnerability triggers (increase warmth)
        if "vulnerability" in self.triggers["adaptation"]:
            vuln_matches = self._semantic_match(
                text, self.triggers["adaptation"]["vulnerability"]
            )
            if vuln_matches:
                return {
                    "dimension": "warmth",
                    "delta": 0.10,  # Significant warmth increase
                    "triggers_found": vuln_matches
                }

        # Check gratitude triggers (increase initiative)
        if "gratitude" in self.triggers["adaptation"]:
            grat_matches = self._semantic_match(
                text, self.triggers["adaptation"]["gratitude"]
            )
            if grat_matches:
                return {
                    "dimension": "initiative",
                    "delta": 0.05,  # Moderate initiative increase
                    "triggers_found": grat_matches
                }

        # No adaptation trigger found
        return {
            "dimension": None,
            "delta": 0.0,
            "triggers_found": []
        }


# Singleton instance
_trigger_manager_instance: Optional[TriggerManager] = None


def get_trigger_manager() -> TriggerManager:
    """
    Get singleton TriggerManager instance.

    Returns:
        Singleton TriggerManager
    """
    global _trigger_manager_instance
    if _trigger_manager_instance is None:
        _trigger_manager_instance = TriggerManager()
    return _trigger_manager_instance
