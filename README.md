# ELIZA Chatbot (with NLP Front End)

A Python implementation of Joseph Weizenbaum's 1966 **ELIZA** conversation program, enhanced with a lightweight Natural Language Processing (NLP) front end.

> **Note:** I built this project at university as a coursework/academic project.

## Overview

Where the original ELIZA matched raw text directly, this version processes each user input through a small NLP pipeline before selecting a response:

```text
tokenise -> expand contractions -> lemmatise -> detect emotions + negation
-> score sentiment -> find names -> pick the highest-ranked rule
-> reflect pronouns with grammar-aware rules -> reassemble a reply
```

## Features

- **Zero-dependency core:** Runs on **Python 3.9+** using only the standard library.
- **Optional spaCy integration:** Automatically uses `spaCy` (`en_core_web_sm`) for lemmatization and named-entity recognition (`PERSON`, `ORG`, `GPE`) if installed; otherwise falls back to built-in analyzers.
- **Emotion & Negation Detection:** Recognizes 8 emotion categories (`sad`, `anxious`, `angry`, `lonely`, `tired`, `guilty`, `confused`, `happy`), checks for nearby negators, and tracks recurring emotions across the session.
- **Grammar-Aware Pronoun Reflection:** Reflects pronouns (`I` ↔ `you`, `my` ↔ `your`, etc.) while distinguishing subject vs. object `you` (`I` vs. `me`).
- **Conversational Memory & Flow:** Remembers `"my ..."` topics for later fallback prompts, detects repeated inputs, and handles farewells cleanly.
- **Crisis Safety Guardrail:** Steps out of character to provide a supportive crisis-resources message if self-harm or crisis keywords are detected.
- **Debug & Self-Test Modes:** Includes a per-turn NLP trace flag and a built-in test suite.

## Requirements

- **Python 3.9+**
- *(Optional)* `spaCy` with the `en_core_web_sm` model:
  ```bash
  pip install spacy
  python -m spacy download en_core_web_sm
  ```

## Usage

Run the chatbot from the project directory:

```bash
# Start interactive chat
python Eliza-chetbot.py

# Start chat and print the NLP analysis trace on each turn
python Eliza-chetbot.py --debug

# Run the built-in self-tests
python Eliza-chetbot.py --test
```

Type `quit`, `exit`, `bye`, or `goodbye` to end a chat session.

## Disclaimer

ELIZA is a rule-based pattern matcher. It does not understand language and is **not** a substitute for a therapist or professional support.
