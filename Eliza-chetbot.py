#!/usr/bin/env python3
"""
ELIZA with an NLP front end (Python 3.9+, standard library only).

A rebuild of the 1966 conversation program by Joseph Weizenbaum. Where the
original matched raw text, this version first runs the input through a small
NLP pipeline:

    tokenise -> expand contractions -> lemmatise -> detect emotions + negation
    -> score sentiment -> find names -> pick the highest-ranked rule
    -> reflect pronouns with grammar-aware rules -> reassemble a reply

If spaCy and en_core_web_sm are installed, they are used for lemmas and named
entities automatically; otherwise the built-in analysers are used.

Usage:
    python eliza_nlp.py             # chat
    python eliza_nlp.py --debug     # chat and show the NLP analysis each turn
    python eliza_nlp.py --test      # run self-tests

ELIZA is a pattern matcher. It does not understand you and is not a therapist.
"""
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

# ------------------------------------------------------------------ optional spaCy
try:  # pragma: no cover - optional dependency
    import spacy
    _SPACY = spacy.load("en_core_web_sm", disable=["parser"])
except Exception:  # not installed, or model missing
    _SPACY = None

# ------------------------------------------------------------------ lexicons
EMOTIONS = {
    "sad": "sad unhappy depressed down miserable cry grief grieve hopeless empty heartbroken gloomy "
           "sorrow blue".split(),
    "anxious": "anxious worried worry nervous afraid scared fear panic stress stressed tense "
               "overwhelmed anxiety frightened terrified uneasy".split(),
    "angry": "angry mad furious annoyed hate irritated resent resentment frustrated rage "
             "livid bitter".split(),
    "lonely": "lonely alone isolated ignored abandoned neglected".split(),
    "tired": "tired exhausted drained sleepy fatigued burnout burnt weary insomnia".split(),
    "guilty": "guilty ashamed embarrassed regret shame remorse".split(),
    "confused": "confused lost unsure uncertain stuck torn puzzled".split(),
    "happy": "happy glad joy excited great wonderful proud hopeful relieved grateful cheerful "
             "delighted".split(),
}
POSITIVE = {"happy"}
NEGATORS = {"not", "never", "no", "cannot", "hardly", "without"}

IRREGULAR = {
    "was": "be", "were": "be", "am": "be", "are": "be", "is": "be", "been": "be", "being": "be",
    "had": "have", "has": "have", "did": "do", "does": "do", "done": "do", "went": "go",
    "gone": "go", "felt": "feel", "thought": "think", "knew": "know", "said": "say",
    "made": "make", "took": "take", "got": "get", "came": "come", "saw": "see", "slept": "sleep",
    "lost": "lose", "left": "leave", "kept": "keep", "told": "tell", "children": "child",
    "men": "man", "women": "woman", "wept": "weep", "hurt": "hurt",
}
VOCAB = {w for ws in EMOTIONS.values() for w in ws} | set(IRREGULAR.values()) | {
    "want", "need", "feel", "think", "believe", "remember", "dream", "love", "like", "miss"}

CONTRACTIONS = {
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would", "you're": "you are",
    "you've": "you have", "you'll": "you will", "you'd": "you would", "it's": "it is",
    "that's": "that is", "what's": "what is", "he's": "he is", "she's": "she is",
    "there's": "there is", "let's": "let us", "can't": "can not", "cannot": "can not",
    "won't": "will not", "ain't": "is not", "y'all": "you all",
}
SUBJECT_CUES = {"that", "if", "when", "because", "and", "but", "think", "believe", "know", "so",
                "whether", "while", "though", "than"}
VERB_CUES = {"are", "were", "have", "do", "did", "will", "would", "can", "could", "should", "must",
             "may", "might", "think", "know", "want", "need", "feel", "like", "love", "hate",
             "said", "say", "never", "always", "just", "really", "too", "seem", "look", "mean",
             "can", "not", "had", "cannot", "get", "got", "make", "tell", "see", "keep"}
NAME_STOP = {"I", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
             "January", "February", "March", "April", "May", "June", "July", "August",
             "September", "October", "November", "December", "Eliza", "ELIZA"}

# ------------------------------------------------------------------ NLP pipeline
def tokenize(text):
    """Lower-case, expand contractions, return word tokens (no punctuation)."""
    t = text.lower().replace("\u2019", "'")
    for short, full in CONTRACTIONS.items():
        t = re.sub(rf"\b{re.escape(short)}\b", full, t)
    t = re.sub(r"\b(\w+)n't\b", r"\1 not", t)
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", t)


def lemmatize(tok):
    """Return the base form, preferring forms the lexicon knows."""
    if tok in IRREGULAR:
        return IRREGULAR[tok]
    cands = [tok]
    if tok.endswith("ies"):
        cands.append(tok[:-3] + "y")
    if tok.endswith("ied"):
        cands.append(tok[:-3] + "y")
    if tok.endswith("es"):
        cands.append(tok[:-2])
    if tok.endswith("s"):
        cands.append(tok[:-1])
    if tok.endswith("ed"):
        cands += [tok[:-2], tok[:-1]]
        if len(tok) > 4 and tok[-3] == tok[-4]:
            cands.append(tok[:-3])
    if tok.endswith("ing"):
        cands += [tok[:-3], tok[:-3] + "e"]
        if len(tok) > 5 and tok[-4] == tok[-5]:
            cands.append(tok[:-4])
    for c in cands:
        if c in VOCAB:
            return c
    return tok


@dataclass
class Emotion:
    label: str
    word: str
    negated: bool


@dataclass
class Analysis:
    raw: str
    tokens: list
    lemmas: list
    text: str
    is_question: bool
    emotions: list = field(default_factory=list)
    sentiment: int = 0
    entities: list = field(default_factory=list)


def find_names(raw):
    """Capitalised words that are not sentence-initial (a light NER stand-in)."""
    names = []
    for m in re.finditer(r"\b[A-Z][a-z]+\b", raw):
        before = raw[:m.start()].rstrip()
        if not before or before[-1] in ".?!":
            continue
        if m.group() not in NAME_STOP and m.group() not in names:
            names.append(m.group())
    return names


def analyze(raw):
    tokens = tokenize(raw)
    lemmas = [lemmatize(t) for t in tokens]
    entities = find_names(raw)
    if _SPACY is not None:  # pragma: no cover
        doc = _SPACY(raw)
        lemmas = [t.lemma_.lower() for t in doc if t.is_alpha] or lemmas
        entities = [e.text for e in doc.ents if e.label_ in ("PERSON", "ORG", "GPE")] or entities
    a = Analysis(raw, tokens, lemmas, " ".join(tokens), raw.strip().endswith("?"),
                 entities=entities)
    for i, (tok, lem) in enumerate(zip(tokens, lemmas)):
        for label, words in EMOTIONS.items():
            if tok in words or lem in words:
                neg = any(t in NEGATORS for t in tokens[max(0, i - 3):i])
                a.emotions.append(Emotion(label, tok, neg))
                break
    for e in a.emotions:
        v = 1 if e.label in POSITIVE else -1
        a.sentiment += -v if e.negated else v
    return a


def reflect(span):
    """Swap perspective (I<->you) using grammar-aware rules, not a blind word map."""
    toks = span.split()
    out = []
    for i, t in enumerate(toks):
        prev = toks[i - 1] if i else ""
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if t in ("i", "me"):
            out.append("you")
        elif t == "my":
            out.append("your")
        elif t == "mine":
            out.append("yours")
        elif t == "myself":
            out.append("yourself")
        elif t == "am":
            out.append("are")
        elif t == "was" and prev == "i":
            out.append("were")
        elif t == "were" and prev == "you":
            out.append("was")
        elif t == "are" and prev == "you":
            out.append("am")
        elif t == "your":
            out.append("my")
        elif t == "yours":
            out.append("mine")
        elif t == "yourself":
            out.append("myself")
        elif t == "you":
            subject = prev in SUBJECT_CUES or nxt in VERB_CUES or (i == 0 and bool(nxt))
            out.append("I" if subject else "me")
        else:
            out.append(t)
    return " ".join(out)


# ------------------------------------------------------------------ script
CRISIS = re.compile(r"\b(kill myself|suicid\w*|end my life|want to die|hurt myself|harm myself|"
                    r"self harm|do not want to live|no reason to live)\b")
CRISIS_REPLY = ("I am going to step out of the script. What you just said matters, and a "
                "pattern-matching program is not the right support. Please reach out to someone you "
                "trust, or contact a local crisis line or emergency service now. If you are in "
                "danger, call your local emergency number.")

# (rank, name, regex over normalised text, reassembly rules; {0} = reflected capture)
RULES = [
    (90, "farewell", r"\b(?:bye|goodbye|quit|exit)\b",
     ["Goodbye. Thank you for talking with me.", "Goodbye. Take care of yourself."]),
    (55, "feel", r"\bi (?:feel|am feeling|have been feeling) (.+)",
     ["Tell me more about feeling {0}.", "Do you often feel {0}?",
      "When you feel {0}, what do you do?"]),
    (51, "want-to", r"\bi want to (.+)",
     ["Why do you want to {0}?", "What would it mean to you to {0}?",
      "What is stopping you from trying to {0}?"]),
    (50, "need", r"\bi need (.+)",
     ["Why do you need {0}?", "Would it really help you to get {0}?",
      "Are you sure you need {0}?"]),
    (50, "want", r"\bi want (.+)",
     ["What would it mean to you if you got {0}?", "Why do you want {0}?",
      "Suppose you got {0} soon. What then?"]),
    (45, "am", r"\bi am (.+)",
     ["How long have you been {0}?", "Do you believe it is normal to be {0}?",
      "Do you enjoy being {0}?", "Why do you tell me you are {0}?"]),
    (45, "cannot", r"\bi can not (.+)",
     ["How do you know you cannot {0}?", "Perhaps you could {0} if you tried.",
      "What would it take for you to {0}?"]),
    (42, "think", r"\bi (?:think|believe) (.+)",
     ["Do you really think so?", "But you are not sure {0}?", "Why do you think {0}?"]),
    (41, "remember", r"\bi (?:remember|recall) (.+)",
     ["Do you often think of {0}?", "What else does thinking of {0} bring to mind?",
      "Why do you recall {0} right now?"]),
    (41, "i-verb-you", r"\bi (?:love|hate|like|miss|trust|fear|admire|respect|envy|dislike) you\b",
     ["Why do you say you {V} me?", "We were discussing you, not me."]),
    (40, "family", r"\b(mother|mom|father|dad|sister|brother|wife|husband|family|parents|son|"
                   r"daughter|girlfriend|boyfriend|friend)\b",
     ["Tell me more about your {0}.", "Who else in your life comes to mind?",
      "How do you feel about your {0}?"]),
    (38, "dream", r"\b(?:dream|dreamed|dreamt|nightmare)\w*",
     ["What does that dream suggest to you?", "Do you dream often?",
      "What persons appear in your dreams?"]),
    (36, "sorry", r"\bsorry\b",
     ["Please do not apologize.", "Apologies are not necessary.",
      "What feelings do you have when you apologize?"]),
    (35, "machine", r"\b(?:computer|machine|program|robot|ai|chatbot)\b",
     ["Do machines worry you?", "Why do you mention machines?",
      "What do you think machines have to do with your problem?"]),
    (34, "you-are", r"\byou are (.+)",
     ["What makes you think I am {0}?", "Does it please you to believe I am {0}?",
      "Perhaps you would like to be {0}."]),
    (30, "my", r"\bmy (.+)",
     ["Your {0}?", "Why do you say your {0}?", "Is it important to you that your {0}?",
      "Does that suggest anything else which belongs to you?"]),
    (25, "because", r"\bbecause\b",
     ["Is that the real reason?", "What other reasons come to mind?",
      "Does that reason explain anything else?"]),
    (22, "absolutes", r"\b(?:always|never|everyone|everybody|nobody|no one)\b",
     ["Can you think of a specific example?", "When?", "Really, always?"]),
    (20, "maybe", r"\b(?:maybe|perhaps|possibly)\b",
     ["You do not seem quite certain.", "Why the uncertain tone?"]),
    (15, "yes", r"^(?:yes|yeah|yep)\b",
     ["You seem quite sure.", "I see.", "Can you say more about that?"]),
    (15, "no", r"^(?:no|nope)\b",
     ["Why not?", "You are being a bit negative.", "Does saying no matter to you?"]),
    (12, "hello", r"\b(?:hello|hi|hey)\b",
     ["Hello. What is on your mind?", "Hi. Please tell me what brings you here."]),
]
COMPILED = [(r, n, re.compile(p), out) for r, n, p, out in RULES]

EMOTION_REPLIES = {
    "sad": ["I am sorry to hear you feel sad. What do you think is behind it?",
            "How long have you been feeling sad?",
            "When you feel sad, what usually happens next?"],
    "anxious": ["It sounds like you are feeling anxious. What feels most uncertain right now?",
                "What do you imagine might happen?",
                "When did you first notice feeling anxious?"],
    "angry": ["You sound angry. What happened just before you felt that way?",
              "What would you like to be different?",
              "Who or what is that anger directed at?"],
    "lonely": ["Feeling lonely can be hard. When do you notice it most?",
               "Who would you like to feel closer to?",
               "What does being alone mean to you?"],
    "tired": ["You sound tired. Is it more your body or your mind?",
              "How long have you felt this worn out?",
              "What takes the most energy from you lately?"],
    "guilty": ["What do you feel responsible for?",
               "Who do you feel you have let down?",
               "If a friend told you this, what would you say to them?"],
    "confused": ["What part feels most unclear?",
                 "What would a clear answer look like to you?",
                 "What have you already tried to sort it out?"],
    "happy": ["I am glad to hear that. What brought it on?",
              "What does feeling happy mean to you?",
              "Has anything else been going well?"],
}
FALLBACK = ["Please go on.", "Tell me more.", "I see.", "Why do you say that?",
            "Can you elaborate on that?", "That is interesting. Please continue.",
            "How does that make you feel?"]
QUESTION = ["Why do you ask?", "Does that question interest you?", "What do you think?"]
REPEAT = ["You already said that. Is it important to you?",
          "You are repeating yourself. What do you think that means?"]
GREETING = "How do you do. Please tell me what is on your mind."


# ------------------------------------------------------------------ the bot
class Eliza:
    def __init__(self, debug=False):
        self.debug = debug
        self.memory = []
        self.counters = Counter()
        self.emotion_counts = Counter()
        self.asked = set()
        self.last = None
        self.ended = False
        self.trace = []

    def _pick(self, key, options):
        n = self.counters[key]
        self.counters[key] += 1
        return options[n % len(options)]

    def reply(self, raw):
        a = analyze(raw)
        self.trace = [f"tokens     {a.tokens}", f"lemmas     {a.lemmas}",
                      f"emotions   {[(e.label, e.word, 'negated' if e.negated else '') for e in a.emotions]}",
                      f"sentiment  {a.sentiment:+d}", f"entities   {a.entities}"]
        if CRISIS.search(a.text):
            self.trace.append("rule       crisis (breaks character)")
            return CRISIS_REPLY
        if not a.tokens:
            return "Please say something."
        if self.last == a.text:
            self.trace.append("rule       repeat")
            return self._pick("repeat", REPEAT)
        self.last = a.text

        # candidates: (rank, name, reply-builder)
        cands = []
        for rank, name, rx, outs in COMPILED:
            m = rx.search(a.text)
            if not m:
                continue
            grp = reflect(m.group(1)) if m.groups() else ""
            if name == "i-verb-you":
                grp = m.group(0).split()[1]
            cands.append((rank, name, outs, grp))
        for e in a.emotions:
            if e.negated:
                cands.append((58, "negated-emotion",
                              [f"You say you are not {e.label}. What made you mention it?"], ""))
            elif e.label in POSITIVE:
                cands.append((40, "emotion", EMOTION_REPLIES[e.label], ""))
                self.emotion_counts[e.label] += 1
            else:
                self.emotion_counts[e.label] += 1
                if self.emotion_counts[e.label] >= 3:
                    cands.append((62, "emotion-recurring",
                                  [f"You have brought up feeling {e.label} several times now. "
                                   "What do you think keeps bringing it back?"], ""))
                else:
                    cands.append((60, "emotion", EMOTION_REPLIES[e.label], ""))
            break
        for name in a.entities:
            if name not in self.asked:
                cands.append((28, "entity", [f"Who is {name} to you?"], ""))
                self.asked.add(name)
                break

        if cands:
            rank, name, outs, grp = max(cands, key=lambda c: c[0])
            key = name + (a.emotions[0].label if name.startswith("emotion") and a.emotions else "")
            text = self._pick(key, outs).replace("{0}", grp).replace("{V}", grp)
            if name == "my" and grp:
                self.memory.append(grp)
            if name == "farewell":
                self.ended = True
            self.trace.append(f"rule       {name} (rank {rank})")
            return self._finish(text)

        self.trace.append("rule       fallback")
        if a.is_question:
            return self._pick("question", QUESTION)
        if self.memory and self.counters["fallback"] % 3 == 2:
            self.counters["fallback"] += 1
            return self._finish(f"Earlier you said your {self.memory.pop(0)}. Tell me more about that.")
        return self._pick("fallback", FALLBACK)

    @staticmethod
    def _finish(text):
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([?.!,])", r"\1", text)
        return text[:1].upper() + text[1:]


# ------------------------------------------------------------------ self-test
TESTS = [
    ("I am sad", "sad"),
    ("I feel really worried about my exam", "anxious"),
    ("I need a vacation", "Why do you need a vacation?"),
    ("I am not happy", "You say you are not happy"),
    ("My mother never listens to me", "Tell me more about your mother."),
    ("I can't sleep", "cannot sleep"),
    ("I am a teacher", "How long have you been a teacher?"),
    ("You are rude", "What makes you think I am rude?"),
    ("I love you", "Why do you say you love me?"),
    ("I need you", "Why do you need you?"),
    ("I talked to Sam yesterday", "Who is Sam to you?"),
    ("I want to travel more", "Why do you want to travel more?"),
    ("I want to kill myself", "step out of the script"),
    ("I was crying all night", "sad"),
]


def self_test():
    bad = 0
    for text, want in TESTS:
        got = Eliza().reply(text)
        ok = want.lower() in got.lower()
        if text == "I need you":  # reflection check: object 'you' becomes 'me'
            ok = got == "Why do you need me?"
        bad += not ok
        print(("PASS " if ok else "FAIL ") + f"{text!r} -> {got!r}")
    bot = Eliza()
    bot.reply("It rains")
    ok = "already" in bot.reply("It rains").lower()
    bad += not ok
    print(("PASS " if ok else "FAIL ") + "repeat detection")
    bot = Eliza()
    bot.reply("goodbye")
    ok = bot.ended
    bad += not ok
    print(("PASS " if ok else "FAIL ") + "farewell ends session")
    print(f"\n{len(TESTS) + 2 - bad}/{len(TESTS) + 2} passed")
    return bad


def main():
    args = sys.argv[1:]
    if "--test" in args:
        sys.exit(1 if self_test() else 0)
    bot = Eliza(debug="--debug" in args)
    print("ELIZA, after Joseph Weizenbaum (1966). Type a sentence; 'quit' to leave.")
    print("(Pattern matching, not a therapist.)\n")
    print("eliza> " + GREETING)
    while not bot.ended:
        try:
            line = input("you>   ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        answer = bot.reply(line)
        if bot.debug:
            print("\n".join("  [nlp] " + t for t in bot.trace))
        print("eliza> " + answer)


if __name__ == "__main__":
    main()