"""Matching keys for terms, so "virtue" and "virtues" are found to be the same concept.

Only the key that decides whether two mentions are the same thing is changed. The text and
the character positions of every mention stay as they are in the transcript.

For the types named in the settings (concepts, schools, groups, languages, periods) the key
is built like this:

1. Unicode form, lowercase and single spaces, as for every other key.
2. A leading "the", "a" or "an" is dropped ("the soul" and "soul").
3. Hyphens become spaces ("self-awareness" and "self awareness").
4. The last word is lemmatized with the tables of spaCy's English model (en_core_web_sm):
   an irregular plural from its exceptions (theses, thesis); a word the model knows as a
   base noun stays as it is (analysis, ethics, logos); otherwise its suffix rules are tried
   and the first result that is a known noun is taken (virtues, virtue; senses, sense).
   An unknown word that ends in a plain "s" loses it (Platonists, Platonist), except after
   "ss", "us", "is" and "os", so Greek and Latin words like "nous" or "physis" stay.

A last word with a digit is left alone, so "1270" and "1270s" stay apart. Names of people
and places are never lemmatized: "Aquinas" must not become "Aquina".
"""

import re
from functools import lru_cache

KEEP_ENDINGS = ("ss", "us", "is", "os")


class Lemmatizer:
    """spaCy's English noun tables, loaded once."""

    def __init__(self, name: str, version: str):
        import spacy
        nlp = spacy.load(name, exclude=["parser", "ner", "tok2vec", "tagger", "attribute_ruler"])
        if nlp.meta["version"] != version:
            raise ValueError(f"the settings ask for {name} {version}, but {nlp.meta['version']} is installed")
        lookups = nlp.get_pipe("lemmatizer").lookups
        self.known = set(lookups.get_table("lemma_index").get("noun", []))
        self.exceptions = lookups.get_table("lemma_exc").get("noun", {})
        self.rules = lookups.get_table("lemma_rules").get("noun", [])

    @lru_cache(maxsize=None)
    def noun(self, word: str) -> str:
        if word in self.exceptions:
            return self.exceptions[word][0]
        if word in self.known:
            return word
        for old, new in self.rules:
            if word.endswith(old) and (candidate := word[:len(word) - len(old)] + new) in self.known:
                return candidate
        if word.endswith("s") and not word.endswith(KEEP_ENDINGS) and len(word) > 3:
            return word[:-1]
        return word


@lru_cache(maxsize=4)
def lemmatizer(name: str, version: str) -> Lemmatizer:
    return Lemmatizer(name, version)


def term_key(normalized: str, entity_type: str, settings) -> str:
    """The matching key of an already normalized surface (see the module docstring)."""
    if settings is None:
        return normalized
    words = normalized.split()
    if entity_type in settings.article_types and len(words) > 1 and words[0] in settings.articles:
        words = words[1:]
    if entity_type not in settings.lemmatize_types:
        return " ".join(words)
    words = " ".join(words).replace("-", " ").split()
    if words and not re.search(r"\d", words[-1]):
        words[-1] = lemmatizer(settings.lemmatizer, settings.lemmatizer_version).noun(words[-1])
    return " ".join(words)
