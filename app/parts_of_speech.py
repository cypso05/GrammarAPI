# parts_of_speech.py

# List (or dictionary) of common adjectives
ADJECTIVES = [
    "happy", "sad", "big", "small", "bright", "dark", "tall", "short",
    "old", "young", "good", "bad", "beautiful", "ugly"
]

# List of common adverbs
ADVERBS = [
    "quickly", "slowly", "silently", "loudly", "happily", "sadly",
    "gracefully", "eagerly", "badly", "well"
]

# List of common interjections
INTERJECTIONS = [
    "oh", "wow", "ouch", "oops", "hey", "yikes", "alas", "bravo"
]

# Irregular verbs dictionary
IRREGULAR_VERBS = {
    "took": "take", "taken": "take", "thought": "think",
    # ... (other verb mappings)
}

# Pronouns dictionary
# parts_of_speech.py

PRONOUNS = {
    "i": "I", "me": "me", "my": "my", "mine": "mine",
    "you": "you", "your": "your", "yours": "yours", "he": "he",
    "him": "him", "his": "his", "she": "she", "her": "her",
    "hers": "hers", "it": "it", "its": "its", "we": "we",
    "us": "us", "our": "our", "ours": "ours", "they": "they",
    "them": "them", "their": "their", "theirs": "theirs",
    "myself": "myself", "yourself": "yourself", "himself": "himself",
    "herself": "herself", "itself": "itself", "ourselves": "ourselves",
    "yourselves": "yourselves", "themselves": "themselves",
    "who": "who", "whom": "whom", "whose": "whose", "which": "which",
    "that": "that", "this": "this", "these": "these", "those": "those"
}


# Conjunctions dictionary
CONJUNCTIONS = {
    "and": ["&", "as well as"], "but": ["however", "yet"], "or": ["either", "otherwise"],
    "nor": ["neither"], "for": ["because"], "so": ["therefore", "thus"], "yet": ["however"]
}

# Prepositions dictionary
PREPOSITIONS = {
    "about": ["regarding", "concerning"], "above": ["over"], "across": ["through"],
    "after": ["following"], "against": ["opposed to"], "along": ["beside"],
    "among": ["amid", "between"], "around": ["about"], "at": ["in", "on"],
    "before": ["prior to"], "behind": ["beyond"], "below": ["under"],
    "beneath": ["under"], "beside": ["next to"], "between": ["among"],
    "beyond": ["outside"], "by": ["via"], "despite": ["in spite of"],
    "down": ["below"], "during": ["throughout"], "except": ["excluding"],
    "for": ["on behalf of"], "from": ["out of"], "in": ["inside", "within"],
    "inside": ["in"], "into": ["to"], "like": ["similar to"], "near": ["close to"],
    "of": ["about"], "off": ["away from"], "on": ["upon", "onto"], "onto": ["on"],
    "out": ["outside"], "over": ["above"], "past": ["beyond"], "since": ["from"],
    "through": ["via"], "to": ["toward", "until"], "toward": ["to"], "under": ["beneath"],
    "until": ["till"], "up": ["above"], "upon": ["on"], "with": ["using"],
    "within": ["inside"], "without": ["lacking"]
}

# Determiners dictionary
DETERMINERS = {
    "this": "this", "that": "that", "these": "these", "those": "those",
    "each": "each", "every": "every", "some": "some", "any": "any",
    "many": "many", "few": "few", "all": "all", "both": "both", "several": "several"
}

# Interjections dictionary
INTERJECTIONS_DICT = {
    "oh": "oh", "wow": "wow", "ouch": "ouch", "hey": "hey", "oops": "oops"
}

# Articles dictionary
ARTICLES = {
    "a": ["an", "the"], "an": ["a", "the"], "the": ["a", "an"]
}

# Contractions dictionary
CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "don't": "do not",
    # ... etc.
}

# Common errors dictionary
COMMON_ERRORS = {
    "your": "you're", "there": "their",
    # ... etc.
}

# Medical terms dictionary
MEDICAL_TERMS = {
    "bp": "blood pressure",
    # ... etc.
}

# UK to US spelling dictionary
UK_US_SPELLING = {
    "colour": "color",
    # ... etc.
}
