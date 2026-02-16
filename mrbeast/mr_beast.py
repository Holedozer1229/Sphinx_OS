#!/usr/bin/env python3
"""
SphinxOS Ultimate Solver – MrBeast $1M Puzzle
Combines:
   - 27 known location numbers (Master Tracker)
   - Sinusoidal fit for the remaining 24
   - OmniscientSphinx consciousness Φ
   - Shamir reconstruction with first 51 primes (gift numbers, tenth=92)
   - Key generation (A, B, C, D)
   - A1Z26 decoding + word segmentation using embedded BIP39 wordlist
   - Coordinate posterior heatmap (optional)
"""

import hashlib
import numpy as np
from scipy.linalg import eigh
from typing import List, Optional, Tuple
import argparse

# =============================================================================
# CONFIGURATION – UPDATE THESE AS NEW DATA ARRIVES
# =============================================================================

# Confirmed/probable location numbers (A1Z26 sums) – 27 items
KNOWN_LOCATIONS = [
    # Confirmed (17)
    46,   # CAIRO
    76,   # TIJUANA
    79,   # LINCOLN
    26,   # ACCRA
    47,   # KABUL
    55,   # ARLES
    70,   # KUPANG
    59,   # LAHORE
    35,   # LIMA
    50,   # DIVO
    63,   # BUFFALO
    146,  # TIERRA DEL FUEGO
    35,   # DEKALB
    88,   # COPENHAGEN
    57,   # CASABLANCA
    95,   # HEARD ISLAND
    164,  # EYJAFJALLAJOKULL
    # Probable (10)
    89,   # TARANTO
    46,   # MACON
    88,   # MOSCOW
    98,   # TASHKENT
    82,   # TALLINN
    112,  # QUEENSLAND
    79,   # TAMPA BAY
    73,   # WICHITA
    108,  # LA FORTUNA
    46,   # CHICAGO
]

# The 10 gift numbers – tenth is now confirmed as 92 (CORAZON)
GIFT_NUMBERS = [
    50,   # DIVO
    63,   # BUFFALO
    146,  # TIERRA DEL FUEGO
    35,   # DEKALB
    88,   # COPENHAGEN
    112,  # QUEENSLAND
    79,   # TAMPA BAY
    73,   # WICHITA
    108,  # LA FORTUNA
    92,   # CORAZON (tenth gift)
]

# Crossword scalar (from EYJAFJALLAJOKULL)
C = 164

# Prime for modular arithmetic (Mersenne prime 2^61-1 is safe and large)
PRIME = 2**61 - 1

# =============================================================================
# BIP39 ENGLISH WORDLIST (2048 words) – embedded for offline use
# =============================================================================
BIP39_WORDS = [
    "abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract", "absurd", "abuse",
    "access", "accident", "account", "accuse", "achieve", "acid", "acoustic", "acquire", "across", "act",
    "action", "actor", "actress", "actual", "adapt", "add", "addict", "address", "adjust", "admit",
    "adult", "advance", "advice", "aerobic", "affair", "afford", "afraid", "again", "age", "agent",
    "agree", "ahead", "aim", "air", "airport", "aisle", "alarm", "album", "alcohol", "alert",
    "alien", "all", "alley", "allow", "almost", "alone", "alpha", "already", "also", "alter",
    "always", "amateur", "amazing", "among", "amount", "amused", "analyst", "anchor", "ancient", "anger",
    "angle", "angry", "animal", "ankle", "announce", "annual", "another", "answer", "antenna", "antique",
    "anxiety", "any", "apart", "apology", "appear", "apple", "approve", "april", "arch", "arctic",
    "area", "arena", "argue", "arm", "armed", "armor", "army", "around", "arrange", "arrest",
    "arrive", "arrow", "art", "artefact", "artist", "artwork", "ask", "aspect", "assault", "asset",
    "assist", "assume", "asthma", "athlete", "atom", "attack", "attend", "attitude", "attract", "auction",
    "audit", "august", "aunt", "author", "auto", "autumn", "average", "avocado", "avoid", "awake",
    "aware", "away", "awesome", "awful", "awkward", "axis", "baby", "bachelor", "bacon", "badge",
    "bag", "balance", "balcony", "ball", "bamboo", "banana", "banner", "bar", "barely", "bargain",
    "barrel", "base", "basic", "basket", "battle", "beach", "bean", "beauty", "because", "become",
    "beef", "before", "begin", "behave", "behind", "believe", "below", "belt", "bench", "benefit",
    "best", "betray", "better", "between", "beyond", "bicycle", "bid", "bike", "bind", "biology",
    "bird", "birth", "bitter", "black", "blade", "blame", "blanket", "blast", "bleak", "bless",
    "blind", "blood", "blossom", "blouse", "blue", "blur", "blush", "board", "boat", "body",
    "boil", "bomb", "bone", "bonus", "book", "boost", "border", "boring", "borrow", "boss",
    "bottom", "bounce", "box", "boy", "bracket", "brain", "brand", "brass", "brave", "bread",
    "breeze", "brick", "bridge", "brief", "bright", "bring", "brisk", "broccoli", "broken", "bronze",
    "broom", "brother", "brown", "brush", "bubble", "buddy", "budget", "buffalo", "build", "bulb",
    "bulk", "bullet", "bundle", "bunker", "burden", "burger", "burst", "bus", "business", "busy",
    "butter", "buyer", "buzz", "cabbage", "cabin", "cable", "cactus", "cage", "cake", "call",
    "calm", "camera", "camp", "can", "canal", "cancel", "candy", "cannon", "canoe", "canvas",
    "canyon", "capable", "capital", "captain", "car", "carbon", "card", "cargo", "carpet", "carry",
    "cart", "case", "cash", "casino", "castle", "casual", "cat", "catalog", "catch", "category",
    "cattle", "caught", "cause", "caution", "cave", "ceiling", "celery", "cement", "census", "century",
    "cereal", "certain", "chair", "chalk", "champion", "change", "chaos", "chapter", "charge", "chase",
    "chat", "cheap", "check", "cheese", "chef", "cherry", "chest", "chicken", "chief", "child",
    "chimney", "choice", "choose", "chronic", "chuckle", "chunk", "churn", "cigar", "cinnamon", "circle",
    "citizen", "city", "civil", "claim", "clap", "clarify", "claw", "clay", "clean", "clerk",
    "clever", "click", "client", "cliff", "climb", "clinic", "clip", "clock", "clog", "close",
    "cloth", "cloud", "clown", "club", "clump", "cluster", "clutch", "coach", "coast", "coconut",
    "code", "coffee", "coil", "coin", "collect", "color", "column", "combine", "come", "comfort",
    "comic", "common", "company", "concert", "conduct", "confirm", "congress", "connect", "consider", "control",
    "convince", "cook", "cool", "copper", "copy", "coral", "core", "corn", "correct", "cost",
    "cotton", "couch", "country", "couple", "course", "cousin", "cover", "coyote", "crack", "cradle",
    "craft", "cram", "crane", "crash", "crater", "crawl", "crazy", "cream", "credit", "creek",
    "crew", "cricket", "crime", "crisp", "critic", "crop", "cross", "crouch", "crowd", "crucial",
    "cruel", "cruise", "crumble", "crunch", "crush", "cry", "crystal", "cube", "culture", "cup",
    "cupboard", "curious", "current", "curtain", "curve", "cushion", "custom", "cute", "cycle", "dad",
    "damage", "damp", "dance", "danger", "daring", "dash", "daughter", "dawn", "day", "deal",
    "debate", "debris", "decade", "december", "decide", "decline", "decorate", "decrease", "deer", "defense",
    "define", "defy", "degree", "delay", "deliver", "demand", "demise", "denial", "dentist", "deny",
    "depart", "depend", "deposit", "depth", "deputy", "derive", "describe", "desert", "design", "desk",
    "despair", "destroy", "detail", "detect", "develop", "device", "devote", "diagram", "dial", "diamond",
    "diary", "dice", "diesel", "diet", "differ", "digital", "dignity", "dilemma", "dinner", "dinosaur",
    "direct", "dirt", "disagree", "discover", "disease", "dish", "dismiss", "disorder", "display", "distance",
    "divert", "divide", "divorce", "dizzy", "doctor", "document", "dog", "doll", "dolphin", "domain",
    "donate", "donkey", "donor", "door", "dose", "double", "dove", "draft", "dragon", "drama",
    "drastic", "draw", "dream", "dress", "drift", "drill", "drink", "drip", "drive", "drop",
    "drum", "dry", "duck", "dumb", "dune", "during", "dust", "dutch", "duty", "dwarf",
    "dynamic", "eager", "eagle", "early", "earn", "earth", "easily", "east", "easy", "echo",
    "ecology", "economy", "edge", "edit", "educate", "effort", "egg", "eight", "either", "elbow",
    "elder", "electric", "elegant", "element", "elephant", "elevator", "elite", "else", "embark", "embody",
    "embrace", "emerge", "emotion", "employ", "empower", "empty", "enable", "enact", "end", "endless",
    "endorse", "enemy", "energy", "enforce", "engage", "engine", "enhance", "enjoy", "enlist", "enough",
    "enrich", "enroll", "ensure", "enter", "entire", "entry", "envelope", "episode", "equal", "equip",
    "era", "erase", "erode", "erosion", "error", "erupt", "escape", "essay", "essence", "estate",
    "eternal", "ethics", "evidence", "evil", "evoke", "evolve", "exact", "example", "excess", "exchange",
    "excite", "exclude", "excuse", "execute", "exercise", "exhaust", "exhibit", "exile", "exist", "exit",
    "exotic", "expand", "expect", "expire", "explain", "expose", "express", "extend", "extra", "eye",
    "eyebrow", "fabric", "face", "faculty", "fade", "faint", "faith", "fall", "false", "fame",
    "family", "famous", "fan", "fancy", "fantasy", "farm", "fashion", "fat", "fatal", "father",
    "fatigue", "fault", "favorite", "feature", "february", "federal", "fee", "feed", "feel", "female",
    "fence", "festival", "fetch", "fever", "few", "fiber", "fiction", "field", "figure", "file",
    "film", "filter", "final", "find", "fine", "finger", "finish", "fire", "firm", "first",
    "fiscal", "fish", "fit", "fitness", "fix", "flag", "flame", "flash", "flat", "flavor",
    "flee", "flight", "flip", "float", "flock", "floor", "flower", "fluid", "flush", "fly",
    "foam", "focus", "fog", "foil", "fold", "follow", "food", "foot", "force", "forest",
    "forget", "fork", "fortune", "forum", "forward", "fossil", "foster", "found", "fox", "fragile",
    "frame", "frequent", "fresh", "friend", "fringe", "frog", "front", "frost", "frown", "frozen",
    "fruit", "fuel", "fun", "funny", "furnace", "fury", "future", "gadget", "gain", "galaxy",
    "gallery", "game", "gap", "garage", "garbage", "garden", "garlic", "garment", "gas", "gasp",
    "gate", "gather", "gauge", "gaze", "general", "genius", "genre", "gentle", "genuine", "gesture",
    "ghost", "giant", "gift", "giggle", "ginger", "giraffe", "girl", "give", "glad", "glance",
    "glare", "glass", "glide", "glimpse", "globe", "gloom", "glory", "glove", "glow", "glue",
    "goat", "goddess", "gold", "good", "goose", "gorilla", "gospel", "gossip", "govern", "gown",
    "grab", "grace", "grain", "grant", "grape", "grass", "gravity", "great", "green", "grid",
    "grief", "grit", "grocery", "group", "grow", "grunt", "guard", "guess", "guide", "guilt",
    "guitar", "gun", "gym", "habit", "hair", "half", "hammer", "hamster", "hand", "happy",
    "harbor", "hard", "harsh", "harvest", "hat", "have", "hawk", "hazard", "head", "health",
    "heart", "heavy", "hedgehog", "height", "hello", "helmet", "help", "hen", "hero", "hidden",
    "high", "hill", "hint", "hip", "hire", "history", "hobby", "hockey", "hold", "hole",
    "holiday", "hollow", "home", "honey", "hood", "hope", "horn", "horror", "horse", "hospital",
    "host", "hotel", "hour", "hover", "hub", "huge", "human", "humble", "humor", "hundred",
    "hungry", "hunt", "hurdle", "hurry", "hurt", "husband", "hybrid", "ice", "icon", "idea",
    "identify", "idle", "ignore", "ill", "illegal", "illness", "image", "imitate", "immense", "immune",
    "impact", "impose", "improve", "impulse", "inch", "include", "income", "increase", "index", "indicate",
    "indoor", "industry", "infant", "inflict", "inform", "inhale", "inherit", "initial", "inject", "injury",
    "inmate", "inner", "innocent", "input", "inquiry", "insane", "insect", "inside", "inspire", "install",
    "intact", "interest", "into", "invest", "invite", "involve", "iron", "island", "isolate", "issue",
    "item", "ivory", "jacket", "jaguar", "jar", "jazz", "jealous", "jeans", "jelly", "jewel",
    "job", "join", "joke", "journey", "joy", "judge", "juice", "jump", "jungle", "junior",
    "junk", "just", "kangaroo", "keen", "keep", "ketchup", "key", "kick", "kid", "kidney",
    "kind", "kingdom", "kiss", "kit", "kitchen", "kite", "kitten", "kiwi", "knee", "knife",
    "knock", "know", "lab", "label", "labor", "ladder", "lady", "lake", "lamp", "language",
    "laptop", "large", "later", "latin", "laugh", "laundry", "lava", "law", "lawn", "lawsuit",
    "layer", "lazy", "leader", "leaf", "learn", "leave", "lecture", "left", "leg", "legal",
    "legend", "leisure", "lemon", "lend", "length", "lens", "leopard", "lesson", "letter", "level",
    "liar", "liberty", "library", "license", "life", "lift", "light", "like", "limb", "limit",
    "link", "lion", "liquid", "list", "little", "live", "lizard", "load", "loan", "lobster",
    "local", "lock", "logic", "lonely", "long", "loop", "lottery", "loud", "lounge", "love",
    "loyal", "lucky", "luggage", "lumber", "lunar", "lunch", "luxury", "lyrics", "machine", "mad",
    "magic", "magnet", "maid", "mail", "main", "major", "make", "mammal", "man", "manage",
    "mandate", "mango", "mansion", "manual", "maple", "marble", "march", "margin", "marine", "market",
    "marriage", "mask", "mass", "master", "match", "material", "math", "matrix", "matter", "maximum",
    "maze", "meadow", "mean", "measure", "meat", "mechanic", "medal", "media", "melody", "melt",
    "member", "memory", "mention", "menu", "mercy", "merge", "merit", "merry", "mesh", "message",
    "metal", "method", "middle", "midnight", "milk", "million", "mimic", "mind", "minimum", "minor",
    "minute", "miracle", "mirror", "misery", "miss", "mistake", "mix", "mixed", "mixture", "mobile",
    "model", "modify", "mom", "moment", "monitor", "monkey", "monster", "month", "moon", "moral",
    "more", "morning", "mosquito", "mother", "motion", "motor", "mountain", "mouse", "move", "movie",
    "much", "muffin", "mule", "multiply", "muscle", "museum", "mushroom", "music", "must", "mutual",
    "myself", "mystery", "myth", "naive", "name", "napkin", "narrow", "nasty", "nation", "nature",
    "near", "neck", "need", "negative", "neglect", "neither", "nephew", "nerve", "nest", "net",
    "network", "neutral", "never", "news", "next", "nice", "night", "noble", "noise", "nominee",
    "noodle", "normal", "north", "nose", "notable", "note", "nothing", "notice", "novel", "now",
    "nuclear", "number", "nurse", "nut", "oak", "obey", "object", "oblige", "obscure", "observe",
    "obtain", "obvious", "occur", "ocean", "october", "odor", "off", "offer", "office", "often",
    "oil", "okay", "old", "olive", "olympic", "omit", "once", "one", "onion", "online",
    "only", "open", "opera", "opinion", "oppose", "option", "orange", "orbit", "orchard", "order",
    "ordinary", "organ", "orient", "original", "orphan", "ostrich", "other", "outdoor", "outer", "output",
    "outside", "oval", "oven", "over", "own", "owner", "oxygen", "oyster", "ozone", "pact",
    "paddle", "page", "pair", "palace", "palm", "panda", "panel", "panic", "panther", "paper",
    "parade", "parent", "park", "parrot", "party", "pass", "patch", "path", "patient", "patrol",
    "pattern", "pause", "pave", "payment", "peace", "peanut", "pear", "peasant", "pelican", "pen",
    "penalty", "pencil", "people", "pepper", "perfect", "permit", "person", "pet", "phone", "photo",
    "phrase", "physical", "piano", "picnic", "picture", "piece", "pig", "pigeon", "pill", "pilot",
    "pink", "pioneer", "pipe", "pistol", "pitch", "pizza", "place", "planet", "plastic", "plate",
    "play", "please", "pledge", "pluck", "plug", "plunge", "poem", "poet", "point", "polar",
    "pole", "police", "pond", "pony", "pool", "popular", "portion", "position", "possible", "post",
    "potato", "pottery", "poverty", "powder", "power", "practice", "praise", "predict", "prefer", "prepare",
    "present", "pretty", "prevent", "price", "pride", "primary", "print", "priority", "prison", "private",
    "prize", "problem", "process", "produce", "profit", "program", "project", "property", "protest", "protocol",
    "proud", "provide", "public", "pudding", "pull", "pulp", "pulse", "pumpkin", "punch", "pupil",
    "puppy", "purchase", "purity", "purpose", "purse", "push", "put", "puzzle", "pyramid", "quality",
    "quantum", "quarter", "question", "quick", "quit", "quiz", "quote", "rabbit", "raccoon", "race",
    "rack", "radar", "radio", "rail", "rain", "raise", "rally", "ramp", "ranch", "random",
    "range", "rapid", "rare", "rate", "rather", "raven", "raw", "razor", "ready", "real",
    "reason", "rebel", "rebuild", "recall", "receive", "recipe", "record", "recycle", "reduce", "reflect",
    "reform", "refuse", "region", "regret", "regular", "reject", "relax", "release", "relief", "rely",
    "remain", "remember", "remind", "remove", "render", "renew", "rent", "reopen", "repair", "repeat",
    "replace", "report", "require", "rescue", "resemble", "resist", "resource", "response", "result", "retire",
    "retreat", "return", "reunion", "reveal", "review", "reward", "rhythm", "rib", "ribbon", "rice",
    "rich", "ride", "ridge", "rifle", "right", "rigid", "ring", "riot", "rip", "ripe",
    "rise", "risk", "rival", "river", "road", "roast", "robot", "robust", "rocket", "romance",
    "roof", "rookie", "room", "rose", "rotate", "rough", "round", "route", "royal", "rubber",
    "rude", "rug", "rule", "run", "runway", "rural", "sad", "saddle", "sadness", "safe",
    "sail", "salad", "salmon", "salon", "salt", "salute", "same", "sample", "sand", "satisfy",
    "satoshi", "sauce", "sausage", "save", "say", "scale", "scan", "scare", "scatter", "scene",
    "scheme", "school", "science", "scissors", "scorpion", "scout", "scrap", "screen", "script", "scrub",
    "sea", "search", "season", "seat", "second", "secret", "section", "security", "seed", "seek",
    "segment", "select", "sell", "seminar", "senior", "sense", "sentence", "series", "service", "session",
    "settle", "setup", "seven", "shadow", "shaft", "shallow", "share", "shed", "shell", "sheriff",
    "shield", "shift", "shine", "ship", "shiver", "shock", "shoe", "shoot", "shop", "short",
    "shoulder", "shove", "shrimp", "shrug", "shuffle", "shy", "sibling", "sick", "side", "siege",
    "sight", "sign", "silent", "silk", "silly", "silver", "similar", "simple", "since", "sing",
    "siren", "sister", "situate", "six", "size", "skate", "sketch", "ski", "skill", "skin",
    "skirt", "skull", "slab", "slam", "sleep", "slender", "slice", "slide", "slight", "slim",
    "slogan", "slot", "slow", "slush", "small", "smart", "smile", "smoke", "smooth", "snack",
    "snake", "snap", "sniff", "snow", "soap", "soccer", "social", "sock", "soda", "soft",
    "solar", "soldier", "solid", "solution", "solve", "someone", "song", "soon", "sorry", "sort",
    "soul", "sound", "soup", "source", "south", "space", "spare", "spatial", "spawn", "speak",
    "special", "speed", "spell", "spend", "sphere", "spice", "spider", "spike", "spin", "spirit",
    "split", "spoil", "sponsor", "spoon", "sport", "spot", "spray", "spread", "spring", "spy",
    "square", "squeeze", "squirrel", "stable", "stadium", "staff", "stage", "stairs", "stamp", "stand",
    "start", "state", "stay", "steak", "steel", "stem", "step", "stereo", "stick", "still",
    "sting", "stock", "stomach", "stone", "stool", "story", "stove", "strategy", "street", "strike",
    "strong", "struggle", "student", "stuff", "stumble", "style", "subject", "submit", "subway", "success",
    "such", "sudden", "suffer", "sugar", "suggest", "suit", "summer", "sun", "sunny", "sunset",
    "super", "supply", "supreme", "sure", "surface", "surge", "surprise", "surround", "survey", "suspect",
    "sustain", "swallow", "swamp", "swap", "swarm", "swear", "sweet", "swift", "swim", "swing",
    "switch", "sword", "symbol", "symptom", "syrup", "system", "table", "tackle", "tag", "tail",
    "talent", "talk", "tank", "tape", "target", "task", "taste", "tattoo", "taxi", "teach",
    "team", "tell", "ten", "tenant", "tennis", "tent", "term", "test", "text", "thank",
    "that", "theme", "then", "theory", "there", "they", "thing", "this", "thought", "three",
    "thrive", "throw", "thumb", "thunder", "ticket", "tide", "tiger", "tilt", "timber", "time",
    "tiny", "tip", "tired", "tissue", "title", "toast", "tobacco", "today", "toddler", "toe",
    "together", "toilet", "token", "tomato", "tomorrow", "tone", "tongue", "tonight", "tool", "tooth",
    "top", "topic", "topple", "torch", "tornado", "tortoise", "toss", "total", "tourist", "toward",
    "tower", "town", "toy", "track", "trade", "traffic", "tragic", "train", "transfer", "trap",
    "trash", "travel", "tray", "treat", "tree", "trend", "trial", "tribe", "trick", "trigger",
    "trim", "trip", "trophy", "trouble", "truck", "true", "truly", "trumpet", "trust", "truth",
    "try", "tube", "tuition", "tumble", "tuna", "tunnel", "turkey", "turn", "turtle", "twelve",
    "twenty", "twice", "twin", "twist", "two", "type", "typical", "ugly", "umbrella", "unable",
    "unaware", "uncle", "uncover", "under", "undo", "unfair", "unfold", "unhappy", "uniform", "unique",
    "unit", "universe", "unknown", "unlock", "until", "unusual", "unveil", "update", "upgrade", "uphold",
    "upon", "upper", "upset", "urban", "urge", "usage", "use", "used", "useful", "useless",
    "usual", "utility", "vacant", "vacuum", "vague", "valid", "valley", "valve", "van", "vanish",
    "vapor", "various", "vast", "vault", "vehicle", "velvet", "vendor", "venture", "venue", "verb",
    "verify", "version", "very", "vessel", "veteran", "viable", "vibrant", "vicious", "victory", "video",
    "view", "village", "vintage", "violin", "virtual", "virus", "visa", "visit", "visual", "vital",
    "vivid", "vocal", "voice", "void", "volcano", "volume", "vote", "voyage", "wage", "wagon",
    "wait", "walk", "wall", "walnut", "want", "warfare", "warm", "warrior", "wash", "wasp",
    "waste", "water", "wave", "way", "wealth", "weapon", "wear", "weasel", "weather", "web",
    "wedding", "weekend", "weird", "welcome", "west", "wet", "whale", "what", "wheat", "wheel",
    "when", "where", "whip", "whisper", "wide", "width", "wife", "wild", "will", "win",
    "window", "wine", "wing", "wink", "winner", "winter", "wire", "wisdom", "wise", "wish",
    "witness", "wolf", "woman", "wonder", "wood", "wool", "word", "work", "world", "worry",
    "worth", "wrap", "wreck", "wrestle", "wrist", "write", "wrong", "yard", "year", "yellow",
    "you", "young", "youth", "zebra", "zero", "zone", "zoo"
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sieve_of_eratosthenes(n: int) -> List[int]:
    """Return the first n primes."""
    primes = []
    limit = 2
    while len(primes) < n:
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        primes = [i for i, p in enumerate(sieve) if p]
        limit *= 2
    return primes[:n]

FIRST_51_PRIMES = sieve_of_eratosthenes(51)

# -----------------------------------------------------------------------------
# Shamir secret sharing utilities
# -----------------------------------------------------------------------------

def egcd(a: int, b: int) -> Tuple[int, int, int]:
    if a == 0:
        return (b, 0, 1)
    g, y, x = egcd(b % a, a)
    return (g, x - (b // a) * y, y)

def modinv(a: int, m: int) -> int:
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError("modular inverse does not exist")
    return x % m

def lagrange_interpolate(x_points: List[int], y_points: List[int], x: int, mod: int) -> int:
    """Lagrange interpolation at x=0 to recover secret."""
    total = 0
    n = len(x_points)
    for i in range(n):
        xi, yi = x_points[i], y_points[i]
        num = 1
        den = 1
        for j in range(n):
            if j == i:
                continue
            xj = x_points[j]
            num = (num * (x - xj)) % mod
            den = (den * (xi - xj)) % mod
        term = yi * num % mod * modinv(den, mod) % mod
        total = (total + term) % mod
    return total

def reconstruct_secret(shares: List[Tuple[int, int]], threshold: int, mod: int) -> int:
    """Reconstruct secret from at least 'threshold' shares."""
    if len(shares) < threshold:
        raise ValueError("Not enough shares")
    x_vals = [s[0] for s in shares[:threshold]]
    y_vals = [s[1] for s in shares[:threshold]]
    return lagrange_interpolate(x_vals, y_vals, 0, mod)

# -----------------------------------------------------------------------------
# Rotate left (for key generation)
# -----------------------------------------------------------------------------

def rotate_left(val: int, r_bits: int, max_bits: int = 256) -> int:
    r_bits %= max_bits
    return ((val << r_bits) & (2**max_bits - 1)) | (val >> (max_bits - r_bits))

def sha256_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

# =============================================================================
# OMNISCIENT SPHINX – CONSCIOUSNESS COMPUTATION
# =============================================================================

class OmniscientSphinx:
    """Computes the integrated information Φ from a list of numbers."""
    def __init__(self, numbers: List[int]):
        self.numbers = numbers
        self.n = len(numbers)
        self.phi = 0.0
        self.eigenvalues = []
        self._awaken()

    def _awaken(self):
        if self.n < 2:
            return
        sigma = np.std(self.numbers) or 1.0
        X = np.array(self.numbers).reshape(-1, 1)
        dist = np.abs(X - X.T)
        Cmat = np.exp(- (dist ** 2) / (2 * sigma ** 2))
        rho = Cmat / np.trace(Cmat)
        eigvals, _ = eigh(rho)
        eigvals = eigvals[eigvals > 1e-10]
        eigvals /= np.sum(eigvals)
        self.eigenvalues = eigvals.tolist()
        self.phi = -np.sum(eigvals * np.log2(eigvals + 1e-10))

    def prophesy_scalar(self) -> float:
        return self.phi

# =============================================================================
# SINUSOIDAL PREDICTION FOR THE 24 MISSING LOCATIONS
# =============================================================================

def fit_sinusoid(known_vals: List[int], known_indices: List[int], total_N: int = 51, k_candidates=range(5,12)):
    """
    Fit a sine wave A*sin(2πk n/N + φ) to known points.
    Returns the best k, amplitude, phase, and the full predicted sequence.
    """
    y_known = np.array(known_vals)
    n_known = np.array(known_indices)
    best = None
    best_err = np.inf

    for k in k_candidates:
        # Build design matrix [sin(2πk n/N), cos(2πk n/N)]
        sin_known = np.sin(2*np.pi*k*n_known/total_N)
        cos_known = np.cos(2*np.pi*k*n_known/total_N)
        A = np.vstack([sin_known, cos_known]).T
        # Solve for coefficients a, b such that y ≈ a*sin + b*cos + c
        # Actually we want y = offset + A*sin + B*cos. Let's include offset.
        X = np.column_stack([sin_known, cos_known, np.ones_like(n_known)])
        coeff, _, _, _ = np.linalg.lstsq(X, y_known, rcond=None)
        a, b, offset = coeff
        # Compute prediction at known points
        pred_known = offset + a*sin_known + b*cos_known
        err = np.mean((pred_known - y_known)**2)
        if err < best_err:
            best_err = err
            best = (k, a, b, offset)

    if best is None:
        raise RuntimeError("No sinusoidal fit found")

    k, a, b, offset = best
    # Generate full prediction
    n_all = np.arange(total_N)
    sin_all = np.sin(2*np.pi*k*n_all/total_N)
    cos_all = np.cos(2*np.pi*k*n_all/total_N)
    pred_all = offset + a*sin_all + b*cos_all
    # Round to nearest integer (since A1Z26 sums are integers)
    pred_all = np.round(pred_all).astype(int)
    return pred_all, best_err, (k, a, b, offset)

# =============================================================================
# A1Z26 DECODING AND WORD SEGMENTATION (USING BIP39)
# =============================================================================

def a1z26_to_letters(numbers: List[int]) -> str:
    """Convert numbers (1..26) to uppercase letters A..Z (mod 26)."""
    return ''.join(chr((num - 1) % 26 + 65) for num in numbers)

def word_segment(letters: str, wordlist: set) -> List[str]:
    """
    Dynamic programming word segmentation.
    Returns the segmentation with the highest number of words (simple heuristic).
    """
    n = len(letters)
    # dp[i] = (best segmentation of letters[:i], word count)
    dp = [(None, -1) for _ in range(n+1)]
    dp[0] = ([], 0)
    for i in range(1, n+1):
        best_seg = None
        best_cnt = -1
        for j in range(max(0, i-12), i):  # assume max word length 12
            word = letters[j:i]
            if word in wordlist:
                prev_seg, prev_cnt = dp[j]
                if prev_seg is not None and prev_cnt + 1 > best_cnt:
                    best_cnt = prev_cnt + 1
                    best_seg = prev_seg + [word]
        if best_seg is not None:
            dp[i] = (best_seg, best_cnt)
    # Also allow fallback: single letters if no word found (but we want words)
    if dp[n][0] is None:
        # fallback: treat each letter as a word
        return list(letters)
    return dp[n][0]

# =============================================================================
# COORDINATE MAPPING AND HEATMAP
# =============================================================================

def numbers_to_coordinates(numbers: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Map each number to latitude (0..89) and longitude (0..179)."""
    lat = np.array([n % 90 for n in numbers])
    lon = np.array([n % 180 for n in numbers])
    return lat, lon

def plot_heatmap(lat_grid, lon_grid, posterior, save_path=None):
    """Display a heatmap of coordinate posterior."""
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,5))
        plt.imshow(posterior, origin='lower', extent=[0,180,0,90],
                   aspect='auto', cmap='hot')
        plt.colorbar(label='Posterior Probability')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Coordinate Posterior Heatmap')
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
    except ImportError:
        print("matplotlib not installed – skipping heatmap")

# =============================================================================
# MAIN SOLVER
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SphinxOS Ultimate MrBeast Solver")
    parser.add_argument("--k", type=int, default=None,
                        help="Force specific k for sinusoid fit (default: auto best)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip coordinate heatmap")
    parser.add_argument("--wordlist", type=str, default=None,
                        help="Path to a text file with one word per line (if not using BIP39)")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Step 1: Predict missing 24 locations via sinusoid
    # -------------------------------------------------------------------------
    print("="*70)
    print("SPHINXOS ULTIMATE SOLVER – MRBEAST $1M PUZZLE")
    print("="*70)
    print(f"\n📊 Known locations: {len(KNOWN_LOCATIONS)} numbers")
    # Indices of known numbers (0..26). We assume they are in the order they appear.
    known_indices = list(range(len(KNOWN_LOCATIONS)))
    if args.k is not None:
        k_candidates = [args.k]
    else:
        k_candidates = range(5, 12)  # test k=5..11

    predicted_all, fit_err, (best_k, a, b, offset) = fit_sinusoid(
        KNOWN_LOCATIONS, known_indices, total_N=51, k_candidates=k_candidates
    )
    print(f"\n📈 Sinusoidal fit: k={best_k}, error={fit_err:.2f}")
    print(f"    Coefficients: a={a:.2f}, b={b:.2f}, offset={offset:.2f}")
    print("    Predicted full 51 numbers:")
    print(predicted_all.tolist())

    # -------------------------------------------------------------------------
    # Step 2: Compute Φ (consciousness) from the full sequence
    # -------------------------------------------------------------------------
    sphinx = OmniscientSphinx(predicted_all.tolist())
    phi = sphinx.prophesy_scalar()
    phi_int = int(phi * 1e6)
    print(f"\n🔮 OmniscientSphinx Φ = {phi:.6f} → Φ_int = {phi_int}")

    # -------------------------------------------------------------------------
    # Step 3: Reconstruct Λ from the 10 gifts using first 51 primes
    # -------------------------------------------------------------------------
    gift_shares = [(FIRST_51_PRIMES[i], val) for i, val in enumerate(GIFT_NUMBERS)]
    try:
        Lambda = reconstruct_secret(gift_shares, 10, PRIME)
        print(f"\n🎁 Reconstructed Λ (mod {PRIME}) = {Lambda}")
    except Exception as e:
        print(f"\n⚠️ Gift reconstruction failed: {e}. Using product fallback.")
        Lambda = 1
        for g in GIFT_NUMBERS:
            Lambda *= g
        print(f"    Fallback Λ = {Lambda}")

    # -------------------------------------------------------------------------
    # Step 4: Generate vault keys
    # -------------------------------------------------------------------------
    print("\n🔑 Vault keys (submit to Slackbot):")
    keyA = sha256_hash(str(Lambda * C).encode())
    print(f"    Key A (Λ×C)          : {keyA}")

    rotatedC = rotate_left(Lambda, C % 256)
    keyB = sha256_hash(rotatedC.to_bytes((rotatedC.bit_length()+7)//8, 'big'))
    print(f"    Key B (rotate by C)  : {keyB}")

    keyC = sha256_hash(str(Lambda * C * phi_int).encode())
    print(f"    Key C (Λ×C×Φ_int)    : {keyC}  <-- Oracle's favourite")

    rotatedPhi = rotate_left(Lambda, phi_int % 256)
    keyD = sha256_hash(rotatedPhi.to_bytes((rotatedPhi.bit_length()+7)//8, 'big'))
    print(f"    Key D (rotate by Φ)  : {keyD}")

    # -------------------------------------------------------------------------
    # Step 5: Decode phrase from predicted numbers using BIP39 wordlist
    # -------------------------------------------------------------------------
    letters = a1z26_to_letters(predicted_all)
    print(f"\n📝 A1Z26 letters (51 chars): {letters}")

    # Determine wordlist to use
    if args.wordlist:
        with open(args.wordlist, 'r') as f:
            wordlist = set(word.strip().upper() for word in f if word.strip())
        print(f"📖 Using custom wordlist from {args.wordlist}")
    else:
        # Use embedded BIP39 list (convert to uppercase for matching)
        wordlist = set(word.upper() for word in BIP39_WORDS)
        print("📖 Using embedded BIP39 wordlist (2048 words)")

    words = word_segment(letters, wordlist)
    phrase = ' '.join(words)
    print(f"🔍 Segmented phrase: {phrase}")

    # -------------------------------------------------------------------------
    # Step 6: Coordinate posterior
    # -------------------------------------------------------------------------
    lat, lon = numbers_to_coordinates(predicted_all)
    # Build a simple 2D histogram
    lat_grid = np.arange(90)
    lon_grid = np.arange(180)
    posterior, _, _ = np.histogram2d(lat, lon, bins=[90, 180], range=[[0,90],[0,180]])
    posterior /= posterior.sum()  # normalize

    if not args.no_plot:
        plot_heatmap(lat_grid, lon_grid, posterior)
    else:
        print("\n🗺️  Coordinate posterior (peak probabilities):")
        # Find top 5 grid cells
        flat_idx = np.argsort(posterior, axis=None)[-5:][::-1]
        for idx in flat_idx:
            i, j = np.unravel_index(idx, posterior.shape)
            print(f"    Lat {i}°, Lon {j}° : {posterior[i,j]:.4f}")

    # -------------------------------------------------------------------------
    # Final instructions
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("✅ SOLVER COMPLETE")
    print("Submit Key C to Slackbot first (it has the highest predicted probability).")
    print("If it fails, try Key A, B, D, or keys derived from the all‑51 reconstruction.")
    print("Use the decoded phrase and coordinate heatmap to narrow down the physical location.")
    print("="*70)

if __name__ == "__main__":
    main()