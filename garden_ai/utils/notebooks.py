from copy import deepcopy
import json
import random

MAX_BYTES = 5 * 1024 * 1024


def is_over_size_limit(notebook_json: dict) -> bool:
    """
    Return True if notebook contents is over 5MB.
    """
    # Convert to byte string to account for longer unicode characters
    as_byte_string = json.dumps(notebook_json).encode("utf-8")
    return len(as_byte_string) > MAX_BYTES


def clear_cells(notebook_json: dict) -> dict:
    """
    Returns new notebook with all cell outputs cleared.
    """
    new_nb = deepcopy(notebook_json)
    for cell in new_nb["cells"]:
        if "outputs" in cell:
            cell["outputs"] = []
    return new_nb


def generate_botanical_filename(extension="ipynb"):
    number = random.randint(1, 10000)
    colors = [
        "Emerald",
        "Jade",
        "Green",
        "Pink",
        "Red",
        "Teal",
        "Indigo",
        "Ochre",
        "Cerulean",
        "Crimson",
        "Amber",
        "Azure",
        "Magenta",
        "Maroon",
        "Mauve",
        "Purple",
    ]
    adjectives = [
        "Blooming",
        "Flourishing",
        "Lush",
        "Verdant",
        "Fragrant",
        "Prickly",
        "Rustling",
        "Dappled",
        "Shimmering",
        "Dewy",
        "Sun-kissed",
        "Whispering",
        "Serene",
        "Bountiful",
        "Wild",
    ]
    nouns = [
        "Blossoms",
        "Petals",
        "Potatoes",
        "Vines",
        "Leaves",
        "Twigs",
        "Roots",
        "Blooms",
        "Flowers",
        "Seeds",
        "Sprouts",
        "Stems",
        "Ferns",
        "Cacti",
        "Berries",
        "Orchids",
        "Turnips",
    ]

    color = random.choice(colors)
    adjective = random.choice(adjectives)
    noun = random.choice(nouns)

    filename = f"{number}-{color}-{adjective}-{noun}.{extension}"
    return filename
