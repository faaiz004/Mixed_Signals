"""
Prompt templates for the Mixed Signals tasks (Pezeshkpour et al., 2025, Appendix B).

Each prompt is assembled from three parts, so the two conditions differ ONLY in
the output-format block:

    instruction    task description (identical in both conditions)
    output format  "reasoning": Explanation + Answer   (the paper's prompt as written)
                   "direct":    Answer only            (our variant; not in the paper)
    inputs         the sample's data, filled into named {slots}

This module is model-independent: it returns the plain user-message text.
The chat template is applied later, in run.py, with each checkpoint's processor.
"""

CONDITIONS = ("direct", "reasoning")

OUTPUT_FORMAT = {
    "reasoning": (
        "The output should be in the following format:\n"
        "Explanation: <....>\n"
        "Answer: {answer_spec}"
    ),
    "direct": (
        "The output should be in the following format:\n"
        "Answer: {answer_spec}"
    ),
}

TASKS = {
    "connectivity": {
        "instruction": (
            "You are given the adjacency matrix of a graph and two target nodes. "
            "The goal is to determine whether the target nodes are connected. "
            "Additionally, you are provided with an image of the graph, where the "
            "target nodes are highlighted in {node_color}. Use both the image and "
            "the adjacency matrix to answer the question."
        ),
        "answer_spec": "<True or False>",
        "inputs": (
            "Adjacency matrix:\n"
            "{adjacency_matrix}\n"
            "Target nodes = {node_1} and {node_2}"
        ),
    },
    "convexity": {
        "instruction": (
            "You are given the mathematical expression of a function. The goal is "
            "to determine whether the function is convex or concave. Additionally, "
            "you are provided with the plot of the function. Use both the plot and "
            "the mathematical expression of the function to identify the convexity "
            "of the function."
        ),
        "answer_spec": "<convex or concave>",
        "inputs": (
            "The mathematical expression:\n"
            "{expression}"
        ),
    },
    "roots": {
        "instruction": (
            # "the the" is in the original paper; kept for fidelity.
            "You are given the mathematical expression of a polynomial. The goal is "
            "to find the roots of this polynomial. Additionally, you are provided "
            "with an image of the plot of that polynomial, where the roots are "
            "highlighted in red. Use both the image and the the mathematical "
            "expression of a polynomial to find the roots."
        ),
        "answer_spec": "<[The list of roots, comma separated]>",
        "inputs": (
            "The polynomial:\n"
            "{polynomial}"
        ),
    },
    "science": {
        "instruction": (
            "You are given a question, multiple choices, a textual description, and "
            "an image. Your task is to analyze both the text and the image and then "
            "choose the answer of the question from the provided choices."
        ),
        "answer_spec": "<A or B or ...>",
        "inputs": (
            "The textual description:\n"
            "{description}\n"
            "The question:\n"
            "{question}\n"
            "The Choices:\n"
            "{choices}"
        ),
    },
    "vsr": {
        "instruction": (
            "You are given a statement, a paragraph, and an image. Your task is to "
            "analyze both the paragraph and the image to decide whether the "
            "statement is true or false."
        ),
        "answer_spec": "<true or false>",
        "inputs": (
            "The paragraph:\n"
            "{paragraph}\n"
            "The statement:\n"
            "{statement}"
        ),
    },
}

# Slots each task's inputs need, so build.py can be checked against them.
REQUIRED_FIELDS = {
    "connectivity": {"node_color", "adjacency_matrix", "node_1", "node_2"},
    "convexity": {"expression"},
    "roots": {"polynomial"},
    "science": {"description", "question", "choices"},
    "vsr": {"paragraph", "statement"},
}


def format_choices(choices):
    """['Solution B', 'neither', 'Solution A'] -> 'A. Solution B\\nB. neither\\nC. Solution A'

    The paper does not show how choices were rendered; lettering them makes the
    requested '<A or B or ...>' answer well defined. Letter i maps to label index i.
    """
    return "\n".join(f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))


def render(task, condition, fields):
    """Return the user-message text for one sample.

    task:      one of TASKS
    condition: "direct" or "reasoning"
    fields:    dict with the slots in REQUIRED_FIELDS[task]; build.py decides
               whether each slot gets the original or the _modify text.
    """
    if task not in TASKS:
        raise KeyError(f"Unknown task {task!r}")
    if condition not in OUTPUT_FORMAT:
        raise KeyError(f"Unknown condition {condition!r}")
    missing = REQUIRED_FIELDS[task] - fields.keys()
    if missing:
        raise KeyError(f"{task}: missing fields {sorted(missing)}")

    t = TASKS[task]
    # Only the instruction (connectivity's node colour) and inputs take sample fields.
    instruction = t["instruction"].format(**fields)
    output_format = OUTPUT_FORMAT[condition].format(answer_spec=t["answer_spec"])
    inputs = t["inputs"].format(**fields)
    return f"{instruction}\n\n{output_format}\n\n{inputs}"


if __name__ == "__main__":
    # Quick visual check: print every task in both conditions with dummy fields.
    demo = {
        "connectivity": {"node_color": "gray", "adjacency_matrix": "[[0, 1], [1, 0]]",
                         "node_1": 0, "node_2": 1},
        "convexity": {"expression": "f(x) = 2.23*Abs(x) + 6.64"},
        "roots": {"polynomial": "3.0x + 4.2"},
        "science": {"description": "Solution A and Solution B have the same number of green particles.",
                    "question": "Which solution has a higher concentration of green particles?",
                    "choices": format_choices(["Solution B", "neither; their concentrations are the same", "Solution A"])},
        "vsr": {"paragraph": "In a cozy kitchen ... The person is inside the refrigerator. ...",
                "statement": "The person is inside the refrigerator."},
    }
    for task, fields in demo.items():
        for condition in CONDITIONS:
            print(f"===== {task} / {condition} =====")
            print(render(task, condition, fields))
            print()


            