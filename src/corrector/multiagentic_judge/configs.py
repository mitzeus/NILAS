"""
Config file with custom setups for difference mutliagentic tasks
"""

from dataclasses import dataclass, field

# TODO modify to fit with the specific use-case (language learning)

# OLD_DEFAULT_RUBRIC = [  # Default setup
#     {
#         "dimension": "Factual Correctness",
#         "weight": 0.35,
#         "preferred_family": "openai",  # GPT-4o is strong on factual benchmarks
#         "description": "Are all factual claims accurate and verifiable?",
#         "anchors": {
#             1: "Contains critical factual errors that would mislead the reader.",
#             2: "Several inaccuracies or unsupported claims.",
#             3: "Mostly correct with minor errors or omissions.",
#             4: "Accurate throughout; any caveats are appropriately noted.",
#             5: "Completely accurate, well-sourced reasoning, no misleading statements.",
#         },
#     },
#     {
#         "dimension": "Clarity & Communication",
#         "weight": 0.25,
#         "preferred_family": "anthropic",  # Claude excels at linguistic/structural awareness
#         "description": "Is the response clear, well-structured, and appropriately concise?",
#         "anchors": {
#             1: "Extremely confusing or incoherent; ideas are not communicated.",
#             2: "Hard to follow; significant structural or language issues.",
#             3: "Generally understandable but could be clearer or better organised.",
#             4: "Clear, logical flow; reader can follow without effort.",
#             5: "Exceptionally clear; perfectly calibrated length and structure.",
#         },
#     },
#     {
#         "dimension": "Depth & Completeness",
#         "weight": 0.20,
#         "preferred_family": "google",  # Gemini's large context + training suits coverage detection
#         "description": "Does the response adequately address the full scope of the query?",
#         "anchors": {
#             1: "Superficial; misses the core of what was asked.",
#             2: "Addresses only part of the question.",
#             3: "Covers the main points but omits important nuances.",
#             4: "Thorough; handles edge cases or complexity where relevant.",
#             5: "Comprehensive; anticipates follow-up questions and addresses them.",
#         },
#     },
#     {
#         "dimension": "Reasoning Quality",
#         "weight": 0.20,
#         "preferred_family": "anthropic",  # Strongest reasoning model (Opus) used here
#         "description": "Is the logical structure of the argument sound and well-supported?",
#         "anchors": {
#             1: "Reasoning is absent, circular, or logically fallacious.",
#             2: "Weak reasoning; conclusions don't follow from evidence.",
#             3: "Adequate reasoning but some logical gaps.",
#             4: "Sound reasoning; conclusions are well-supported.",
#             5: "Rigorous, transparent reasoning; trade-offs are explicitly considered.",
#         },
#     },
# ]


DEFAULT_RUBRIC = [  # Default setup
    {
        "dimension": "Factual Correctness",
        "weight": 0.35,
        "profile": "factuality",
        "description": "Are all factual claims accurate and verifiable?",
        "anchors": {
            1: "Contains critical factual errors that would mislead the reader.",
            2: "Several inaccuracies or unsupported claims.",
            3: "Mostly correct with minor errors or omissions.",
            4: "Accurate throughout; any caveats are appropriately noted.",
            5: "Completely accurate, well-sourced reasoning, no misleading statements.",
        },
    },
    {
        "dimension": "Clarity & Communication",
        "weight": 0.25,
        "profile": "linguistics",
        "description": "Is the response clear, well-structured, and appropriately concise?",
        "anchors": {
            1: "Extremely confusing or incoherent; ideas are not communicated.",
            2: "Hard to follow; significant structural or language issues.",
            3: "Generally understandable but could be clearer or better organised.",
            4: "Clear, logical flow; reader can follow without effort.",
            5: "Exceptionally clear; perfectly calibrated length and structure.",
        },
    },
    {
        "dimension": "Depth & Completeness",
        "weight": 0.20,
        "profile": "depth",
        "description": "Does the response adequately address the full scope of the query?",
        "anchors": {
            1: "Superficial; misses the core of what was asked.",
            2: "Addresses only part of the question.",
            3: "Covers the main points but omits important nuances.",
            4: "Thorough; handles edge cases or complexity where relevant.",
            5: "Comprehensive; anticipates follow-up questions and addresses them.",
        },
    },
    {
        "dimension": "Reasoning Quality",
        "weight": 0.20,
        "profile": "reasoning",
        "description": "Is the logical structure of the argument sound and well-supported?",
        "anchors": {
            1: "Reasoning is absent, circular, or logically fallacious.",
            2: "Weak reasoning; conclusions don't follow from evidence.",
            3: "Adequate reasoning but some logical gaps.",
            4: "Sound reasoning; conclusions are well-supported.",
            5: "Rigorous, transparent reasoning; trade-offs are explicitly considered.",
        },
    },
]
