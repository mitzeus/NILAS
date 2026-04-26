"""
Config file with custom setups for difference mutliagentic tasks
"""

# TODO modify to fit with the specific use-case (language learning)


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

CORRECTOR_RUBRIC = [  # Setup for the Corrector. Accomodates both Correctness and naturalness
    {
        "dimension": "Factual Correctness",
        "weight": 0.25,
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
        "weight": 0.20,
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
        "dimension": "Grammar & Naturalness",
        "weight": 0.25,
        "profile": "linguistics",
        "description": "Is the response grammatically correct and follows a natural speech pattern?",
        "anchors": {
            1: "Uninteligible; critical grammatical errors or fully unnatural response.",
            2: "Hard to follow; significant grammatical inconsistencies or very unnatural.",
            3: "Grammar and response naturalness is generally acceptable although inconsistent.",
            4: "Good grammar and natural flow with minimal or insignificant mistakes",
            5: "Near or fully flawless grammar and exceptionally natural and human-like in flow.",
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
        "weight": 0.10,
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
