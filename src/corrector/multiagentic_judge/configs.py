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
        "profile": "sceptic",
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
        "weight": 0.325,
        "profile": "teacher",
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
        "weight": 0.325,
        "profile": "editor",
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
        "dimension": "Task Match & Adequacy",
        "weight": 0.10,
        "profile": "analyst",
        "description": "Does the response provide exactly what the user asked for, with appropriate level of detail—neither under-informative nor unnecessarily expanded?",
        "anchors": {
            1: "Misses the intent of the question or is largely irrelevant; does not answer what was asked.",
            2: "Partially relevant but either omits key required information or is too vague to be useful.",
            3: "Correctly answers the question at a basic level, but is either slightly under-detailed or slightly over-expanded beyond what was asked.",
            4: "Well-targeted response that clearly answers the question with appropriate detail and no significant missing or extraneous information.",
            5: "Excellent task alignment: directly answers exactly what was asked with the right amount of detail, including any necessary clarifications or examples without going beyond scope.",
        },
    },
    # {
    #     "dimension": "Language Compliance",
    #     "weight": 0.30,
    #     "profile": "enforcer",
    #     "description": "Is the response written in user's target language?",
    #     "anchors": {
    #         1: "Response was not answered in the user's target language (which the prompt is asking *about*) at all.",
    #         2: "Response largely misses to use user's target language (which the prompt is asking *about*).",
    #         3: "Language compliance is mixed and mixes user target language with prompt language or other language.",
    #         4: "Major compliance; response is mostly made up of user's target language (which the prompt is asking *about*).",
    #         5: "Perfect; strictly used user's target language only (which the prompt is asking *about*).",
    #     },
    # },
]
