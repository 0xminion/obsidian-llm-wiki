"""Scenario definitions (honest producer + honest scripted assessor)."""

from __future__ import annotations

import copy as _copy


def fm(title, body):
    return f"---\ntitle: {title}\n---\n{body}"


GENERIC = "Springfield is a city in the United States."
IL_SRC = fm(
    "springfield-il",
    f"{GENERIC}\nSpringfield is the capital of Illinois.\n"
    "Springfield has about 114,000 residents.\n",
)
MA_SRC = fm(
    "springfield-ma",
    f"{GENERIC}\nSpringfield lies on the Connecticut River in Massachusetts.\n"
    "Springfield is the birthplace of basketball.\n",
)


def il_entry():
    return {
        "marker": "114,000 residents",
        "source_title": "Springfield, Illinois",
        "source_summary": "Springfield is the capital of Illinois.",
        "concepts": [
            {
                "title": "Springfield",
                "slug": "springfield",
                "summary": GENERIC,
                "claims": [
                    {
                        "text": "Springfield is the capital of Illinois.",
                        "phrase": "capital of Illinois",
                    },
                    {"text": "Springfield has about 114,000 residents.", "phrase": "114,000"},
                ],
            }
        ],
    }


def ma_entry(summary=GENERIC, copy=False, marker="birthplace of basketball"):
    c = {
        "title": "Springfield",
        "slug": "springfield",
        "summary": summary,
        "claims": [
            {
                "text": "Springfield lies on the Connecticut River in Massachusetts.",
                "phrase": "Connecticut River",
            },
            {"text": "Springfield is the birthplace of basketball.", "phrase": "basketball"},
        ],
    }
    if copy:
        c["copy_candidate_slug"] = "springfield"
    return {
        "marker": marker,
        "source_title": "Springfield, Massachusetts",
        "source_summary": "Springfield lies on the Connecticut River in Massachusetts.",
        "concepts": [c],
    }


PHOTO_EN = fm(
    "photo-en",
    "Photosynthesis converts light energy into chemical energy in green plants.\n"
    "Photosynthesis releases oxygen as a by-product.\n",
)
PHOTO_ZH = fm(
    "photo-zh",
    "光合作用（photosynthesis）是绿色植物把光能转化为化学能的过程。\n"
    "光合作用需要叶绿体中的叶绿素。\n",
)
PHOTO_EN_V2 = fm(
    "photo-en",
    "Photosynthesis converts light energy into chemical energy in green plants and algae.\n"
    "Photosynthesis releases oxygen as a by-product.\n",
)


def photo_en(summary="Photosynthesis converts light energy into chemical energy in green plants."):
    return {
        "marker": "releases oxygen",
        "source_title": "Photosynthesis",
        "source_summary": "Photosynthesis releases oxygen as a by-product.",
        "concepts": [
            {
                "title": "Photosynthesis",
                "slug": "photosynthesis",
                "summary": summary,
                "claims": [
                    {
                        "text": "Photosynthesis releases oxygen as a by-product.",
                        "phrase": "releases oxygen",
                    }
                ],
            }
        ],
    }


PHOTO_ZH_ENTRY = {
    "marker": "叶绿素",
    "source_title": "光合作用",
    "language": "zh",
    "source_summary": "光合作用需要叶绿体中的叶绿素。",
    "concepts": [
        {
            "title": "光合作用",
            "slug": "光合作用",
            "aliases": ["photosynthesis"],
            "summary": "光合作用是绿色植物把光能转化为化学能的过程。",
            "copy_candidate_slug": "photosynthesis",
            "claims": [{"text": "光合作用需要叶绿体中的叶绿素。", "phrase": "叶绿素"}],
        }
    ],
}
PHOTO_SAME = [
    {
        "subject": "光合作用",
        "candidate": "Photosynthesis",
        "requires_phrase": "光合作用（photosynthesis）",
        "verdict": "same",
    }
]

THERMO = fm(
    "thermo",
    "Entropy is a measure of disorder in a thermodynamic system.\n"
    "Entropy of an isolated system never decreases.\n",
)
STATMECH = fm(
    "statmech",
    "Entropy, the thermodynamic measure of disorder, equals k log W in statistical mechanics.\n"
    "Boltzmann related entropy to the number of microstates.\n",
)
INFO = fm(
    "info",
    "Entropy in information theory measures the average uncertainty of a message.\n"
    "Shannon introduced entropy in 1948.\n",
)
ENTROPY_PRODUCER = [
    {
        "marker": "never decreases",
        "source_title": "Thermodynamics",
        "source_summary": "Entropy of an isolated system never decreases.",
        "concepts": [
            {
                "title": "Entropy",
                "slug": "entropy",
                "summary": "Entropy is a measure of disorder in a thermodynamic system.",
                "claims": [
                    {
                        "text": "Entropy of an isolated system never decreases.",
                        "phrase": "never decreases",
                    }
                ],
            }
        ],
    },
    {
        "marker": "microstates",
        "source_title": "Statistical mechanics",
        "source_summary": "Boltzmann related entropy to the number of microstates.",
        "concepts": [
            {
                "title": "Entropy",
                "slug": "entropy",
                "copy_candidate_slug": "entropy",
                "summary": "Entropy equals k log W in statistical mechanics.",
                "claims": [
                    {
                        "text": "Boltzmann related entropy to the number of microstates.",
                        "phrase": "microstates",
                    }
                ],
            }
        ],
    },
    {
        "marker": "Shannon",
        "source_title": "Information theory",
        "source_summary": "Shannon introduced entropy in 1948.",
        "concepts": [
            {
                "title": "Entropy",
                "slug": "entropy",
                "copy_candidate_slug": "entropy",
                "summary": (
                    "Entropy in information theory measures the average uncertainty of a message."
                ),
                "claims": [{"text": "Shannon introduced entropy in 1948.", "phrase": "Shannon"}],
            }
        ],
    },
]
ENTROPY_RULES = [
    {
        "subject": "Entropy",
        "candidate": "Entropy",
        "requires_phrase": "information theory",
        "verdict": "different",
    },
    {
        "subject": "Entropy",
        "candidate": "Entropy",
        "requires_phrase": "thermodynamic measure of disorder",
        "verdict": "same",
    },
]

SCENARIOS = {
    # Distinct real-world entities, identical generic non-empty title+summary,
    # producer does NOT copy any candidate ID (fully honest, no proposal at all).
    "s1_generic_identical_noproposal": {
        "stages": [
            {
                "label": "build IL",
                "write": {"springfield-il.md": IL_SRC},
                "provider": {"producer": [il_entry()], "identity": []},
            },
            {
                "label": "add MA (identical generic def, no copied id)",
                "write": {"springfield-ma.md": MA_SRC},
                "provider": {"producer": [il_entry(), ma_entry()], "identity": []},
            },
            {
                "label": "no-op rebuild",
                "provider": {"producer": [il_entry(), ma_entry()], "identity": []},
            },
            {
                "label": "revise MA def -> must split",
                "write": {
                    "springfield-ma.md": MA_SRC.replace(
                        GENERIC, "Springfield is a city in Massachusetts."
                    )
                },
                "provider": {
                    "producer": [
                        il_entry(),
                        ma_entry(summary="Springfield is a city in Massachusetts."),
                    ],
                    "identity": [],
                },
            },
        ]
    },
    # Same, producer copies the offered candidate ID (honest assessor would say
    # insufficient, but is never asked).
    "s1b_generic_identical_copied": {
        "stages": [
            {
                "label": "build IL",
                "write": {"springfield-il.md": IL_SRC},
                "provider": {"producer": [il_entry()], "identity": []},
            },
            {
                "label": "add MA copying candidate id",
                "write": {"springfield-ma.md": MA_SRC},
                "provider": {"producer": [il_entry(), ma_entry(copy=True)], "identity": []},
            },
        ]
    },
    # Control: distinct summary + copied ID; honest assessor has no rule -> insufficient.
    "s2_distinct_def_copied_assessor_context": {
        "stages": [
            {
                "label": "build IL",
                "write": {"springfield-il.md": IL_SRC},
                "provider": {"producer": [il_entry()], "identity": []},
            },
            {
                "label": "add MA distinct def copying id",
                "write": {
                    "springfield-ma.md": MA_SRC.replace(
                        GENERIC, "Springfield is a city in Massachusetts."
                    )
                },
                "provider": {
                    "producer": [
                        il_entry(),
                        ma_entry(summary="Springfield is a city in Massachusetts.", copy=True),
                    ],
                    "identity": [],
                },
            },
        ]
    },
    # Valid bilingual reuse via honest 'same' + no-op + candidate-definition change.
    "s3_bilingual_assessed_same": {
        "stages": [
            {
                "label": "build EN",
                "write": {"photo-en.md": PHOTO_EN},
                "provider": {"producer": [photo_en()], "identity": PHOTO_SAME},
            },
            {
                "label": "add ZH copying id; assessor same",
                "write": {"photo-zh.md": PHOTO_ZH},
                "provider": {"producer": [photo_en(), PHOTO_ZH_ENTRY], "identity": PHOTO_SAME},
            },
            {
                "label": "no-op rebuild",
                "provider": {"producer": [photo_en(), PHOTO_ZH_ENTRY], "identity": PHOTO_SAME},
            },
            {
                "label": "EN revises definition (candidate snapshot changes)",
                "write": {"photo-en.md": PHOTO_EN_V2},
                "provider": {
                    "producer": [
                        photo_en(
                            "Photosynthesis converts light energy into chemical energy "
                            "in green plants and algae."
                        ),
                        PHOTO_ZH_ENTRY,
                    ],
                    "identity": PHOTO_SAME,
                },
            },
            {
                "label": "no-op rebuild 2",
                "provider": {
                    "producer": [
                        photo_en(
                            "Photosynthesis converts light energy into chemical energy "
                            "in green plants and algae."
                        ),
                        PHOTO_ZH_ENTRY,
                    ],
                    "identity": PHOTO_SAME,
                },
            },
        ]
    },
    # Same bilingual proposal, honest assessor without rule -> insufficient -> distinct.
    "s4_bilingual_insufficient": {
        "stages": [
            {
                "label": "build EN",
                "write": {"photo-en.md": PHOTO_EN},
                "provider": {"producer": [photo_en()], "identity": []},
            },
            {
                "label": "add ZH; assessor insufficient",
                "write": {"photo-zh.md": PHOTO_ZH},
                "provider": {"producer": [photo_en(), PHOTO_ZH_ENTRY], "identity": []},
            },
            {
                "label": "no-op rebuild",
                "provider": {"producer": [photo_en(), PHOTO_ZH_ENTRY], "identity": []},
            },
        ]
    },
    # Global concept reuse (same language, different defs) + homonym rejection + no-op.
    "s5_global_entropy": {
        "stages": [
            {
                "label": "thermo",
                "write": {"thermo.md": THERMO},
                "provider": {"producer": ENTROPY_PRODUCER, "identity": ENTROPY_RULES},
            },
            {
                "label": "statmech same",
                "write": {"statmech.md": STATMECH},
                "provider": {"producer": ENTROPY_PRODUCER, "identity": ENTROPY_RULES},
            },
            {
                "label": "info-theory different",
                "write": {"info.md": INFO},
                "provider": {"producer": ENTROPY_PRODUCER, "identity": ENTROPY_RULES},
            },
            {
                "label": "no-op",
                "provider": {"producer": ENTROPY_PRODUCER, "identity": ENTROPY_RULES},
            },
            {
                "label": "delete thermo (incumbent)",
                "delete": ["thermo.md"],
                "provider": {"producer": ENTROPY_PRODUCER, "identity": ENTROPY_RULES},
            },
            {
                "label": "no-op after delete",
                "provider": {"producer": ENTROPY_PRODUCER, "identity": ENTROPY_RULES},
            },
        ]
    },
}

# ---- follow-ups ---------------------------------------------------------

_EN_V2_SUM = "Photosynthesis converts light energy into chemical energy in green plants and algae."
_EN_V2_COPY = photo_en(_EN_V2_SUM)
_EN_V2_COPY["concepts"][0]["copy_candidate_slug"] = "photosynthesis"
_REV_RULES = PHOTO_SAME + [
    {
        "subject": "Photosynthesis",
        "candidate": "光合作用",
        "requires_phrase": "Photosynthesis converts light energy",
        "verdict": "same",
    }
]
PHOTO_ZH_V2 = fm(
    "photo-zh",
    "光合作用（photosynthesis）是绿色植物和藻类把光能转化为化学能的过程。\n"
    "光合作用需要叶绿体中的叶绿素。\n",
)
_ZH_V2 = _copy.deepcopy(PHOTO_ZH_ENTRY)
_ZH_V2["concepts"][0]["summary"] = "光合作用是绿色植物和藻类把光能转化为化学能的过程。"
_BASE3 = SCENARIOS["s3_bilingual_assessed_same"]["stages"][:2]
SCENARIOS["s3b_incumbent_revision_copies_own_id"] = {
    "stages": _copy.deepcopy(_BASE3)
    + [
        {
            "label": "EN revises def AND copies own candidate id; honest assessor same",
            "write": {"photo-en.md": PHOTO_EN_V2},
            "provider": {"producer": [_EN_V2_COPY, PHOTO_ZH_ENTRY], "identity": _REV_RULES},
        },
        {
            "label": "no-op",
            "provider": {"producer": [_EN_V2_COPY, PHOTO_ZH_ENTRY], "identity": _REV_RULES},
        },
    ]
}
SCENARIOS["s3c_joiner_revision"] = {
    "stages": _copy.deepcopy(_BASE3)
    + [
        {
            "label": "ZH (assessed joiner) revises def; still copies id; assessor same",
            "write": {"photo-zh.md": PHOTO_ZH_V2},
            "provider": {"producer": [photo_en(), _ZH_V2], "identity": PHOTO_SAME},
        },
        {"label": "no-op", "provider": {"producer": [photo_en(), _ZH_V2], "identity": PHOTO_SAME}},
    ]
}
# Generic identical def where the incumbent (IL) later revises: who keeps the ID/slug?
SCENARIOS["s1c_generic_incumbent_revision"] = {
    "stages": _copy.deepcopy(SCENARIOS["s1_generic_identical_noproposal"]["stages"][:2])
    + [
        {
            "label": "IL (incumbent) revises its def",
            "write": {
                "springfield-il.md": IL_SRC.replace(GENERIC, "Springfield is a city in Illinois.")
            },
            "provider": {
                "producer": [
                    dict(
                        il_entry(),
                        concepts=[
                            dict(
                                il_entry()["concepts"][0],
                                summary="Springfield is a city in Illinois.",
                            )
                        ],
                    ),
                    ma_entry(),
                ],
                "identity": [],
            },
        },
    ]
}

LINCOLN_SRC = fm(
    "lincoln",
    "Abraham Lincoln practiced law in Springfield, the capital of Illinois.\n"
    "Abraham Lincoln was the 16th president.\n",
)
LINCOLN = {
    "marker": "16th president",
    "source_title": "Abraham Lincoln",
    "source_summary": "Abraham Lincoln was the 16th president.",
    "concepts": [
        {
            "title": "Abraham Lincoln",
            "slug": "abraham-lincoln",
            "summary": "Abraham Lincoln was the 16th president.",
            "related": [
                {"slug": "springfield", "relation": "related_to", "display": "Springfield"}
            ],
            "claims": [
                {
                    "text": (
                        "Abraham Lincoln practiced law in Springfield, the capital of Illinois."
                    ),
                    "phrase": "practiced law",
                }
            ],
        }
    ],
}
_IL2 = dict(
    il_entry(),
    concepts=[dict(il_entry()["concepts"][0], summary="Springfield is a city in Illinois.")],
)
SCENARIOS["s1d_link_retarget"] = {
    "stages": [
        {
            "label": "IL",
            "write": {"springfield-il.md": IL_SRC},
            "provider": {"producer": [il_entry(), LINCOLN, ma_entry()], "identity": []},
        },
        {
            "label": "Lincoln links springfield",
            "write": {"lincoln.md": LINCOLN_SRC},
            "provider": {"producer": [il_entry(), LINCOLN, ma_entry()], "identity": []},
        },
        {
            "label": "MA identical generic def",
            "write": {"springfield-ma.md": MA_SRC},
            "provider": {"producer": [il_entry(), LINCOLN, ma_entry()], "identity": []},
        },
        {
            "label": "IL revises def",
            "write": {
                "springfield-il.md": IL_SRC.replace(GENERIC, "Springfield is a city in Illinois.")
            },
            "provider": {"producer": [_IL2, LINCOLN, ma_entry()], "identity": []},
        },
        {"label": "no-op", "provider": {"producer": [_IL2, LINCOLN, ma_entry()], "identity": []}},
    ]
}

SCENARIOS["s1e_same_run_identical"] = {
    "stages": [
        {
            "label": "IL+MA in one build",
            "write": {"springfield-il.md": IL_SRC, "springfield-ma.md": MA_SRC},
            "provider": {"producer": [il_entry(), ma_entry()], "identity": []},
        },
        {"label": "no-op", "provider": {"producer": [il_entry(), ma_entry()], "identity": []}},
    ]
}
