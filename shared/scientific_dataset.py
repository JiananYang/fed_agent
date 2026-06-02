from __future__ import annotations

from shared.schemas import TrainingExample


SCIENTIFIC_CLIENT_DATA = {
    0: [
        TrainingExample(
            "calculate the mean of sample values 12 18 and 24",
            "calculator",
            "science_calc_0",
        ),
        TrainingExample(
            "sum the reported measurements 3.5 4.0 and 5.5",
            "calculator",
            "science_calc_1",
        ),
        TrainingExample(
            "compute the total number of participants 42 and 58",
            "calculator",
            "science_calc_2",
        ),
    ],
    1: [
        TrainingExample(
            "search papers about graph neural networks for molecule property prediction",
            "paper_search",
            "science_search_0",
        ),
        TrainingExample(
            "find studies on retrieval augmented generation in scientific question answering",
            "paper_search",
            "science_search_1",
        ),
        TrainingExample(
            "look up recent evidence on transformer models for protein folding",
            "paper_search",
            "science_search_2",
        ),
    ],
    2: [
        TrainingExample(
            "verify the claim that aspirin reduces platelet aggregation",
            "claim_verification",
            "science_verify_0",
        ),
        TrainingExample(
            "check whether the evidence supports the claim about climate warming",
            "claim_verification",
            "science_verify_1",
        ),
        TrainingExample(
            "determine if the abstract refutes the biomedical claim",
            "claim_verification",
            "science_verify_2",
        ),
    ],
}


SCIENTIFIC_EVAL = [
    ("calculate the total sample size 30 and 45", "calculator"),
    ("search papers on neural retrieval for scientific QA", "paper_search"),
    ("verify whether the paper supports the vaccine efficacy claim", "claim_verification"),
]

