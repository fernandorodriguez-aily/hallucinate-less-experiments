from typing import Literal

from pydantic import BaseModel


class ClassificationOutput(BaseModel):
    """
    Represents the output of a classification task.

    Attributes:
        reasoning (str): Text field for the reasoning behind the classification.
        relevance_label (Literal): The relevance label, constrained to three
                                   allowed values.
    """

    reasoning: str
    relevance_label: Literal["Highly Relevant", "Relevant", "Irrelevant"]
