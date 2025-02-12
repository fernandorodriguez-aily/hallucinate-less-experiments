from typing import Generic, Literal, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=str)


class BaseClassificationOutput(BaseModel, Generic[T]):
    """
    Base class for classification outputs.

    Attributes:
        reasoning (str): Text field for the reasoning behind the classification.
        relevance_label (T): The relevance label, constrained in subclasses.
    """

    reasoning: str
    relevance_label: T


class ClassificationOutput3Class(
    BaseClassificationOutput[Literal["Highly Relevant", "Relevant", "Irrelevant"]]
):
    """
    Represents the output of a classification task with 3 possible labels.
    """

    pass


class ClassificationOutput2Class(
    BaseClassificationOutput[Literal["Relevant", "Irrelevant"]]
):
    """
    Represents the output of a classification task with 2 possible labels.
    """

    pass
