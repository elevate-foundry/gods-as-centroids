"""Expanded corpus parts — import all and combine."""

from .part1_abrahamic import PART1_ABRAHAMIC
from .part2_asian import PART2_ASIAN
from .part3_african_indigenous_ancient import PART3_AFRICAN_INDIGENOUS_ANCIENT
from .part4_philosophical_diverse import PART4_PHILOSOPHICAL_DIVERSE

EXPANDED_CORPUS = (
    PART1_ABRAHAMIC
    + PART2_ASIAN
    + PART3_AFRICAN_INDIGENOUS_ANCIENT
    + PART4_PHILOSOPHICAL_DIVERSE
)
