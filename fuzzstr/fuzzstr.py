from itertools import product
from typing import Callable, Dict, List, Sequence

import numpy as np
import textdistance
from scipy.optimize import linear_sum_assignment


def get_closest(
    query: str,
    candidates: Sequence[str],
    n_closest: int = 1,
    scorer: Callable = textdistance.ratcliff_obershelp.normalized_distance,
):

    scores = [{"candidate": c, "distance": scorer(query, c)} for c in candidates]
    if n_closest > 1:
        return sorted(scores, key=lambda x: x["distance"])[:n_closest]
    else:
        return min(scores, key=lambda x: x["distance"])


def hungarian_fuzz(
    queries: Sequence[str],
    candidates: Sequence[str],
    key: Callable = lambda x: x,
    scorer: Callable = textdistance.jaro_winkler.distance,
    maximize: bool = False,
) -> List[Dict]:
    """
    Assign an element from the target sequence to an element from the source sequence.

    Use the Hungarian Method to solve an assignment problem where the goal is to minimize the total distance calculated by the scorer.

    Parameters
    ----------
    queries : Sequence[str]
        A list of strings that must be matched to elements from the target list.
    candidates : Sequence[str]
        A list of strings to wich elements from the source list must be matched.
    key : Callable
        A function that transforms elements in the source list into a sequence of strings.
    scorer : Callable
        A function that returns a distance based on the input of two strings.
    maximize : bool
        Whether to maximize the value provided by the scorer if set to True, or to minimize it otherwise.
    """

    # TODO: preprocess the list to remove common elements that can be checked without distance calculation

    # The key functionality is added thanks to https://stackoverflow.com/a/18296812

    pairs = product(queries, candidates)

    scores = np.array([scorer(key(q), c) for q, c in pairs]).reshape(
        (len(queries), len(candidates))
    )

    row_ind, col_ind = linear_sum_assignment(scores, maximize)

    return [
        {"query": queries[i], "candidate": candidates[j], "distance": scores[i, j]}
        for i, j in zip(row_ind, col_ind)
    ]
