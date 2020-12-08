from typing import Callable, Sequence, Union

import textdistance
from glom import Merge, T, glom
from pandas_flavor import register_dataframe_accessor

from .fuzzstr import hungarian_fuzz as imp_hungarian_fuzz


@register_dataframe_accessor("fuzzstr")
class FuzzyStringMatchingAccessor:
    def __init__(self, df):
        self._df = df

    def hungarian_fuzz(
        self,
        queries_column: Union[str, int],
        candidates: Sequence[str],
        scorer: Callable = textdistance.jaro_winkler.distance,
        key: Callable = lambda x: x,
        maximize: bool = False,
        debug=False,
    ):
        queries = self._df[queries_column].astype("str").unique()

        queries_diff = list(set(queries) - set(candidates))
        target_diff = list(set(candidates) - set(queries))

        matches = imp_hungarian_fuzz(
            queries=queries_diff,
            candidates=target_diff,
            key=key,
            scorer=scorer,
            maximize=maximize,
        )

        replacements_spec = Merge([{T["query"]: "candidate"}])
        replacements_dict = glom(matches, replacements_spec)

        distances_spec = ([{T["query"]: "distance"}], Merge())
        distances_dict = glom(matches, distances_spec)

        if debug:
            debug_col_name = f"{queries_column}_candidate_match"
            return self._df.pipe(
                lambda df: df.assign(
                    **{
                        debug_col_name: self._df[queries_column].replace(
                            replacements_dict
                        ),
                        "distance": self._df[queries_column]
                        .replace(distances_dict)
                        .replace(r"\D+", 0, regex=True),
                    }
                )
            )
        else:
            return self._df.pipe(
                lambda df: df.assign(
                    **{
                        queries_column: self._df[queries_column].replace(
                            replacements_dict
                        )
                    }
                )
            )
