from typing import Callable, Iterable, Union

import textdistance
from glom import Merge, T, glom
from pandas_flavor import register_dataframe_accessor


@register_dataframe_accessor("fuzz")
class FuzzyStringMatchingAccessor:
    def __init__(self, df):
        self._df = df

    def fuzzy_match(
        self,
        source_column: Union[str, int],
        target: Iterable[str],
        scorer: Callable = textdistance.jaro_winkler.distance,
        key: Callable = lambda x: x,
        maximize: bool = False,
        debug=False,
    ):
        source = self._df[source_column].astype("str").unique()

        source_diff = list(set(source) - set(target))
        target_diff = list(set(target) - set(source))

        matches = fuzzy_matching_best(
            source=source_diff,
            target=target_diff,
            key=key,
            scorer=scorer,
            maximize=maximize,
        )

        replacements_spec = Merge([{T["source"]: "target"}])
        replacements_dict = glom(matches, replacements_spec)

        distances_spec = ([{T["source"]: "distance"}], Merge())
        distances_dict = glom(matches, distances_spec)

        if debug:
            debug_col_name = f"{source_column}_match_from_target"
            return self._df.pipe(
                lambda df: df.assign(
                    **{
                        debug_col_name: self._df[source_column].replace(
                            replacements_dict
                        ),
                        "distance": self._df[source_column]
                        .replace(distances_dict)
                        .replace(r"\D+", 0, regex=True),
                    }
                )
            ).set_index([source_column, debug_col_name, "distance"])
        else:
            return self._df.pipe(
                lambda df: df.assign(
                    **{
                        source_column: self._df[source_column].replace(
                            replacements_dict
                        )
                    }
                )
            )
