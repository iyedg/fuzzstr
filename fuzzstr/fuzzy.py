# TODO: integrate visuall diffs for debugging https://github.com/google/diff-match-patch/wiki/Language:-Python


def ordered_difference(left: List[Any], right: List[Any]) -> List[Any]:
    return [i for i in left if i not in right]
