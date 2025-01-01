def get_layouts_from_cklist(ck_list):
    assert ck_list is not None
    scores, _, _ = ck_list[0]
    assert isinstance(scores, dict)
    return list(scores.keys())