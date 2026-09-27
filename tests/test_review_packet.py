from evals.review_packet import select


def _items():
    out = []
    for i in range(30):
        out.append(
            {
                "id": f"q{i}",
                "type": "unanswerable" if i % 5 == 0 else "single",
                "refused": i % 7 == 0,
            }
        )
    return out


def test_selection_is_deterministic_and_filtered():
    a = select(_items(), 10, 7)
    assert [i["id"] for i in a] == [i["id"] for i in select(_items(), 10, 7)]
    assert len(a) == 10
    assert all(i["type"] != "unanswerable" and not i["refused"] for i in a)
    assert [i["id"] for i in a] != [i["id"] for i in select(_items(), 10, 8)]
