import base64

from spoken_record.search_index import SearchConfig, quantize, search_payload

CONFIG = SearchConfig(repo_id="m", revision="a" * 40, browser_repo_id="b", browser_revision="b" * 40, dimension=2)


def fake_encode(texts):
    return [{"vector": [0.6, 0.8] if "soul" in text else [1.0, 0.0]} for text in texts]


def test_quantize_keeps_the_direction_of_each_vector():
    data = base64.b64decode(quantize([[0.6, -0.8], [1.0, 0.0]]))
    values = [b - 256 if b > 127 else b for b in data]
    assert values == [76, -102, 127, 0]


def test_payload_has_passages_and_only_topic_names():
    passages = [{"id": "p1", "text": "the soul is the form of the body"}, {"id": "p2", "text": "on logic"}]
    nodes = [{"id": "soul", "type": "Concept", "label": "soul"},
             {"id": "ar", "type": "Person", "label": "Aristotle"},
             {"id": "man", "type": "Concept", "label": "man", "generic": True}]
    payload = search_payload(CONFIG, passages, nodes, fake_encode)
    assert payload["passages"]["ids"] == ["p1", "p2"]
    assert payload["topics"]["ids"] == ["soul"], "people and generic words are found by name only"
    assert len(base64.b64decode(payload["passages"]["data"])) == 2 * CONFIG.dimension
    assert payload["model"]["repo_id"] == "b" and payload["model"]["normalize"] is True
