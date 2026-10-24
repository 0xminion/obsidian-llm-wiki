from obsidian_llm_wiki.core.registry import _normalize_url


def test_query_order_is_not_destroyed_by_identity_normalization():
    first = "https://example.org/?sort=price&sort=date"
    second = "https://example.org/?sort=date&sort=price"
    assert _normalize_url(first) != _normalize_url(second)


def test_query_encoding_is_not_reinterpreted_as_form_data():
    assert _normalize_url("https://example.org/?q=a+b") != _normalize_url(
        "https://example.org/?q=a%20b"
    )


def test_ipv6_identity_remains_a_well_formed_url():
    assert _normalize_url("https://[2606:4700:4700::1111]:443/path") == (
        "https://[2606:4700:4700::1111]/path"
    )
