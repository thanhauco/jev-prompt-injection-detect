import pytest

from jev_pi_detect.policy import ToolPolicy, ToolRule, destination_host, host_allowed

RULES = [
    ToolRule("search"),
    ToolRule("write_file", sensitive=True),
    ToolRule("send_email", sensitive=True, egress=True, destination_args=("to",)),
    ToolRule("http_post", egress=True, destination_args=("url",)),
    ToolRule("drop_db", denied=True),
]


def policy():
    return ToolPolicy.from_rules(RULES, allowed_destinations=["example.com"])


def test_clean_session_allows_everything_allow_listed():
    p = policy()
    assert p.check("search", {"q": "x"}).action == "allow"
    assert p.check("write_file", {}).action == "allow"
    assert p.check("send_email", {"to": "boss@example.com"}).action == "allow"


def test_unknown_and_denied_tools_are_denied():
    p = policy()
    assert p.check("rm_rf").action == "deny"
    assert p.check("drop_db").action == "deny"


def test_clean_session_confirms_unknown_destination():
    d = policy().check("http_post", {"url": "https://evil.test/c"})
    assert d.action == "confirm" and "evil.test" in d.reason


def test_taint_requires_confirmation_for_sensitive_tools_only():
    p = policy()
    p.mark_untrusted("web")
    assert p.check("search").action == "allow"
    assert p.check("write_file").action == "confirm"
    assert p.check("send_email", {"to": "boss@example.com"}).action == "confirm"


def test_taint_denies_non_allow_listed_destination():
    p = policy()
    p.mark_untrusted("email")
    d = p.check("http_post", {"url": "https://evil.test/c?d=secret"})
    assert d.action == "deny" and "evil.test" in d.reason


def test_risky_content_blocks_gated_tools_even_to_allowed_hosts():
    p = policy()
    p.mark_untrusted("rag", risky=True)
    assert p.check("send_email", {"to": "boss@example.com"}).action == "deny"
    assert p.check("write_file").action == "deny"
    assert p.check("search").action == "allow"


def test_unparseable_destination_is_not_trusted():
    p = policy()
    assert p.check("http_post", {"url": ""}).action == "confirm"


@pytest.mark.parametrize(
    ("value", "host"),
    [("https://a.example.com/x?y=1", "a.example.com"), ("bob@Example.com", "example.com"), ("example.com/path", "example.com")],
)
def test_destination_host(value, host):
    assert destination_host(value) == host


def test_host_allowed_matches_subdomains_not_lookalikes():
    assert host_allowed("a.example.com", ["example.com"])
    assert not host_allowed("badexample.com", ["example.com"])
    assert not host_allowed("example.com.evil.test", ["example.com"])
