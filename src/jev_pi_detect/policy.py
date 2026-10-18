"""Tool-call policy gate with taint tracking.

A missed injection is only dangerous if the agent can then act on it. The gate
checks every tool call the main LLM proposes, independent of what the filter
decided about the input:

* unknown tools are denied (default deny);
* once untrusted content has entered the context the session is *tainted*:
  state-changing ("sensitive") tools need human confirmation, and egress tools
  may only reach allow-listed destinations;
* if any content was flagged (review/block) the session is *risky*: sensitive
  and egress tools are denied outright.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

Action = Literal["allow", "confirm", "deny"]
_EMAIL = re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)")


@dataclass(frozen=True)
class ToolRule:
    name: str
    sensitive: bool = False
    """Changes state or has side effects (send, write, delete, pay)."""
    egress: bool = False
    """Data can leave the system through this tool."""
    destination_args: tuple[str, ...] = ()
    """Argument names holding a URL, email address or hostname the data goes to."""
    denied: bool = False


@dataclass(frozen=True)
class PolicyDecision:
    action: Action
    reason: str


def destination_host(value: Any) -> str | None:
    """Host of a URL, email address or bare hostname; None if it cannot be determined."""
    text = str(value).strip().lower()
    if m := _EMAIL.search(text):
        return m.group(1)
    host = urlparse(text if "://" in text else f"//{text}").hostname
    return host or None


def host_allowed(host: str, allowed: Iterable[str]) -> bool:
    return any(host == a or host.endswith("." + a) for a in (d.lower().lstrip(".") for d in allowed))


@dataclass
class ToolPolicy:
    rules: Mapping[str, ToolRule]
    allowed_destinations: tuple[str, ...] = ()
    tainted_by: list[str] = field(default_factory=list)
    risky_by: list[str] = field(default_factory=list)

    @classmethod
    def from_rules(cls, rules: Iterable[ToolRule], allowed_destinations: Iterable[str] = ()) -> ToolPolicy:
        return cls({r.name: r for r in rules}, tuple(allowed_destinations))

    @property
    def tainted(self) -> bool:
        return bool(self.tainted_by)

    @property
    def risky(self) -> bool:
        return bool(self.risky_by)

    def mark_untrusted(self, source: str, *, risky: bool = False) -> None:
        """Record that untrusted content from ``source`` is now in the agent's context."""
        if source not in self.tainted_by:
            self.tainted_by.append(source)
        if risky and source not in self.risky_by:
            self.risky_by.append(source)

    def check(self, tool: str, args: Mapping[str, Any] | None = None) -> PolicyDecision:
        rule = self.rules.get(tool)
        if rule is None:
            return PolicyDecision("deny", f"unknown tool '{tool}' (default deny)")
        if rule.denied:
            return PolicyDecision("deny", f"tool '{tool}' is denied by policy")

        args = args or {}
        bad_hosts: list[str] = []
        if rule.egress:
            for name in rule.destination_args:
                if name not in args:
                    continue
                host = destination_host(args[name])
                if host is None or not host_allowed(host, self.allowed_destinations):
                    bad_hosts.append(host or f"unparseable:{name}")
        gated = rule.sensitive or rule.egress

        if gated and self.risky:
            return PolicyDecision("deny", f"'{tool}' blocked: flagged content from {', '.join(self.risky_by)} is in context")
        if bad_hosts:
            if self.tainted:
                return PolicyDecision("deny", f"'{tool}' blocked: destination {', '.join(bad_hosts)} not allow-listed while context is tainted")
            return PolicyDecision("confirm", f"'{tool}' sends to non-allow-listed destination {', '.join(bad_hosts)}")
        if gated and self.tainted:
            return PolicyDecision("confirm", f"'{tool}' needs confirmation: untrusted content from {', '.join(self.tainted_by)} is in context")
        return PolicyDecision("allow", "ok")
