#!/usr/bin/env python3
"""Static audit: every $SSH / $SCP use in the orchestrator must be either
(a) a comment, (b) inside the transport helpers on_loadgen/from_loadgen, or
(c) inside the ssh-only `else` arm of a `MODE = local` guard.

Tracks then/else nesting so an `else` arm is not misreported as local.
"""
import re
import sys

P = "stage7_28/exp13jul/run1h25k_jul13_fullcapture.sh"
lines = open(P).read().splitlines()

stack = []          # each: dict(mode_guard=bool, in_else=bool)
violations = []
rows = []
fn = None

for i, ln in enumerate(lines, 1):
    st = ln.strip()

    m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\(\)\s*\{", st)
    if m:
        fn = m.group(1)

    # track MODE guards
    if re.search(r'\[ "\$MODE" = local \]', st):
        stack.append({"mode": True, "else": False})
    elif re.search(r'\[ "\$MODE" != local \]', st):
        stack.append({"mode": True, "else": False})
    elif st == "fi":
        if stack:
            stack.pop()
    elif st == "else" and stack:
        stack[-1]["else"] = True
    elif st == "}":
        pass

    m2 = re.search(r"\$(SSH|SCP)\b", ln)
    if not m2:
        continue

    in_ssh_else = any(f["mode"] and f["else"] for f in stack)
    kind = "comment" if st.startswith("#") else "code"
    branch = "ssh-else" if in_ssh_else else "other"
    rows.append((i, fn or "-", kind, branch, st[:100]))
    if kind == "comment":
        continue
    if fn in ("on_loadgen", "from_loadgen"):
        continue
    if branch == "ssh-else":
        continue
    violations.append(rows[-1])

print("=== every $SSH / $SCP occurrence ===")
print("%-5s %-20s %-8s %-9s %s" % ("line", "function", "kind", "branch", "text"))
for r in rows:
    print("%-5d %-20s %-8s %-9s %s" % r)

print()
print("=== ssh-only arms (inside `else` of a MODE=local guard) ===")
for r in rows:
    if r[2] == "code" and r[3] == "ssh-else":
        print("  line %d (%s): %s" % (r[0], r[1], r[4]))

print()
print("=== violations: code that runs in --local mode and would need ssh ===")
if violations:
    for r in violations:
        print("  VIOLATION line %d (%s): %s" % (r[0], r[1], r[4]))
    sys.exit(1)
print("  none")
print()
print("RESULT: the only remote-transport code lives in on_loadgen()/from_loadgen()")
print("        and in ssh-only else arms; nothing on the --local path invokes ssh.")