#!/usr/bin/env bash
# commit-msg gate: enforce the agent-commit header grammar
#
#     type(scope): subject        (scope is MANDATORY)
#     type(scope)!: subject       (breaking change)
#
# `type` must be one of the tags in docs/development/tags.md and `scope` one of
# the scopes in docs/development/scopes.md. Both lists are parsed from those
# files at run time, so this hook never duplicates them -- they stay the single
# source of truth.
#
# Wired as a pre-commit `commit-msg` stage hook (.pre-commit-config.yaml).
set -euo pipefail

msg_file="${1:?commit-msg hook expects the message file as $1}"

# Header = first non-blank, non-comment line (git strips '#' lines after us).
header="$(grep -v '^#' "$msg_file" | sed '/^[[:space:]]*$/d' | head -n1 || true)"

# Skip machine-generated messages that git creates without an author-typed
# header (merges, reverts, and rebase/amend fixup markers).
case "$header" in
  "Merge "* | "Revert "* | "fixup! "* | "squash! "* | "amend! "*) exit 0 ;;
  "") echo "commit-msg: empty commit message." >&2; exit 1 ;;
esac

repo_root="$(git rev-parse --show-toplevel)"

# Allowed tokens: bullet lines of the form  - `token` — ...
extract() { grep -oE '^- `[a-z][a-z0-9-]*`' "$1" | tr -d '`' | sed 's/^- //'; }
types="$(extract "$repo_root/docs/development/tags.md")"
scopes="$(extract "$repo_root/docs/development/scopes.md")"

fail() {
  echo "commit-msg: $1" >&2
  echo "  header: $header" >&2
  exit 1
}

# type ( scope ) optional-! : SPACE subject
re='^([a-z]+)\(([a-z0-9-]+)\)!?:[[:space:]].+'
[[ "$header" =~ $re ]] || fail \
  "header must be 'type(scope): subject' with a mandatory scope (docs/development/{tags,scopes}.md)."

type="${BASH_REMATCH[1]}"
scope="${BASH_REMATCH[2]}"

grep -qxF "$type" <<<"$types" \
  || fail "unknown type '$type'. Allowed (docs/development/tags.md): $(echo $types | tr '\n' ' ')"
grep -qxF "$scope" <<<"$scopes" \
  || fail "unknown scope '$scope'. Allowed (docs/development/scopes.md): $(echo $scopes | tr '\n' ' ')"
