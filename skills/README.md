# Agent skills

Vendor-neutral, portable skills for AI coding agents — plain Markdown, no lock-in
to any single tool. Each `skills/<name>/SKILL.md` is a self-contained guide with
YAML frontmatter (`name`, `description`) an agent reads to learn a task; the
frontmatter is optional metadata any tool can parse or ignore.

- [`omnirun/SKILL.md`](omnirun/SKILL.md) — install, configure, and use omnirun
  (run jobs on Slurm/SSH/Kaggle/Colab/marketplace GPUs), plus a Nix appendix.

## Install into your agent

Because these are ordinary files in the repo, any agent can use them — pick
whatever your tool supports:

- **Read it directly.** Point the agent at
  `skills/omnirun/SKILL.md` (raw URL or local path) and tell it to follow it.
- **Claude Code / Agent-Skills-compatible tools.** Copy or symlink the skill dir
  into the discovery path, e.g.
  `ln -s "$PWD/skills/omnirun" ~/.claude/skills/omnirun` (project or user scope).
- **Cursor / Continue / Aider / Codex, etc.** Reference `skills/omnirun/SKILL.md`
  from your rules/context file (`AGENTS.md`, `.cursor/rules`, …), or paste it in.

The content is the contract; where a given tool looks for it is that tool's
detail, so nothing here depends on one vendor's directory layout.
