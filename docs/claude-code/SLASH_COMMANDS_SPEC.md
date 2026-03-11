# Claude Code Slash Commands - Summary (2026-02-20)

## What they are
- Slash commands are commands starting with `/` used inside Claude Code sessions.
- Built-in commands are provided by Claude Code; custom commands are user-defined Markdown files.

## Custom command locations
- Project scope: `.claude/commands/`
- Personal scope: `~/.claude/commands/`

## How command names are derived
- The Markdown filename (without `.md`) becomes the command name.
- Example: `.claude/commands/refactor.md` => `/refactor`.

## Arguments
- All arguments: `$ARGUMENTS` placeholder.
- Positional arguments: `$1`, `$2`, ...
- `argument-hint` frontmatter can describe expected args.

## Frontmatter (optional)
- `description`: short description used in `/help` and for the SlashCommand tool context.
- `argument-hint`: optional arg hint displayed in help.
- `allowed-tools`: allow bash/tool execution inside a command.
- `disable-model-invocation: true`: hides the command from the SlashCommand tool.

## SlashCommand tool (programmatic invocation)
- Only custom commands (not built-in) are available to the tool.
- The `description` frontmatter is required for the tool to expose the command.
- Permissions can allow/deny specific SlashCommand invocations.

## MCP and plugins
- Plugins can provide commands in `commands/` within a plugin root.
- MCP servers can expose prompts as slash commands in the `/mcp__<server>__<prompt>` format.

Sources:
- https://docs.claude.com/en/docs/claude-code/slash-commands
- https://docs.anthropic.com/en/docs/claude-code/slash-commands
- https://platform.claude.com/docs/en/agent-sdk/slash-commands
