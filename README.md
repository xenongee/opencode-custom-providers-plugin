# OpenCode Custom Providers Plugin

An OpenCode V2 plugin that discovers chat models from custom
OpenAI-compatible providers and registers them with OpenCode.

It filters identifiable non-chat models, enriches metadata with models.dev,
and converts configured provider prices to USD. Discovery runs on plugin load
and credential changes. If a refresh fails, the last valid inventory is kept
for the same provider and credentials.

[Unlicense](LICENSE)
