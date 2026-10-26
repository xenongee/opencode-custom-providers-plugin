import type { Model } from "@opencode/plugin"

export type ProviderSpec = {
    id: string
    name: string
    baseURL: string
    package: string
    enabled: boolean
    authEnv?: string
    aliases: Record<string, string>
    providerAliases: Record<string, string>
    modelsPrefixes: string[]
    modelsExcludePrefixes: string[]
    modelPackages: Record<string, string>
    fallbackLimits?: { context: number; output: number }
    priceCurrency?: string
}

/** Currency units per 1 USD, keyed by uppercase ISO 4217 code. */
export type FxRates = Readonly<Record<string, number>>

export type BuildOptions = {
    /** models.dev catalog payload used to enrich live model metadata. */
    catalog?: unknown
    /** Exchange rates used to convert non-USD prices. USD is implicit. */
    fxRates?: FxRates
    /** Previously published models used as last-good metadata. */
    lastGood?: readonly Model.Info[]
}

type Dict = Record<string, unknown>
type CatalogEntry = Dict & { modalities?: Dict; limit?: Dict; reasoning_options?: unknown[] }
const EFFORTS = new Set(["none", "minimal", "low", "medium", "high", "xhigh", "max"])
const MODALITIES = new Set(["text", "audio", "image", "video", "pdf"])
const LEGACY_PACKAGE = "@ai-sdk/openai-compatible"
const NATIVE_PACKAGE = "@opencode/ai/providers/openai-compatible"
const NON_CHAT_ID = /embed|rerank|moderat|whisper|transcri|(^|[-_/.:])tts([-_/.:]|$)|dall-?e|imagen/i

export function record(value: unknown): Dict | undefined {
    return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Dict : undefined
}
function strings(value: unknown): string[] {
    if (value === undefined) return []
    if (!Array.isArray(value) || value.some(x => typeof x !== "string" || !x.trim())) throw new TypeError("provider registry contains an invalid string list")
    return value.map(x => (x as string).trim())
}
function externalStrings(value: unknown): string[] {
    return Array.isArray(value) ? value.filter((x): x is string => typeof x === "string" && !!x.trim()).map(x => x.trim()) : []
}
/** Output modalities declared by the live payload, else by its modality string, else by the catalog. Empty when unknown. */
function declaredOutputs(architecture: Dict | undefined, catalogModalities: Dict | undefined): string[] {
    const live = externalStrings(architecture?.output_modalities).map(x => x.toLowerCase())
    if (live.length) return live
    const rawModality = architecture?.modality
    const modality = typeof rawModality === "string" ? rawModality.toLowerCase() : ""
    const arrow = modality.indexOf("->")
    if (arrow >= 0) {
        const fromModality = modality.slice(arrow + 2).split("+").map(x => x.trim()).filter(Boolean)
        if (fromModality.length) return fromModality
    }
    return externalStrings(catalogModalities?.output).map(x => x.toLowerCase())
}
function stringMap(value: unknown, lowercaseKeys = true): Record<string, string> {
    if (value === undefined) return {}
    const obj = record(value)
    if (!obj || Object.entries(obj).some(([k, v]) => !k.trim() || typeof v !== "string" || !v.trim())) throw new TypeError("provider registry contains an invalid string map")
    return Object.fromEntries(Object.entries(obj).map(([k, v]) => {
        const key = k.trim()
        return [lowercaseKeys ? key.toLowerCase() : key, (v as string).trim()]
    }))
}
function positive(value: unknown): number | undefined {
    const n = typeof value === "number" ? value : typeof value === "string" && /^\d+$/.test(value.trim()) ? Number(value) : NaN
    return Number.isFinite(n) && n > 0 ? Math.floor(n) : undefined
}
function currencyCode(value: unknown): string | undefined {
    if (typeof value !== "string") return undefined
    const code = value.trim().toUpperCase()
    return /^[A-Z]{3}$/.test(code) ? code : undefined
}
function nativePackage(value: string): string {
    return value === LEGACY_PACKAGE ? NATIVE_PACKAGE : value
}
function allowed(spec: ProviderSpec, id: string): boolean {
    const matchesPrefix = !spec.modelsPrefixes.length || spec.modelsPrefixes.some(prefix => id.startsWith(prefix))
    const excluded = spec.modelsExcludePrefixes.some(prefix => id.startsWith(prefix))
    return matchesPrefix && !excluded
}
function normalizeModalities(value: unknown): string[] {
    return strings(value).map(v => v.toLowerCase() === "file" ? "pdf" : v.toLowerCase()).filter((v, i, all) => MODALITIES.has(v) && all.indexOf(v) === i)
}
function catalogIndex(payload: unknown): Map<string, CatalogEntry> {
    const out = new Map<string, CatalogEntry>()
    const root = record(payload)
    if (!root) return out
    const providers = record(root.providers) ?? root
    const entries: Array<{ providerId: string; id: string; model: CatalogEntry }> = []
    for (const [providerId, rawProvider] of Object.entries(providers)) {
        const provider = record(rawProvider)
        const models = record(provider?.models)
        if (!models) continue
        for (const [id, rawModel] of Object.entries(models)) {
            const model = record(rawModel) as CatalogEntry | undefined
            if (!model) continue
            entries.push({ providerId: providerId.toLowerCase(), id: id.toLowerCase(), model })
        }
    }
    // First reserve every supplier-qualified key. Fallback keys are filled only
    // afterward, so catalog IDs/aliases cannot shadow a qualified native model.
    entries.sort((a, b) => a.providerId.localeCompare(b.providerId) || a.id.localeCompare(b.id))
    for (const entry of entries) {
        out.set(`${entry.providerId}/${entry.id}`, entry.model)
    }
    const fallback: Array<{ key: string; model: CatalogEntry; canonical: boolean }> = []
    for (const entry of entries) {
        const slash = entry.id.indexOf("/")
        const name = slash >= 0 ? entry.id.slice(slash + 1) : entry.id
        const qualified = slash >= 0 ? entry.id : `${entry.providerId}/${entry.id}`
        const standardProvider = name.startsWith("claude-") ? "anthropic" : name.startsWith("gpt-") ? "openai" : undefined
        fallback.push({ key: qualified, model: entry.model, canonical: entry.providerId === standardProvider || (slash >= 0 && entry.id.slice(0, slash) === entry.providerId) })
        fallback.push({ key: name, model: entry.model, canonical: entry.providerId === standardProvider })
        for (const alias of [entry.model.id, entry.model.slug, entry.model.canonical_slug, ...(Array.isArray(entry.model.aliases) ? entry.model.aliases : [])])
            if (typeof alias === "string" && alias.trim()) fallback.push({ key: alias.trim().toLowerCase(), model: entry.model, canonical: alias.trim().toLowerCase().startsWith(`${entry.providerId}/`) })
    }
    fallback.sort((a, b) => Number(b.canonical) - Number(a.canonical) || a.key.localeCompare(b.key))
    for (const item of fallback) if (!out.has(item.key)) out.set(item.key, item.model)
    const labs = record(root.providers) ? record(root.models) : undefined
    for (const [id, value] of Object.entries(labs ?? {})) {
        const model = record(value) as CatalogEntry | undefined
        if (model && !out.has(id.toLowerCase())) out.set(id.toLowerCase(), model)
    }
    return out
}

export function isValidCatalogPayload(payload: unknown): boolean {
    const root = record(payload)
    if (!root) return false
    const providers = record(root.providers) ?? root
    let count = 0
    for (const provider of Object.values(providers)) {
        const models = record(record(provider)?.models)
        if (models) count += Object.values(models).filter(value => !!record(value)).length
    }
    return count > 0
}
const DEFAULT_PROVIDER_ALIASES: Record<string, string> = { "z-ai": "zai", "zai-org": "zai", qwen: "alibaba", minimaxai: "minimax", "x-ai": "xai" }
function catalogMatch(id: string, index: Map<string, CatalogEntry>, spec: ProviderSpec): CatalogEntry | undefined {
    const lower = id.toLowerCase()
    const strip = (v: string) => v.replace(/:free$|-free$/g, "")
    const explicit = spec.aliases[lower] ?? spec.aliases[id]
    if (explicit) {
        const found = index.get(explicit.toLowerCase()) ?? index.get(strip(explicit.toLowerCase()))
        if (found) return found
    }
    const direct = index.get(lower) ?? index.get(strip(lower))
    if (direct) return direct
    const slash = lower.indexOf("/")
    const alias = slash > 0 ? spec.providerAliases[lower.slice(0, slash)] ?? DEFAULT_PROVIDER_ALIASES[lower.slice(0, slash)] : undefined
    if (alias) {
        const candidate = `${alias}/${lower.slice(slash + 1)}`
        return index.get(candidate.toLowerCase()) ?? index.get(strip(candidate.toLowerCase()))
    }
    if (slash < 0) for (const [prefix, target] of [["claude-", "anthropic/"], ["gpt-", "openai/"]]) {
        if (lower.startsWith(prefix)) return index.get(`${target}${lower}`) ?? index.get(strip(`${target}${lower}`))
    }
    return undefined
}
export function isValidLiveModelPayload(payload: unknown): boolean {
    const root = record(payload)
    const rows = Array.isArray(payload) ? payload : root?.data
    return Array.isArray(rows) && (rows.length === 0 || rows.some(row => {
        const item = record(row)
        return !!item && typeof item.id === "string" && !!item.id.trim()
    }))
}
function cost(live: Dict, spec: ProviderSpec, fxRates: FxRates): Model.Info["cost"] {
    const configured = currencyCode(spec.priceCurrency)
    if (!configured) return []
    const pricing = record(live.pricing)
    const top = record(live.top_provider)
    const source = pricing ?? record(top?.pricing)
    if (!source) return []
    const perToken = !!pricing
    // A present but malformed currency makes the price unknown; it must not fall back to the configured currency.
    const currency = source.currency === undefined ? configured : currencyCode(source.currency)
    if (!currency) return []
    const rate = currency === "USD" ? 1 : fxRates[currency]
    // Without a trustworthy rate the price is unknown, not free: publish nothing.
    if (typeof rate !== "number" || !Number.isFinite(rate) || rate <= 0) return []
    const convert = (key: string): number | undefined => {
        const value = source[perToken ? key : `${key}_per_million`]
        const n = typeof value === "number" ? value : typeof value === "string" && value.trim() !== "" ? Number(value) : NaN
        if (!Number.isFinite(n) || n < 0) return undefined
        const converted = (perToken ? n * 1e6 : n) / rate
        return Number.isFinite(converted) ? converted : undefined
    }
    const input = convert("prompt"), output = convert("completion")
    if (input === undefined || output === undefined) return []
    return [{ input, output, cache: { read: convert("input_cache_read") ?? input, write: convert("input_cache_write") ?? input } }] as unknown as Model.Info["cost"]
}

export function parseRegistry(input: unknown): ProviderSpec[] {
    const root = record(input)
    const rows = Array.isArray(input) ? input : root?.providers
    if (!Array.isArray(rows)) throw new TypeError("provider registry must contain a providers array")
    if (rows.length === 0) throw new TypeError("provider registry must contain at least one provider")
    const seen = new Set<string>()
    const result: ProviderSpec[] = []
    for (const row of rows) {
        const raw = record(row)
        if (!raw || typeof raw.id !== "string" || !/^[a-zA-Z0-9._-]+$/.test(raw.id)) throw new TypeError("provider registry contains a malformed provider")
        const key = raw.id.toLowerCase()
        if (seen.has(key)) throw new TypeError("provider registry contains a duplicate provider ID")
        seen.add(key)
        if (raw.name !== undefined && (typeof raw.name !== "string" || !raw.name.trim())) throw new TypeError("provider registry contains an invalid provider name")
        const enabled = raw.enabled === undefined ? true : raw.enabled
        if (typeof enabled !== "boolean") throw new TypeError("provider registry contains an invalid enabled flag")
        for (const key of ["baseURL", "baseUrl", "package", "npm", "priceCurrency", "authEnv"] as const) {
            if (raw[key] !== undefined && typeof raw[key] !== "string") throw new TypeError("provider registry contains an invalid optional field")
        }
        const baseURL = (typeof raw.baseURL === "string" ? raw.baseURL : typeof raw.baseUrl === "string" ? raw.baseUrl : "").trim()
        if (enabled) {
            let parsedURL: URL
            try { parsedURL = new URL(baseURL) } catch { throw new TypeError("enabled provider requires an HTTP(S) base URL") }
            if (!/^https?:$/.test(parsedURL.protocol) || parsedURL.username || parsedURL.password) throw new TypeError("enabled provider requires a safe HTTP(S) base URL")
            if (parsedURL.search || parsedURL.hash) throw new TypeError("enabled provider baseURL must not contain a query or fragment")
        }
        const packageValue = typeof raw.package === "string" ? raw.package : typeof raw.npm === "string" ? raw.npm : NATIVE_PACKAGE
        if (!packageValue.trim()) throw new TypeError("provider registry contains an invalid package")
        const aliases = stringMap(raw.aliases)
        const providerAliases = stringMap(raw.providerAliases)
        const prefixes = strings(raw.modelsPrefixes), excludePrefixes = strings(raw.modelsExcludePrefixes)
        const modelPackages = Object.fromEntries(Object.entries(stringMap(raw.modelPackages, false)).map(([prefix, pkg]) => [prefix, nativePackage(pkg)]))
        const authEnv = raw.authEnv
        if (authEnv !== undefined && (typeof authEnv !== "string" || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(authEnv))) throw new TypeError("provider registry contains an invalid authEnv name")
        const priceCurrency = raw.priceCurrency === undefined ? undefined : currencyCode(raw.priceCurrency)
        if (raw.priceCurrency !== undefined && !priceCurrency) throw new TypeError("provider registry contains an invalid priceCurrency")
        let fallbackLimits: ProviderSpec["fallbackLimits"]
        if (raw.fallbackLimits !== undefined) {
            const limits = record(raw.fallbackLimits)
            if (!limits || !Number.isSafeInteger(limits.context) || (limits.context as number) <= 0 || !Number.isSafeInteger(limits.output) || (limits.output as number) <= 0)
                throw new TypeError("provider registry contains invalid fallbackLimits")
            fallbackLimits = { context: limits.context as number, output: limits.output as number }
        }
        result.push({
            id: raw.id, name: typeof raw.name === "string" ? raw.name.trim() : raw.id,
            baseURL, package: nativePackage(packageValue.trim()), enabled,
            ...(typeof authEnv === "string" ? { authEnv } : {}),
            aliases, providerAliases, modelsPrefixes: prefixes,
            modelsExcludePrefixes: excludePrefixes, modelPackages,
            ...(priceCurrency ? { priceCurrency } : {}),
            ...(fallbackLimits ? { fallbackLimits } : {}),
        })
    }
    return result
}

export function buildModelInfos(spec: ProviderSpec, livePayload: unknown, options: BuildOptions = {}): Model.Info[] {
    const liveRoot = record(livePayload)
    const rows = Array.isArray(livePayload) ? livePayload : liveRoot?.data
    if (!Array.isArray(rows)) return []
    const catalog = catalogIndex(options.catalog), fxRates = options.fxRates ?? {}
    const previousById = new Map<string, Model.Info>((options.lastGood ?? []).map(model => [model.id, model]))
    const packages = Object.entries(spec.modelPackages).sort(([left], [right]) => right.length - left.length)
    const result: Model.Info[] = []
    const seen = new Set<string>()
    for (const value of rows) {
        const raw = record(value)
        if (!raw || typeof raw.id !== "string") continue
        const id = raw.id.trim()
        if (!id || !allowed(spec, id) || seen.has(id.toLowerCase())) continue
        const md = catalogMatch(id, catalog, spec)
        const previous = previousById.get(id)
        const architecture = record(raw.architecture), top = record(raw.top_provider)
        const type = typeof raw.type === "string" ? raw.type.toLowerCase() : undefined
        if (type && type !== "chat") continue
        const liveInput = externalStrings(architecture?.input_modalities)
        const mdMods = record(md?.modalities), mdInput = externalStrings(mdMods?.input)
        const input = liveInput.length ? liveInput : mdInput
        if (input.length && !input.some(x => x.toLowerCase() === "text")) continue
        const outputs = declaredOutputs(architecture, mdMods)
        if (outputs.length && !outputs.includes("text")) continue
        if (!outputs.length && NON_CHAT_ID.test(id)) continue
        const knownContext = positive(raw.context_length)
            ?? positive(top?.context_length)
            ?? positive(record(md?.limit)?.context)
            ?? previous?.limit.context
        const context = knownContext ?? spec.fallbackLimits?.context ?? 32_000
        const knownOutput = positive(top?.max_completion_tokens)
            ?? positive(raw.max_completion_tokens)
            ?? positive(record(md?.limit)?.output)
            ?? previous?.limit.output
        const output = Math.min(knownOutput ?? (spec.fallbackLimits?.output ?? 4096), context)
        const mdEfforts: string[] = []
        for (const opt of Array.isArray(md?.reasoning_options) ? md.reasoning_options : []) {
            const option = record(opt)
            if (option?.type === "effort") for (const e of externalStrings(option.values).map(x => x.toLowerCase())) if (EFFORTS.has(e) && !mdEfforts.includes(e)) mdEfforts.push(e)
        }
        const efforts = mdEfforts.length ? mdEfforts : (previous?.variants.map(variant => variant.id) ?? [])
        const supportedRaw = Array.isArray(top?.supported_parameters) ? top.supported_parameters : raw.supported_parameters
        const supported = Array.isArray(supportedRaw) ? supportedRaw.map(x => String(x).toLowerCase()) : undefined
        const variants = efforts.map(e => ({ id: e as Model.Info["variants"][number]["id"], settings: { reasoningEffort: e } }))
        const liveInMods = normalizeModalities(externalStrings(architecture?.input_modalities))
        const liveOutMods = normalizeModalities(externalStrings(architecture?.output_modalities))
        const catalogInput = normalizeModalities(mdInput)
        const catalogOutput = normalizeModalities(externalStrings(mdMods?.output))
        const inMods = liveInMods.length ? liveInMods : catalogInput.length ? catalogInput : previous?.capabilities.input ?? []
        const outMods = liveOutMods.length ? liveOutMods : catalogOutput.length ? catalogOutput : previous?.capabilities.output ?? []
        const currentInterleaved = typeof md?.interleaved === "string" ? md.interleaved : typeof record(md?.interleaved)?.field === "string" ? String(record(md?.interleaved)?.field) : undefined
        const compatibility: Model.Info["compatibility"] = currentInterleaved ? { reasoningField: currentInterleaved } : previous?.compatibility ?? {}
        const tools = supported ? supported.includes("tools") : typeof md?.tool_call === "boolean" ? md.tool_call : previous?.capabilities.tools ?? true
        const currentCost = cost(raw, spec, fxRates)
        const packageEntry = packages.find(([prefix]) => id.startsWith(prefix))
        const name = (typeof raw.name === "string" && raw.name)
            || (typeof md?.name === "string" && md.name)
            || previous?.name
            || id
        const model: Model.Info = {
            id: id as Model.Info["id"],
            modelID: id as Model.Info["modelID"],
            providerID: spec.id as Model.Info["providerID"],
            name,
            capabilities: {
                tools,
                input: inMods.length ? inMods : ["text"],
                output: outMods.length ? outMods : ["text"],
            },
            variants,
            time: previous?.time ?? { released: 0 },
            cost: currentCost.length ? currentCost : previous?.cost ?? [],
            status: "active",
            enabled: true,
            limit: { context, output },
            ...(packageEntry ? { package: packageEntry[1] } : {}),
            ...(Object.keys(compatibility).length ? { compatibility } : {}),
        }
        result.push(model)
        seen.add(id.toLowerCase())
    }
    return result
}
