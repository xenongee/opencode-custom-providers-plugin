import type { Model, Plugin, Provider } from "@opencode/plugin"
import { createHash } from "node:crypto"
import { readFile } from "node:fs/promises"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { setTimeout as delay } from "node:timers/promises"
import {
    buildModelInfos, isValidCatalogPayload, isValidLiveModelPayload, parseRegistry, record,
    type FxRates, type ProviderSpec,
} from "./models.ts"

const MODELS_DEV_URLS = ["https://models.dev/api.json", "https://models.dev/catalog.json"]
const frankfurterRateUrl = (currency: string) => `https://api.frankfurter.dev/v2/rate/USD/${encodeURIComponent(currency)}`
const STORAGE_KEY = "custom-model-providers:v1"
const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/
const CACHE_TTL_MS = 24 * 60 * 60 * 1000

export type RuntimeDependencies = {
    fetch: typeof globalThis.fetch
    /** Replaces reading the provider registry file (used by tests). */
    readRegistry?: () => Promise<unknown>
    now: () => number
    env: NodeJS.ProcessEnv
}
const defaults: Omit<RuntimeDependencies, "readRegistry"> = {
    fetch: globalThis.fetch.bind(globalThis),
    now: Date.now,
    env: process.env,
}

type SavedInventory = {
    providerId: string
    scope: string
    savedAt: number
    models: Model.Info[]
}
type SavedCatalog = {
    savedAt: number
    payload: unknown
}
type SavedRate = {
    currency: string
    rate: number
    date: string
    savedAt: number
}
type RecordEntry = {
    spec: ProviderSpec
    models: Model.Info[]
    connection?: Awaited<ReturnType<Plugin.Context["integration"]["connection"]["active"]>>
    scope?: string
}
function isFiniteNumber(value: unknown): value is number {
    return typeof value === "number" && Number.isFinite(value)
}
function isPositiveInteger(value: unknown): boolean {
    return Number.isInteger(value) && Number(value) > 0
}
function isStringArray(value: unknown): value is string[] {
    return Array.isArray(value) && value.every(item => typeof item === "string")
}
function sha256(value: string): string {
    return createHash("sha256").update(value).digest("hex")
}
function headerValue(value: unknown): string | undefined {
    const credential = record(value)
    if (credential?.type === "key" && typeof credential.key === "string") return credential.key
    if (credential?.type === "oauth" && typeof credential.access === "string") return credential.access
    return undefined
}
function describeReadError(error: unknown): string {
    if (error instanceof SyntaxError) return "invalid JSON"
    return (error as NodeJS.ErrnoException | undefined)?.code ?? "read error"
}
function validCapabilities(value: unknown): boolean {
    const capabilities = record(value)
    return !!capabilities
        && typeof capabilities.tools === "boolean"
        && isStringArray(capabilities.input)
        && isStringArray(capabilities.output)
}
function validCosts(value: unknown): boolean {
    return Array.isArray(value) && value.every(cost => {
        const entry = record(cost), cache = record(entry?.cache)
        return !!entry && [entry.input, entry.output, cache?.read, cache?.write].every(amount => isFiniteNumber(amount) && amount >= 0)
    })
}
function validLimit(value: unknown): boolean {
    const limit = record(value)
    return !!limit && isPositiveInteger(limit.context) && isPositiveInteger(limit.output)
}
function modelInfo(value: unknown, providerID: string): value is Model.Info {
    const item = record(value)
    const time = record(item?.time)
    if (!item
        || typeof item.id !== "string"
        || item.id.length === 0
        || item.modelID !== item.id
        || item.providerID !== providerID
        || typeof item.name !== "string") return false
    if ("temperature" in item || "reasoning" in item || "attachment" in item) return false
    if (!validCapabilities(item.capabilities)) return false
    if (!Array.isArray(item.variants) || !item.variants.every(v => typeof record(v)?.id === "string")) return false
    if (!time || !isFiniteNumber(time.released)) return false
    if (!validCosts(item.cost)) return false
    return validLimit(item.limit)
        && typeof item.enabled === "boolean"
        && ["active", "alpha", "beta", "deprecated"].includes(String(item.status))
}
/** Returns the saved models that still validate for this provider and scope; one malformed entry does not discard the rest. */
function savedModels(value: unknown, id: string, scope: string): Model.Info[] | undefined {
    const item = record(value)
    if (item?.providerId !== id || item.scope !== scope || !isFiniteNumber(item.savedAt) || !Array.isArray(item.models)) return undefined
    const models = item.models.filter((model): model is Model.Info =>
        modelInfo(model, id) && model.capabilities.output.some(output => output.toLowerCase() === "text"))
    return models.length ? models : undefined
}
function validRate(value: unknown, currency: string): value is SavedRate {
    const item = record(value)
    return item?.currency === currency
        && isFiniteNumber(item.rate)
        && item.rate > 0
        && typeof item.date === "string"
        && ISO_DATE.test(item.date)
        && isFiniteNumber(item.savedAt)
}
/** Parses a Frankfurter v2 `/rate/USD/{currency}` response. */
function parsedRate(value: unknown, currency: string): { rate: number; date: string } | undefined {
    const item = record(value)
    if (!item) return undefined
    if (typeof item.quote !== "string" || item.quote.trim().toUpperCase() !== currency) return undefined
    if (item.base !== undefined && (typeof item.base !== "string" || item.base.trim().toUpperCase() !== "USD")) return undefined
    const rate = typeof item.rate === "number" ? item.rate : typeof item.rate === "string" && item.rate.trim() ? Number(item.rate) : NaN
    if (!Number.isFinite(rate) || rate <= 0) return undefined
    const date = typeof item.date === "string" ? item.date.trim() : ""
    if (!ISO_DATE.test(date)) return undefined
    return { rate, date }
}
function providerInfo(entry: RecordEntry): Provider.Info {
    return {
        id: entry.spec.id as Provider.Info["id"], name: entry.spec.name, activation: "enabled", package: entry.spec.package,
        integrationID: entry.spec.id as NonNullable<Provider.Info["integrationID"]>,
        settings: { baseURL: entry.spec.baseURL },
    }
}

export async function setupPlugin(ctx: Plugin.Context, overrides: Partial<RuntimeDependencies> = {}): Promise<() => Promise<void>> {
    const deps = { ...defaults, ...overrides }
    if (deps.env.OPENCODE_SYNC_MODELS_DISABLE === "1") {
        return async () => {}
    }
    let rawRegistry: unknown
    try {
        if (deps.readRegistry) rawRegistry = await deps.readRegistry()
        else {
            const optionPath = typeof ctx.options.providersFile === "string" && ctx.options.providersFile.trim() ? ctx.options.providersFile.trim() : undefined
            const envPath = deps.env.OPENCODE_PROVIDERS_FILE?.trim() || undefined
            const path = optionPath ?? envPath ?? resolve(dirname(fileURLToPath(import.meta.url)), "providers.json")
            rawRegistry = JSON.parse(await readFile(resolve(path), "utf8")) as unknown
        }
    } catch (error) {
        console.warn(`[custom-model-providers] provider registry unavailable (${describeReadError(error)}); no providers loaded`)
        return async () => {}
    }
    let specs: ProviderSpec[]
    try {
        specs = parseRegistry(rawRegistry).filter(spec => spec.enabled)
    } catch (error) {
        console.warn(`[custom-model-providers] invalid provider registry (${error instanceof Error ? error.message : "unknown error"}); no providers loaded`)
        return async () => {}
    }
    if (!specs.length) return async () => {}

    // Legacy registry fields are accepted but ignored. Never echo their values.
    const registryRows = Array.isArray(rawRegistry) ? rawRegistry : record(rawRegistry)?.providers
    const legacy = new Set<string>()
    for (const row of Array.isArray(registryRows) ? registryRows : []) {
        const item = record(row)
        if (!item) continue
        if (item.priceRatePerUsd !== undefined) legacy.add("priceRatePerUsd is ignored; exchange rates are fetched from Frankfurter")
        if (item.preset !== undefined) legacy.add("preset is ignored; models are discovered from the provider /models endpoint")
    }
    for (const notice of legacy) console.warn(`[custom-model-providers] registry notice: ${notice}`)

    let stopped = false
    const warn = (message: string): void => {
        if (!stopped) console.warn(`[custom-model-providers] ${message}`)
    }
    let pendingUpdate = false
    let updatePromise: Promise<void> | undefined
    let subscription: Promise<void> | undefined
    const eventController = new AbortController()
    let registration: Awaited<ReturnType<typeof ctx.provider.transform>> | undefined
    let integrationRegistration: Awaited<ReturnType<typeof ctx.integration.transform>> | undefined
    const controllers = new Set<AbortController>()
    const entries = new Map(specs.map(spec => [spec.id, { spec, models: [] } as RecordEntry]))
    const beginRequest = (timeoutMs: number): { signal: AbortSignal; done: () => void } | undefined => {
        if (stopped) return undefined
        const controller = new AbortController()
        controllers.add(controller)
        return {
            signal: AbortSignal.any([controller.signal, AbortSignal.timeout(timeoutMs)]),
            done: () => controllers.delete(controller),
        }
    }
    const fetchJson = async (url: string, timeoutMs: number): Promise<unknown> => {
        const request = beginRequest(timeoutMs)
        if (!request) return undefined
        try {
            const response = await deps.fetch(url, { headers: { Accept: "application/json" }, signal: request.signal })
            if (!response.ok) return undefined
            return await response.json()
        } catch {
            return undefined
        } finally {
            request.done()
        }
    }
    /** Undefined means the lookup failed. That is different from "no connection", which returns {}. */
    const credentialFor = async (spec: ProviderSpec): Promise<{ token?: string; connection?: RecordEntry["connection"] } | undefined> => {
        try {
            const connection = await ctx.integration.connection.active(spec.id)
            if (!connection) return {}
            const token = headerValue(await ctx.integration.connection.resolve(connection))
            if (!token?.trim() || /[\r\n\x00]/.test(token)) return undefined
            return { token, connection }
        } catch {
            return undefined
        }
    }
    const scopeFor = (spec: ProviderSpec, token: string | undefined, connection?: RecordEntry["connection"]): string => {
        const connectionRecord = record(connection)
        const connectionIdentity = connectionRecord?.type === "credential"
            ? { type: "credential", id: connectionRecord.id }
            : connectionRecord?.type === "env"
                ? { type: "env", name: connectionRecord.name }
                : undefined
        const identity = JSON.stringify({
            id: spec.id, name: spec.name, baseURL: spec.baseURL, package: spec.package,
            aliases: spec.aliases, providerAliases: spec.providerAliases, modelsPrefixes: spec.modelsPrefixes,
            modelsExcludePrefixes: spec.modelsExcludePrefixes, modelPackages: spec.modelPackages,
            priceCurrency: spec.priceCurrency, fallbackLimits: spec.fallbackLimits,
            credentialHash: token ? sha256(token) : "none",
            connection: connectionIdentity ? sha256(JSON.stringify(connectionIdentity)) : "none",
        })
        return sha256(identity)
    }
    const authScope = async (spec: ProviderSpec): Promise<{ scope: string; token?: string; connection?: RecordEntry["connection"] } | undefined> => {
        const auth = await credentialFor(spec)
        if (!auth) return undefined
        return { scope: scopeFor(spec, auth.token, auth.connection), token: auth.token, connection: auth.connection }
    }
    /** Returns the saved models that still validate for this scope, or undefined when nothing usable is saved. */
    const loadCache = async (spec: ProviderSpec, scope: string): Promise<Model.Info[] | undefined> => {
        try {
            return savedModels(await ctx.storage.get(`${STORAGE_KEY}:inventory:${spec.id}`), spec.id, scope)
        } catch {
            return undefined
        }
    }
    const writeStorage = async (key: string, value: unknown): Promise<void> => {
        if (stopped) return
        await ctx.storage.set(key, value as never)
    }
    const disposeRegistrations = async (): Promise<void> => {
        try {
            await registration?.dispose()
        } catch { /* host may already be unloading */ }
        try {
            await integrationRegistration?.dispose()
        } catch { /* host may already be unloading */ }
    }
    try {
        integrationRegistration = await ctx.integration.transform(editor => {
            for (const spec of specs) {
                editor.method.update({ integrationID: spec.id, method: { type: "key", label: spec.name } })
                if (spec.authEnv) editor.method.update({ integrationID: spec.id, method: { type: "env", names: [spec.authEnv] } })
                editor.update(spec.id, integration => { integration.name = spec.name })
            }
        })
        registration = await ctx.provider.transform(editor => {
            for (const entry of entries.values()) {
                editor.remove(entry.spec.id)
                editor.add({ info: providerInfo(entry), models: entry.models, ...(entry.connection ? { sourceConnection: entry.connection } : {}) })
            }
        })
    } catch {
        console.warn("[custom-model-providers] provider registration failed")
        await disposeRegistrations()
        return async () => {}
    }

    /** models.dev metadata. A saved copy younger than CACHE_TTL_MS is used without network; otherwise models.dev is fetched, and the saved copy is the fallback. */
    const loadCatalog = async (): Promise<unknown> => {
        const key = `${STORAGE_KEY}:catalog`
        let saved: SavedCatalog | undefined = undefined
        try {
            saved = await ctx.storage.get(key) as SavedCatalog | undefined
        } catch { /* optional last-good catalog */ }
        const savedPayload = isValidCatalogPayload(saved?.payload) ? saved?.payload : undefined
        if (saved && savedPayload !== undefined && isFiniteNumber(saved.savedAt) && saved.savedAt <= deps.now() && deps.now() - saved.savedAt < CACHE_TTL_MS) return savedPayload
        for (const url of MODELS_DEV_URLS) {
            if (stopped) return savedPayload
            const payload = await fetchJson(url, 12_000)
            if (isValidCatalogPayload(payload)) {
                try {
                    await writeStorage(key, { savedAt: deps.now(), payload } satisfies SavedCatalog)
                } catch { /* cache is optional */ }
                return payload
            }
        }
        warn("models.dev catalog unavailable; using the saved copy when present")
        return savedPayload
    }
    /** Units per USD for one currency: a saved rate younger than CACHE_TTL_MS, else a fresh rate, else the last valid saved rate. */
    const resolveRate = async (currency: string): Promise<number | undefined> => {
        const key = `${STORAGE_KEY}:fx:${currency}`
        let saved: unknown = undefined
        try {
            saved = await ctx.storage.get(key)
        } catch { /* optional last-good rate */ }
        if (validRate(saved, currency) && saved.savedAt <= deps.now() && deps.now() - saved.savedAt < CACHE_TTL_MS) return saved.rate
        const fresh = parsedRate(await fetchJson(frankfurterRateUrl(currency), 8_000), currency)
        if (stopped) return undefined
        if (fresh) {
            try {
                await writeStorage(key, { currency, rate: fresh.rate, date: fresh.date, savedAt: deps.now() } satisfies SavedRate)
            } catch { /* rate cache is optional */ }
            return fresh.rate
        }
        if (validRate(saved, currency)) {
            warn(`${currency} exchange rate unavailable; using the cached rate from ${saved.date}`)
            return saved.rate
        }
        return undefined
    }
    const resolveRates = async (): Promise<FxRates> => {
        const rates: Record<string, number> = { USD: 1 }
        const currencies = [...new Set(specs.map(spec => spec.priceCurrency).filter((value): value is string => !!value && value !== "USD"))].sort()
        await Promise.all(currencies.map(async currency => {
            const rate = await resolveRate(currency)
            if (rate !== undefined) rates[currency] = rate
            else warn(`no ${currency} exchange rate available; prices will be omitted`)
        }))
        return rates
    }
    const reload = async (): Promise<void> => {
        if (stopped) return
        try {
            await ctx.provider.reload()
        } catch {
            warn("provider reload failed")
        }
    }
    const update = async (): Promise<void> => {
        if (stopped) return
        const enrichmentPromise = loadCatalog().catch((): undefined => undefined)
        const ratesPromise = resolveRates()
        await Promise.all(specs.map(async spec => {
            try {
                if (stopped) return
                const entry = entries.get(spec.id)!
                const auth = await authScope(spec)
                if (stopped) return
                if (!auth) {
                    warn(`${spec.id} credential lookup failed; keeping the current models`)
                    return
                }
                const scope = auth.scope
                const changedScope = entry.scope !== undefined && entry.scope !== scope
                entry.scope = scope
                entry.connection = auth.connection
                if (changedScope) {
                    entry.models = []
                    await reload()
                }
                const cached = await loadCache(spec, scope)
                if (stopped) return
                if (!entry.models.length && cached) {
                    entry.models = cached
                    await reload()
                }
                const request = beginRequest(12_000)
                if (!request) return
                let live: unknown
                try {
                    const response = await deps.fetch(`${spec.baseURL.replace(/\/$/, "")}/models`, {
                        headers: { Accept: "application/json", ...(auth.token ? { Authorization: `Bearer ${auth.token}` } : {}) },
                        signal: request.signal,
                    })
                    if (response.ok) live = await response.json()
                    else warn(`${spec.id} refresh failed (http ${response.status})`)
                } catch (error) {
                    warn(`${spec.id} refresh failed (${request.signal.aborted ? "timeout/aborted" : error instanceof SyntaxError ? "invalid-json" : "network"})`)
                } finally {
                    request.done()
                }
                if (stopped || live === undefined) return
                if (!isValidLiveModelPayload(live)) {
                    warn(`${spec.id} refresh failed (invalid-payload)`)
                    return
                }
                const [enrichment, fxRates] = await Promise.all([enrichmentPromise, ratesPromise])
                if (stopped) return
                const newest = await authScope(spec)
                if (stopped || !newest || newest.scope !== scope) return
                const updated = buildModelInfos(spec, live, { catalog: enrichment, fxRates, lastGood: entry.models.length ? entry.models : cached })
                if (!updated.length && entry.models.length) {
                    warn(`${spec.id} returned no usable models; keeping the last valid inventory`)
                    return
                }
                const changedModels = JSON.stringify(entry.models) !== JSON.stringify(updated)
                const payload: SavedInventory = { providerId: spec.id, scope, savedAt: deps.now(), models: updated }
                try {
                    await writeStorage(`${STORAGE_KEY}:inventory:${spec.id}`, payload)
                } catch { /* in-memory publication can continue */ }
                if (stopped) return
                const finalAuth = await authScope(spec)
                if (stopped || !finalAuth || finalAuth.scope !== scope) return
                entry.models = updated
                entry.connection = finalAuth.connection
                if (changedModels) await reload()
            } catch (error) {
                warn(`${spec.id} refresh failed (${error instanceof TypeError ? "invalid-data" : "provider-error"})`)
            }
        }))
        await Promise.all([enrichmentPromise, ratesPromise])
    }
    const startUpdate = (): void => {
        if (stopped) return
        if (updatePromise) {
            pendingUpdate = true
            return
        }
        updatePromise = update().catch(() => {
            warn("refresh cycle failed")
        }).finally(() => {
            updatePromise = undefined
            if (pendingUpdate && !stopped) {
                pendingUpdate = false
                startUpdate()
            }
        })
    }

    for (const spec of specs) {
        const entry = entries.get(spec.id)!
        const auth = await authScope(spec)
        if (stopped) break
        if (!auth) continue // Lookup failed: no scope yet, so nothing is shown until the first refresh.
        entry.connection = auth.connection
        entry.scope = auth.scope
        const cached = await loadCache(spec, auth.scope)
        if (cached && !stopped) entry.models = cached
    }
    if (entries.size && !stopped) await reload()
    // Credential changes are event-driven; the V2 event stream is the only refresh
    // trigger besides the initial load. Events during an active cycle are queued.
    subscription = (async () => {
        if (!ctx.event || typeof ctx.event.subscribe !== "function") {
            warn("credential events unavailable; model lists refresh on the next load")
            return
        }
        while (!stopped) {
            try {
                for await (const event of ctx.event.subscribe({ signal: eventController.signal })) {
                    if (stopped) break
                    const item = record(event)
                    if (!item) continue
                    // credential.updated carries no payload, so every provider is refreshed.
                    if (item.type === "credential.updated") {
                        startUpdate()
                        continue
                    }
                    if (item.type === "credential.switched") {
                        const integrationID = record(item.data)?.integrationID
                        if (typeof integrationID !== "string" || entries.has(integrationID)) startUpdate()
                    }
                }
            } catch {
                if (stopped) break
                warn("credential event subscription failed; retrying in 30 seconds")
            }
            if (stopped) break
            try {
                await delay(30_000, undefined, { signal: eventController.signal })
            } catch {
                break
            }
        }
    })()
    startUpdate()
    return async () => {
        stopped = true
        pendingUpdate = false
        eventController.abort()
        if (subscription) await subscription
        for (const controller of controllers) controller.abort()
        controllers.clear()
        const trackedUpdate = updatePromise
        if (trackedUpdate) await trackedUpdate
        // The SDK cannot cancel credential resolution; wait for it to settle. Network
        // requests have been aborted, and stopped guards prevent future writes/reloads.
        await disposeRegistrations()
    }
}
