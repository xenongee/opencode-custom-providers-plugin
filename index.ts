import { Plugin } from "@opencode/plugin"
import { setupPlugin } from "./runtime.ts"

export default Plugin.define({ id: "custom-model-providers", setup: setupPlugin })
