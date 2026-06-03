/* tslint:disable */
/* eslint-disable */

export class GeometricAttn {
    free(): void;
    [Symbol.dispose](): void;
    constructor(backend: string, seq_len: number, head_dim: number);
    /**
     * Run the selected backend. q/k/v are Float32Arrays of length seq_len*head_dim.
     * Returns a new Vec<f32> (becomes Float32Array on the JS side).
     */
    run(q: Float32Array, k: Float32Array, v: Float32Array): Float32Array;
}

export function on_start(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_geometricattn_free: (a: number, b: number) => void;
    readonly geometricattn_new: (a: number, b: number, c: number, d: number) => number;
    readonly geometricattn_run: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly on_start: () => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
