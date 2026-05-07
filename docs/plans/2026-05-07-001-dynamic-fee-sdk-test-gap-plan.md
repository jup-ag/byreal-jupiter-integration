---
title: test: Cover Dynamic-Fee SDK Gaps From Contract Cases
type: test
status: active
date: 2026-05-07
origin: external dynamic_fee_test_cases.md
---

# test: Cover Dynamic-Fee SDK Gaps From Contract Cases

## Summary

This plan maps the external dynamic-fee test-case document to the Jupiter SDK surface and adds only the missing SDK-owned tests. The SDK should cover adapter behavior that Jupiter depends on: off-chain quote inputs, direction mapping, Pyth account handling, account metas, fail-closed routing, and parity with contract helper math. Chain-owned admin instruction behavior and internal on-chain math branches stay outside this SDK test scope except where the SDK calls the same helper and can regress the mapping.

## Problem Frame

The external test-case document enumerates dynamic-fee behavior across math helpers, Pyth validation, admin parameter updates, dynamic flag state, and `swap_v3_dyn` integration. Current repo coverage already includes a meaningful subset in `byreal-clmm-common/tests/dynamic_fee_parity.rs`, `byreal-clmm-jupiter-integration/tests/dynamic_fee_routing.rs`, and LiteSVM parity tests, but the high-risk SDK boundary is not fully covered.

The main gap for this task is not adding every contract case verbatim. The risk is that the SDK can pass through the wrong off-chain inputs to the contract helpers or build the wrong Jupiter route surface even though the on-chain program's own unit tests pass.

## Requirements Trace

- R1. Compare the external `dynamic_fee_test_cases.md` checklist against current SDK tests and classify cases as SDK-owned, already covered, or chain-owned/out of scope.
- R2. Add focused SDK tests for any missing Jupiter adapter risks, prioritizing P0/P1 cases from the source document.
- R3. Preserve existing dynamic-fee route matrix: feature off fails closed for dynamic pools; feature on uses `swap_v3_dyn`-compatible metas; non-dynamic pools do not require Pyth oracles.
- R4. Do not run `cargo fmt` or auto-formatting tools.
- R5. Run `cargo check` after code changes, per repo guidance, and run targeted tests that exercise the new coverage.

## Scope Boundaries

- In scope: SDK tests under `byreal-clmm-common` and `byreal-clmm-jupiter-integration` that validate quote input mapping, Pyth price freshness/non-positive handling, final Pyth oracle meta ordering, directional account metas, and fail-closed dynamic routing.
- In scope: SDK `update()` tests for Pyth account owner and feed-id mismatch, because Jupiter relies on the adapter to reject bad oracle accounts before quoting dynamic pools.
- In scope: SDK fee-input mapping tests for normalized quote trade size and base-vault quote-value conversion, because these values are produced by the adapter before it calls the shared contract helper.
- In scope: Small test-only helpers inside existing modules when needed to observe private SDK mapping logic without changing runtime API.
- Out of scope: On-chain admin instruction tests for `set_swap_dynamic_fee_params`, pool-manager authorization, partial parameter updates, and raw bit setters unless the SDK directly depends on those APIs.
- Out of scope: Rewriting dynamic-fee math formulas or changing production routing behavior.
- Out of scope: Adding new live-RPC or new LiteSVM fixtures unless existing fixtures already make the case straightforward.

## Assumptions

- The source document is a coverage checklist, not an instruction to duplicate contract unit tests one-for-one in the SDK.
- For cases where the SDK imports on-chain helper functions directly, the SDK test should validate the SDK's input mapping rather than reimplementing helper internals.
- Exact numeric expectations should follow the current contract dependency's helper output when it differs from prose examples in the source document.

## Existing Patterns To Follow

- `byreal-clmm-common/tests/dynamic_fee_parity.rs` already validates contract helper parity for arbitrage, trade-slippage, imbalance, total dynamic fee, and base/decay fee.
- `byreal-clmm-jupiter-integration/tests/dynamic_fee_routing.rs` already validates dynamic-pool account update lists and feature-off fail-closed behavior.
- Unit tests in `byreal-clmm-jupiter-integration/src/lib.rs` already validate V2/V3 dynamic account-meta ordering, reverse direction ordering, missing directional tick-array rejection, and Pyth PDA derivation.
- Existing error assertions use substring checks on `anyhow` context.

## Implementation Units

### U1. SDK Coverage Matrix

**Goal:** Record the test-case mapping so future reviewers can see why some source cases are intentionally not copied into Jupiter SDK tests.

**Files:**
- `docs/plans/2026-05-07-001-dynamic-fee-sdk-test-gap-plan.md`

**Approach:** Keep the mapping inside this plan's Test Coverage Matrix. Mark SDK-owned missing cases as implementation units below, and mark chain-owned cases as out of scope.

**Test scenarios:** Documentation-only unit; no direct test execution.

### U2. Quote Direction Mapping Tests

**Goal:** Cover source scenario 11's high-risk `is_buying_base` mapping across quote-token placement and swap direction.

**Files:**
- `byreal-clmm-common/src/lib.rs`

**Approach:** Add focused unit tests that call the SDK's internal fee-rate mapping path with dynamic fee enabled, neutral arbitrage/trade-size fee inputs, and skewed vault balances. Cover token1-as-quote and token0-as-quote pools, with both quote-input and base-input directions. Assert that trades increasing imbalance receive the imbalance fee and trades reducing imbalance do not.

**Source cases covered:** case69, case70, case71, plus the missing fourth combination from the source checklist.

**Test scenarios:**
- token1 is quote, token0 to token1 is selling base and receives imbalance fee when base is already overweight.
- token1 is quote, token1 to token0 is buying base and receives no imbalance fee when it reduces base overweight.
- token0 is quote, token0 to token1 is buying base and receives no imbalance fee when it reduces base overweight.
- token0 is quote, token1 to token0 is selling base and receives imbalance fee when base is already overweight.

### U3. Pyth Price Boundary Tests

**Goal:** Cover SDK-side Pyth price validation from source scenarios 6 and 10 where validation happens during quote computation.

**Files:**
- `byreal-clmm-common/src/lib.rs`

**Approach:** Add unit tests around the SDK's dynamic price loading path through fee computation. Validate non-positive prices, stale prices older than 3600 seconds, and the exactly-3600-second freshness boundary.

**Source cases covered:** case39, case40, case41, and the quote-path part of case66.

**Test scenarios:**
- token0 price equal to zero returns the existing non-positive dynamic-fee error.
- token1 price with `publish_time < current_timestamp - 3600` returns the existing stale dynamic-fee error.
- token prices with `publish_time == current_timestamp - 3600` are accepted and allow fee computation.

### U4. SDK Amount And Vault-Value Mapping Tests

**Goal:** Cover source scenario 5 where the SDK converts raw user amounts and vault balances into dynamic-fee helper inputs.

**Files:**
- `byreal-clmm-common/src/lib.rs`

**Approach:** Add tests through the SDK internal fee-rate path with all unrelated fee components neutralized. Use the trade-slippage component to observe normalized quote trade size and the imbalance component to observe base-vault quote-value conversion. Cover quote token on token1 and quote token on token0 so both conversion branches are exercised.

**Source cases covered:** case32, case33, case34, plus the token0-quote variant implied by SDK support.

**Test scenarios:**
- token1 is quote and a raw 5,001 USDC input with 6 quote decimals triggers only the minimum positive trade-size fee for normalized size 5,001 over threshold 5,000.
- token1 is quote and a sub-1 quote-token input normalizes to zero and does not trigger trade-size fee.
- token1 is quote and base-vault quote-value conversion identifies an overweight base side for imbalance-fee calculation.
- token0 is quote and base-vault quote-value conversion identifies the same overweight base side through the inverse branch.

### U5. Pyth Feed And Meta Regression Tests

**Goal:** Close adapter-level gaps around Pyth feed validation and oracle account ordering.

**Files:**
- `byreal-clmm-jupiter-integration/src/lib.rs`
- `byreal-clmm-jupiter-integration/tests/dynamic_fee_routing.rs`

**Approach:** Extend existing tests rather than adding new fixtures. Add a token1-zero-feed test alongside the existing token0-zero-feed test. Add feature-on account-update tests that prove zero feed ids fail closed by omitting invalid Pyth PDAs while still requiring `update()` to reject invalid dynamic pools. Add direct `update()` tests for wrong Pyth account owner and wrong feed-id content. Keep existing meta-order tests as the source of truth for case67's final `[token0 oracle, token1 oracle]` ordering.

**Source cases covered:** case37, case38, case53, case61, and case67.

**Test scenarios:**
- `get_dynamic_pyth_oracle_addresses` rejects token1 zero feed id with a clear error.
- With dynamic-pool feature enabled, `get_accounts_to_update()` for an invalid dynamic pool includes vaults but does not derive zero-feed Pyth accounts.
- `update()` rejects a dynamic pool with either zero feed id before decoding vault/Pyth accounts.
- `update()` rejects token0 or token1 Pyth accounts owned by a non-Pyth program.
- `update()` rejects a decoded Pyth account whose feed id does not match the pool state's expected feed id.

## Test Coverage Matrix

| Source scenarios | SDK decision |
| --- | --- |
| 1, 2, 3, 4, 7 | Mostly covered by `byreal-clmm-common/tests/dynamic_fee_parity.rs`; this task only adds SDK input-mapping gaps rather than duplicating every helper branch. |
| 5 | SDK-owned where raw user amounts and vault balances are mapped into helper inputs; add missing mapping tests. |
| 6, 10 | SDK-owned where quote/update loads Pyth data and builds metas; add missing non-positive/stale/boundary, owner/feed mismatch, and feed/meta regressions. |
| 8, 9 | Chain/admin-owned; do not add SDK tests except existing dynamic flag route-matrix checks. |
| 11 | SDK-owned high-risk mapping; add the four-combination direction test. |

## Verification Plan

- `cargo check`
- `cargo check -p byreal-clmm-jupiter-integration --features dynamic-pool`
- `cargo test -p byreal-clmm-common dynamic_fee -- --nocapture`
- `cargo test -p byreal-clmm-common direction -- --nocapture`
- `cargo test -p byreal-clmm-jupiter-integration --test dynamic_fee_routing -- --nocapture`
- `cargo test -p byreal-clmm-jupiter-integration --features dynamic-pool --test dynamic_fee_routing -- --nocapture`
- `cargo test -p byreal-clmm-jupiter-integration --lib dynamic_pool -- --nocapture`
- `cargo test -p byreal-clmm-jupiter-integration --features dynamic-pool --lib dynamic_pool -- --nocapture`
