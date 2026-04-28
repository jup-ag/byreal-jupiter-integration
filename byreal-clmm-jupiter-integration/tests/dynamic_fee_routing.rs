use anchor_lang::{AccountDeserialize, Discriminator};
use byreal_clmm_common::{PoolState, TickArrayBitmapExtension};
use byreal_clmm_jupiter_integration::{ByrealClmm, BYREAL_CLMM_PROGRAM};
use jupiter_amm_interface::{Amm, AmmContext, KeyedAccount, QuoteParams, SwapMode, SwapParams};
use solana_sdk::{account::Account, pubkey::Pubkey};

#[test]
fn dynamic_pool_accounts_to_update_include_vaults_and_pyth_oracles() {
    let (amm, pool_state) = dynamic_test_amm();
    let accounts = amm.get_accounts_to_update();

    assert!(accounts.contains(&pool_state.token_vault_0));
    assert!(accounts.contains(&pool_state.token_vault_1));
    assert!(accounts.contains(&TickArrayBitmapExtension::key(amm.key())));
    assert!(accounts.contains(&pyth_price_feed_address(&pool_state.token0_pyth_feed_id)));
    assert!(accounts.contains(&pyth_price_feed_address(&pool_state.token1_pyth_feed_id)));
}

#[cfg(not(feature = "dynamic-pool"))]
#[test]
fn dynamic_pool_rejects_quote_and_swap_metas_when_sdk_flag_off() {
    let (amm, pool_state) = dynamic_test_amm();

    let quote_err = amm
        .quote(&QuoteParams {
            amount: 1_000,
            input_mint: pool_state.token_mint_0,
            output_mint: pool_state.token_mint_1,
            swap_mode: SwapMode::ExactIn,
        })
        .err()
        .expect("dynamic pool disabled should reject swap metas");
    assert!(format!("{quote_err:#}").contains("dynamic pool disabled by compile-time feature"));

    let jupiter_program = Pubkey::new_unique();
    let swap_err = amm
        .get_swap_and_account_metas(&SwapParams {
            source_mint: pool_state.token_mint_0,
            destination_mint: pool_state.token_mint_1,
            source_token_account: Pubkey::new_unique(),
            destination_token_account: Pubkey::new_unique(),
            token_transfer_authority: Pubkey::new_unique(),
            quote_mint_to_referrer: None,
            jupiter_program_id: &jupiter_program,
            in_amount: 1_000,
            out_amount: 1,
            missing_dynamic_accounts_as_default: false,
            swap_mode: SwapMode::ExactIn,
        })
        .err()
        .expect("dynamic pool route should fail closed");
    assert!(format!("{swap_err:#}").contains("dynamic pool disabled by compile-time feature"));
}

#[cfg(feature = "dynamic-pool")]
#[test]
fn dynamic_pool_feature_on_still_fails_closed_until_swap_v3_dyn_route_exists() {
    let (amm, pool_state) = dynamic_test_amm();

    let quote_err = amm
        .quote(&QuoteParams {
            amount: 1_000,
            input_mint: pool_state.token_mint_0,
            output_mint: pool_state.token_mint_1,
            swap_mode: SwapMode::ExactIn,
        })
        .err()
        .expect("dynamic pool route should fail closed");
    assert!(format!("{quote_err:#}").contains("dynamic CLMM tx path is not supported yet"));

    let jupiter_program = Pubkey::new_unique();
    let swap_err = amm
        .get_swap_and_account_metas(&SwapParams {
            source_mint: pool_state.token_mint_0,
            destination_mint: pool_state.token_mint_1,
            source_token_account: Pubkey::new_unique(),
            destination_token_account: Pubkey::new_unique(),
            token_transfer_authority: Pubkey::new_unique(),
            quote_mint_to_referrer: None,
            jupiter_program_id: &jupiter_program,
            in_amount: 1_000,
            out_amount: 1,
            missing_dynamic_accounts_as_default: false,
            swap_mode: SwapMode::ExactIn,
        })
        .err()
        .expect("dynamic pool route should fail closed");
    assert!(format!("{swap_err:#}").contains("dynamic CLMM tx path is not supported yet"));
}

fn dynamic_test_amm() -> (ByrealClmm, PoolState) {
    let pool_key = Pubkey::new_unique();
    let mut pool_state = PoolState::default();
    pool_state.amm_config = Pubkey::new_unique();
    pool_state.token_mint_0 = Pubkey::new_unique();
    pool_state.token_mint_1 = Pubkey::new_unique();
    pool_state.token_vault_0 = Pubkey::new_unique();
    pool_state.token_vault_1 = Pubkey::new_unique();
    pool_state.observation_key = Pubkey::new_unique();
    pool_state.tick_spacing = 1;
    pool_state.token0_pyth_feed_id = [1u8; 32];
    pool_state.token1_pyth_feed_id = [2u8; 32];
    pool_state.set_swap_dynamic_fee_enabled(true);

    let data = pool_state_account_data(&pool_state);
    let keyed_account = KeyedAccount {
        key: pool_key,
        account: Account {
            lamports: 1_000_000,
            data,
            owner: BYREAL_CLMM_PROGRAM,
            executable: false,
            rent_epoch: 0,
        },
        params: None,
    };
    let amm = ByrealClmm::from_keyed_account(
        &keyed_account,
        &AmmContext {
            clock_ref: Default::default(),
        },
    )
    .unwrap();

    (amm, pool_state)
}

fn pool_state_account_data(pool_state: &PoolState) -> Vec<u8> {
    let mut data = Vec::with_capacity(PoolState::LEN);
    data.extend_from_slice(&PoolState::DISCRIMINATOR);
    data.extend_from_slice(bytemuck::bytes_of(pool_state));
    let mut verify = data.as_slice();
    PoolState::try_deserialize(&mut verify).unwrap();
    data
}

fn pyth_price_feed_address(feed_id: &[u8; 32]) -> Pubkey {
    Pubkey::find_program_address(&[&0u16.to_le_bytes(), feed_id], &pyth_solana_receiver_sdk::ID).0
}
