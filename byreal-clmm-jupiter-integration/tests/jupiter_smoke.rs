use byreal_clmm_jupiter_integration::ByrealClmm;
use jupiter_amm_interface::{
    AccountMap, Amm, AmmContext, ClockRef, KeyedAccount, QuoteParams, SwapMode, SwapParams,
};
use solana_client::rpc_client::RpcClient;
use solana_sdk::pubkey::Pubkey;
use std::str::FromStr;

#[test]
fn jupiter_integration_smoke() {
    if std::env::var("RUN_INTEGRATION_TESTS").is_err() {
        println!("Skipping integration test. Set RUN_INTEGRATION_TESTS=1 to run.");
        return;
    }

    let pool_address = Pubkey::from_str("J4jiEPEu8c8nLdpkiMa7k1P8rL1HCJSNxCvzA5DsmYds").unwrap();
    let rpc = RpcClient::new("https://api.mainnet-beta.solana.com");
    let account = rpc.get_account(&pool_address).unwrap();
    let keyed_account = KeyedAccount {
        key: pool_address,
        account: account.into(),
        params: None,
    };
    let amm_context = AmmContext {
        clock_ref: ClockRef::default(),
    };
    let mut amm = ByrealClmm::from_keyed_account(&keyed_account, &amm_context).unwrap();

    println!("Pool: {}", amm.key());
    println!("Label: {}", amm.label());
    println!("Program: {}", amm.program_id());

    let accounts_to_update = amm.get_accounts_to_update();
    println!("Accounts to update: {}", accounts_to_update.len());
    let accounts = rpc.get_multiple_accounts(&accounts_to_update).unwrap();
    let account_map: AccountMap = accounts_to_update
        .iter()
        .enumerate()
        .filter_map(|(i, key)| accounts[i].as_ref().map(|account| (*key, account.clone().into())))
        .collect();
    amm.update(&account_map).unwrap();

    let mints = amm.get_reserve_mints();
    println!("Reserve mints: {:?}", mints);

    let quote_in = amm
        .quote(&QuoteParams {
            amount: 100_000,
            input_mint: mints[0],
            output_mint: mints[1],
            swap_mode: SwapMode::ExactIn,
        })
        .unwrap();
    println!(
        "Quote exact-in: in={}, out={}, fee={}",
        quote_in.in_amount, quote_in.out_amount, quote_in.fee_amount
    );

    let quote_out = amm
        .quote(&QuoteParams {
            amount: quote_in.out_amount,
            input_mint: mints[0],
            output_mint: mints[1],
            swap_mode: SwapMode::ExactOut,
        })
        .unwrap();
    println!(
        "Quote exact-out: in={}, out={}, fee={}",
        quote_out.in_amount, quote_out.out_amount, quote_out.fee_amount
    );

    let jupiter_program = Pubkey::new_unique();
    let swap_params = SwapParams {
        source_mint: mints[0],
        destination_mint: mints[1],
        source_token_account: Pubkey::new_unique(),
        destination_token_account: Pubkey::new_unique(),
        token_transfer_authority: Pubkey::new_unique(),
        quote_mint_to_referrer: None,
        jupiter_program_id: &jupiter_program,
        in_amount: quote_in.in_amount,
        out_amount: quote_in.out_amount,
        missing_dynamic_accounts_as_default: false,
        swap_mode: SwapMode::ExactIn,
    };
    let swap_and_metas = amm.get_swap_and_account_metas(&swap_params).unwrap();
    println!("Swap accounts: {}", swap_and_metas.account_metas.len());
    assert!(amm.has_dynamic_accounts());
}
