use byreal_clmm::libraries::{
    dynamic_fee_math::{
        calculate_arbitrage_fee, calculate_dynamic_fee_rate, calculate_imbalance_fee,
        calculate_trade_slippage_fee, normalize_trade_size, DynamicFeeInputs,
    },
    fixed_point_64,
};
use byreal_clmm::states::{AmmConfig, PoolState};

#[test]
fn normalize_trade_size_matches_contract_cases() {
    assert_eq!(normalize_trade_size(250_000_000u128, 6).unwrap(), 250);
    assert_eq!(normalize_trade_size(250u128, 0).unwrap(), 250);
}

#[test]
fn arbitrage_fee_matches_contract_cases() {
    let p_index = fixed_point_64::Q64;
    let p_0 = fixed_point_64::Q64 * 101 / 100;

    assert_eq!(
        calculate_arbitrage_fee(p_0, p_index, 0, 1_000).unwrap(),
        9_000
    );
    assert_eq!(
        calculate_arbitrage_fee(p_0, p_index, 9_000, 1_000).unwrap(),
        0
    );
    assert_eq!(
        calculate_arbitrage_fee(fixed_point_64::Q64 * 99 / 100, p_index, 0, 1_000).unwrap(),
        9_001
    );
    assert_eq!(
        calculate_arbitrage_fee(fixed_point_64::Q64 * 99 / 100, p_index, 9_000, 1_000).unwrap(),
        1
    );
    assert_eq!(
        calculate_arbitrage_fee(fixed_point_64::Q64, fixed_point_64::Q64, 0, 1_000).unwrap(),
        0
    );
    assert_eq!(
        calculate_arbitrage_fee(fixed_point_64::Q64 * 105 / 100, p_index, 5_000, 1_000).unwrap(),
        44_000
    );
}

#[test]
fn trade_slippage_fee_matches_contract_cases() {
    assert_eq!(calculate_trade_slippage_fee(100, 10, 1).unwrap(), 0);
    assert_eq!(calculate_trade_slippage_fee(200, 10, 1).unwrap(), 100_000);
    assert_eq!(calculate_trade_slippage_fee(500, 0, 1).unwrap(), 0);
    assert_eq!(calculate_trade_slippage_fee(100, 10, 0).unwrap(), 100_000);
    assert_eq!(calculate_trade_slippage_fee(101, 10, 1).unwrap(), 10_000);
}

#[test]
fn imbalance_fee_matches_contract_cases() {
    assert_eq!(
        calculate_imbalance_fee(150, 50, 5, 10, false).unwrap(),
        200_000
    );
    assert_eq!(calculate_imbalance_fee(150, 50, 5, 10, true).unwrap(), 0);
    assert_eq!(calculate_imbalance_fee(100, 100, 5, 10, false).unwrap(), 0);
    assert_eq!(
        calculate_imbalance_fee(50, 150, 5, 10, true).unwrap(),
        200_000
    );
    assert_eq!(calculate_imbalance_fee(0, 0, 5, 10, false).unwrap(), 0);
    assert_eq!(calculate_imbalance_fee(550, 450, 5, 10, false).unwrap(), 0);
}

#[test]
fn dynamic_fee_rate_matches_contract_cases() {
    let p_index = fixed_point_64::Q64;
    let p_0 = fixed_point_64::Q64 * 101 / 100;

    let result = calculate_dynamic_fee_rate(&DynamicFeeInputs {
        p_0,
        p_index,
        trade_size: 200,
        quote_value_of_base: 150,
        quote_balance: 50,
        is_buying_base: false,
        fee_base: 1_000,
        arbitrage_fee_buffer_ppm: 0,
        trade_slippage_fee_base: 10,
        trade_slippage_fee_trade_size_threshold: 1,
        imbalance_fee_base: 5,
        imbalance_fee_x: 100,
    })
    .unwrap();
    assert_eq!(result.arbitrage_fee, 9_000);
    assert_eq!(result.trade_slippage_fee, 100_000);
    assert_eq!(result.imbalance_fee, 0);
    assert_eq!(result.swap_dynamic_fee, 109_000);
    assert_eq!(result.total_fee_rate, 110_000);

    let zero_result = calculate_dynamic_fee_rate(&DynamicFeeInputs {
        p_0: fixed_point_64::Q64,
        p_index: fixed_point_64::Q64,
        trade_size: 0,
        quote_value_of_base: 100,
        quote_balance: 100,
        is_buying_base: false,
        fee_base: 1_000,
        arbitrage_fee_buffer_ppm: 0,
        trade_slippage_fee_base: 10,
        trade_slippage_fee_trade_size_threshold: 1,
        imbalance_fee_base: 5,
        imbalance_fee_x: 10,
    })
    .unwrap();
    assert_eq!(zero_result.arbitrage_fee, 0);
    assert_eq!(zero_result.trade_slippage_fee, 0);
    assert_eq!(zero_result.imbalance_fee, 0);
    assert_eq!(zero_result.swap_dynamic_fee, 0);
    assert_eq!(zero_result.total_fee_rate, 1_000);

    assert!(calculate_dynamic_fee_rate(&DynamicFeeInputs {
        p_0: fixed_point_64::Q64 * 3,
        p_index,
        trade_size: 0,
        quote_value_of_base: 100,
        quote_balance: 100,
        is_buying_base: false,
        fee_base: 500_000,
        arbitrage_fee_buffer_ppm: 0,
        trade_slippage_fee_base: 0,
        trade_slippage_fee_trade_size_threshold: 1,
        imbalance_fee_base: 0,
        imbalance_fee_x: 10,
    })
    .is_err());

    let all_nonzero = calculate_dynamic_fee_rate(&DynamicFeeInputs {
        p_0: fixed_point_64::Q64 * 105 / 100,
        p_index,
        trade_size: 500,
        quote_value_of_base: 700,
        quote_balance: 300,
        is_buying_base: false,
        fee_base: 1_000,
        arbitrage_fee_buffer_ppm: 5_000,
        trade_slippage_fee_base: 5,
        trade_slippage_fee_trade_size_threshold: 1,
        imbalance_fee_base: 3,
        imbalance_fee_x: 5,
    })
    .unwrap();
    assert_eq!(all_nonzero.arbitrage_fee, 44_000);
    assert_eq!(all_nonzero.trade_slippage_fee, 100_000);
    assert_eq!(all_nonzero.imbalance_fee, 105_000);
    assert_eq!(all_nonzero.swap_dynamic_fee, 249_000);
    assert_eq!(all_nonzero.total_fee_rate, 250_000);
}

#[test]
fn base_trade_fee_matches_contract_cases() {
    let mut pool_state = PoolState::default();
    let mut amm_config = AmmConfig::default();
    amm_config.trade_fee_rate = 1_000;

    assert_eq!(
        pool_state
            .calculate_base_trade_fee_rate(&amm_config, true, 0)
            .unwrap(),
        1_000
    );

    pool_state.trade_fee_rate = 5_000;
    assert_eq!(
        pool_state
            .calculate_base_trade_fee_rate(&amm_config, true, 0)
            .unwrap(),
        5_000
    );

    pool_state
        .initialize_decay_fee(true, false, 80, 10, 10)
        .unwrap();
    assert_eq!(
        pool_state
            .calculate_base_trade_fee_rate(&amm_config, true, 0)
            .unwrap(),
        800_000
    );
    assert_eq!(
        pool_state
            .calculate_base_trade_fee_rate(&amm_config, false, 0)
            .unwrap(),
        5_000
    );
}
