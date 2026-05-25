use anchor_lang::prelude::*;
use anyhow::{anyhow, ensure, Result};
use solana_address::{Address, address};
use pyth_solana_receiver_sdk::price_update::{Price, PriceUpdateV2};
use solana_program_pack::Pack;

use spl_token_2022_interface::{
    extension::{transfer_fee::TransferFeeConfig, BaseStateWithExtensions, StateWithExtensions},
    state::{Account as Token2022Account, Mint as Token2022Mint},
};

use byreal_clmm_common::
    PoolState
;

// Program IDs
#[cfg(feature = "mainnet")]
pub const BYREAL_CLMM_PROGRAM: Address =
    address!("REALQqNEomY6cQGZJUGwywTBD2UmDT32rZcNnfxQ5N2");

#[cfg(feature = "devnet")]
pub const BYREAL_CLMM_PROGRAM: Address =
    address!("45iBNkaENereLKMjLm2LHkF3hpDapf6mnvrM5HWFg9cY");

#[cfg(not(any(feature = "mainnet", feature = "devnet")))]
pub const BYREAL_CLMM_PROGRAM: Address =
    address!("REALQqNEomY6cQGZJUGwywTBD2UmDT32rZcNnfxQ5N2");

pub const ID: Address = BYREAL_CLMM_PROGRAM;