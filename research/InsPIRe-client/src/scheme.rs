use std::collections::HashMap;
use std::time::Instant;

use log::debug;
use rand::{thread_rng, Rng};

use spiral_rs::aligned_memory::AlignedMemory64;
use spiral_rs::arith::rescale;
use spiral_rs::poly::{PolyMatrix, PolyMatrixRaw};
use spiral_rs::{client::*, params::*};

use crate::bits::{read_bits, u64s_to_contiguous_bytes};
use crate::modulus_switch::ModulusSwitch;
use crate::noise_analysis::YPIRSchemeParams;
use crate::packing::{PackingKeys, PackingType, ToStr};

use super::{client::*, lwe::LWEParams, measurement::*, params::*};
use strum_macros::{Display, EnumString};

pub const STATIC_PUBLIC_SEED: [u8; 32] = [0u8; 32];
pub const SEED_0: u8 = 0;
pub const SEED_1: u8 = 1;

pub const STATIC_SEED_2: [u8; 32] = [
    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

pub const W_SEED: [u8; 32] = [
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
];
pub const V_SEED: [u8; 32] = [
    8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
];

// have three protocol types called "simplepir", "doublepir", and "InsPIRe"
// implement a display function too
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
pub enum ProtocolType {
    #[strum(serialize = "SimplePIR")]
    SimplePIR,
    #[strum(serialize = "DoublePIR")]
    DoublePIR,
    #[strum(serialize = "InsPIRe")]
    InsPIRe,
}

impl Default for ProtocolType {
    fn default() -> Self {
        ProtocolType::DoublePIR // Specify the default variant
    }
}

impl ToStr for ProtocolType {
    fn to_str(&self) -> String {
        match self {
            ProtocolType::SimplePIR => "SimplePIR",
            ProtocolType::DoublePIR => "DoublePIR",
            ProtocolType::InsPIRe => "InsPIRe",            
        }.to_string()
    } 
}

pub trait Sample {
    fn sample() -> Self;
}

impl Sample for u8 {
    fn sample() -> Self {
        fastrand::u8(..)
    }
}

impl Sample for u16 {
    fn sample() -> Self {
        fastrand::u16(..)
    }
}

impl Sample for u32 {
    fn sample() -> Self {
        fastrand::u32(..)
    }
}

pub fn mean(xs: &[usize]) -> f64 {
    xs.iter().map(|x| *x as f64).sum::<f64>() / xs.len() as f64
}

pub fn mean_f64(xs: &[f64]) -> f64 {
    xs.iter().map(|x| *x as f64).sum::<f64>() / xs.len() as f64
}

pub fn std_dev(xs: &[usize]) -> f64 {
    let mean = mean(xs);
    let mut variance = 0.;
    for x in xs {
        variance += (*x as f64 - mean).powi(2);
    }
    (variance / xs.len() as f64).sqrt()
}

pub fn std_dev_f64(xs: &[f64]) -> f64 {
    let mean = mean_f64(xs);
    let mut variance = 0.;
    for x in xs {
        variance += (*x as f64 - mean).powi(2);
    }
    (variance / xs.len() as f64).sqrt()
}
