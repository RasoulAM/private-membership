pub mod bits;
pub mod client;
pub mod convolution;
pub mod lwe;
pub mod measurement;
pub mod modulus_switch;
pub mod noise_analysis;
pub mod packing;
pub mod params;
pub mod scheme;
pub mod transpose;
pub mod util;
pub mod commons;

// WebAssembly bindings
use wasm_bindgen::prelude::*;
use web_sys;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use spiral_rs::{
    client::*,
    gadget::*,
};
use spiral_rs::poly::{PolyMatrix, PolyMatrixRaw, PolyMatrixNTT};
use crate::{client::*, measurement::*, modulus_switch::*, packing::*, params::*, scheme::*};
use crate::commons::*;

// WebAssembly-exposed struct that holds state
#[wasm_bindgen]
pub struct ClientQueryState {
    num_items: usize, 
    dim0: usize, 
    item_size_bits: usize, 
}

#[wasm_bindgen]
impl ClientQueryState {
    
    #[wasm_bindgen(constructor)]
    pub fn new(num_items: usize, dim0: usize, item_size_bits: usize) -> ClientQueryState {
        ClientQueryState {
            num_items,
            dim0,
            item_size_bits,
        }
    }

    #[wasm_bindgen]
    pub fn query(&self, which_item: usize) -> Vec<u8> {
        // --- 1. Initial computation to generate a query ---
        let mut _measurement = Measurement::default();

        let num_items = self.num_items;
        let dim0 = self.dim0;
        let item_size_bits = self.item_size_bits;
        
        let (params, interpolate_degree, _) 
            = params_rgswpir_given_input_size_and_dim0(num_items, item_size_bits, dim0);

        let gamma = params.poly_len;
        let packing_type = PackingType::InspiRING;

        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_cols = params.instances * params.poly_len;

        // RLWE reduced moduli
        let _rlwe_q_prime_1 = params.get_q_prime_1();
        let _rlwe_q_prime_2 = params.get_q_prime_2();

        let _db_cols_prime = db_cols / gamma;

        // ================================================================
        // QUERY GENERATION PHASE
        // ================================================================
        let mut client = Client::init(&params);

        let items_per_row = num_items / db_rows;
        let target_row = which_item / items_per_row;
        let target_sub_col = which_item % items_per_row;

        let sk_reg = client.get_sk_reg();

        let packing_params = PackParams::new_fast(&params, gamma);

        let mut packing_keys = match packing_type {
            PackingType::InspiRING => {
                if gamma <= params.poly_len / 2 {
                    PackingKeys::init(&packing_params, sk_reg, W_SEED)
                } else {
                    PackingKeys::init_full(&packing_params, sk_reg, W_SEED, V_SEED)
                }
            },
            PackingType::CDKS => {
                PackingKeys::init_cdks(&params, sk_reg, STATIC_SEED_2)
            },
            PackingType::NoPacking => {
                panic!("Shouldn't be here");
            }
        };

        let y_client = YClient::new(&mut client, &params);
        let mut ct_gsw_body = PolyMatrixNTT::zero(&params, 1, 2 * params.t_gsw);

        let bits_per = get_bits_per(&params, params.t_gsw);
        for j in 0..params.t_gsw {
            let mut sigma = PolyMatrixRaw::zero(&params, 1, 1);
            let exponent = (2 * params.poly_len * target_sub_col / interpolate_degree) % (2 * params.poly_len);
            sigma.get_poly_mut(0, 0)[exponent % params.poly_len] = if exponent < params.poly_len {
                1u64 << (bits_per * j)
            } else {
                params.modulus - (1u64 << (bits_per * j))
            };
            let sigma_ntt = sigma.ntt();
            let ct = y_client.client().encrypt_matrix_reg(
                &sigma_ntt,
                &mut ChaCha20Rng::from_entropy(),
                &mut ChaCha20Rng::from_seed(RGSW_SEEDS[2*j+1]),
            );
            ct_gsw_body.copy_into(&ct.submatrix(1, 0, 1, 1), 0, 2 * j + 1);
            
            let prod = &y_client.client().get_sk_reg().ntt() * &sigma_ntt;
            let ct = &y_client.client().encrypt_matrix_reg(
                &prod,
                &mut ChaCha20Rng::from_entropy(),
                &mut ChaCha20Rng::from_seed(RGSW_SEEDS[2*j]),
            );
            ct_gsw_body.copy_into(&ct.submatrix(1, 0, 1, 1), 0, 2 * j);
        }

        // Generate query row and pack it
        let query_row = y_client.generate_query_over_prime(
            SEED_0,
            params.db_dim_1,
            packing_type,
            target_row,
        );

        assert_eq!(query_row.len(), (params.poly_len + 1) * db_rows);
        let query_row_last_row: &[u64] = &query_row[params.poly_len * db_rows..];
        
        assert_eq!(query_row_last_row.len(), db_rows);
        let packed_query_row = pack_query(&params, query_row_last_row);
        
        // Serialize the query for transmission
        serialize_everything(&params, &mut packing_keys, packed_query_row, ct_gsw_body)
    }

    #[wasm_bindgen]
    pub fn extract_result(&self, response_data: &[u8], which_item: usize) -> Vec<u32> {
        let num_items = self.num_items;
        let dim0 = self.dim0;
        let item_size_bits = self.item_size_bits;

        web_sys::console::log_1(&format!("num_items={}", num_items).into());
        web_sys::console::log_1(&format!("dim0={}", dim0).into());
        web_sys::console::log_1(&format!("item_size_bits={}", item_size_bits).into());

        let (params, interpolate_degree, _) 
            = params_rgswpir_given_input_size_and_dim0(num_items, item_size_bits, dim0);

        let gamma = params.poly_len;
        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_cols = params.instances * params.poly_len;

        web_sys::console::log_1(&format!("params.instances={}", params.instances).into());


        // RLWE reduced moduli
        let rlwe_q_prime_1 = params.get_q_prime_1();
        let rlwe_q_prime_2 = params.get_q_prime_2();

        let db_cols_prime = db_cols / gamma;

        let mut client = Client::init(&params);
        let y_client = YClient::new(&mut client, &params);
        web_sys::console::log_1(&"here".into());

        // let items_per_row = num_items / db_rows;
        // let target_row = which_item / items_per_row;
        // let target_sub_col = which_item % items_per_row;

        let q_1_bits = (rlwe_q_prime_1 as f64).log2().ceil() as usize;
        let q_2_bits = (rlwe_q_prime_2 as f64).log2().ceil() as usize;
        let total_sz_bits = ((q_2_bits * params.poly_len + q_1_bits * params.poly_len + 63) / 64) * 64;
        let total_sz_bytes = (total_sz_bits + 7) / 8;
        web_sys::console::log_1(&"here".into());

        let c = db_cols_prime / interpolate_degree;
        web_sys::console::log_1(&format!("db_cols_prime: {}", db_cols_prime).into());
        web_sys::console::log_1(&format!("interpolate_degree: {}", interpolate_degree).into());
        web_sys::console::log_1(&format!("response_data length: {}", response_data.len()).into());
        
        let chunk_size = response_data.len() / c;

        let sum_switched: Vec<Vec<u8>> = response_data
            .chunks_exact(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        web_sys::console::log_1(&format!("c={}", c).into());
        web_sys::console::log_1(&format!("c={}", c).into());
        
        let mut results = Vec::new();
        web_sys::console::log_1(&format!("c={}", c).into());
        web_sys::console::log_1(&format!("sum_switched length: {}", sum_switched.len()).into());
        for which_poly in 0..c {
            let sum = PolyMatrixRaw::recover_how_many(&params, rlwe_q_prime_1, rlwe_q_prime_2, gamma, sum_switched[which_poly].as_slice());
            results.push(sum);
        }

        web_sys::console::log_1(&"here".into());
        let rgsw_ans = results.iter().flat_map(|ct| {
            decrypt_ct_reg_measured(y_client.client(), &params, &ct.ntt(), params.poly_len).as_slice()[..gamma]
                .iter()
                .map(|&x| x as u32)  // Convert u64 to u32 since rescaled values fit in u32
                .collect::<Vec<u32>>()
        }).collect::<Vec<_>>();
        
        // Log initial data
        web_sys::console::log_1(&format!("rgsw_ans length: {}", rgsw_ans.len()).into());
        web_sys::console::log_1(&format!("First 10 rgsw_ans values: {:?}", &rgsw_ans[..std::cmp::min(10, rgsw_ans.len())]).into());
        
        // Convert u32 array (with 16-bit data per element) to byte array
        // Based on commons.rs encoding: each u16 is packed as (high_byte << 8 | low_byte)
        let mut bytes = Vec::new();
        for value in &rgsw_ans {
            // Extract the lower 16 bits and convert to 2 bytes
            let value_16bit = *value as u16;
            // Unpack as: high byte first, then low byte (matching commons.rs encoding)
            bytes.push((value_16bit >> 8) as u8);  // high byte
            bytes.push((value_16bit & 0xFF) as u8); // low byte
        }
        
        web_sys::console::log_1(&format!("Bytes length: {}", bytes.len()).into());
        web_sys::console::log_1(&format!("First 16 bytes: {:?}", &bytes[..std::cmp::min(16, bytes.len())]).into());
        
        // Extract the first 8 bytes as big-endian content length
        if bytes.len() < 8 {
            web_sys::console::log_1(&"Not enough data - less than 8 bytes".into());
            return Vec::new(); // Not enough data
        }
        
        let length_bytes: [u8; 8] = [
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7]
        ];
        let content_length = u64::from_be_bytes(length_bytes) as usize;
        
        web_sys::console::log_1(&format!("Length bytes: {:?}", length_bytes).into());
        web_sys::console::log_1(&format!("Content length: {}", content_length).into());
        
        // Skip the first 8 bytes (length header) and take only the specified amount of content
        let content_start = 8;
        let content_end = content_start + content_length;
        
        web_sys::console::log_1(&format!("Content range: {} to {}", content_start, content_end).into());
        web_sys::console::log_1(&format!("Available bytes: {}", bytes.len()).into());
        
        if content_end > bytes.len() {
            web_sys::console::log_1(&format!("Not enough data - need {} bytes but only have {}", content_end, bytes.len()).into());
            return Vec::new(); // Not enough data for the specified content length
        }
        
        // For u32 array indices (since user changed to return u32 values)
        let u32_content_start = content_start / 2;
        let u32_content_end = content_end / 2;
        
        web_sys::console::log_1(&format!("U32 content range: {} to {}", u32_content_start, u32_content_end).into());
        web_sys::console::log_1(&format!("Result length: {}", u32_content_end - u32_content_start).into());
        
        if u32_content_end <= rgsw_ans.len() {
            let result = rgsw_ans[u32_content_start..u32_content_end].to_vec();
            web_sys::console::log_1(&format!("First 10 result values: {:?}", &result[..std::cmp::min(10, result.len())]).into());
            result
        } else {
            web_sys::console::log_1(&format!("U32 range out of bounds - need {} but rgsw_ans has {}", u32_content_end, rgsw_ans.len()).into());
            Vec::new()
        }
    }
}  