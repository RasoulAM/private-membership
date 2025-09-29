use log::debug;

use spiral_rs::{
    arith::*, discrete_gaussian::*, gadget::*, ntt::*, number_theory::invert_uint_mod, params::*,
    poly::*,
};

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::str::FromStr;

use super::{client::*, measurement::*};

use super::util::*;

pub fn gadget_invert_transposed_alloc<'a>(
    inp: &PolyMatrixRaw<'a>,
    num_digits: usize,
) -> PolyMatrixRaw<'a> {
    assert_eq!(inp.cols, 1);

    let params = inp.params;
    let mut out = PolyMatrixRaw::zero(&params, inp.rows, num_digits);

    let num_elems = num_digits;
    let bits_per = get_bits_per(params, num_elems);
    let mask = (1u64 << bits_per) - 1;

    for i in 0..inp.rows {
        for z in 0..params.poly_len {
            let val = inp.get_poly(i, 0)[z];
            for k in 0..num_elems {
                let bit_offs = k * bits_per;
                let piece = if bit_offs >= 64 { 0 } else { (val >> bit_offs) & mask };
                out.get_poly_mut(i, k)[z] = piece;
            }
        }
    }
    out
}

fn homomorphic_automorph<'a>(
    params: &'a Params,
    t: usize,
    t_exp: usize,
    ct: &PolyMatrixNTT<'a>,
    pub_param: &PolyMatrixNTT<'a>,
) -> PolyMatrixNTT<'a> {
    assert_eq!(ct.rows, 2);
    assert_eq!(ct.cols, 1);

    let ct_raw = ct.raw();
    let ct_auto = automorph_alloc(&ct_raw, t);

    let mut ginv_ct = PolyMatrixRaw::zero(params, t_exp, 1);
    gadget_invert_rdim(&mut ginv_ct, &ct_auto, 1);
    let mut ginv_ct_ntt = PolyMatrixNTT::zero(params, t_exp, 1);
    for i in 1..t_exp {
        let pol_src: &[u64] = ginv_ct.get_poly(i, 0);
        let pol_dst = ginv_ct_ntt.get_poly_mut(i, 0);
        reduce_copy(params, pol_dst, pol_src);
        ntt_forward(params, pol_dst);
    }
    // let ginv_ct_ntt = ginv_ct.ntt();
    let w_times_ginv_ct = pub_param * &ginv_ct_ntt;

    let mut ct_auto_1 = PolyMatrixRaw::zero(params, 1, 1);
    ct_auto_1.data.as_mut_slice().copy_from_slice(ct_auto.get_poly(1, 0));
    let ct_auto_1_ntt = ct_auto_1.ntt();

    &ct_auto_1_ntt.pad_top(1) + &w_times_ginv_ct
}

pub fn pack_lwes_inner<'a>(
    params: &'a Params,
    ell: usize,
    start_idx: usize,
    rlwe_cts: &[PolyMatrixNTT<'a>],
    pub_params: &[PolyMatrixNTT<'a>],
    y_constants: &(Vec<PolyMatrixNTT<'a>>, Vec<PolyMatrixNTT<'a>>),
) -> PolyMatrixNTT<'a> {
    assert_eq!(pub_params.len(), params.poly_len_log2);

    if ell == 0 {
        return rlwe_cts[start_idx].clone();
    }

    let step = 1 << (params.poly_len_log2 - ell);
    let even = start_idx;
    let odd = start_idx + step;

    let mut ct_even = pack_lwes_inner(params, ell - 1, even, rlwe_cts, pub_params, y_constants);
    let ct_odd = pack_lwes_inner(params, ell - 1, odd, rlwe_cts, pub_params, y_constants);

    let (y, neg_y) = (&y_constants.0[ell - 1], &y_constants.1[ell - 1]);

    let y_times_ct_odd = scalar_multiply_alloc(&y, &ct_odd);
    let neg_y_times_ct_odd = scalar_multiply_alloc(&neg_y, &ct_odd);

    let mut ct_sum_1 = ct_even.clone();
    add_into(&mut ct_sum_1, &neg_y_times_ct_odd);
    add_into(&mut ct_even, &y_times_ct_odd);

    // let now = Instant::now();
    let ct_sum_1_automorphed = homomorphic_automorph(
        params,
        (1 << ell) + 1,
        params.t_exp_left,
        &ct_sum_1,
        &pub_params[params.poly_len_log2 - 1 - (ell - 1)],
    );
    // debug!("Homomorphic automorph in {} us", now.elapsed().as_micros());

    &ct_even + &ct_sum_1_automorphed
}

pub fn add_all_rotations<'a>(
    params: &'a Params,
    pub_params: &[PolyMatrixNTT<'a>],
    lwe_ct: &PolyMatrixNTT<'a>,
) -> PolyMatrixNTT<'a> {
    // computing:
    // r0 = f
    // r1 = r0 + automorph(r0, ts[0])
    // r2 = r1 + automorph(r1, ts[1])
    // ...
    // r_\log d = ...

    let mut cur_r = lwe_ct.clone();
    for i in 0..params.poly_len_log2 {
        let t = (params.poly_len / (1 << i)) + 1;
        let pub_param = &pub_params[i];
        let tau_of_r = homomorphic_automorph(params, t, params.t_exp_left, &cur_r, pub_param);
        add_into(&mut cur_r, &tau_of_r);
    }
    cur_r
}

pub fn fast_barrett_raw_u64(input: u64, const_ratio_1: u64, modulus: u64) -> u64 {
    let tmp = (((input as u128) * (const_ratio_1 as u128)) >> 64) as u64;

    // Barrett subtraction
    let res = input - tmp * modulus;

    res
}

pub fn fast_add_into(res: &mut PolyMatrixNTT, a: &PolyMatrixNTT) {
    assert!(res.rows == a.rows);
    assert!(res.cols == a.cols);

    let params = res.params;
    for i in 0..res.rows {
        for j in 0..res.cols {
            let res_poly = res.get_poly_mut(i, j);
            let a_poly = a.get_poly(i, j);
            for c in 0..params.crt_count {
                for i in 0..params.poly_len {
                    let idx = c * params.poly_len + i;
                    unsafe {
                        let p_res = res_poly.as_mut_ptr().add(idx);
                        let p_a = a_poly.as_ptr().add(idx);
                        let val = *p_res + *p_a;
                        let reduced =
                            fast_barrett_raw_u64(val, params.barrett_cr_1[c], params.moduli[c]);
                        *p_res = reduced;
                    }
                }
            }
        }
    }
}

pub fn condense_matrix<'a>(params: &'a Params, a: &PolyMatrixNTT<'a>) -> PolyMatrixNTT<'a> {
    if params.crt_count == 2 {
        let mut res = PolyMatrixNTT::zero(params, a.rows, a.cols);
        for i in 0..a.rows {
            for j in 0..a.cols {
                let res_poly = &mut res.get_poly_mut(i, j);
                let a_poly = a.get_poly(i, j);
                for z in 0..params.poly_len {
                    res_poly[z] = a_poly[z] | (a_poly[z + params.poly_len] << 32); // TODO: Does this assume that the digit decomposition is exactly 2?
                }
            }
        }
        res
    } else {
        a.clone()
    }
}

pub fn uncondense_matrix<'a>(params: &'a Params, a: &PolyMatrixNTT<'a>) -> PolyMatrixNTT<'a> {
    let mut res = PolyMatrixNTT::zero(params, a.rows, a.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            let res_poly = &mut res.get_poly_mut(i, j);
            let a_poly = a.get_poly(i, j);
            for z in 0..params.poly_len {
                res_poly[z] = a_poly[z] & ((1u64 << 32) - 1);
                res_poly[z + params.poly_len] = a_poly[z] >> 32;
            }
        }
    }
    res
}


pub fn fast_add_into_no_reduce(res: &mut PolyMatrixNTT, a: &PolyMatrixNTT) {
    assert!(res.rows == a.rows);
    assert!(res.cols == a.cols);

    let a_slc = a.as_slice();
    let res_slc = res.as_mut_slice();
    for (res_chunk, a_chunk) in res_slc.chunks_exact_mut(8).zip(a_slc.chunks_exact(8)) {
        for i in 0..8 {
            res_chunk[i] += a_chunk[i];
        }
    }
}

pub fn fast_reduce(res: &mut PolyMatrixNTT) {
    let params = res.params;
    let res_slc = res.as_mut_slice();
    for m in 0..params.crt_count {
        for i in 0..params.poly_len {
            let idx = m * params.poly_len + i;
            // res_slc[idx] = barrett_coeff_u64(params, res_slc[idx], m);
            unsafe {
                let p = res_slc.as_mut_ptr().add(idx);
                *p = barrett_coeff_u64(params, *p, m);
            }
        }
    }
}

pub fn combine<'a>(
    params: &'a Params,
    cur_ell: usize,
    ct_even: &mut PolyMatrixNTT<'a>,
    ct_odd: &PolyMatrixNTT<'a>,
    pub_params: &[PolyMatrixNTT<'a>],
    y_constants: &(Vec<PolyMatrixNTT<'a>>, Vec<PolyMatrixNTT<'a>>),
) {
    let (y, neg_y) = (&y_constants.0[cur_ell - 1], &y_constants.1[cur_ell - 1]);

    let y_times_ct_odd = scalar_multiply_alloc(&y, &ct_odd);
    let neg_y_times_ct_odd = scalar_multiply_alloc(&neg_y, &ct_odd);

    let mut ct_sum_1 = ct_even.clone();
    add_into(&mut ct_sum_1, &neg_y_times_ct_odd);
    add_into(ct_even, &y_times_ct_odd);

    let ct_sum_1_automorphed = homomorphic_automorph(
        params,
        (1 << cur_ell) + 1,
        params.t_exp_left,
        &ct_sum_1,
        &pub_params[params.poly_len_log2 - 1 - (cur_ell - 1)],
    );

    add_into(ct_even, &ct_sum_1_automorphed);
}

pub fn prep_pack_lwes<'a>(
    params: &'a Params,
    lwe_cts: &[u64],
    // packed_per_rlwe: usize,
) -> Vec<PolyMatrixNTT<'a>> {
    let lwe_cts_size = params.poly_len * (params.poly_len + 1);
    assert_eq!(lwe_cts.len(), lwe_cts_size);

    // assert!(cols_to_do == params.poly_len);

    let mut rlwe_cts: Vec<PolyMatrixNTT<'_>> = Vec::new();
    for i in 0..params.poly_len {
        let mut rlwe_ct = PolyMatrixRaw::zero(params, 2, 1);

        // 'a' vector
        // put this in negacyclic order
        let mut poly = Vec::new();
        for j in 0..params.poly_len {
            poly.push(lwe_cts[j * params.poly_len + i])
        }
        let nega = negacyclic_perm(&poly, 0, params.modulus);

        for j in 0..params.poly_len {
            rlwe_ct.get_poly_mut(0, 0)[j] = nega[j];
        }
        // 'b' scalar (skip)

        rlwe_cts.push(rlwe_ct.ntt());
    }

    rlwe_cts
}

// (d+1) * t * d
// v = (0 -> d) + (td -> td+d) + (2td -> 2td+d) ... + ()

pub fn prep_pack_many_lwes<'a>(
    params: &'a Params,
    lwe_cts: &[u64],
    num_rlwe_outputs: usize, // num_rlwe_outputs = db_cols / gamma
    gamma: usize,
) -> Vec<Vec<PolyMatrixNTT<'a>>> {
    let lwe_cts_size = (params.poly_len + 1) * (num_rlwe_outputs * gamma); // = (params.poly_len + 1) * db_cols
    assert_eq!(lwe_cts.len(), lwe_cts_size);

    let mut res = Vec::new();
    let ratio = params.poly_len / gamma;
    for i in 0..(num_rlwe_outputs / ratio) {
        // db_cols / params.poly_len

        let mut v = Vec::new();
        for j in 0..params.poly_len + 1 {
            let index = j * ((num_rlwe_outputs / ratio) * params.poly_len) + i * params.poly_len; // = j * db_cols + i * params.poly_len
            v.extend(&lwe_cts[index..index + params.poly_len]);
        }
        let to_push = prep_pack_lwes(params, &v);
        for k in 0..ratio {
            res.push(to_push[k * gamma..(k + 1) * gamma].to_vec());
        }
    }

    res
}


fn swap_midpoint<T>(a: &mut [T]) {
    let len = a.len();
    let (a, b) = a.split_at_mut(len / 2);
    a.swap_with_slice(b);
}

pub fn produce_table(poly_len: usize, chunk_size: usize) -> Vec<usize> {
    let mut cur = (0..poly_len).collect::<Vec<_>>();

    let outer_chunk_size = poly_len / (chunk_size / 2);
    println!("outer_chunk_size {}", outer_chunk_size);

    let mut do_it = true;
    for outer_chunk in cur.chunks_mut(outer_chunk_size) {
        if !do_it {
            do_it = true;
            continue;
        }
        do_it = false;

        for chunk in outer_chunk.chunks_mut(chunk_size) {
            let mut offs = 0;
            let mut to_add_to_offs = (chunk_size / 2).min(chunk.len() / 2); // weird hack
            while to_add_to_offs > 0 {
                swap_midpoint(&mut chunk[offs..]);
                offs += to_add_to_offs;
                to_add_to_offs /= 2;
            }
        }
    }

    cur
}

pub fn automorph_ntt_tables(poly_len: usize, log2_poly_len: usize) -> Vec<Vec<usize>> {
    let mut tables = Vec::new();
    for i in 0..log2_poly_len {
        let chunk_size = 1 << i;
        println!("table {}", i);
        let table = produce_table(poly_len, 2 * chunk_size);
        println!("table {:?}", &table.as_slice()[..32]);
        tables.push(table);
    }

    tables
}

pub fn bit_reversal(x: usize, log_len: usize) -> usize {
    let mut reversed = 0;
    let mut i = 0;
    let mut j = log_len - 1;

    while i < j {
        let bit_i = (x >> i) & 1;
        let bit_j = (x >> j) & 1;

        reversed |= (bit_i << j) | (bit_j << i);
        i += 1;
        j -= 1;
    }

    // If the length is odd, the middle bit remains unchanged
    if log_len % 2 == 1 {
        reversed |= (x >> (log_len / 2)) << (log_len / 2);
    }

    reversed
}
pub fn generate_automorph_tables_brute_force(params: &Params) -> Vec<Vec<usize>> {
    let mut tables = Vec::new();
    // for i in (1..=params.poly_len_log2) {
    for t in (1..2 * params.poly_len).step_by(2) {
        let mut table_candidate = vec![0usize; params.poly_len];

        // for 2048 balls and 2^28 bins, we will have a collision ~1% of the time
        // so, we redo if necessary
        loop {
            // let t = (1 << i) + 1;

            let poly = PolyMatrixRaw::random(&params, 1, 1);
            let poly_ntt = poly.ntt();

            let poly_auto = automorph_alloc(&poly, t);
            let poly_auto_ntt = poly_auto.ntt();

            let pol_orig = (&poly_ntt.get_poly(0, 0)[..params.poly_len]).to_vec();
            let pol_auto = (&poly_auto_ntt.get_poly(0, 0)[..params.poly_len]).to_vec();

            let mut must_redo = false;

            for i in 0..params.poly_len {
                let mut total = 0;
                let mut found = None;
                for j in 0..params.poly_len {
                    if pol_orig[i] == pol_auto[j] {
                        total += 1;
                        found = Some(j);
                    }
                }
                table_candidate[found.unwrap()] = i;
                if total != 1 {
                    must_redo = true;
                    break;
                }
            }

            if !must_redo {
                break;
            }
        }
        tables.push(table_candidate);
    }
    tables
}

// pub fn generate_automorph_tables_brute_force(params: &Params) -> Vec<Vec<usize>> {
//     let mut tables = Vec::new();
//     let n = params.poly_len as usize;
//     let m = 2 * n;

//     // Iterate through the same odd automorphism factors
//     for t in (1..m).step_by(2) {
//         let mut table_candidate = vec![0usize; n as usize];

//         // We need the modular inverse of t to reverse the permutation
//         let t_inv = invert_uint_mod(t as u64, m as u64).unwrap() as usize;

//         // For each new index 'j', find the original index 'i' it came from
//         for j in 0..n {
//             let j_exp = 2 * j + 1;

//             // Apply the inverse permutation on the exponent
//             let i_exp = (t_inv * j_exp) % m;
            
//             // Convert the old exponent back to an index
//             let i = (i_exp - 1) / 2;

//             // Store the mapping: table[new_index] = old_index
//             table_candidate[j as usize] = i as usize;
//         }
//         tables.push(table_candidate);
//     }
//     tables
// }


pub fn apply_automorph_ntt_raw<'a>(
    params: &Params,
    poly: &[u64],
    out: &mut [u64],
    t: usize,
    tables: &[Vec<usize>],
) {
    let poly_len = params.poly_len;
    // table_idx = log2(poly_len / (t - 1))
    // let table_idx = (poly_len / (t - 1)).trailing_zeros() as usize;
    let table_idx = (t - 1) / 2;
    let table = &tables[table_idx];

    for i in 0..poly_len {
        out[i] += poly[table[i]];
    }
}

pub fn apply_automorph_ntt<'a>(
    params: &'a Params,
    tables: &[Vec<usize>],
    mat: &PolyMatrixNTT<'a>,
    res: &mut PolyMatrixNTT<'a>,
    t: usize,
) {
    // run apply_automorph_ntt on each poly in the matrix
    // let mut res = PolyMatrixNTT::zero(params, mat.rows, mat.cols);
    for i in 0..mat.rows {
        for j in 0..mat.cols {
            let poly = mat.get_poly(i, j);
            let mut res_poly: Vec<&mut [u64]> = res.get_poly_mut(i, j).chunks_exact_mut(params.poly_len).collect();
            // for (chunk, res_chunk) in
                // poly.chunks_exact(params.poly_len).zip(res_poly.chunks_exact_mut(params.poly_len))
            for (index, chunk) in poly.chunks_exact(params.poly_len).enumerate()
            {
                apply_automorph_ntt_raw(params, chunk, res_poly[index], t, tables);
            }
        }
    }
    // res
}

pub fn apply_automorph_ntt_double<'a>(
    params: &'a Params,
    tables: &[Vec<usize>],
    mat: &PolyMatrixNTT<'a>,
    res_1: &mut PolyMatrixNTT<'a>,
    res_2: &mut PolyMatrixNTT<'a>,
    t: usize,
) {
    // run apply_automorph_ntt on each poly in the matrix
    // let mut res = PolyMatrixNTT::zero(params, mat.rows, mat.cols);
    for i in 0..mat.rows {
        for j in 0..mat.cols {
            let poly = mat.get_poly(i, j);
            let mut res_1_poly: Vec<&mut [u64]> = res_1.get_poly_mut(i, j).chunks_exact_mut(params.poly_len).collect();
            let mut res_2_poly: Vec<&mut [u64]> = res_2.get_poly_mut(i, j).chunks_exact_mut(params.poly_len).collect();
            // for (chunk, res_1_chunk, res_2_chunk) in
            //     izip!(poly.chunks_exact(params.poly_len), res_1_poly.chunks_exact_mut(params.poly_len), res_2_poly.chunks_exact_mut(params.poly_len))
            for (index, chunk) in poly.chunks_exact(params.poly_len).enumerate()
            {
                apply_automorph_ntt_raw(params, chunk, res_1_poly[index], t, tables);
                apply_automorph_ntt_raw(params, chunk, res_2_poly[index], 2*params.poly_len - t, tables);
            }
        }
    }
    // res
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PackingType {
    NoPacking,
    CDKS,
    InspiRING,
}

impl FromStr for PackingType {
    type Err = String;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input.to_lowercase().as_str() {
            "nopacking" => Ok(PackingType::NoPacking),
            "cdks" => Ok(PackingType::CDKS),
            "inspiring" => Ok(PackingType::InspiRING),
            _ => Err(format!("Invalid input: {}", input)),
        }
    }
}
/// Trait to convert an enum to a string representation
pub trait ToStr {
    fn to_str(&self) -> String;
}

impl ToStr for PackingType {
    fn to_str(&self) -> String {
        match self {
            PackingType::NoPacking => "NoPacking",
            PackingType::CDKS => "CDKS",
            PackingType::InspiRING => "InspiRING",
        }
        .to_string()
    }
}

impl Default for PackingType {
    fn default() -> Self {
        PackingType::CDKS // Specify the default variant
    }
}

#[derive(Clone)]
pub struct PackParams<'a> {
    pub params: &'a Params,
    pub num_to_pack: usize,
    pub tables: Vec<Vec<usize>>,
    pub gen_pows: Vec<usize>,
    pub mod_inv_poly: PolyMatrixNTT<'a>,
    pub monomial_ntts: Vec<PolyMatrixNTT<'a>>,
    pub neg_monomial_ntts: Vec<PolyMatrixNTT<'a>>,
}

impl PackParams<'_> {
    pub fn new<'a>(params: &'a Params, num_to_pack: usize) -> PackParams<'a> {
        debug!("Starting tables");
        let tables = generate_automorph_tables_brute_force(params);
        debug!("Got tables");
        let gen: usize =
            if num_to_pack < params.poly_len { (2 * params.poly_len / num_to_pack) + 1 } else { 5 };

        let mut gen_pows = Vec::new();
        for i in 0..params.poly_len {
            gen_pows
                .push(exponentiate_uint_mod(gen as u64, i as u64, 2 * params.poly_len as u64)
                    as usize);
        }
        let mod_inv = invert_uint_mod(num_to_pack as u64, params.modulus).unwrap();
        let mod_inv_poly = single_poly(params, mod_inv).ntt();

        let mut monomial_ntts = Vec::new();
        let mut neg_monomial_ntts = Vec::new();
        for j in 0..params.poly_len {
            let mut monomial = PolyMatrixRaw::zero(params, 1, 1);
            monomial.get_poly_mut(0, 0)[j] = 1;
            let mono_ntt = monomial.ntt();
            monomial_ntts.push(mono_ntt.clone());
            neg_monomial_ntts.push(-&mono_ntt);
        }
        PackParams {
            params,
            num_to_pack,
            tables,
            gen_pows,
            mod_inv_poly,
            monomial_ntts,
            neg_monomial_ntts,
        }
    }

    // An incomplete version of the packing parameters, used by the client
    pub fn new_fast<'a>(params: &'a Params, num_to_pack: usize) -> PackParams<'a> {
        let gen: usize =
            if num_to_pack < params.poly_len { (2 * params.poly_len / num_to_pack) + 1 } else { 5 };

        let mut gen_pows = Vec::new();
        for i in 0..params.poly_len {
            gen_pows
                .push(exponentiate_uint_mod(gen as u64, i as u64, 2 * params.poly_len as u64)
                    as usize);
        }

        PackParams {
            params,
            num_to_pack,
            tables: vec![],
            gen_pows,
            mod_inv_poly:PolyMatrixNTT::zero(&params, 1, 1),
            monomial_ntts: vec![],
            neg_monomial_ntts: vec![],
        }
    }
    
    pub fn empty<'a>(params: &'a Params) -> PackParams<'_> {
        PackParams {
            params,
            num_to_pack: 0,
            tables: vec![],
            gen_pows: vec![],
            mod_inv_poly: PolyMatrixNTT::zero(&params, 1, 1),
            monomial_ntts: vec![],
            neg_monomial_ntts: vec![],
        }
    }
}

#[derive(Clone)]
pub struct PrecompInsPIR<'a> {
    pub a_hat: PolyMatrixRaw<'a>,
    pub bold_t_condensed: PolyMatrixNTT<'a>,
    pub bold_t_bar_condensed: PolyMatrixNTT<'a>,
    pub bold_t_hat_condensed: PolyMatrixNTT<'a>,
}

pub fn generate_rotations<'a>(
    packing_params: &PackParams<'a>,
    to_rotate: &PolyMatrixNTT<'a>,
) -> PolyMatrixNTT<'a> {
    let params = packing_params.params;
    let num_to_pack = packing_params.num_to_pack;
    let tables = &packing_params.tables;
    let gen_pows = &packing_params.gen_pows;

    let mut rotations_all = PolyMatrixNTT::zero(&params, num_to_pack - 1, params.t_exp_left);
    for i in 0..num_to_pack - 1 {
        let mut rotated_w = PolyMatrixNTT::zero(&params, 1, params.t_exp_left);
        apply_automorph_ntt(&params, &tables, &to_rotate, &mut rotated_w, gen_pows[i]);
        rotations_all.copy_into(&rotated_w, i, 0);
    }
    rotations_all
}

pub fn generate_rotations_double<'a>(
    packing_params: &PackParams<'a>,
    to_rotate: &PolyMatrixNTT<'a>,
) -> (PolyMatrixNTT<'a>, PolyMatrixNTT<'a>) {
    let params = packing_params.params;
    let num_to_pack = packing_params.num_to_pack;
    let tables = &packing_params.tables;
    let gen_pows = &packing_params.gen_pows;
    assert_eq!(num_to_pack, params.poly_len);

    let num_rotations = params.poly_len / 2 - 1;

    let mut rotations_all = PolyMatrixNTT::zero(&params, num_rotations, params.t_exp_left);
    let mut rotations_bar_all = PolyMatrixNTT::zero(&params, num_rotations, params.t_exp_left);
    for i in 0..num_rotations {
        let mut rotated_w_1 = PolyMatrixNTT::zero(&params, 1, params.t_exp_left);
        let mut rotated_w_2 = PolyMatrixNTT::zero(&params, 1, params.t_exp_left);
        apply_automorph_ntt_double(&params, &tables, &to_rotate, &mut rotated_w_1, &mut rotated_w_2, gen_pows[i]);
        rotations_all.copy_into(&rotated_w_1, i, 0);
        rotations_bar_all.copy_into(&rotated_w_2, i, 0);
    }
    (rotations_all, rotations_bar_all)
}

#[derive(Clone)]
pub struct OfflinePackingKeys<'a> {

    pub packing_params: Option<&'a PackParams<'a>>,
    pub full_key: bool,

    pub w_seed: [u8; 32],
    pub v_seed: [u8; 32],

    pub w_mask: Option<PolyMatrixNTT<'a>>,
    pub v_mask: Option<PolyMatrixNTT<'a>>,

    pub w_all: Option<PolyMatrixNTT<'a>>,
    pub w_bar_all: Option<PolyMatrixNTT<'a>>,

}

impl OfflinePackingKeys<'_> {

    pub fn init_empty<'a>() -> OfflinePackingKeys<'a> {
        OfflinePackingKeys {
            packing_params: None,
            full_key: false,
            w_seed: [0; 32],
            v_seed: [0; 32],
            w_mask: None,
            w_all: None,
            w_bar_all: None,
            v_mask: None,
        }
    }

    pub fn init<'a>(packing_params: &'a PackParams<'a>, w_seed: [u8; 32]) -> OfflinePackingKeys<'_> {
        let w_mask = PolyMatrixNTT::random_rng(
            &packing_params.params,
            1,
            packing_params.params.t_exp_left,
            &mut ChaCha20Rng::from_seed(w_seed),
        );
        // let w_mask = PolyMatrixNTT::zero(&packing_params.params, 1, packing_params.params.t_exp_left);
        let x = generate_rotations(&packing_params, &w_mask);
        let w_all = Some(x);
        OfflinePackingKeys{
            packing_params: Some(packing_params),
            full_key: false,
            w_seed,
            v_seed: [0; 32],
            w_mask: Some(w_mask),
            w_all: w_all,
            w_bar_all: None,
            v_mask: None,
        }
    }

    pub fn init_full<'a>(packing_params: &'a PackParams<'a>, w_seed: [u8; 32], v_seed: [u8; 32]) -> OfflinePackingKeys<'a> {
        let w_mask = PolyMatrixNTT::random_rng(
            &packing_params.params,
            1,
            packing_params.params.t_exp_left,
            &mut ChaCha20Rng::from_seed(w_seed),
        );
        let v_mask = PolyMatrixNTT::random_rng(
            &packing_params.params,
            1,
            packing_params.params.t_exp_left,
            &mut ChaCha20Rng::from_seed(v_seed),
        );

        let (x,y) = generate_rotations_double(&packing_params, &w_mask);
        let (w_all, w_bar_all) = (Some(x), Some(y));

        OfflinePackingKeys{
            packing_params: Some(packing_params),
            full_key: true,
            w_seed,
            v_seed,
            w_mask: Some(w_mask),
            w_all: w_all,
            w_bar_all: w_bar_all,
            v_mask: Some(v_mask),
        }
    }

}

#[derive(Clone)]
pub struct PackingKeys<'a> {

    pub packing_type: PackingType,

    // Inspiring Stuff
    pub full_key: bool,
    pub packing_params: Option<PackParams<'a>>,
    pub y_body: Option<PolyMatrixNTT<'a>>,
    pub z_body: Option<PolyMatrixNTT<'a>>,
    pub y_body_condensed: Option<PolyMatrixNTT<'a>>,
    pub z_body_condensed: Option<PolyMatrixNTT<'a>>,    

    pub expanded: bool,
    pub y_all_condensed: Option<PolyMatrixNTT<'a>>,
    pub y_bar_all_condensed: Option<PolyMatrixNTT<'a>>,    

    // CDKS stuff
    pub params: Option<Params>,
    pub pack_pub_params_row_1s: Vec<PolyMatrixNTT<'a>>,    
    pub fake_pack_pub_params: Vec<PolyMatrixNTT<'a>>,
}

impl PackingKeys<'_> {

    pub fn init_full<'a>(
        packing_params: &PackParams<'a>,
        sk_reg: &PolyMatrixRaw<'a>, 
        w_seed :[u8; 32],
        v_seed :[u8; 32],
    ) -> PackingKeys<'a> {
        let w_mask: PolyMatrixNTT<'_> = PolyMatrixNTT::random_rng(
            &packing_params.params,
            1,
            packing_params.params.t_exp_left,
            &mut ChaCha20Rng::from_seed(w_seed),
        );
        let v_mask: PolyMatrixNTT<'_> = PolyMatrixNTT::random_rng(
            &packing_params.params,
            1,
            packing_params.params.t_exp_left,
            &mut ChaCha20Rng::from_seed(v_seed),
        );                
        let y_body = generate_ksk_body(
            &packing_params.params,
            &sk_reg,
            packing_params.gen_pows[1],
            &w_mask,
            &mut ChaCha20Rng::from_entropy(),
        );
        let z_body = generate_ksk_body(
            &packing_params.params,
            &sk_reg,
            2 * packing_params.params.poly_len - 1,
            &v_mask,
            &mut ChaCha20Rng::from_entropy(),
        );
        let y_body_condensed = condense_matrix(&packing_params.params, &y_body);
        let z_body_condensed = condense_matrix(&packing_params.params, &z_body);
        PackingKeys {
            packing_type: PackingType::InspiRING,
            packing_params: Some(packing_params.clone()),
            full_key: true,
            y_body: Some(y_body),
            z_body: Some(z_body),
            y_body_condensed: Some(y_body_condensed),
            z_body_condensed: Some(z_body_condensed),
            expanded: false,
            y_all_condensed: None,
            y_bar_all_condensed: None,
            params: None,
            pack_pub_params_row_1s: vec![],
            fake_pack_pub_params: vec![],
        }
    }

    pub fn init<'a> (
        packing_params: &PackParams<'a>,
        sk_reg: &PolyMatrixRaw<'a>, 
        w_seed :[u8; 32],
    ) -> PackingKeys<'a> {
        let w_mask: PolyMatrixNTT<'_> = PolyMatrixNTT::random_rng(
            &packing_params.params,
            1,
            packing_params.params.t_exp_left,
            &mut ChaCha20Rng::from_seed(w_seed),
        );
        // let w_mask = PolyMatrixNTT::zero(&packing_params.params, 1, packing_params.params.t_exp_left);
        let y_body = generate_ksk_body(
            &packing_params.params,
            &sk_reg,
            packing_params.gen_pows[1],
            &w_mask,
            &mut ChaCha20Rng::from_entropy(),
        );
        let y_body_condensed = condense_matrix(&packing_params.params, &y_body);
        PackingKeys {
            packing_type: PackingType::InspiRING,
            packing_params: Some(packing_params.clone()),
            full_key: false,
            y_body : Some(y_body),
            z_body : None,
            y_body_condensed : Some(y_body_condensed),
            z_body_condensed : None,
            expanded: false,
            y_all_condensed: None,
            y_bar_all_condensed: None,
            params: None,
            pack_pub_params_row_1s: vec![],
            fake_pack_pub_params: vec![],
        }        
    }

    pub fn init_cdks<'a>(
        params: &'a Params,
        sk_reg: &PolyMatrixRaw<'a>,
        static_seed_2: [u8; 32],  
    ) -> PackingKeys<'a> {
        let pack_pub_params = raw_generate_expansion_params(
            &params,
            &sk_reg,
            params.poly_len_log2,
            params.t_exp_left,
            &mut ChaCha20Rng::from_entropy(),
            &mut ChaCha20Rng::from_seed(static_seed_2),
        );
        // let pub_params_size = get_vec_pm_size_bytes(&pack_pub_params) / 2;
        let mut pack_pub_params_row_1s = pack_pub_params.to_vec();
        for i in 0..pack_pub_params.len() {
            pack_pub_params_row_1s[i] =
                pack_pub_params[i].submatrix(1, 0, 1, pack_pub_params[i].cols);
            pack_pub_params_row_1s[i] =
                condense_matrix(&params, &pack_pub_params_row_1s[i]);
        }

        let mut fake_pack_pub_params = pack_pub_params.clone();
        // zero out all of the second rows
        for i in 0..pack_pub_params.len() {
            for col in 0..pack_pub_params[i].cols {
                fake_pack_pub_params[i].get_poly_mut(1, col).fill(0);
            }
        }

        PackingKeys {
            packing_type: PackingType::CDKS,
            packing_params: None,
            full_key: false,
            y_body : None,
            z_body : None,
            y_body_condensed : None,
            z_body_condensed : None,
            expanded: false,
            y_all_condensed: None,
            y_bar_all_condensed: None,
            params: Some(params.clone()),
            pack_pub_params_row_1s : pack_pub_params_row_1s,
            fake_pack_pub_params: fake_pack_pub_params,
        }
    }

    pub fn get_gamma(&self) -> usize {
        if self.packing_type == PackingType::InspiRING {
            self.packing_params.as_ref().unwrap().num_to_pack
        } else {
            self.params.as_ref().unwrap().poly_len
        }
    }

    pub fn get_size_bytes(&self) -> usize {
        if self.packing_type == PackingType::InspiRING {
            if self.full_key {
                get_vec_pm_size_bytes(&vec![self.y_body.as_ref().unwrap().clone()])
                + get_vec_pm_size_bytes(&vec![self.z_body.as_ref().unwrap().clone()])
            } else {
                get_vec_pm_size_bytes(&vec![self.y_body.as_ref().unwrap().clone()])
            }
        } else {
            get_vec_pm_size_bytes(&self.pack_pub_params_row_1s)
        }
    }

    pub fn expand<'a>(&mut self) {
        assert_eq!(self.packing_type, PackingType::InspiRING);
        if !self.expanded {
            let packing_params = self.packing_params.as_ref().unwrap();
            if self.full_key {
                let (x,y) = generate_rotations_double(&packing_params, &self.y_body_condensed.as_ref().unwrap());
                (self.y_all_condensed, self.y_bar_all_condensed) = (Some(x), Some(y));
            } else {
                let x = generate_rotations(&packing_params, &self.y_body_condensed.as_ref().unwrap());
                self.y_all_condensed = Some(x);
            }
            self.expanded = true;
        }
    }

}

pub fn generate_ksk_body<'a>(
    params: &'a Params,
    sk_reg: &PolyMatrixRaw<'a>,
    gen: usize,
    mask: &PolyMatrixNTT<'a>,
    rng: &mut ChaCha20Rng,
) -> PolyMatrixNTT<'a> {
    let tau_sk_reg = automorph_alloc(&sk_reg, gen);
    let minus_s_times_mask = &sk_reg.ntt() * &(-mask);
    let error_poly = PolyMatrixRaw::noise(
        &params,
        1,
        params.t_exp_left,
        &DiscreteGaussian::init(params.noise_width),
        rng,
    );
    // let error_poly = PolyMatrixRaw::zero(&params, 1, params.t_exp_left);
    let g_exp_ntt = build_gadget(&params, 1, params.t_exp_left).ntt();
    let ksk = &tau_sk_reg.ntt() * &g_exp_ntt;
    let body = &minus_s_times_mask + &error_poly.ntt();
    let result = &body + &ksk;
    result
}


pub fn query_gen<'a>(
    packing_params: &'a PackParams,
    sk_reg: &PolyMatrixRaw<'a>,
    w_mask: &PolyMatrixNTT<'a>,
    v_mask: &PolyMatrixNTT<'a>,
    messages: &Vec<u64>,
    a_ct_tilde: &Vec<PolyMatrixNTT<'a>>,
    rng_y: &mut ChaCha20Rng,
    rng_z: &mut ChaCha20Rng,
) -> (PolyMatrixRaw<'a>, PolyMatrixNTT<'a>, PolyMatrixNTT<'a>) {
    let params = packing_params.params;
    let gen = packing_params.gen_pows[1];
    // let num_to_pack = packing_params.num_to_pack;
    let gamma = a_ct_tilde.len();
    let y_body = generate_ksk_body(&params, &sk_reg, gen, &w_mask, rng_y);
    let z_body = generate_ksk_body(&params, &sk_reg, 2 * params.poly_len - 1, &v_mask, rng_z);

    let mut b_poly = PolyMatrixRaw::zero(&params, 1, 1);
    for i in 0..gamma {
        let mut pt: PolyMatrixRaw<'_> = PolyMatrixRaw::zero(&params, 1, 1);
        pt.get_poly_mut(0, 0)[0] = rescale(messages[i], params.pt_modulus, params.modulus);

        let mut b_ntt = &sk_reg.ntt() * &(-&a_ct_tilde[i]);
        fast_add_into(&mut b_ntt, &pt.ntt());

        b_poly.get_poly_mut(0, 0)[i] = b_ntt.raw().get_poly(0, 0)[0];
    }
    (b_poly, y_body, z_body)
}