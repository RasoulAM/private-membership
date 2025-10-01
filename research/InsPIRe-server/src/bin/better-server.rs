// use std::io::prelude::*;
use std::time::Duration;
use std::thread;

use std::sync::Arc;

// use std::net::{TcpListener, TcpStream};
use actix_web::{web, App, HttpServer, HttpResponse, Responder};
use actix_cors::Cors;

use inspire::packing::PackingType;
use inspire::scheme::ProtocolType;

use log::debug;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde_json::json;

// use serde::Serialize;
use spiral_rs::arith::multiply_uint_mod;
use spiral_rs::arith;

use spiral_rs::aligned_memory::{AlignedMemory64};
use spiral_rs::{
    params::*,
    poly::*,
    gadget::*,
};

use spiral_rs::poly::{PolyMatrix, PolyMatrixRaw, PolyMatrixNTT};
use spiral_rs::number_theory::invert_uint_mod;

use inspire::{bits::*, kernel::*, measurement::*, modulus_switch::*, packing::*, params::*, server::*};

use inspire::interpolate::*;

use std::{marker::PhantomData, time::Instant};
use std::collections::HashMap;
use std::path::Path;

use inspire::commons::*;
use clap::Parser;

pub trait RGSWPIR<'a> {
    
    fn new_rgswpir<'b>(
        params: &'a Params,
        db: &[u16],
        interpolate_degree: usize,
    ) -> Self;

    fn perform_online_computation_simplepir_and_rgsw(
        &self,
        gamma: usize,
        first_dim_queries_packed: &[u64],
        ct_gsw: PolyMatrixNTT<'a>,
        offline_vals: &OfflinePrecomputedValues<'a>,
        packing_keys: &mut PackingKeys<'a>,
        measurement: Option<&mut Measurement>,
    ) -> Vec<Vec<u8>>;

    fn handle_connection(
        &self,
        query_bytes: &Vec<u8>,
        params: &Params,
        packing_params: &'a PackParams,        
        interpolate_degree: usize,
        offline_vals: &OfflinePrecomputedValues<'a>
    ) -> Vec<u8>;
    
}

impl<'a, T: Sync> RGSWPIR<'a> for YServer<'a, T> where
T: Sized + Copy + ToU64 + Default + std::marker::Sync,
*const T: ToM512,
u64: From<T>,
{

    fn new_rgswpir<'b>(
        params: &'a Params,
        db: &[u16],
        interpolate_degree: usize,
    ) -> Self {
        let second_level_packing_mask = PackingType::InspiRING;
        let bytes_per_pt_el = std::mem::size_of::<u16>();
        debug!("bytes_per_pt_el: {}", bytes_per_pt_el);

        let poly_len = params.poly_len;

        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_cols = params.instances * params.poly_len;

        let sz_bytes = db_rows * db_cols * bytes_per_pt_el;

        let gamma = poly_len;
        let db_cols_prime = db_cols / gamma;
        let c = db_cols_prime / interpolate_degree;

        let mut db_buf_aligned = AlignedMemory64::new(sz_bytes / 8);
        let db_buf_mut = as_bytes_mut(&mut db_buf_aligned);
        let db_buf_ptr = db_buf_mut.as_mut_ptr() as *mut u16;

        let can_use_pt_ntt = params.pt_modulus % (2*poly_len as u64) == 1;

        let mod_str = format!("[{}]", params.pt_modulus);
        let plaintext_params = if can_use_pt_ntt {
            internal_params_for(poly_len, 0, 0, 1, 28, 3, &mod_str)
        } else {
            let modulus = params.pt_modulus;
            let moduli = [params.pt_modulus; 4];
            let modulus_log2 = arith::log2_ceil(modulus);
            let (barrett_cr_0, barrett_cr_1) = arith::get_barrett(&moduli);
            let (barrett_cr_0_modulus, barrett_cr_1_modulus) = arith::get_barrett_crs(modulus);
            Params {
                poly_len: poly_len,
                poly_len_log2: params.poly_len_log2,
                ntt_tables: Vec::new(),
                scratch: Vec::new(),
                crt_count: 1,
                barrett_cr_0: barrett_cr_0,
                barrett_cr_1: barrett_cr_1,
                barrett_cr_0_modulus: barrett_cr_0_modulus,
                barrett_cr_1_modulus: barrett_cr_1_modulus,
                mod0_inv_mod1: 0,
                mod1_inv_mod0: 0,
                moduli: moduli,
                modulus: params.pt_modulus,
                modulus_log2: modulus_log2 ,
                noise_width: 0.,
                n: 0,
                pt_modulus: 0,
                q2_bits: 0,
                t_conv: 0,
                t_exp_left: 0,
                t_exp_right: 0,
                t_gsw: 0,
                expand_queries: false,
                db_dim_1: 0,
                db_dim_2: 0,
                instances: 0,
                db_item_size: 0,
                version: 0
            }
        };


        let mut monomials = Vec::with_capacity(2*poly_len);

        if can_use_pt_ntt {
            for i in 0..poly_len {
                let mut monomial = PolyMatrixRaw::zero(&plaintext_params, 1, 1);
                monomial.get_poly_mut(0, 0)[i] = 1;
                monomials.push(monomial.ntt());
            }
            for i in 0..poly_len {
                monomials.push(-&monomials[i]);
            }
        }

        let num_inv = invert_uint_mod(interpolate_degree as u64, plaintext_params.modulus).unwrap();

        let step = db_rows / 8;
        for i in 0..db_rows {
            if i % step == 0 {
                print!("i={} -> ",i);
            }

            for which_poly in 0..c {

                let mut points = Vec::new();
    
                for j_prime in 0..interpolate_degree {
                    let mut point = PolyMatrixRaw::zero(&plaintext_params, 1, 1);
                    for k in 0..gamma {
                        let j = which_poly * interpolate_degree * gamma + j_prime * gamma + k;
                        let idx = j * db_rows + i;
                        point.get_poly_mut(0, 0)[k] = db[idx] as u64;
                    }
                    points.push(point);
                }
                let coeffs = if can_use_pt_ntt {
                    cooley_tukey(
                        &plaintext_params,
                        points.iter().map(|x| x.ntt()).collect::<Vec<_>>(),
                        monomials.as_slice()).iter().map(|x| x.raw()).collect::<Vec<_>>()
                } else {
                    cooley_tukey_without_ntt(&plaintext_params, points)
                } ;
    
                for j_prime in 0..interpolate_degree {
                    let raw_coeff = &coeffs[j_prime];
                    let temp = raw_coeff.get_poly(0, 0);
                    for k in 0..gamma {
                        let j = which_poly * interpolate_degree * gamma + j_prime * gamma + k;
                        let idx = j * db_rows + i;
                        
                        unsafe {
                            *db_buf_ptr.add(idx) = multiply_uint_mod(temp[k] as u64, num_inv, plaintext_params.modulus) as u16;
                        }
                        
                    }
                }

            }

        }

        // Parameters for the second round (the "DoublePIR" round)
        let smaller_params = params.clone();

        let mut packing_params_set: HashMap<usize, PackParams> = HashMap::new(); 
        let mut half_packing_params_set: HashMap<usize, PackParams> = HashMap::new(); 
        for gamma in vec![poly_len] {
            if !packing_params_set.contains_key(&gamma) {
                let packing_params = PackParams::new(&params, gamma);
                let half_packing_params = PackParams::new(&params, gamma >> 1);
                packing_params_set.insert(gamma, packing_params);
                half_packing_params_set.insert(gamma, half_packing_params);
            }

        }

        Self {
            params,
            packing_params_set: packing_params_set,
            half_packing_params_set: half_packing_params_set,
            smaller_params,
            db_buf_aligned,
            phantom: PhantomData,
            protocol_type: ProtocolType::SimplePIR,
            second_level_packing_mask,
            second_level_packing_body:PackingType::NoPacking,
        }
    }

    /// Perform SimplePIR-style YPIR
    fn perform_online_computation_simplepir_and_rgsw(
        &self,
        interpolate_degree: usize,
        first_dim_queries_packed: &[u64],
        ct_gsw_body: PolyMatrixNTT<'a>,
        offline_vals: &OfflinePrecomputedValues<'a>,
        packing_keys: &mut PackingKeys<'a>,
        measurement: Option<&mut Measurement>,
    ) -> Vec<Vec<u8>> {
        assert_eq!(self.protocol_type, ProtocolType::SimplePIR);

        let params = self.params;
        let gamma = params.poly_len;

        // RLWE reduced moduli
        let rlwe_q_prime_1 = params.get_q_prime_1();
        let rlwe_q_prime_2 = params.get_q_prime_2();

        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_cols = params.instances * params.poly_len;
        let db_cols_prime = (db_cols as f64 / gamma as f64).ceil() as usize;

        assert_eq!(first_dim_queries_packed.len(), db_rows);

        // Begin online computation

        let first_pass = Instant::now();
        debug!("Performing mul...");
        let mut intermediate = AlignedMemory64::new(db_cols);
        fast_batched_dot_product_generic::<T>(
            &params,
            intermediate.as_mut_slice(),
            first_dim_queries_packed,
            db_rows,
            self.db(),
            db_rows,
            db_cols,
        );
        debug!("Done w mul...");

        let first_pass_time_us = first_pass.elapsed().as_micros();

        let mut packing_key_rotations_time_us = 0;
        let ring_packing = Instant::now();

        let packed = match self.second_level_packing_mask {
            PackingType::InspiRING => {
                let precomp_inspir_vec = &offline_vals.precomp_inspir_vec;
                let packing_key_rotations_time = Instant::now();
                packing_keys.expand();
                packing_key_rotations_time_us = packing_key_rotations_time.elapsed().as_micros();
                pack_many_lwes_inspir(
                    &self.packing_params_set[&gamma],
                    &precomp_inspir_vec,
                    intermediate.as_slice(),
                    &packing_keys,
                    gamma,
                )                
            },
            PackingType::CDKS => {
                let y_constants = &offline_vals.y_constants;
                let precomp = &offline_vals.precomp;
                pack_many_lwes(
                    &params,
                    &precomp,
                    intermediate.as_slice(),
                    db_cols_prime,
                    &packing_keys.pack_pub_params_row_1s,
                    &y_constants,
                )
            },
            PackingType::NoPacking => {
                panic!("Shouldn't be here!");
            },
        };

        let total_ring_packing_time_us = ring_packing.elapsed().as_micros();

        let rgsw_time = Instant::now();

        let mut ct_gsw = ct_gsw_body.pad_top(1);

        for i in 0..ct_gsw.cols {
            let a = PolyMatrixRaw::random_rng(params, 1, 1, &mut ChaCha20Rng::from_seed(RGSW_SEEDS[i]));
            ct_gsw.copy_into(&(-&a).ntt(), 0, i);
        }

        let ell = ct_gsw.cols / 2;
        let mut ginv_c = PolyMatrixRaw::zero(&params, 2 * ell, 1);
        let mut ginv_c_ntt = PolyMatrixNTT::zero(&params, 2 * ell, 1);
        
        let mut results = Vec::new();

        assert_eq!(db_cols_prime, packed.len());
        assert!(db_cols_prime % interpolate_degree == 0);
        let c = db_cols_prime / interpolate_degree;

        for which_poly in 0..c {
            let mut sum = PolyMatrixRaw::zero(&params, 2, 1);
            for i in (0..interpolate_degree).rev() {
                let mut prod = PolyMatrixNTT::zero(&params, 2, 1);
                gadget_invert(&mut ginv_c, &sum);
                to_ntt(&mut ginv_c_ntt, &ginv_c);
                multiply(
                    &mut prod,
                    &ct_gsw,
                    &ginv_c_ntt,
                );
                sum = &prod.raw() + &packed[which_poly * interpolate_degree + i];
                sum.reduce_mod(params.modulus);
            }
            results.push(sum);
        }
        
        let total_rgsw_time_us = rgsw_time.elapsed().as_micros();
        debug!("RGSW Time: {} us", total_rgsw_time_us);

        debug!("Packed...");
        if let Some(m) = measurement {
            m.online.first_pass_time_us = first_pass_time_us as usize;
            m.online.packing_key_rotations_time_us = packing_key_rotations_time_us as usize;
            m.online.first_pack_time_us = total_ring_packing_time_us as usize;
            m.online.rgsw_time_us = total_rgsw_time_us as usize;
        }

        let mut packed_mod_switched = Vec::with_capacity(results.len());
        for ct in results.iter() {
            let res_switched = ct.switch_and_keep(rlwe_q_prime_1, rlwe_q_prime_2, gamma);
            packed_mod_switched.push(res_switched);
        }

        packed_mod_switched

    }

    fn handle_connection(&self, received_data: &Vec<u8>, params: &Params, packing_params: &'a PackParams, interpolate_degree: usize, offline_vals: &OfflinePrecomputedValues<'a>) -> Vec<u8> {

        println!("Processing request...");

        let (packing_keys, packed_query_row, ct_gsw_body) = deserialize_everything(&params, &packing_params, received_data.to_vec());

        let mut measurement = Measurement::default();

        println!("Starting online...");

        let sum_switched = self.perform_online_computation_simplepir_and_rgsw(
            interpolate_degree,
            packed_query_row.as_slice(),
            ct_gsw_body,
            &offline_vals,
            &mut packing_keys.clone(),
            Some(&mut measurement),
        );

        let response = sum_switched.concat();
        println!("...Processing complete.");

        response

    }

}


/// Holds all the application state that needs to be shared across requests.
struct AppState<'a, T: std::marker::Sync> {
    num_items: usize,
    dim0: usize,
    item_size_bits: usize,
    key_to_row: HashMap<String, usize>,
    y_server: Arc<YServer<'a, T>>,
    params: Arc<Params>,
    packing_params: Arc<PackParams<'a>>,
    interpolate_degree: usize,
    offline_values: Arc<OfflinePrecomputedValues<'a>>,
}

/// Handler for the "/setup" endpoint.
async fn setup_endpoint<T: std::marker::Sync>(
    data: web::Data<AppState<'_, T>>,
) -> impl Responder {
    // You can implement your setup logic here.

    let obj = json!({
        "num_items": data.num_items,
        "dim0": data.dim0,
        "item_size_bits": data.item_size_bits,
        "key_to_row": data.key_to_row
    });

    let json_str = obj.to_string();
    HttpResponse::Ok().body(json_str)

}

/// Handler for the "/query" endpoint.
async fn query_endpoint<T>(
    data: web::Data<AppState<'_, T>>,
    query_bytes: web::Bytes,
) -> impl Responder where
T: Sized + Copy + ToU64 + Default + std::marker::Sync,
*const T: ToM512,
u64: From<T>, {
    // This function now receives the entire query as a byte payload from the client.
    // It calls a separate function to process the query and get the response bytes.
    let response_bytes = process_pir_query::<T>(
        &data.y_server,
        &data.params,
        &data.packing_params,
        data.interpolate_degree,
        &data.offline_values,
        &query_bytes,
    );

    // Return the response bytes to the client with a binary content type.
    HttpResponse::Ok()
        .content_type("application/octet-stream")
        .body(response_bytes)
}

/// This function should contain the logic from your original `handle_connection`.
///
/// It needs to be adapted to work with byte slices instead of a stream. It takes the client's
/// query as input (`query_bytes`) and should return the server's response as a `Vec<u8>`.
///
/// For example, where `handle_connection` would have done `stream.read(...)`, this function
/// will use the `query_bytes` slice. Where it would have done `stream.write(...)`, this
/// function will build and return a `Vec<u8>`.
fn process_pir_query<T>(
    y_server: &YServer<T>,
    params: &Params,
    packing_params: &PackParams,
    interpolate_degree: usize,
    offline_values: &Arc<OfflinePrecomputedValues<'_>>,
    query_bytes: &[u8],
) -> Vec<u8> where
T: Sized + Copy + ToU64 + Default + std::marker::Sync,
*const T: ToM512,
u64: From<T>, {

    println!("Received a query of {} bytes.", query_bytes.len());

	y_server.handle_connection(&query_bytes.to_vec(), &params, &packing_params, interpolate_degree, &offline_values)
}


/// Run the YPIR scheme with the given parameters
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[clap(long)]
    path_to_db: Option<String>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {

	let args = Args::parse();
	let Args {
			path_to_db
	} = args;

	let path_to_db = path_to_db.unwrap();
	
	// --- Initial processing before starting the server ---
	println!("🚀 Performing initial setup...");

    // let num_items = 1 << 12;
    let dim0 = 2048;
    let item_size_base_bits = 16 * 2048;
    let number_of_sub_databases = 8;
    let item_size_bits = number_of_sub_databases * item_size_base_bits;

    let item_size_base_elements= item_size_base_bits / 16;
    let item_size_elements= item_size_bits / 16;

	let filename = path_to_db;
	let (num_items, read_db, key_to_row) = read_json_into_matrix(Path::new(&filename), item_size_bits).unwrap();
    
    assert_eq!(read_db.len(), num_items);
    assert_eq!(read_db[0].len(), item_size_elements);

    println!("num_items: {}", num_items);
	println!(
			"Protocol=InsPIRe, DB={} KB",
			(num_items * item_size_bits) / 8192
	);

	let (params_raw, interpolate_degree, _) 
			= params_rgswpir_given_input_size_and_dim0(num_items, item_size_bits, dim0);
	let params_static: &'static Params = Box::leak(Box::new(params_raw.clone()));

    // println!("interpolate_degree: {}", interpolate_degree);

	let params = Arc::new(params_raw);

	let packing_params_raw = PackParams::new(&params_static, params.poly_len);
	let packing_params = Arc::new(packing_params_raw);

	let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
	let db_cols = params.instances * params.poly_len;

	println!("📦 Reading and processing '{}'...", filename);

    let items_per_row = num_items / db_rows;

    assert!(items_per_row == interpolate_degree);

    // split the database into number_of_sub_databases sub-databases by splitting each row into number_of_sub_databases sub-rows
    // each database takes one of the large columns
    let mut sub_databases: Vec<Vec<Vec<u16>>> = Vec::with_capacity(number_of_sub_databases);
    for k in 0..number_of_sub_databases {
        let mut temp_sub_database: Vec<Vec<u16>> = Vec::with_capacity(db_rows);
        for i in 0..num_items {
            temp_sub_database.push(read_db[i][k*item_size_base_elements .. (k + 1) * item_size_base_elements].to_vec());
        }
        sub_databases.push(temp_sub_database);
    }

    assert_eq!(sub_databases.len(), number_of_sub_databases);
    assert_eq!(sub_databases[0].len(), num_items);
    assert_eq!(sub_databases[0][0].len(), item_size_base_elements);

    let mut resized_sub_databases: Vec<Vec<Vec<u16>>> = Vec::with_capacity(number_of_sub_databases);
    for k in 0..number_of_sub_databases {
        // take the k-th sub-database
        // flatten `items_per_row` consecutive rows into one
        let mut temp_resized_sub_database: Vec<Vec<u16>> = Vec::with_capacity(db_rows);
        for i in 0..db_rows {
            temp_resized_sub_database.push(sub_databases[k][i*items_per_row .. (i + 1) * items_per_row].iter().flatten().copied().collect());
        }
        resized_sub_databases.push(temp_resized_sub_database);
    }

    assert_eq!(resized_sub_databases.len(), number_of_sub_databases);
    assert_eq!(resized_sub_databases[0].len(), db_rows);
    assert_eq!(resized_sub_databases[0][0].len(), items_per_row*item_size_base_elements);

    // append the corresponding rows of the sub-databases
    let mut resized_db: Vec<Vec<u16>> = Vec::with_capacity(db_rows);
    for i in 0..db_rows {
        let mut temp_row: Vec<u16> = Vec::with_capacity(item_size_elements);
        for k in 0..number_of_sub_databases {
            temp_row.extend(resized_sub_databases[k][i].iter().copied());
        }
        resized_db.push(temp_row);
    }

    assert_eq!(resized_db.len(), db_rows);
    assert_eq!(resized_db[0].len(), items_per_row*item_size_elements);
    assert_eq!(resized_db[0].len(), db_cols);

    let mut actual_db: Vec<u16> = Vec::with_capacity(db_rows*db_cols);
    for j in 0..db_cols {
        for i in 0..db_rows {
            actual_db.push(resized_db[i][j]);
        }
    }

	type T = u16;
	let gamma = params.poly_len;

	let y_server = Arc::new(YServer::<T>::new_rgswpir(
			params_static,
			actual_db.as_slice(),
			interpolate_degree,
	));

	let y_server_static: &'static YServer::<T> = Box::leak(Box::new(y_server.clone()));

	assert_eq!(y_server.db().len(), db_rows * db_cols);

	// ================================================================
	// OFFLINE PHASE
	// ================================================================
	let mut measurement = Measurement::default();

	println!("Starting offline...");
	let offline_values_raw = y_server_static
			.perform_offline_precomputation_simplepir(gamma, Some(&mut measurement), false);
  let offline_values = Arc::new(offline_values_raw);
	println!("Done offline...");

	thread::sleep(Duration::from_secs(1));
	println!("✅ Setup complete. Server is starting.");

	// --- Create the shared application state for Actix ---
	let app_state = web::Data::new(AppState {
            num_items,
            dim0,
            item_size_bits,
            key_to_row,
			y_server: Arc::clone(&y_server),
			params: Arc::clone(&params),
			packing_params: Arc::clone(&packing_params),
			interpolate_degree,
			offline_values: Arc::clone(&offline_values),
	});

	// --- Start the Actix HTTP server ---
	// This replaces your TcpListener and ThreadPool loop.
	println!("📡 Starting HTTP server on http://127.0.0.1:8081");
	HttpServer::new(move || {
			App::new()
					.wrap(
							Cors::default()
									.allow_any_origin()
									.allow_any_method()
									.allow_any_header()
									.max_age(3600)
					)
					.app_data(app_state.clone()) // Share the state with all handlers
					.app_data(web::PayloadConfig::new(10 * 1024 * 1024)) // Set 100MB payload limit for large PIR queries
					.route("/setup", web::get().to(setup_endpoint::<T>))
					.route("/query", web::post().to(query_endpoint::<T>))
	})
	.bind(("127.0.0.1", 8081))?
	.run()
	.await

}
