mod supports;
mod functions;

use crate::functions::matmul;
use crate::supports::build_ocl_proque;

use std::time::{Duration, Instant};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray::prelude::*;

use ocl::{Device,Platform};

pub fn main() {
    /*
    let v = vec![1.;20];
    match multiply_by_scalar(v,0.5) {
        Ok(_) => (),
        Err(err) => println!("{}", err),
    }*/

    //let a: Array2<f32> = array![[1.,2.,3.],[4.,5.,6.]];
    //let b: Array2<f32> = array![[1.,1.],[1.,1.],[1.,1.]];
    let iterations = 1;
    let a = Array::random((60_000, 784), Uniform::new(0., 1.));
    let b = Array::random((784, 10), Uniform::new(0., 1.));

    let a_start = Instant::now();
    for _ in 0..iterations {
        let c_ndarray = a.dot(&b);
    }
    let a_end = a_start.elapsed().as_millis();
    println!(
        "Time for {} loops on CPU: {}",
        iterations,
        a_end
    );
    

    let mut ocl_pq: ocl::ProQue = build_ocl_proque("GeForce".to_string());
    let b_start = Instant::now();
    for _ in 0..iterations {
        let c = matmul(&mut ocl_pq, &a, &b).expect("Couldn't multiply a.dot(b)");
    }
    let b_end = b_start.elapsed().as_millis();
    println!(
        "Time for {} loops on GPU: {}",
        iterations,
        b_end
    );

    match a_end < b_end {
        true => println!("The CPU computation is {} times quicker",b_end as f32/ a_end as f32),
        false => println!("The GPU computaiton is {} quicker",a_end as f32/b_end as f32),
        _ => panic!("Something's gone wrong...")
    }

    //let c_ndarray = a.clone().dot(&b);
    //assert_eq!(c, c_ndarray);
}
