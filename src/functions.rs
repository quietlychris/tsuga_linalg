use ocl::{Buffer, MemFlags, ProQue, SpatialDims::*};

use ndarray::prelude::*;

use std::iter::FromIterator;
use std::time::{Duration, Instant};

// Note: From benchmarking, the highest contribution to the runtime of this function is the conversion from an Array2 struct into a vector. In the context of a dense neural network, it's probably possible to do all of that overhead at the beginning, then keep exchanging the already-built vectors back and forth.

// TO_DO: We're not accessing the NVIDIA GPU by default at the moment, which doens't seem right. Even though the built-in Intel still seems like it could offer a speed-up, that needs to be investigated. (Currently done, but not in a great way)
// Follow-up: Hmmm, actually looks like the discrete GPU is slower. Maybe because it has to pass back and forth?
// From a naive benchmark, it looks like there's something around a 26% speedup using a GPU right now using the naive implementation
// TO_DO: We need to build/test kernels for a few more operations, then integrate into a minimal neural net

pub fn matmul(ocl_pq: &mut ProQue, a: &Array2<f32>, b: &Array2<f32>) -> ocl::Result<Array2<f32>> {
    let now = Instant::now();
    let (n, m, k): (usize, usize, usize) = (a.nrows(), a.ncols(), b.ncols());
    //println!("(n,m,k): ({},{},{})",n,m,k);

    ocl_pq.set_dims([n,m]);
    let a_vec = &Array::from_iter(a.iter().cloned()).to_vec();
    //println!("a_vec: {:?}", a_vec);
    let source_buffer_a = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(ocl_pq.dims().clone())
        .copy_host_slice(&a_vec)
        .build()?;
    let built_a = now.elapsed().as_millis();
    println!("a_buffer has been built at: {}",built_a);

    ocl_pq.set_dims([m,k]);
    let b_vec = &Array::from_iter(b.iter().cloned()).to_vec();
    //println!("b_vec: {:?}", b_vec);
    let source_buffer_b = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len([m,k])
        .copy_host_slice(&b_vec)
        .build()?;
    let built_b = now.elapsed().as_millis() - built_a;
    println!("b_buffer has been built at: {}",built_b);

    ocl_pq.set_dims([n,k]);
    let result_buffer: Buffer<f32> = ocl_pq.create_buffer()?;

    // Create a kernel with arguments corresponding to those in the kernel.
    // Just for fun, one argument will be 'named':
    let mut kern = ocl_pq
        .kernel_builder("matmul")
        .arg(&source_buffer_a)
        .arg(&source_buffer_b)
        .arg(&result_buffer)
        .arg(&m)
        .arg(&k)
        .build()?;

    kern.set_default_global_work_size(Two(m,k)); // This one alone works for MNIST-size sets
    let built_kern = now.elapsed().as_millis() - (built_b + built_a);
    println!("kernel has been built at: {}",built_kern);

    // println!("Kernel global work size: {:?}", kern.default_global_work_size());
    // println!("Kernel local work size: {:?}", kern.default_local_work_size());

    // Enqueue kernel:
    unsafe {
        kern.enq()?;
    }

    // Read results from the device into result_buffer's local vector:
    let mut vec_result = vec![0.; n * k];
    result_buffer.read(&mut vec_result).enq()?;

    let retrieved_result = now.elapsed().as_millis() - (built_b + built_a + built_kern);
    println!("result have been retrived from GPU at: {}",retrieved_result);

    let result_array: Array2<f32> = Array::from_shape_vec((n, k), vec_result)
        .expect("Coudn't convert result to properly sized array");
    let array_convert = now.elapsed().as_millis() - (built_b + built_a + built_kern + retrieved_result);
    println!("result have been retrived from GPU at: {}",array_convert);
    //println!("vec_result is: {:?} to array:\n{}",vec_result,result_array);

    Ok(result_array)
}


// Our arbitrary data set size (about a million) and coefficent:
const WORK_SIZE: usize = 1 << 20;

pub fn multiply_by_scalar(input: Vec<f32>, coeff: f32) -> ocl::Result<()> {
    // Create a big ball of OpenCL-ness (see ProQue and ProQueBuilder docs for info):

    let src = include_str!("cl/multiply_by_scalar.cl");
    println!("The WORK_SIZE is {}", WORK_SIZE);
    let ocl_pq = ProQue::builder()
        .src(src)
        .dims(WORK_SIZE)
        .build()
        .expect("Build ProQue");

    // Create a temporary init vector and the source buffer. Initialize them
    // with random floats between 0.0 and 20.0:
    // let vec_source = ocl_extras::scrambled_vec((0.0, 2.0), ocl_pq.dims().to_len());
    let mut vec_source = vec![1.2; ocl_pq.dims().to_len()];
    let source_buffer = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(WORK_SIZE)
        .copy_host_slice(&vec_source)
        .build()?;

    let mut vec_result = vec![0.0f32; WORK_SIZE];
    let result_buffer: Buffer<f32> = ocl_pq.create_buffer()?;

    // Create a kernel with arguments corresponding to those in the kernel.
    // Just for fun, one argument will be 'named':
    let kern = ocl_pq
        .kernel_builder("multiply_by_scalar")
        .arg(coeff)
        .arg(None::<&Buffer<f32>>)
        .arg_named("result", None::<&Buffer<f32>>)
        .build()?;

    kern.set_arg(0, &coeff)?;
    kern.set_arg(1, Some(&source_buffer))?;
    kern.set_arg(2, &result_buffer)?;

    println!(
        "Kernel global work size: {:?}",
        kern.default_global_work_size()
    );

    // Enqueue kernel:
    unsafe {
        kern.enq()?;
    }

    // Read results from the device into result_buffer's local vector:
    result_buffer.read(&mut vec_result).enq()?;

    // Check results and print the first 20:

    /*for idx in 0..WORK_SIZE {
        if idx < input.len() {
            println!(
                "source[{idx}]: {:.03}, \t coeff: {}, \tresult[{idx}]: {}",
                vec_source[idx],
                coeff,
                vec_result[idx],
                idx = idx
            );
        }
        assert_eq!(vec_source[idx] * coeff, vec_result[idx]);
    }*/

    Ok(())
}
