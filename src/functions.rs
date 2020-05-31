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

pub fn hadamard(ocl_pq: &mut ProQue, a: &Array2<f32>, b: &Array2<f32>) -> ocl::Result<Array2<f32>> {
    let now = Instant::now();
    let (n, m): (usize, usize) = (a.nrows(), a.ncols());

    assert_eq!(&a.dim(),&b.dim());

    ocl_pq.set_dims(One(n*m));
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

    let b_vec = &Array::from_iter(b.iter().cloned()).to_vec();
    //println!("b_vec: {:?}", b_vec);
    let source_buffer_b = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(ocl_pq.dims().clone())
        .copy_host_slice(&b_vec)
        .build()?;
    let built_b = now.elapsed().as_millis() - built_a;
    println!("b_buffer has been built at: {}",built_b);

    let result_buffer: Buffer<f32> = ocl_pq.create_buffer()?;

    // Create a kernel with arguments corresponding to those in the kernel.
    // Just for fun, one argument will be 'named':
    let mut kern = ocl_pq
        .kernel_builder("hadamard")
        .arg(&source_buffer_a)
        .arg(&source_buffer_b)
        .arg(&result_buffer)
        .build()?;

    kern.set_default_global_work_size(One(n*m)); 
    kern.set_default_local_work_size(One(n*m)); 

    // Enqueue kernel:
    unsafe {
        kern.enq()?;
    }

    // Read results from the device into result_buffer's local vector:
    let mut vec_result = vec![0.; n * m];
    result_buffer.read(&mut vec_result).enq()?;

    let result_array: Array2<f32> = Array::from_shape_vec((n, m), vec_result.clone())
        .expect("Coudn't convert result to properly sized array");
    println!("vec_result is: {:?} to array:\n{}",vec_result,result_array);

    Ok(result_array)
}

// const WORK_SIZE: usize = 1 << 20;
pub fn multiply_by_scalar(ocl_pq: &mut ProQue,input: Vec<f32>, coeff: f32) -> ocl::Result<Vec<f32>> {

    let WORK_SIZE: usize = input.len();
    println!("The WORK_SIZE is {}", WORK_SIZE);
    ocl_pq.set_dims(One(input.len()));

    let source_buffer = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(WORK_SIZE)
        .copy_host_slice(&input)
        .build()?;

    let mut vec_result = vec![0.0f32; WORK_SIZE];
    let result_buffer: Buffer<f32> = ocl_pq.create_buffer()?;

    // Create a kernel with arguments corresponding to those in the kernel.
    // Just for fun, one argument will be 'named':
    let mut kern = ocl_pq
        .kernel_builder("multiply_by_scalar")
        .arg(coeff)
        .arg(None::<&Buffer<f32>>)
        .arg_named("result", None::<&Buffer<f32>>)
        .build()?;

    kern.set_default_global_work_size(One(input.len())); // This one alone works for MNIST-size sets

    kern.set_arg(0, &coeff)?;
    kern.set_arg(1, Some(&source_buffer))?;
    kern.set_arg(2, &result_buffer)?;

    // Enqueue kernel:
    unsafe {
        kern.enq()?;
    }

    result_buffer.read(&mut vec_result).enq()?;

    Ok(vec_result)
}

pub fn transpose(ocl_pq: &mut ProQue, a: &Array2<f32>) -> ocl::Result<Array2<f32>> {
    let now = Instant::now();
    let (n, m): (usize, usize) = (a.nrows(), a.ncols());

    ocl_pq.set_dims(Two(n,m));
    let a_vec = &Array::from_iter(a.iter().cloned()).to_vec();
    //println!("a_vec: {:?}", a_vec);
    let source_buffer_a = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(ocl_pq.dims().clone())
        .copy_host_slice(&a_vec)
        .build()?;

    let result_buffer: Buffer<f32> = ocl_pq.create_buffer()?;

    // Create a kernel with arguments corresponding to those in the kernel.
    // Just for fun, one argument will be 'named':
    let mut kern = ocl_pq
        .kernel_builder("transpose")
        .arg(&source_buffer_a)
        .arg(&result_buffer)
        .arg(&n)
        .arg(&m)
        .build()?;

    kern.set_default_global_work_size(Two(n,m)); 
    kern.set_default_local_work_size(Two(n,m)); 

    // Enqueue kernel:
    unsafe {
        kern.enq()?;
    }

    // Read results from the device into result_buffer's local vector:
    let mut vec_result = vec![0.; n * m];
    result_buffer.read(&mut vec_result).enq()?;

    let result_array: Array2<f32> = Array::from_shape_vec((m, n), vec_result.clone())
        .expect("Coudn't convert result to properly sized array");
    println!("vec_result is: {:?} to array:\n{}",vec_result,result_array);

    Ok(result_array)
}

pub fn sigmoid(ocl_pq: &mut ProQue,input: &Array2<f32>) -> ocl::Result<Array2<f32>> {

    let (n, m): (usize, usize) = (input.nrows(), input.ncols());
    let WORK_SIZE: usize = n*m;
    println!("The WORK_SIZE is {}", WORK_SIZE);
    ocl_pq.set_dims(One(input.len()));

    let a_vec = &Array::from_iter(input.iter().cloned()).to_vec();
    let source_buffer = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(WORK_SIZE)
        .copy_host_slice(&a_vec)
        .build()?;

    let mut vec_result = vec![0.0f32; WORK_SIZE];
    let result_buffer: Buffer<f32> = ocl_pq.create_buffer()?;

    // Create a kernel with arguments corresponding to those in the kernel.
    // Just for fun, one argument will be 'named':
    let mut kern = ocl_pq
        .kernel_builder("sigmoid")
        .arg(&source_buffer)
        .arg(&result_buffer)
        .build()?;

    kern.set_default_global_work_size(One(n*m));

    // Enqueue kernel:
    unsafe {
        kern.enq()?;
    }

    result_buffer.read(&mut vec_result).enq()?;
    let result_array: Array2<f32> = Array::from_shape_vec((n,m), vec_result.clone())
        .expect("Coudn't convert result to properly sized array");
    println!("vec_result is: {:?} to array:\n{}",vec_result,result_array);

    Ok(result_array)
}