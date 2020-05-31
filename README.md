## tsuga_linalg

**What is this?**

This is a sketch of an OpenCL linear algebra library that's primarily being built to support the `tsuga` machine learning project [here](https://github.com/quietlychris/tsuga). Under the hood, it sends `Vec<f32>` data structures to the GPU to have operations run based on them, but the interop is based around being able to do that from a user's perspective with `ndarray`'s `Array2<f32>` types. 


**What does it do?**

You can find the full list of supported functions in `src/functions.rs`. The preliminary list of operations is:
    
- Dot product
- Transpose
- Hadamard product/element-wise array multiplication
- Scalar multiplication
- Sigmoid operation

although not all of these are currrently implemented.

**Where can I learn more?**

This library is a combination of a couple different previous open-source projects, including:
- [gpuarray-rs](https://github.com/tedsta/gpuarray-rs) by tedsta
- [ocl-algebra](https://github.com/timohaas/ocl-algebra/) by timohaas

although `gpuarray-rs` doesn't use the `ocl` crate and builds off of a custom type for the linear algebra, and `ocl-algebra` isn't the primary source of the OpenCL linear algebra (`gpuarray-rs` is). As far as I know, neither one has existing support for `ndarray`. 