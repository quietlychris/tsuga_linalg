__kernel void matmul(__global const float *a,
                     __global const float *b,
                           __global float *c,
                               const ulong M,
                               const ulong K)
{
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    //const ulong wa = 3;
    //const ulong wb = 2;

    float val = 0.0;
    for (ulong k = 0; k < M; k++) {
        val += a[i*M + k] * b[k*K + j];
    }
    c[i*K + j] = val;
}
