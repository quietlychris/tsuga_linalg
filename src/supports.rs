use ocl::{Buffer, MemFlags, ProQue, SpatialDims::*,Platform, Device};
use ocl::enums::DeviceSpecifier::*;

struct Context {
    pq: ProQue
}

pub fn build_ocl_proque(gpu_type: String) -> ProQue {
    let src = include_str!("cl/functions.cl");

    let mut dev = None;
    let platforms = Platform::list();
    for p_idx in 0..platforms.len() {
        let platform = &platforms[p_idx];
        let devices = Device::list_all(platform).unwrap();
        for d_idx in 0..devices.len() {
            let device = devices[d_idx];
            println!("Device: {:?}",device.name());
            if device.name().unwrap().to_string().contains(&gpu_type) {
                dev = Some(device);
            }
            //let deviceinforesult = core::get_device_info(&device, DeviceInfo::MaxComputeUnits);
            //let units = deviceinforesult.to_string().parse().unwrap();
        }
    }

    //println!("The WORK_SIZE is {}",WORK_SIZE);
    let mut ocl_pq = ProQue::builder()
        .src(src)
        .device(dev.unwrap())
        .build()
        .expect("Build ProQue");
    //println!("Built proque: {}",now.elapsed().as_millis());

    println!("The specified device is: {}",ocl_pq.device().name().unwrap());
    println!("It has a maximum working group size of {}",ocl_pq.device().max_wg_size().unwrap());
    assert!(ocl_pq.device().is_available().unwrap());
    ocl_pq
}
