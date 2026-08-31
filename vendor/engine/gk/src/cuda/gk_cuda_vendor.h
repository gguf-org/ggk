#pragma once

// The one file that knows whether this is a CUDA build or a HIP build.
//
// The kernels in this directory are written once. AMD's HIP is a rename of
// CUDA's runtime API rather than a different model - the launch syntax, the
// memory model, the warp primitives and the arithmetic are the same shapes
// with different spellings - so a header of aliases is enough to compile the
// same sources for both, and is far less work to keep correct than a second
// copy of every kernel would be.
//
// What is *not* hidden here is anything where the two really differ: the warp
// width (32 on NVIDIA, 64 on most AMD parts) is exposed as GK_WARP_SIZE and
// the kernels read it rather than assuming, and anything that would need a
// vendor-specific instruction is simply not used.

#if defined(GK_USE_HIP)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define gkStream_t              hipStream_t
#define gkEvent_t               hipEvent_t
#define gkError_t               hipError_t
#define gkSuccess               hipSuccess
#define gkGetErrorString        hipGetErrorString
#define gkGetLastError          hipGetLastError

#define gkStreamPerThread       hipStreamPerThread
#define gkFuncSetAttribute      hipFuncSetAttribute
#define gkFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

#define gkMalloc                hipMalloc
#define gkFree                  hipFree
#define gkHostAlloc(p, s)       hipHostMalloc(p, s, hipHostMallocDefault)
#define gkFreeHost              hipHostFree
#define gkMemset                hipMemset
#define gkMemsetAsync           hipMemsetAsync
#define gkMemcpy                hipMemcpy
#define gkMemcpyAsync           hipMemcpyAsync
#define gkMemcpyFromSymbol      hipMemcpyFromSymbol
#define gkMemcpyHostToDevice    hipMemcpyHostToDevice
#define gkMemcpyDeviceToHost    hipMemcpyDeviceToHost
#define gkMemcpyDeviceToDevice  hipMemcpyDeviceToDevice
#define gkMemcpyPeerAsync       hipMemcpyPeerAsync

// Blocking, for the same reason as the CUDA spelling below: the buffer
// interface copies on the legacy default stream, and only a blocking stream is
// ordered against it.
#define gkStreamCreate(s)       hipStreamCreate(s)
#define gkStreamDestroy         hipStreamDestroy
#define gkStreamSynchronize     hipStreamSynchronize
#define gkDeviceSynchronize     hipDeviceSynchronize

#define gkEventCreate           hipEventCreate
#define gkEventDestroy          hipEventDestroy
#define gkEventRecord           hipEventRecord
#define gkEventSynchronize      hipEventSynchronize
#define gkEventElapsedTime      hipEventElapsedTime
#define gkEventQuery            hipEventQuery

// Stream capture and executable graphs. Relaxed mode, because capture happens
// on a live server: other threads keep uploading tensors on their own streams
// while this one records, and the stricter modes would fail *their* calls for
// it. The instantiate signatures differ between the vendors (and between CUDA
// 11 and 12), so it is wrapped rather than aliased.
#define gkGraph_t               hipGraph_t
#define gkGraphExec_t           hipGraphExec_t
#define gkStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#define gkStreamBeginCapture    hipStreamBeginCapture
#define gkStreamEndCapture      hipStreamEndCapture
#define gkGraphInstantiate(pe, g) hipGraphInstantiate(pe, g, NULL, NULL, 0)
#define gkGraphLaunch           hipGraphLaunch
#define gkGraphDestroy          hipGraphDestroy
#define gkGraphExecDestroy      hipGraphExecDestroy

#define gkSetDevice             hipSetDevice
#define gkGetDevice             hipGetDevice
#define gkGetDeviceCount        hipGetDeviceCount
#define gkDeviceProp_t          hipDeviceProp_t
#define gkGetDeviceProperties   hipGetDeviceProperties
#define gkMemGetInfo            hipMemGetInfo
#define gkDeviceCanAccessPeer   hipDeviceCanAccessPeer
#define gkDeviceEnablePeerAccess hipDeviceEnablePeerAccess

// "this binary has no code this device can run". HIP names it after the
// binary, CUDA after the image; they mean the same thing.
#define gkErrorNoKernelImage    hipErrorNoBinaryForGpu

// The name the device list shows, and the name a user types.
#define GK_CUDA_BACKEND_NAME "ROCm"

#else

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define gkStream_t              cudaStream_t
#define gkEvent_t               cudaEvent_t
#define gkError_t               cudaError_t
#define gkSuccess               cudaSuccess
#define gkGetErrorString        cudaGetErrorString
#define gkGetLastError          cudaGetLastError

#define gkStreamPerThread       cudaStreamPerThread
#define gkFuncSetAttribute      cudaFuncSetAttribute
#define gkFuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributeMaxDynamicSharedMemorySize

#define gkMalloc                cudaMalloc
#define gkFree                  cudaFree
#define gkHostAlloc(p, s)       cudaHostAlloc(p, s, cudaHostAllocDefault)
#define gkFreeHost              cudaFreeHost
#define gkMemset                cudaMemset
#define gkMemsetAsync           cudaMemsetAsync
#define gkMemcpy                cudaMemcpy
#define gkMemcpyAsync           cudaMemcpyAsync
#define gkMemcpyFromSymbol      cudaMemcpyFromSymbol
#define gkMemcpyHostToDevice    cudaMemcpyHostToDevice
#define gkMemcpyDeviceToHost    cudaMemcpyDeviceToHost
#define gkMemcpyDeviceToDevice  cudaMemcpyDeviceToDevice
#define gkMemcpyPeerAsync       cudaMemcpyPeerAsync

// A *blocking* stream, deliberately. The buffer interface copies with plain
// cudaMemcpy/cudaMemset, which run on the legacy default stream, and a
// non-blocking stream has no ordering relationship with that one - a weight
// upload or a result readback would race the kernels instead of being
// sequenced against them. A blocking stream is ordered against the legacy
// stream, which is the guarantee the rest of this backend assumes. Nothing
// here overlaps two streams, so the concurrency a non-blocking stream buys is
// concurrency this backend never uses.
#define gkStreamCreate(s)       cudaStreamCreate(s)
#define gkStreamDestroy         cudaStreamDestroy
#define gkStreamSynchronize     cudaStreamSynchronize
#define gkDeviceSynchronize     cudaDeviceSynchronize

#define gkEventCreate           cudaEventCreate
#define gkEventDestroy          cudaEventDestroy
#define gkEventRecord           cudaEventRecord
#define gkEventSynchronize      cudaEventSynchronize
#define gkEventElapsedTime      cudaEventElapsedTime
#define gkEventQuery            cudaEventQuery

// Stream capture and executable graphs; see the HIP block for why relaxed
// mode and why instantiate is a wrapper. The flags spelling is the one that
// exists on both sides of the CUDA 12 signature change.
#define gkGraph_t               cudaGraph_t
#define gkGraphExec_t           cudaGraphExec_t
#define gkStreamCaptureModeRelaxed cudaStreamCaptureModeRelaxed
#define gkStreamBeginCapture    cudaStreamBeginCapture
#define gkStreamEndCapture      cudaStreamEndCapture
#define gkGraphInstantiate(pe, g) cudaGraphInstantiateWithFlags(pe, g, 0)
#define gkGraphLaunch           cudaGraphLaunch
#define gkGraphDestroy          cudaGraphDestroy
#define gkGraphExecDestroy      cudaGraphExecDestroy

#define gkSetDevice             cudaSetDevice
#define gkGetDevice             cudaGetDevice
#define gkGetDeviceCount        cudaGetDeviceCount
#define gkDeviceProp_t          cudaDeviceProp
#define gkGetDeviceProperties   cudaGetDeviceProperties
#define gkMemGetInfo            cudaMemGetInfo
#define gkDeviceCanAccessPeer   cudaDeviceCanAccessPeer
#define gkDeviceEnablePeerAccess cudaDeviceEnablePeerAccess

#define gkErrorNoKernelImage    cudaErrorNoKernelImageForDevice

#define GK_CUDA_BACKEND_NAME "CUDA"

#endif

// Lanes per warp. AMD's wavefronts are 64 wide on CDNA and on most RDNA parts
// in wave64 mode; NVIDIA is always 32. Every reduction below is written
// against this rather than a literal, because a warp reduction that assumes
// the wrong width does not fail - it silently sums a third of the data.
#if defined(GK_USE_HIP) && defined(__AMDGCN_WAVEFRONT_SIZE)
#define GK_WARP_SIZE __AMDGCN_WAVEFRONT_SIZE
#elif defined(GK_USE_HIP)
#define GK_WARP_SIZE 64
#else
#define GK_WARP_SIZE 32
#endif
