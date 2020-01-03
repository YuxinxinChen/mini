#pragma once
#include "pr_res/pr_res_problem.hxx"

namespace gunrock {
namespace pr_res {

struct pr_res_functor_t {

static __device__ __forceinline__ bool cond_filter(int idx, pr_res_problem_t::data_slice_t *data, int iteration) {
    return idx != -1;
}

static __device__ __forceinline__ bool cond_advance(int src, int dst, int edge_id, int rank, int output_idx, pr_res_problem_t::data_slice_t *data, int iteration) {
    int neighbor_len = data->row_offset[src+1] - (edge_id-rank);
    float add_res = data->d_res2[src]*data->lambda[0]/neighbor_len;
    float old_res = atomicAdd(data->d_res1+dst, add_res);
    if(old_res < data->epsilon[0] && old_res+add_res >= data->epsilon[0])
	return true;
    else return false;
}

static __device__ __forceinline__ bool apply_advance(int src, int dst, int edge_id, int rank, int output_idx, pr_res_problem_t::data_slice_t *data, int iteration) {
    return true;
}

};

}// end of bfs
}// end of gunrock



