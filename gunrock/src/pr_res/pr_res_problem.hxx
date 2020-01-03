#pragma once

#include "problem.hxx"

#include <queue>

namespace gunrock {
namespace pr_res {

struct pr_res_problem_t : problem_t {
  mem_t<float> d_rank;
  mem_t<float> d_res1;
  mem_t<float> d_res2;
  int max_iter;
  mem_t<float> d_epsilon;
  mem_t<float> d_lambda;
  std::vector<float> h_rank;
  std::vector<float> h_res;
  float lambda;
  float epsilon;
  mem_t<int> pivot;

  struct data_slice_t {
      float *d_rank;
      float *d_res1;
      float *d_res2;
      int *pivot;
      int *row_offset;
      float *lambda;
      float *epsilon;

      void init(mem_t<float> &_rank, mem_t<float> &_res1, mem_t<float> &_res2, mem_t<int> &_pivot, mem_t<int> &_row_offset, mem_t<float> &_lambda, mem_t<float> &_epsilon) {
          d_rank = _rank.data();
          d_res1 = _res1.data();
	  d_res2 = _res2.data();
	  pivot = _pivot.data();
	  row_offset = _row_offset.data();
	  lambda = _lambda.data();
	  epsilon = _epsilon.data();
      }
  };

  mem_t<data_slice_t> d_data_slice;
  std::vector<data_slice_t> data_slice;

  pr_res_problem_t() {}

  pr_res_problem_t(const pr_res_problem_t& rhs) = delete;
  pr_res_problem_t& operator=(const pr_res_problem_t& rhs) = delete;

  pr_res_problem_t(std::shared_ptr<graph_device_t> rhs, int max_iter, float lambda, float epsilon, standard_context_t& context) :
      problem_t(rhs),
      max_iter(max_iter),
      lambda(lambda),
      epsilon(epsilon),
      data_slice( std::vector<data_slice_t>(1) ) {
         d_rank = fill(1.0f-lambda, rhs->num_nodes, context);
	 d_res1 = fill(0.0f, rhs->num_nodes, context);
	 d_res2 = fill(0.0f, rhs->num_nodes, context);
	 d_epsilon = fill(epsilon, 1, context);
	 d_lambda = fill(lambda, 1, context);
	 initRes(d_res1, context); 
	 pivot = fill(0, 1, context);
         data_slice[0].init(d_rank, d_res1, d_res2, pivot, rhs->d_row_offsets, d_lambda, d_epsilon);
         d_data_slice = to_mem(data_slice, context);
      }

  void initRes(mem_t<float> &_res, standard_context_t &context) {
     int *row_offsets = gslice->d_row_offsets.data();
     int *col_indices = gslice->d_col_indices.data();
     float *res = _res.data();
     float *lambda= d_lambda.data();
     auto f = [=]__device__(int idx) {
     	int neighbor_offset = row_offsets[idx];
   	int neighbor_len =  row_offsets[idx+1]-neighbor_offset;
   	float add_item = (1.0-lambda[0])*lambda[0]/neighbor_len;
   	for(int item = 0; item < neighbor_len; item++)
   	{
   	    int neighbor = col_indices[neighbor_offset+item];
   	    atomicAdd(res+neighbor, add_item);
   	}
      };
      transform(f, gslice->num_nodes, context);

  }

  void extract(standard_context_t &context) {
      // TODO: output the selected edge ids?
      float *res1 = d_res1.data();
//      float *res2 = d_res2.data();
//      transform([=]MGPU_DEVICE(int index) {
//        res1[index] = res1[index]+res2[index];
//      }, gslice->num_nodes, context);
      h_rank = from_mem(d_rank);
      h_res = from_mem(d_res1);
  }

  void cpu(std::vector<float> &validation_rank, std::vector<float> &validation_res,
                  std::vector<int> &row_offsets,
                  std::vector<int> &col_indices) {
      validation_rank = std::vector<float>(gslice->num_nodes, (1.0-lambda));
      validation_res = std::vector<float>(gslice->num_nodes, 0.0f);
      std::queue<int> f;
      for(int node=0; node < gslice->num_nodes; node++)
      {
	  int neighbor_offset = row_offsets[node];
	  int neighbor_len = row_offsets[node+1]-neighbor_offset;
	  float res = (1.0-lambda)*lambda/neighbor_len; 
	  for(int item = 0; item<neighbor_len; item++)
	  {
	      int neighbor = col_indices[neighbor_offset+item];
	      validation_res[neighbor] = validation_res[neighbor]+res;
	  }
	  f.push(node);
      }

//      while(!f.empty())
//      {
//	  int node = f.front();
//	  f.pop();
//	  int neighbor_offset = row_offsets[node];
//	  int neighbor_len = row_offsets[node+1]-neighbor_offset;
//	  float res = validation_res[node];
//	  validation_rank[node] = validation_rank[node]+res;
//	  res = res*lambda/neighbor_len;
//	  validation_res[node] = 0.0f;
//          for(int item=0; item<neighbor_len; item++)
//	  {
//	      int neighbor = col_indices[neighbor_offset+item];
//	      validation_res[neighbor] = validation_res[neighbor]+res;
//	      if(validation_res[neighbor] >= epsilon && validation_res[neighbor]-res < epsilon)
//		  f.push(neighbor);
//	  }
//      }
      std::queue<int> output;
      std::vector<float>res_reduce(gslice->num_nodes, 0.0f);

      for(int iteration=0; iteration < max_iter; iteration++)
      {
	  int frontier_length = f.size();
	  for(int it=0; it<frontier_length; it++)
	  {
	      int node = f.front();
	      f.pop();
	      int neighbor_offset = row_offsets[node];
	      int neighbor_len = row_offsets[node+1]-neighbor_offset;
	      float res = validation_res[node];
	      validation_rank[node] = validation_rank[node]+res;
	      res = res*lambda/neighbor_len;
	      validation_res[node] = 0.0f;
	      for(int item=0; item<neighbor_len; item++)
	      {
		 int neighbor = col_indices[neighbor_offset+item];
		 res_reduce[neighbor] = res_reduce[neighbor]+res;
	      }
	  }
	  for(int it=0; it<validation_res.size(); it++) {
	     float temp = validation_res[it]+res_reduce[it];
	     if(validation_res[it] < epsilon && temp >= epsilon)
		 output.push(it);
	     validation_res[it] = temp;
	     res_reduce[it] = 0.0f;
	  }
	  frontier_length = output.size();
	  if(frontier_length==0) break;
	  f.swap(output);
      }
  }

};

} //end coloring
} //end gunrock
