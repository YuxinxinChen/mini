#pragma once
#include "graph.hxx"
#include "frontier.hxx"

#include "pr_res_problem.hxx"
#include "pr_res_functor.hxx"

#include "filter.hxx"
#include "advance.hxx"

#include "enactor.hxx"

#include "test_utils.hxx"

using namespace mgpu;

using namespace gunrock::oprtr::advance;
using namespace gunrock::oprtr::filter;

namespace gunrock {
namespace pr_res {

struct pr_res_enactor_t : enactor_t {

    //Constructor
    pr_res_enactor_t(standard_context_t &context, int num_nodes, int num_edges) :
        enactor_t(context, num_nodes, num_edges)
    {
    }

    pr_res_enactor_t(const pr_res_enactor_t& rhs) = delete;
    pr_res_enactor_t& operator=(const pr_res_enactor_t& rhs) = delete;

    void init_frontier(std::shared_ptr<pr_res_problem_t> pr_res_problem) {
	std::vector<int> node_idx(pr_res_problem.get()->gslice->num_nodes);
	int count = 0;
        generate(node_idx.begin(), node_idx.end(), [&](){ return count++; });
        buffers[0]->load(node_idx);
    }
    
    //Enact
    void enact(std::shared_ptr<pr_res_problem_t> pr_res_problem, standard_context_t &context) {
        init_frontier(pr_res_problem);

        int frontier_length = pr_res_problem.get()->gslice->num_nodes;
        int selector = 0;
        int iteration;

	int * frontier = buffers[selector].get()->data()->data();
	float * rank = pr_res_problem.get()->d_rank.data();
	float * res = pr_res_problem.get()->d_res1.data();
	float * push_res = pr_res_problem.get()->d_res2.data();

        for (iteration = 0; iteration < pr_res_problem.get()->max_iter ; ++iteration) {
            std::cout << "push " << iteration << std::endl;
	    transform([=]MGPU_DEVICE(int index) {
		int node = frontier[index];
	        rank[node] = rank[node] + res[node];
	        push_res[node] = res[node]; 	
		res[node] = 0.0f;
		}, frontier_length, context);
            frontier_length = advance_forward_kernel<pr_res_problem_t, pr_res_functor_t, false, true>
                (pr_res_problem,
                 buffers[selector],
                 buffers[selector^1],
                 iteration,
                 context);
            selector ^= 1;
            if (!frontier_length) break;
            frontier_length = filter_kernel<pr_res_problem_t, pr_res_functor_t>
                (pr_res_problem,
                 buffers[selector],
                 buffers[selector^1],
                 iteration,
                 context);
            selector ^= 1;
            if (!frontier_length) break;
	    std::cout << "iteration: "<< iteration << " frontier_size: "<< frontier_length << std::endl;
        }
        std::cout << std::endl << "pushed iterations: " << iteration << std::endl;
    }
   
};

} //bfs
} //gunrock
