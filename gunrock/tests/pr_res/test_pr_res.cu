#include "pr_res/pr_res_enactor.hxx"
#include "test_utils.hxx"

#include <algorithm>
#include <cstdlib>

using namespace gunrock;
using namespace gunrock::pr_res;

int main(int argc, char** argv) {

    // read in graph file
    std::string filename;
    CommandLineArgs args(argc, argv);
    args.GetCmdLineArgument("file", filename);

    // read in source node from cmd line
    int max_iter = 50;
    args.GetCmdLineArgument("max_iter", max_iter);

    float lambda = 0.85;
    args.GetCmdLineArgument("lambda", lambda);

    float epsilon = 0.01;
    args.GetCmdLineArgument("epsilon", epsilon);

    bool undirected = true;
    args.GetCmdLineArgument("undirected", undirected);

    int device = 1;
    args.GetCmdLineArgument("device", device);

    cudaSetDevice(device);
    cout << "max iter: "<< max_iter << " lambda: "<< lambda << " epsilon: "<< epsilon << " undirected: "<< undirected << " device: "<< device <<endl;

    // CUDA context is used for all mgpu transforms
    standard_context_t context;
   
    // Load graph data to device
    std::shared_ptr<graph_t> graph = load_graph(filename.c_str(), undirected);
    std::shared_ptr<graph_device_t> d_graph(std::make_shared<graph_device_t>());
    graph_to_device(d_graph, graph, context);

//    display_csr(graph.get()->csr);

    // Initializes coloring problem object
    std::shared_ptr<pr_res_problem_t> pr_res_problem(std::make_shared<pr_res_problem_t>(d_graph, max_iter, lambda, epsilon, context));
    cout << pr_res_problem.get()->gslice->num_nodes << ", "<< pr_res_problem.get()->gslice->num_edges << endl;


    std::shared_ptr<pr_res_enactor_t> pr_res_enactor(std::make_shared<pr_res_enactor_t>(context, d_graph->num_nodes, d_graph->num_edges));
    std::cout << "start PageRank" << std::endl;

    test_timer_t timer;
    timer.start();
    pr_res_enactor->enact(pr_res_problem, context);
    cout << "elapsed time: " << timer.end() << "s." << std::endl;

//    cout << "rank\n";
//    display_device_data(pr_res_problem.get()->d_rank.data(), pr_res_problem.get()->gslice->num_nodes);
//    cout << "res1\n";
//    display_device_data(pr_res_problem.get()->d_res1.data(), pr_res_problem.get()->gslice->num_nodes);


    pr_res_problem->extract(context);

    std::vector<float> validation_rank;
    std::vector<float> validation_res;

    pr_res_problem->cpu(validation_rank, validation_res, graph->csr->offsets, graph->csr->indices);

//    if (!validate(pr_res_problem.get()->h_rank, validation_rank, pr_res_problem.get()->epsilon*2))
    if (!validate_rank(pr_res_problem.get()->h_rank, validation_rank))
        cout << "Validation Error." << endl;
    else
        cout << "Correct." << endl;


}



