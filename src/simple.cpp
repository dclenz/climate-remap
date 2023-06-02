//--------------------------------------------------------------
// example of encoding / decoding higher dimensional data w/ fixed number of control points and a
// single block in a split model w/ one model containing geometry and other model science variables
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#include <mfa/mfa.hpp>

#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <set>

#include <diy/master.hpp>
#include <diy/reduce-operations.hpp>
#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>
#include <diy/io/block.hpp>

#include "opts.h"

#include "cblock.hpp"
#include "parser.hpp"

using namespace std;
using B = CBlock<real_t>;

int main(int argc, char** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);     // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;               // equivalent of MPI_COMM_WORLD


    MFAParser opts;
    bool proceed = opts.parse_input(argc, argv);
    if (!proceed)
    {
        if (world.rank() == 0)
            std::cout << opts.ops;
        return 1;
    }

    int tot_blocks = opts.tot_blocks;
    int mem_blocks  = -1;                       // everything in core
    int num_threads = 1;                        // needed in order to do timing

    int dom_dim = opts.dom_dim;
    int pt_dim = opts.pt_dim;
    string input = opts.input;

    // print input arguments
    opts.echo_mfa_settings("simple remap example");
    opts.echo_all_data_settings();

    // initialize DIY
    diy::FileStorage          storage("./DIY.XXXXXX"); // used for blocks to be moved out of core
    diy::Master               master(world,
                                     num_threads,
                                     mem_blocks,
                                     &B::create,
                                     &B::destroy,
                                     &storage,
                                     &B::save,
                                     &B::load);
    diy::ContiguousAssigner   assigner(world.size(), tot_blocks);

    // set global domain bounds and decompose
    Bounds<real_t> dom_bounds(dom_dim);
    for (int i = 0; i < dom_bounds.min.dimension(); i++)
    {
        dom_bounds.min[i] = 0.0;
        dom_bounds.max[i] = 1.0;
    }
    
    Decomposer<real_t> decomposer(dom_dim, dom_bounds, tot_blocks);
    decomposer.decompose(world.rank(),
                         assigner,
                         [&](int gid, const Bounds<real_t>& core, const Bounds<real_t>& bounds, const Bounds<real_t>& domain, const RCLink<real_t>& link)
                         { B::add(gid, core, bounds, domain, link, master, dom_dim, pt_dim, 0.0); });

    // If scalar == true, assume all science vars are scalar. Else one vector-valued var
    // We assume that dom_dim == geom_dim
    // Different examples can reset this below
    vector<int> model_dims;
    if (opts.scalar) // Set up (pt_dim - dom_dim) separate scalar variables
    {
        model_dims.assign(pt_dim - dom_dim + 1, 1);
        model_dims[0] = dom_dim;                        // index 0 == geometry
    }
    else    // Set up a single vector-valued variable
    {   
        model_dims = {dom_dim, pt_dim - dom_dim};
    }

    // set up parameters for examples
    MFAInfo     mfa_info(dom_dim, opts.verbose);
    DomainArgs  d_args(dom_dim, model_dims);
    opts.setup_args(model_dims, mfa_info, d_args);

   
    if (input=="roms")
    {
        master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
        {
            b->read_roms_data<double>(cp, mfa_info, d_args);
        });
    }
    else if (input=="mpas")
    {
        master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
        {
            b->read_mpas_data<double>(cp, mfa_info, d_args);
        });
    }
    else
    {
        cerr << "Input keyword \'" << input << "\' not recognized. Exiting." << endl;
        exit(0);
    }

    // compute the MFA
    double encode_time = MPI_Wtime();
    master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
            { b->fixed_encode_block(cp, mfa_info); });
    encode_time = MPI_Wtime() - encode_time;

    // Decode onto original point locations or grid
    double decode_time = MPI_Wtime();
    if (opts.error)
    {
        fprintf(stderr, "\nDecoding at original point locations\n");
        master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
        { 
            b->range_error(cp, true, false);
        });
        decode_time = MPI_Wtime() - decode_time;
    }
    else if (opts.decode_grid.size() == dom_dim)
    {
        fprintf(stderr, "\nDecoding on regular grid of size %s\n", mfa::print_vec(opts.decode_grid).c_str());
        master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
        {
            b->decode_block_grid(cp, opts.decode_grid);
        });
        decode_time = MPI_Wtime() - decode_time;
    }

    // print results
    fprintf(stderr, "\n------- Final block results --------\n");
    master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
            { b->print_block(cp, opts.error); });
    fprintf(stderr, "encoding time         = %.3lf s.\n", encode_time);
    if (opts.error)
        fprintf(stderr, "decoding time         = %.3lf s.\n", decode_time);
    fprintf(stderr, "-------------------------------------\n\n");

    // save the results in diy format
    string outname = input + ".mfa";
    diy::io::write_blocks(outname, world, master);
}
