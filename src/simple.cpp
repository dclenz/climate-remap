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
#include "example-setup.hpp"
#include "cblock.hpp"
#include "parser.hpp"

using namespace std;
using B = CBlock<real_t>;

int main(int argc, char** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);     // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;               // equivalent of MPI_COMM_WORLD

    // default command line arguments
    int         pt_dim          = 3;        // dimension of input points
    int         dom_dim         = 2;        // dimension of domain (<= pt_dim)
    int         scalar          = 1;        // flag for scalar or vector-valued science variables (0 == multiple scalar vars)
    int         geom_degree     = 1;        // degree for geometry (same for all dims)
    int         vars_degree     = 4;        // degree for science variables (same for all dims)
    int         ndomp           = 100;      // input number of domain points (same for all dims)
    int         ntest           = 0;        // number of input test points in each dim for analytical error tests
    int         geom_nctrl      = -1;       // input number of control points for geometry (same for all dims)
    vector<int> vars_nctrl      = {11};     // initial # control points for all science variables (default same for all dims)
    string      input           = "sinc";   // input dataset
    real_t      rot             = 0.0;      // rotation angle in degrees
    real_t      twist           = 0.0;      // twist (waviness) of domain (0.0-1.0)
    real_t      noise           = 0.0;      // fraction of noise
    int         structured      = 1;        // input data format (bool 0/1)
    int         rand_seed       = -1;       // seed to use for random data generation (-1 == no randomization)
    real_t      regularization  = 0;        // smoothing parameter for models with non-uniform input density (0 == no smoothing)
    int         reg1and2        = 0;        // flag for regularizer: 0 = regularize only 2nd derivs. 1 = regularize 1st and 2nd
    int         verbose         = 1;        // MFA verbosity (0 = no extra output)
    int         strong_sc       = 1;        // strong scaling (bool 0 or 1, 0 = weak scaling)
    int         weighted        = 0;        // Use NURBS weights (0/1)
    real_t      ghost           = 0.1;      // amount of ghost zone overlap as a factor of block size (0.0 - 1.0)
    int         tot_blocks      = 1;        // 
    int         adaptive        = 0;        // do analytical encode (0/1)
    real_t      e_threshold     = 1e-1;     // error threshold for adaptive fitting
    int         rounds          = 0;
    bool        help            = false;    // show help

    string romsfile;
    string mpasfile;

    opts::Options ops;
    ops >> opts::Option('d', "pt_dim",      pt_dim,     " dimension of points");
    ops >> opts::Option('m', "dom_dim",     dom_dim,    " dimension of domain");
    ops >> opts::Option('l', "scalar",      scalar,     " flag for scalar or vector-valued science variables");
    ops >> opts::Option('p', "geom_degree", geom_degree," degree in each dimension of geometry");
    ops >> opts::Option('q', "vars_degree", vars_degree," degree in each dimension of science variables");
    ops >> opts::Option('n', "ndomp",       ndomp,      " number of input points in each dimension of domain");
    ops >> opts::Option('a', "ntest",       ntest,      " number of test points in each dimension of domain (for analytical error calculation)");
    ops >> opts::Option('g', "geom_nctrl",  geom_nctrl, " number of control points in each dimension of geometry");
    ops >> opts::Option('v', "vars_nctrl",  vars_nctrl, " number of control points in each dimension of all science variables");
    ops >> opts::Option('i', "input",       input,      " input dataset");
    ops >> opts::Option('h', "help",        help,       " show help");
    ops >> opts::Option('x', "structured",  structured, " input data format (default=structured=true)");
    ops >> opts::Option('y', "rand_seed",   rand_seed,  " seed for random point generation (-1 = no randomization, default)");
    ops >> opts::Option('b', "regularization", regularization, "smoothing parameter for models with non-uniform input density");
    ops >> opts::Option('k', "reg1and2",    reg1and2,   " regularize both 1st and 2nd derivatives (if =1) or just 2nd (if =0)");
    ops >> opts::Option('z' ,"romsfile",     romsfile, "");
    ops >> opts::Option('z', "mpasfile",     mpasfile, "");


    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    int mem_blocks  = -1;                       // everything in core
    int num_threads = 1;                        // needed in order to do timing

    // print input arguments
    echo_mfa_settings("simple remap test", dom_dim, pt_dim, scalar, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                        regularization, reg1and2, adaptive, e_threshold, rounds);
    echo_data_settings(input, "", 0, 0);

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

    // We assume that dom_dim == geom_dim
    // Different examples can reset this below
    vector<int> model_dims = {dom_dim, pt_dim - dom_dim};


    // set up parameters for examples
    MFAInfo     mfa_info(dom_dim, verbose);
    DomainArgs  d_args(dom_dim, model_dims);
    setup_args(dom_dim, pt_dim, model_dims, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                input, "", ndomp, structured, rand_seed, rot, twist, noise,
                reg1and2, regularization, adaptive, verbose, mfa_info, d_args);
    // master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
    // {
    //     b->read_roms_data<double>(cp, romsfile, mfa_info, d_args);
    //     b->read_mpas_data<double>(cp, mpasfile, mfa_info, d_args);
    // });

    double encode_time = MPI_Wtime();
    if (input=="roms")
    {
        master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
        {
            b->read_roms_data<double>(cp, romsfile, mfa_info, d_args);
            b->input = b->roms_input;
            b->roms_input = nullptr;
        });
    }
    else if (input=="mpas")
    {
        master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
        {
            b->read_mpas_data<double>(cp, mpasfile, mfa_info, d_args);
            b->input = b->mpas_input;
            b->mpas_input = nullptr;
        });
    }
    else if (input=="remap")
    {
        master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
        {
            b->read_roms_data<double>(cp, romsfile, mfa_info, d_args);
            b->read_mpas_data<double>(cp, mpasfile, mfa_info, d_args);
            b->remap(cp, mfa_info);
        });        
    }
    else
    {
        cerr << "Input keyword \'" << input << "\' not recognized. Exiting." << endl;
        exit(0);
    }
    encode_time = MPI_Wtime() - encode_time;


    // // compute the MFA
    // double encode_time = MPI_Wtime();
    // master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
    // { 
    //     b->remap(cp, mfa_info);

    //     // b->app
    // });
    // encode_time = MPI_Wtime() - encode_time;

    // // Decode onto original point locations or grid
    // double decode_time = MPI_Wtime();
    // if (error)
    // {
    //     fprintf(stderr, "\nDecoding at original point locations\n");
    //     master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
    //     { 
    //         b->range_error(cp, true, false);
    //     });
    //     decode_time = MPI_Wtime() - decode_time;
    // }
    // else if (opts.decode_grid.size() == dom_dim)
    // {
    //     fprintf(stderr, "\nDecoding on regular grid of size %s\n", mfa::print_vec(opts.decode_grid).c_str());
    //     master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
    //     {
    //         b->decode_block_grid(cp, opts.decode_grid);
    //     });
    //     decode_time = MPI_Wtime() - decode_time;
    // }

    // print results
    fprintf(stderr, "\n------- Final block results --------\n");
    master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
            { b->print_model(cp); });
    fprintf(stderr, "encoding time         = %.3lf s.\n", encode_time);
    // if (opts.error)
    //     fprintf(stderr, "decoding time         = %.3lf s.\n", decode_time);
    fprintf(stderr, "-------------------------------------\n\n");

    // save the results in diy format
    string outname = "remap.mfa";
    diy::io::write_blocks(outname, world, master);
}
