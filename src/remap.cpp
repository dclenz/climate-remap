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

//moab
#include    "moab/ParallelComm.hpp"

using namespace std;
using B = CBlock<real_t>;

#define ERR {if(rval!=MB_SUCCESS)printf("MOAB error at line %d in %s\n", __LINE__, __FILE__);}

int main(int argc, char** argv)
{
    // initialize MPI
    diy::mpi::environment  env(argc, argv);     // equivalent of MPI_Init(argc, argv)/MPI_Finalize()

    // for some reason, local has to be a duplicate of world, not world itself
    diy::mpi::communicator      world;
    MPI_Comm                    local;
    MPI_Comm_dup(world, &local);
    diy::mpi::communicator local_(local);

    int mem_blocks  = -1;                       // everything in core
    int num_threads = 1;                        // needed in order to do timing

    // default command line arguments
    int         pt_dim          = 3;        // dimension of input points
    int         dom_dim         = 2;        // dimension of domain (<= pt_dim)
    int         scalar          = 1;        // flag for scalar or vector-valued science variables (0 == multiple scalar vars)
    int         geom_degree     = 1;        // degree for geometry (same for all dims)
    int         vars_degree     = 4;        // degree for science variables (same for all dims)
    int         ndomp           = 0;      // input number of domain points (same for all dims)
    int         ntest           = 0;        // number of input test points in each dim for analytical error tests
    int         geom_nctrl      = -1;       // input number of control points for geometry (same for all dims)
    vector<int> vars_nctrl      = {11};     // initial # control points for all science variables (default same for all dims)
    string      input           = "sinc";   // input dataset
    int         structured      = 1;        // input data format (bool 0/1)
    int         rand_seed       = -1;       // seed to use for random data generation (-1 == no randomization)
    real_t      regularization  = 0;        // smoothing parameter for models with non-uniform input density (0 == no smoothing)
    int         reg1and2        = 0;        // flag for regularizer: 0 = regularize only 2nd derivs. 1 = regularize 1st and 2nd
    int         verbose         = 1;        // MFA verbosity (0 = no extra output)
    int         weighted        = 0;        // Use NURBS weights (0/1)
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
    ops >> opts::Option('z' ,"romsfile",     romsfile,  " file path for roms data file");
    ops >> opts::Option('z', "mpasfile",     mpasfile,  " file path for mpas data file");


    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // print input arguments
    echo_mfa_settings("simple remap test", dom_dim, pt_dim, scalar, geom_degree, geom_nctrl, vars_degree, vars_nctrl,
                        regularization, reg1and2, adaptive, e_threshold, rounds);
    echo_data_settings(input, "", 0, 0);

    // ---------- debug: test of reading files --------------
    // TODO: REMOVE once I know how to run the actual remap code

    std::string mpas_infile = "mpas_outfile.h5m";
    std::string roms_infile = "roms_outfile.h5m";
    std::string read_opts   = "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS;DEBUG_IO=3;";
    ErrorCode   rval;

    // initialize moab for mpas file
    Interface*              mpas_mbi = new Core();                              // moab interface
    ParallelComm*           mpas_pc  = new ParallelComm(mpas_mbi, local);     // moab communicator
    EntityHandle            mpas_root;
    rval = mpas_mbi->create_meshset(MESHSET_SET, mpas_root); ERR(rval);

    // initialize moab for roms file
    Interface*              roms_mbi = new Core();                              // moab interface
    ParallelComm*           roms_pc  = new ParallelComm(roms_mbi, local);     // moab communicator
    EntityHandle            roms_root;
    rval = roms_mbi->create_meshset(MESHSET_SET, roms_root); ERR(rval);

    // debug
    fmt::print(stderr, "*** consumer before reading files ***\n");

    // read files
    rval = mpas_mbi->load_file(mpas_infile.c_str(), &mpas_root, read_opts.c_str() ); ERR(rval);
    rval = roms_mbi->load_file(roms_infile.c_str(), &roms_root, read_opts.c_str() ); ERR(rval);

    // debug
    fmt::print(stderr, "*** consumer after reading files ***\n");

    return 0;

    // ------------- end of debug: test of reading files

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
    Bounds<real_t> domain(dom_dim);
    domain.min = vector<real_t>(dom_dim, 0);
    domain.max = vector<real_t>(dom_dim, 1);

    Decomposer<real_t> decomposer(dom_dim, domain, tot_blocks);
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
                input, "", ndomp, structured, rand_seed, 0, 0, 0,
                reg1and2, regularization, adaptive, verbose, mfa_info, d_args);

    double encode_time = MPI_Wtime();
    if (input=="roms")
    {
        master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
        {
            b->read_roms_data_3d_mb<real_t>(cp, romsfile);
            b->input = b->roms_input;
            b->roms_input = nullptr;
        });
    }
    else if (input=="mpas")
    {
        master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
        {
            b->read_mpas_data_3d_mb<real_t>(cp, mpasfile, mfa_info);
            b->input = b->mpas_input;
            b->mpas_input = nullptr;
        });
    }
    else if (input=="remap")
    {
        master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
        {
            b->read_roms_data_3d_mb<double>(cp, romsfile); 

            VectorX<real_t> mins = b->roms_input->mins();
            VectorX<real_t> maxs = b->roms_input->maxs();
            b->read_mpas_data_3d_mb<double>(cp, mpasfile, mfa_info, mins, maxs);

            b->remap(cp, mfa_info);
        });
    }
    else
    {
        cerr << "Input keyword \'" << input << "\' not recognized. Exiting." << endl;
        exit(0);
    }
    encode_time = MPI_Wtime() - encode_time;

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
