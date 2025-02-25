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

    int tot_blocks = 1;                     // number of DIY blocks
    int mem_blocks  = -1;                   // everything in core
    int num_threads = 1;                    // set single threaded for timing

    // default command line arguments
    int         pt_dim          = 3;        // dimension of input points
    int         dom_dim         = 2;        // dimension of domain (<= pt_dim)
    int         vars_degree     = 2;        // degree for science variables (same for all dims)
    vector<int> vars_nctrl      = {11};     // initial # control points for all science variables (default same for all dims)
    string      romsfile        = "";       // data file produced by ROMS (remap target)
    string      mpasfile        = "";       // data file produced by MPAS (remap source)
    vector<string> varNames     = {""};
    real_t      regularization  = 0;        // smoothing parameter for models with non-uniform input density (0 == no smoothing)
    int         reg1and2        = 0;        // flag for regularizer: 0 = regularize only 2nd derivs. 1 = regularize 1st and 2nd
    int         adaptive        = 0;        // do analytical encode (0/1)
    real_t      e_threshold     = 1e-1;     // error threshold for adaptive fitting
    int         rounds          = 0;        // maximum rounds of adaptive MFA refinement (0 = no limit)
    int         verbose         = 1;        // MFA verbosity (0 = no extra output)
    bool        help            = false;    // show help

    const int geom_degree = 1;  // assume flat geometry
    const int geom_nctrl = -1;  // assume flat geometry

    // Parse command line
    opts::Options ops;
    ops >> opts::Option('d', "pt_dim",      pt_dim,     " dimension of points");
    ops >> opts::Option('m', "dom_dim",     dom_dim,    " dimension of domain");
    ops >> opts::Option('q', "vars_degree", vars_degree," degree in each dimension of science variables");
    ops >> opts::Option('v', "vars_nctrl",  vars_nctrl, " number of control points in each dimension of all science variables");
    ops >> opts::Option('b', "regularization", regularization, "smoothing parameter for models with non-uniform input density");
    ops >> opts::Option('k', "reg1and2",    reg1and2,   " regularize both 1st and 2nd derivatives (if =1) or just 2nd (if =0)");
    ops >> opts::Option('r', "varNames",    varNames,  " vector of variable names to remap");
    ops >> opts::Option('z' ,"romsfile",    romsfile,  " file path for roms data file");
    ops >> opts::Option('z', "mpasfile",    mpasfile,  " file path for mpas data file");
    ops >> opts::Option('x', "verbose",     verbose,    " MFA verbosity (0 = no output, 1 = limited output, 2 = high output)");
    ops >> opts::Option('h', "help",        help,       " show help");
    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << ops;
        return 1;
    }

    // print input arguments
    fmt::print("--------- MFA Settings ----------\n");
    fmt::print("pt_dim = {}\t dom_dim = {}\n", pt_dim, dom_dim);
    fmt::print("geom_deg = {}\t geom_nctrl = {}\n", geom_degree, geom_nctrl);
    fmt::print("vars_deg = {}\t vars_nctrl = {}\n", vars_degree, mfa::print_vec(vars_nctrl));
    fmt::print("encoding type = {}\n", adaptive ? "adaptive" : "fixed");
    if (adaptive)
        fmt::print("error = {}\t max rounds = {}\n", e_threshold, (rounds == 0 ? "unlimited" : to_string(rounds)));
    
    string reg_type = regularization == 0 ? "N/A" : (reg1and2 > 0 ? "1st and 2nd derivs" : "2nd derivs only");
    fmt::print("regularization = {}, type: {}\n", regularization, reg_type);

    string thread_type = "undefined";
#ifdef MFA_TBB
    thread_type = "TBB";
#endif
#ifdef MFA_SERIAL
    thread_type = "serial";
#endif
    fmt::print("threading: {}\n", thread_type);
    fmt::print("MPAS file name: {}\n", mpasfile);
    fmt::print("ROMS file name: {}\n", romsfile);
    fmt::print("Variables to remap: {}\n", mfa::print_vec(varNames));

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

    // Set up problem dimensionality. Should look like {3, 1, 1, ..., 1}
    //   That is, the geometry is 3-dimensional and every variable is a scalar
    const int geom_dim = 3;  // Assume points are always 3D
    vector<int> model_dims(pt_dim - geom_dim + 1, 1);
    model_dims[0] = geom_dim;

    // Set up MFA
    mfa::MFAInfo mfa_info(dom_dim, verbose, model_dims, geom_degree, geom_nctrl, vars_degree, vars_nctrl);
    mfa_info.weighted         = 0;
    mfa_info.regularization   = regularization;
    mfa_info.reg1and2         = reg1and2;

    // Read in data
    double init_time = MPI_Wtime();
    master.foreach([&](B* b, const diy::Master::ProxyWithLink&cp)
    {
        b->varNames = varNames;
        b->initMOAB(local, dom_dim);
        b->readTargetData<double>(cp, romsfile);
        b->readSourceData<double>(cp, mpasfile);
    });
    init_time = MPI_Wtime() - init_time;

    // Perform the remapping
    double encode_time = MPI_Wtime();
    master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
    {
        // Customize remapper block
        b->verbose = verbose;
        b->dumpMatrices = false;
        b->addBdryData = true;

        // Do the remapping
        b->remap(cp, mfa_info);
    });
    encode_time = MPI_Wtime() - encode_time;

    // Write source file and (remapped) target file to VTK
    master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
    {
        b->writeVTK();
    });

    // print results
    fmt::print("\n------- Final block results --------\n");
    master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
            { b->print_model(cp); });
    fmt::print("startup time          = {:.3} s.\n", init_time);
    fmt::print("encoding time         = {:.3} s.\n", encode_time);
    fmt::print("-------------------------------------\n\n");

    // save the results in diy format
    string outname = "remap.mfa";
    diy::io::write_blocks(outname, world, master);
}
