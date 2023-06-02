//--------------------------------------------------------------
// Command line parser for MFA examples
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//--------------------------------------------------------------
#ifndef _MFA_EX_PARSER_HPP
#define _MFA_EX_PARSER_HPP

#include "opts.h"
#include "domain_args.hpp"
#include <mfa/types.hpp>
#include <mfa/mfa.hpp>
#include <mfa/utilities/util.hpp>

struct MFAParser
{
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
    int         weighted        = 1;        // solve for and use weights (bool 0/1)
    real_t      rot             = 0.0;      // rotation angle in degrees
    real_t      twist           = 0.0;      // twist (waviness) of domain (0.0-1.0)
    real_t      noise           = 0.0;      // fraction of noise
    int         error           = 1;        // decode all input points and check error (bool 0/1)
    string      infile;                     // input file name
    string      infile2;
    int         structured      = 1;        // input data format (bool 0/1)
    int         rand_seed       = -1;       // seed to use for random data generation (-1 == no randomization)
    real_t      regularization  = 0;        // smoothing parameter for models with non-uniform input density (0 == no smoothing)
    int         reg1and2        = 0;        // flag for regularizer: 0 = regularize only 2nd derivs. 1 = regularize 1st and 2nd
    int         verbose         = 1;        // MFA verbosity (0 = no extra output)
    vector<int> decode_grid     = {};       // Grid size for uniform decoding
    int         strong_sc       = 1;        // strong scaling (bool 0 or 1, 0 = weak scaling)
    real_t      ghost           = 0.1;      // amount of ghost zone overlap as a factor of block size (0.0 - 1.0)
    int         tot_blocks      = 1;        // 
    bool        help            = false;    // show help

    // Constants for this example
    bool adaptive = false;
    real_t e_threshold = 0;
    int rounds = 0;

    opts::Options ops;

    MFAParser()
    {
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
        ops >> opts::Option('w', "weights",     weighted,   " solve for and use weights");
        ops >> opts::Option('r', "rotate",      rot,        " rotation angle of domain in degrees");
        ops >> opts::Option('t', "twist",       twist,      " twist (waviness) of domain (0.0-1.0)");
        ops >> opts::Option('s', "noise",       noise,      " fraction of noise (0.0 - 1.0)");
        ops >> opts::Option('c', "error",       error,      " decode entire error field (default=true)");
        ops >> opts::Option('f', "infile",      infile,     " input file name");
        ops >> opts::Option('h', "help",        help,       " show help");
        ops >> opts::Option('x', "structured",  structured, " input data format (default=structured=true)");
        ops >> opts::Option('y', "rand_seed",   rand_seed,  " seed for random point generation (-1 = no randomization, default)");
        ops >> opts::Option('b', "regularization", regularization, "smoothing parameter for models with non-uniform input density");
        ops >> opts::Option('k', "reg1and2",    reg1and2,   " regularize both 1st and 2nd derivatives (if =1) or just 2nd (if =0)");
        ops >> opts::Option('u', "grid_decode", decode_grid," size of regular grid to decode MFA");
        ops >> opts::Option('o', "overlap",     ghost,      " relative ghost zone overlap (0.0 - 1.0)");
        ops >> opts::Option('z', "infile2",     infile2,    " extra data file (some apps require two file paths");
        ops >> opts::Option('z', "strong_sc",   strong_sc,  " strong scaling (1 = strong, 0 = weak)");
        ops >> opts::Option('z', "tot_blocks",  tot_blocks, " total number of blocks");
    }

    // parse command line input and indicate if program should exit
    bool parse_input(int argc, char** argv)
    {
        bool success = ops.parse(argc, argv);
        bool proceed = success && !help;

        return proceed;
    }

    // Print basic info about data set
    void echo_basic_data_settings(ostream& os = std::cerr)
    {
        os << "--------- Data Settings ----------" << endl;
        os << "input: "       << input       << ", " << "infile: " << infile << endl;
        os << "num pts    = " << ndomp       << '\t' << "test pts    = " << (ntest > 0 ? to_string(ntest) : "N/A") << endl;

        return;
    }

    // Print all info about data set
    void echo_all_data_settings(ostream& os = std::cerr)
    {
        echo_basic_data_settings();
        os << "structured = " << boolalpha << (bool)structured << '\t' << "random seed = " << rand_seed << endl;
        os << "rotation   = " << setw(7) << left << rot << '\t' << "twist       = " << twist << endl;
        os << "noise      = " << setw(7) << left << noise << endl;

        return;
    }

    void echo_mfa_settings(string run_name, ostream& os = std::cerr)
    {
        os << ">>> Running \'" << run_name << "\'" << endl;
        os << endl;
        os << "--------- MFA Settings ----------" << endl;
        os << "pt_dim   = " << pt_dim      << '\t' << "dom_dim    = " << dom_dim 
                << '\t' << "scalar: " << boolalpha << (bool)scalar << endl;
        os << "geom_deg = " << geom_degree << '\t' << "geom_nctrl = " << geom_nctrl  << endl;
        os << "vars_deg = " << vars_degree << '\t' << "vars_nctrl = " << mfa::print_vec(vars_nctrl) << endl;
        os << "regularization = " << regularization << ", type: " << 
            (regularization == 0 ? "N/A" : (reg1and2 > 0 ? "1st and 2nd derivs" : "2nd derivs only")) << endl;
        if (adaptive)
        {
        os << "error    = " << e_threshold << '\t' << "max rounds = " << (rounds == 0 ? "unlimited" : to_string(rounds)) << endl;
        }
#ifdef MFA_NO_WEIGHTS
        weighted = 0;
        os << "weighted: false" << endl;
#else
        os << "weighted: " << boolalpha <<  (bool)weighted << endl;
#endif
#ifdef CURVE_PARAMS
        os << "parameterization method: curve" << endl;
#else
        os << "parameterization method: domain" << endl;
#endif
#ifdef MFA_TBB
        os << "threading: TBB" << endl;
#endif
#ifdef MFA_KOKKOS
        os << "threading: Kokkos" << endl;
        os << "KOKKOS execution space: " << Kokkos::DefaultExecutionSpace::name() << "\n";
#endif
#ifdef MFA_SYCL
        os << "threading: SYCL" << endl;
#endif
#ifdef MFA_SERIAL
        os << "threading: serial" << endl;
#endif

        return;
    }

    void echo_multiblock_settings(MFAInfo& mfa_info, DomainArgs& d_args, int nproc, int tot_blocks, vector<int>& divs, int strong_sc, real_t ghost, ostream& os = std::cerr)
    {
        os << "------- Multiblock Settings ---------" << endl;
        os << "Total MPI processes  =  " << nproc << "\t" << "Total blocks = " << tot_blocks << endl;
        os << "Blocks per dimension = " << mfa::print_vec(divs) << endl;
        os << "Ghost overlap  = " << ghost << endl;
        os << "Strong scaling = " << boolalpha << (bool)strong_sc << endl;
        os << "Per-block settings:" << endl;
        os << "    Input pts (each dim):      " << mfa::print_vec(d_args.ndom_pts) << endl;
        os << "    Geom ctrl pts (each dim):  " << mfa::print_vec(mfa_info.geom_model_info.nctrl_pts) << endl;
        for (int k = 0; k < mfa_info.nvars(); k++)
        {
            os << "    Var " << k << " ctrl pts (each dim): " << mfa::print_vec(mfa_info.var_model_infos[k].nctrl_pts) << endl;
        }

        return;
    }

        // Sets the degree and number of control points 
    // For geom and each var, degree is same in each domain dimension
    // For geom, # ctrl points is the same in each domain dimension
    // For each var, # ctrl points varies per domain dimension, but is the same for all of the vars
    void set_mfa_info(vector<int> model_dims, MFAInfo& mfa_info)
    // void set_mfa_info(int dom_dim, vector<int> model_dims, 
    //                     int geom_degree, int geom_nctrl,
    //                     int vars_degree, vector<int> vars_nctrl,
    //                     MFAInfo& mfa_info)
    {
        // Clear any existing data in mfa_info
        mfa_info.reset();

        int nvars       = model_dims.size() - 1;
        int geom_dim    = model_dims[0];

        // If only one value for vars_nctrl was parsed, assume it applies to all dims
        if (vars_nctrl.size() == 1 & dom_dim > 1)
        {
            vars_nctrl = vector<int>(dom_dim, vars_nctrl[0]);
        }

        // Minimal necessary control points
        if (geom_nctrl == -1) geom_nctrl = geom_degree + 1;
        for (int i = 0; i < vars_nctrl.size(); i++)
        {
            if (vars_nctrl[i] == -1) vars_nctrl[i] = vars_degree + 1;
        }

        ModelInfo geom_info(dom_dim, geom_dim, geom_degree, geom_nctrl);
        mfa_info.addGeomInfo(geom_info);

        for (int k = 0; k < nvars; k++)
        {
            ModelInfo var_info(dom_dim, model_dims[k+1], vars_degree, vars_nctrl);
            mfa_info.addVarInfo(var_info);
        }
    }

    // Currently for single block examples only
    void setup_args(vector<int> model_dims, MFAInfo& mfa_info, DomainArgs& d_args)
    {
        // If only one value for vars_nctrl was parsed, assume it applies to all dims
        if (vars_nctrl.size() == 1 & dom_dim > 1)
        {
            vars_nctrl = vector<int>(dom_dim, vars_nctrl[0]);
        }

        // Set basic info for DomainArgs
        d_args.updateModelDims(model_dims);
        d_args.multiblock   = false;
        d_args.r            = rot * M_PI / 180;
        d_args.t            = twist;
        d_args.n            = noise;
        d_args.infile       = infile;
        d_args.infile2      = infile2;
        d_args.structured   = structured;
        d_args.rand_seed    = rand_seed;

        // Specify size, location of domain 
        d_args.ndom_pts     = vector<int>(dom_dim, ndomp);
        d_args.full_dom_pts = vector<int>(dom_dim, ndomp);
        d_args.starts       = vector<int>(dom_dim, 0);
        d_args.tot_ndom_pts = 1;
        for (int i = 0; i < dom_dim; i++)
        {
            d_args.tot_ndom_pts *= ndomp;
        }

        // Set default extents of physical domain
        d_args.min.assign(dom_dim, 0.0);
        d_args.max.assign(dom_dim, 1.0);

        if (input == "roms" || input == "mpas")
        {
            if (dom_dim != 2)
            {
                cerr << "dom_dim must be 2 to run mpas/roms example" << endl;
                exit(1);
            }
            model_dims = {3, 1};
            d_args.updateModelDims(model_dims);
        }

        set_mfa_info(model_dims, mfa_info);
        mfa_info.verbose          = verbose;
        mfa_info.weighted         = weighted;
        mfa_info.regularization   = regularization;
        mfa_info.reg1and2         = reg1and2;
    } // setup_args()
};

#endif  // _MFA_EX_PARSER_HPP