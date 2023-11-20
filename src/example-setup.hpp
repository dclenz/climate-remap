//--------------------------------------------------------------
// Helper functions to set up pre-defined examples
//
// David Lenz
// Argonne National Laboratory
// dlenz@anl.gov
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------

#ifndef MFA_EX_SETUP_HPP
#define MFA_EX_SETUP_HPP

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <set>

// #include "block.hpp"
#include "domain_args.hpp"

using namespace std;
    void echo_mfa_settings(string run_name, int dom_dim, int pt_dim, int scalar,
                            int geom_degree, int geom_nctrl, int vars_degree, vector<int>& vars_nctrl,
                            real_t regularization, int reg1and2,
                            int adaptive, real_t e_threshold, int rounds,
                            ostream& os = std::cerr)
    {
        os << ">>> Running \'" << run_name << "\'" << endl;
        os << endl;
        os << "--------- MFA Settings ----------" << endl;
        os << "pt_dim   = " << pt_dim      << '\t' << "dom_dim    = " << dom_dim 
                << '\t' << "scalar: " << boolalpha << (bool)scalar << endl;
        os << "geom_deg = " << geom_degree << '\t' << "vars_deg = " << vars_degree << endl;
        os << "encoding type = " << (adaptive ? "adaptive" : "fixed") << endl;
        if (adaptive)
        {
        os << "error    = " << e_threshold << '\t' << "max rounds = " << (rounds == 0 ? "unlimited" : to_string(rounds)) << endl;
        }
        else
        {
        os << "geom_nctrl = " << geom_nctrl << '\t' << "vars_nctrl = " << mfa::print_vec(vars_nctrl) << endl;
        }
        os << "regularization = " << regularization << ", type: " << 
            (regularization == 0 ? "N/A" : (reg1and2 > 0 ? "1st and 2nd derivs" : "2nd derivs only")) << endl;

#ifdef MFA_NO_WEIGHTS
        os << "weighted: false" << endl;
#else
        os << "weighted: " << boolalpha <<  (bool)weighted << endl;
#endif
#ifdef CURVE_PARAMS
        os << "parameterization method: curve" << endl;
        os << "ERROR: curve parametrization not currently supported in examples. Exiting." << endl;
        exit(1);
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

    // Print basic info about data set
    void echo_data_settings(string input, string infile, int ndomp, int ntest, ostream& os = std::cerr)
    {
        bool is_analytical = false; // always false for this example

        os << "--------- Data Settings ----------" << endl;
        os << "input: "       << input       << ", " << "infile: " << (is_analytical ? "N/A" : infile) << endl;
        os << "num pts    = " << ndomp       << '\t' << "test pts    = " << (ntest > 0 ? to_string(ntest) : "N/A") << endl;

        return;
    }

    // Print basic info about data set
    void echo_data_settings(string input, string infile, int ndomp, vector<int> ntest, ostream& os = std::cerr)
    {
        bool is_analytical = false; // always false for this example

        os << "--------- Data Settings ----------" << endl;
        os << "input: "       << input       << ", " << "infile: " << (is_analytical ? "N/A" : infile) << endl;
        os << "num pts    = " << ndomp       << '\t' << "test pts    = " << mfa::print_vec(ntest) << endl;

        return;
    }  

    // Print all info about data set
    void echo_data_mod_settings(int structured, int rand_seed, real_t rot, real_t twist, real_t noise, ostream& os = std::cerr)
    {
        os << "structured = " << boolalpha << (bool)structured << '\t' << "random seed = " << rand_seed << endl;
        os << "rotation   = " << setw(7) << left << rot << '\t' << "twist       = " << twist << endl;
        os << "noise      = " << setw(7) << left << noise << endl;

        return;
    }

    // Sets the degree and number of control points 
    // For geom and each var, degree is same in each domain dimension
    // For geom, # ctrl points is the same in each domain dimension
    // For each var, # ctrl points varies per domain dimension, but is the same for all of the vars
    // void set_mfa_info(vector<int> model_dims, MFAInfo& mfa_info)
    void set_mfa_info(int dom_dim, vector<int> model_dims, 
                        int geom_degree, int geom_nctrl,
                        int vars_degree, vector<int> vars_nctrl,
                        MFAInfo& mfa_info)
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

    // Currently for single block examples only
    // void setup_args(vector<int> model_dims, MFAInfo& mfa_info, DomainArgs& d_args)
    void setup_args( int dom_dim, int pt_dim, vector<int> model_dims,
                        int geom_degree, int geom_nctrl, int vars_degree, vector<int> vars_nctrl,
                        string input, string infile, int ndomp,
                        int structured, int rand_seed, real_t rot, real_t twist, real_t noise,
                        int reg1and2, real_t regularization, bool adaptive, int verbose,
                        MFAInfo& mfa_info, DomainArgs& d_args)
    {
        int weighted = 0;

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
        d_args.infile2      = "";
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

        if (input == "roms" || input == "mpas" || input == "remap")
        {
            if (dom_dim == 2)
            {
                model_dims = {3, 1, 1, 1};
            }
            else if (dom_dim == 3)
            {
                model_dims = {3, 1, 1};
            }
            else
            {
                cerr << "ERROR: Incorrect dom_dim" << endl;
                exit(1);
            }

            d_args.updateModelDims(model_dims);
        }

        set_mfa_info(dom_dim, model_dims, geom_degree, geom_nctrl, vars_degree, vars_nctrl, mfa_info);
        mfa_info.verbose          = verbose;
        mfa_info.weighted         = weighted;
        mfa_info.regularization   = regularization;
        mfa_info.reg1and2         = reg1and2;
    } // setup_args()

#endif // MFA_EX_SETUP_HPP