//--------------------------------------------------------------
// MPAS -> ROMS -> MPAS round-trip remapping and error analysis
//--------------------------------------------------------------

#include <mfa/mfa.hpp>

#include <iostream>
#include <string>
#include <vector>

#include <diy/assigner.hpp>
#include <diy/decomposition.hpp>
#include <diy/io/block.hpp>
#include <diy/master.hpp>

#include "opts.h"
#include "cblock2.hpp"

using namespace std;
using B = CBlock2<real_t>;

int main(int argc, char** argv)
{
    diy::mpi::environment env(argc, argv);
    diy::mpi::communicator world;
    MPI_Comm local;
    MPI_Comm_dup(world, &local);

    int dom_dim = 3;
    int vars_degree = 2;
    int reverse_vars_degree = 2;
    vector<int> vars_nctrl = {11};
    vector<int> reverse_vars_nctrl = {17};
    string romsfile;
    string mpasfile;
    vector<string> varNames = {
        "timeDaily_avg_activeTracers_temperature_3d",
        "timeDaily_avg_activeTracers_salinity_3d"
    };
    real_t regularization = 0;
    real_t reverse_regularization = 0;
    int reg1and2 = 0;
    int reverse_reg1and2 = 0;
    int verbose = 1;
    bool help = false;
    int geom_degree = 1;
    int geom_nctrl = -1;

    opts::Options ops;
    ops >> opts::Option('m', "dom_dim", dom_dim, " dimension of domain (round trips require 3)");
    ops >> opts::Option('q', "vars_degree", vars_degree, " forward science-variable degree in each dimension");
    ops >> opts::Option('v', "vars_nctrl", vars_nctrl, " forward control points; repeat once per dimension");
    ops >> opts::Option('e', "reverse_vars_degree", reverse_vars_degree, " reverse science-variable degree in each dimension");
    ops >> opts::Option('w', "reverse_vars_nctrl", reverse_vars_nctrl, " reverse control points; repeat once per dimension");
    ops >> opts::Option('b', "regularization", regularization, " forward smoothing parameter");
    ops >> opts::Option('a', "reverse_regularization", reverse_regularization, " reverse smoothing parameter");
    ops >> opts::Option('k', "reg1and2", reg1and2, " forward regularize first and second derivatives when 1");
    ops >> opts::Option('j', "reverse_reg1and2", reverse_reg1and2, " reverse regularize first and second derivatives when 1");
    ops >> opts::Option('r', "varNames", varNames, " MPAS variable name; repeat for each field");
    ops >> opts::Option('o', "romsfile", romsfile, " ROMS H5M file path");
    ops >> opts::Option('s', "mpasfile", mpasfile, " MPAS H5M file path");
    ops >> opts::Option('g', "geom_degree", geom_degree, " geometry degree in each dimension");
    ops >> opts::Option('c', "geom_nctrl", geom_nctrl, " geometry control points in each dimension");
    ops >> opts::Option('x', "verbose", verbose, " MFA verbosity (0-2)");
    ops >> opts::Option('h', "help", help, " show help");
    if (!ops.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            cout << ops;
        return help ? 0 : 1;
    }

    if (dom_dim != 3)
    {
        fmt::print(stderr, "ERROR: remap_loop requires --dom_dim 3.\n");
        return 1;
    }
    if (mpasfile.empty() || romsfile.empty())
    {
        fmt::print(stderr, "ERROR: --mpasfile and --romsfile are required.\n");
        return 1;
    }
    if (varNames.empty())
    {
        fmt::print(stderr, "ERROR: At least one --varNames value is required.\n");
        return 1;
    }
    for (const string& varName : varNames)
    {
        if (varName.empty())
        {
            fmt::print(stderr, "ERROR: Variable names cannot be empty.\n");
            return 1;
        }
    }
    if ((vars_nctrl.size() != 1 && vars_nctrl.size() != dom_dim) ||
        (reverse_vars_nctrl.size() != 1 && reverse_vars_nctrl.size() != dom_dim))
    {
        fmt::print(stderr, "ERROR: Forward and reverse control counts require one value or {} values.\n",
                   dom_dim);
        return 1;
    }

    const int geom_dim = 3;
    const int pt_dim = geom_dim + varNames.size();
    vector<int> model_dims(varNames.size() + 1, 1);
    model_dims[0] = geom_dim;

    fmt::print(stderr, "--------- Round-Trip MFA Settings ----------\n");
    fmt::print(stderr, "Forward controls: {}\n", mfa::print_vec(vars_nctrl));
    fmt::print(stderr, "Reverse controls: {}\n", mfa::print_vec(reverse_vars_nctrl));
    fmt::print(stderr, "Variables: {}\n", mfa::print_vec(varNames));
    fmt::print(stderr, "MPAS file: {}\nROMS file: {}\n", mpasfile, romsfile);

    diy::FileStorage storage("./DIY.XXXXXX");
    diy::Master master(world, 1, -1, &B::create, &B::destroy, &storage, &B::save, &B::load);
    diy::ContiguousAssigner assigner(world.size(), 1);
    Bounds<real_t> domain(dom_dim);
    domain.min = vector<real_t>(dom_dim, 0);
    domain.max = vector<real_t>(dom_dim, 1);
    Decomposer<real_t> decomposer(dom_dim, domain, 1);
    decomposer.decompose(world.rank(), assigner,
        [&](int gid, const Bounds<real_t>& core, const Bounds<real_t>& bounds,
            const Bounds<real_t>& global, const RCLink<real_t>& link)
        {
            B::add(gid, core, bounds, global, link, master, dom_dim, pt_dim, 0.0);
        });

    mfa::MFAInfo forward_info(dom_dim, verbose, model_dims, geom_degree, geom_nctrl,
                              vars_degree, vars_nctrl);
    forward_info.weighted = 0;
    forward_info.regularization = regularization;
    forward_info.reg1and2 = reg1and2;

    mfa::MFAInfo reverse_info(dom_dim, verbose, model_dims, geom_degree, geom_nctrl,
                              reverse_vars_degree, reverse_vars_nctrl);
    reverse_info.weighted = 0;
    reverse_info.regularization = reverse_regularization;
    reverse_info.reg1and2 = reverse_reg1and2;

    double init_time = MPI_Wtime();
    master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
    {
        b->verbose = verbose;
        b->varNames = varNames;
        b->writeLegacyRemapTags = false;
        b->useDeterministicTargetPlane = true;
        b->initMOAB(local, dom_dim);
        b->readTargetData<double>(cp, romsfile);
        b->readSourceData<double>(cp, mpasfile);
        b->readTargetDataFull<double>();
        b->classifySourceCoverage();
    });
    init_time = MPI_Wtime() - init_time;

    double forward_time = MPI_Wtime();
    master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
    {
        b->remap(cp, forward_info);
        b->writeForwardRemapH5M("forward_remap.h5m");
        b->print_model(cp);
    });
    forward_time = MPI_Wtime() - forward_time;

    diy::io::write_blocks("forward_remap.mfa", world, master);

    double reverse_time = MPI_Wtime();
    master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
    {
        (void)cp;
        b->reverseRemap(reverse_info);
        b->writeRoundtripResults("roundtrip_mpas.h5m", "roundtrip_metrics.csv");
    });
    reverse_time = MPI_Wtime() - reverse_time;

    master.foreach([&](B* b, const diy::Master::ProxyWithLink& cp)
    {
        (void)cp;
        b->activateReverseMFAForSerialization();
    });
    diy::io::write_blocks("reverse_remap.mfa", world, master);

    fmt::print(stderr, "Initialization time: {:.3f} s\n", init_time);
    fmt::print(stderr, "Forward remap time: {:.3f} s\n", forward_time);
    fmt::print(stderr, "Reverse remap time: {:.3f} s\n", reverse_time);
    return 0;
}
