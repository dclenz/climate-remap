//--------------------------------------------------------------
// MPAS -> ROMS -> MPAS round-trip remapping and error analysis
//--------------------------------------------------------------

#include <mfa/mfa.hpp>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef MFA_TBB
#include <tbb/global_control.h>
#endif

#include <diy/assigner.hpp>
#include <diy/decomposition.hpp>
#include <diy/io/block.hpp>
#include <diy/master.hpp>

#include "opts.h"
#include "cblock2.hpp"

using namespace std;
using RemapBlock = CBlock2<real_t>;

namespace
{
struct RoundTripOptions
{
    int domDim = 3;
    int forwardDegree = 2;
    int reverseDegree = 2;
    vector<int> forwardControlPoints = {11};
    vector<int> reverseControlPoints = {17};
    string romsFile;
    string mpasFile;
    vector<string> variableNames = {
        "timeDaily_avg_activeTracers_temperature_3d",
        "timeDaily_avg_activeTracers_salinity_3d"
    };
    real_t forwardRegularization = 1;
    real_t reverseRegularization = 1;
    int forwardRegularizeFirstAndSecond = 0;
    int reverseRegularizeFirstAndSecond = 0;
    int verbose = 1;
    bool help = false;
    int geometryDegree = 1;
    int geometryControlPoints = -1;
};

void registerOptions(opts::Options& options, RoundTripOptions& config)
{
    options >> opts::Option('m', "dom_dim", config.domDim, " dimension of domain (round trips require 3)");
    options >> opts::Option('q', "vars_degree", config.forwardDegree, " forward science-variable degree in each dimension");
    options >> opts::Option('v', "vars_nctrl", config.forwardControlPoints, " forward control points; repeat once per dimension");
    options >> opts::Option('e', "reverse_vars_degree", config.reverseDegree, " reverse science-variable degree in each dimension");
    options >> opts::Option('w', "reverse_vars_nctrl", config.reverseControlPoints, " reverse control points; repeat once per dimension");
    options >> opts::Option('b', "regularization", config.forwardRegularization, " forward smoothing parameter");
    options >> opts::Option('a', "reverse_regularization", config.reverseRegularization, " reverse smoothing parameter");
    options >> opts::Option('k', "reg1and2", config.forwardRegularizeFirstAndSecond, " forward regularize first and second derivatives when 1");
    options >> opts::Option('j', "reverse_reg1and2", config.reverseRegularizeFirstAndSecond, " reverse regularize first and second derivatives when 1");
    options >> opts::Option('r', "varNames", config.variableNames, " MPAS variable name; repeat for each field");
    options >> opts::Option('o', "romsfile", config.romsFile, " ROMS H5M file path");
    options >> opts::Option('s', "mpasfile", config.mpasFile, " MPAS H5M file path");
    options >> opts::Option('g', "geom_degree", config.geometryDegree, " geometry degree in each dimension");
    options >> opts::Option('c', "geom_nctrl", config.geometryControlPoints, " geometry control points in each dimension");
    options >> opts::Option('x', "verbose", config.verbose, " MFA verbosity (0-2)");
    options >> opts::Option('h', "help", config.help, " show help");
}

bool validateOptions(const RoundTripOptions& config)
{
    if (config.domDim != 3)
    {
        fmt::print(stderr, "ERROR: remap_loop requires --dom_dim 3.\n");
        return false;
    }
    if (config.mpasFile.empty() || config.romsFile.empty())
    {
        fmt::print(stderr, "ERROR: --mpasfile and --romsfile are required.\n");
        return false;
    }
    if (config.variableNames.empty())
    {
        fmt::print(stderr, "ERROR: At least one --varNames value is required.\n");
        return false;
    }
    for (const string& variableName : config.variableNames)
    {
        if (variableName.empty())
        {
            fmt::print(stderr, "ERROR: Variable names cannot be empty.\n");
            return false;
        }
    }
    const auto validControlPoints = [&](const vector<int>& values)
    {
        return values.size() == 1 || values.size() == static_cast<size_t>(config.domDim);
    };
    if (!validControlPoints(config.forwardControlPoints) ||
        !validControlPoints(config.reverseControlPoints))
    {
        fmt::print(stderr, "ERROR: Forward and reverse control counts require one value or {} values.\n",
                   config.domDim);
        return false;
    }
    return true;
}

mfa::MFAInfo makeMFAInfo(const RoundTripOptions& config, bool reverse)
{
    const vector<int> modelDimensions = {3, static_cast<int>(config.variableNames.size())};
    mfa::MFAInfo info(config.domDim, config.verbose, modelDimensions,
                      config.geometryDegree, config.geometryControlPoints,
                      reverse ? config.reverseDegree : config.forwardDegree,
                      reverse ? config.reverseControlPoints : config.forwardControlPoints);
    info.weighted = 0;
    info.regularization = reverse ? config.reverseRegularization : config.forwardRegularization;
    info.reg1and2 = reverse ? config.reverseRegularizeFirstAndSecond :
                             config.forwardRegularizeFirstAndSecond;
    return info;
}

#ifdef MFA_TBB
unique_ptr<tbb::global_control> configureThreading()
{
    const char* value = std::getenv("MFA_NUM_THREADS");
    if (!value)
        return nullptr;

    try
    {
        size_t parsedCharacters = 0;
        const int threadCount = std::stoi(value, &parsedCharacters);
        if (threadCount <= 0 || parsedCharacters != string(value).size())
            throw std::invalid_argument("invalid thread count");
        return make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, threadCount);
    }
    catch (const std::exception&)
    {
        fmt::print(stderr, "ERROR: MFA_NUM_THREADS must be a positive integer.\n");
        return nullptr;
    }
}
#endif
} // namespace

int main(int argc, char** argv)
{
#ifdef MFA_TBB
    const char* requestedThreadCount = std::getenv("MFA_NUM_THREADS");
    auto threadControl = configureThreading();
    if (requestedThreadCount && !threadControl)
        return 1;
#endif

    diy::mpi::environment env(argc, argv);
    diy::mpi::communicator world;
    MPI_Comm localComm;
    MPI_Comm_dup(world, &localComm);

    RoundTripOptions config;
    opts::Options options;
    registerOptions(options, config);
    const bool parsed = options.parse(argc, argv);
    if (!parsed || config.help)
    {
        if (world.rank() == 0)
            cout << options;
        return config.help ? 0 : 1;
    }
    if (!validateOptions(config))
        return 1;

    const int pointDimension = 3 + config.variableNames.size();
    fmt::print(stderr, "--------- Round-Trip MFA Settings ----------\n");
    fmt::print(stderr, "Forward controls: {}\n", mfa::print_vec(config.forwardControlPoints));
    fmt::print(stderr, "Reverse controls: {}\n", mfa::print_vec(config.reverseControlPoints));
    fmt::print(stderr, "Variables: {}\n", mfa::print_vec(config.variableNames));
    fmt::print(stderr, "MPAS file: {}\nROMS file: {}\n", config.mpasFile, config.romsFile);

    diy::FileStorage storage("./DIY.XXXXXX");
    diy::Master master(world, 1, -1, &RemapBlock::create, &RemapBlock::destroy,
                       &storage, &RemapBlock::save, &RemapBlock::load);
    diy::ContiguousAssigner assigner(world.size(), 1);
    Bounds<real_t> domain(config.domDim);
    domain.min = vector<real_t>(config.domDim, 0);
    domain.max = vector<real_t>(config.domDim, 1);
    Decomposer<real_t> decomposer(config.domDim, domain, 1);
    decomposer.decompose(world.rank(), assigner,
        [&](int gid, const Bounds<real_t>& core, const Bounds<real_t>& bounds,
            const Bounds<real_t>& global, const RCLink<real_t>& link)
        {
            RemapBlock::add(gid, core, bounds, global, link, master,
                            config.domDim, pointDimension, 0.0);
        });

    mfa::MFAInfo forwardInfo = makeMFAInfo(config, false);
    mfa::MFAInfo reverseInfo = makeMFAInfo(config, true);


    double init_time = MPI_Wtime();
    master.foreach([&](RemapBlock* block, const diy::Master::ProxyWithLink& cp)
    {
        block->verbose = config.verbose;
        block->varNames = config.variableNames;
        block->configureForRoundTrip();
        block->initMOAB(localComm);
        block->readTargetData<double>(cp, config.romsFile);
        block->readSourceData<double>(cp, config.mpasFile);
        block->readTargetDataFull<double>();
        block->classifySourceCoverage();
    });
    init_time = MPI_Wtime() - init_time;

    double forward_time = MPI_Wtime();
    master.foreach([&](RemapBlock* block, const diy::Master::ProxyWithLink& cp)
    {
        block->remap(cp, forwardInfo);
        block->writeForwardRemapH5M("forward_remap.h5m");
        block->print_model(cp);
    });
    forward_time = MPI_Wtime() - forward_time;

    double forward_serialization_time = MPI_Wtime();
    diy::io::write_blocks("forward_remap.mfa", world, master);
    forward_serialization_time = MPI_Wtime() - forward_serialization_time;

    double reverse_time = MPI_Wtime();
    master.foreach([&](RemapBlock* block, const diy::Master::ProxyWithLink&)
    {
        block->reverseRemap(reverseInfo);
        block->writeRoundtripResults("roundtrip_mpas.h5m", "roundtrip_metrics.csv");
    });
    reverse_time = MPI_Wtime() - reverse_time;

    master.foreach([&](RemapBlock* block, const diy::Master::ProxyWithLink&)
    {
        block->activateReverseMFAForSerialization();
    });
    double reverse_serialization_time = MPI_Wtime();
    diy::io::write_blocks("reverse_remap.mfa", world, master);
    reverse_serialization_time = MPI_Wtime() - reverse_serialization_time;

    fmt::print(stderr, "Initialization time: {:.3f} s\n", init_time);
    fmt::print(stderr, "Forward remap and H5M time: {:.3f} s\n", forward_time);
    fmt::print(stderr, "Forward MFA serialization time: {:.3f} s\n", forward_serialization_time);
    fmt::print(stderr, "Reverse remap and output time: {:.3f} s\n", reverse_time);
    fmt::print(stderr, "Reverse MFA serialization time: {:.3f} s\n", reverse_serialization_time);
    return 0;
}
