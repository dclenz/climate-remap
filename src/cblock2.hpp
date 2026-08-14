//--------------------------------------------------------------
// A custom block for climate remapping
//--------------------------------------------------------------
#ifndef MFA_REMAP_CBLOCK2_HPP
#define MFA_REMAP_CBLOCK2_HPP

#include <memory>
#include <string>
#include <vector>

#include <diy/assigner.hpp>
#include <diy/decomposition.hpp>
#include <diy/io/block.hpp>
#include <diy/master.hpp>
#include <mfa/block_base.hpp>
#include <mfa/mfa.hpp>

#include "remap_domain.hpp"
#include "remap_models.hpp"
#include "remap_output.hpp"

template <typename T>
struct CBlock2 : public BlockBase<T>
{
    using Base = BlockBase<T>;
    using Base::approx;
    using Base::dom_dim;
    using Base::input;
    using Base::mfa;

    static constexpr int geomDim = RemapDomain<T>::GeometryDimension;

    int verbose = 0;
    std::vector<std::string> varNames;
    bool dumpMatrices = false;
    bool writeLegacyRemapTags = true;
    bool useDeterministicTargetPlane = false;
    bool combineScienceVariables = false;
    bool reverseSerializationActivated = false;

    std::unique_ptr<RemapMeshes<T>> remapMeshes;
    std::unique_ptr<RemapDomain<T>> remapDomain;
    RemapModels<T> remapModels;

    CBlock2() = default;
    virtual ~CBlock2() = default;

    static void* create() { return mfa::create<CBlock2>(); }
    static void destroy(void* block) { mfa::destroy<CBlock2>(block); }

    static void add(int gid, const Bounds<T>& core, const Bounds<T>& bounds,
                    const Bounds<T>& domain, const RCLink<T>& link,
                    diy::Master& master, int domainDimension, int pointDimension,
                    T ghostFactor = 0.0)
    {
        mfa::add<CBlock2, T>(gid, core, bounds, domain, link, master,
                             domainDimension, pointDimension, ghostFactor);
    }

    static void save(const void* block, diy::BinaryBuffer& buffer)
    {
        mfa::save<CBlock2, T>(block, buffer);
    }

    static void load(void* block, diy::BinaryBuffer& buffer)
    {
        mfa::load<CBlock2, T>(block, buffer);
    }

    void configureForRoundTrip()
    {
        writeLegacyRemapTags = false;
        useDeterministicTargetPlane = true;
        combineScienceVariables = true;
    }

    void initMOAB(MPI_Comm communicator, int dimension)
    {
        (void)dimension;
        remapMeshes = std::make_unique<RemapMeshes<T>>(communicator);
        typename RemapDomain<T>::Configuration configuration;
        configuration.verbose = verbose;
        configuration.variableNames = varNames;
        configuration.deterministicTargetPlane = useDeterministicTargetPlane;
        configuration.combineScienceVariables = combineScienceVariables;
        configuration.domainDimension = dom_dim;
        remapDomain = std::make_unique<RemapDomain<T>>(*remapMeshes, std::move(configuration));
    }

    template <typename V>
    void readTargetData(const diy::Master::ProxyWithLink& cp, std::string filename)
    {
        remapDomain->template readTargetData<V>(cp, filename);
    }

    template <typename V>
    void readSourceData(const diy::Master::ProxyWithLink& cp, std::string filename)
    {
        remapDomain->template readSourceData<V>(cp, filename);
    }

    template <typename V>
    void readTargetDataFull()
    {
        remapDomain->template readTargetDataFull<V>();
    }

    void classifySourceCoverage()
    {
        remapDomain->classifySourceCoverage();
    }

    void setParameterizations()
    {
        remapDomain->setParameterizations();
    }

    void computeRemap(mfa::MFAInfo& info)
    {
        auto forwardApproximation = remapModels.applyForward(
            *mfa, remapDomain->sourceInput(), remapDomain->targetInput(), info,
            dom_dim, geomDim, dumpMatrices, remapDomain->sourceLevelMeanHeights(),
            remapDomain->minHeight(), remapDomain->maxHeight());

        using Database = typename RemapMeshes<T>::Database;
        moab::Tag locationTag = remapMeshes->targetTag("ROMS3DLocations");
        const int locationTagLength = remapMeshes->tagLength(
            Database::Target, locationTag, "ROMS3DLocations");

        const size_t expectedTargetPoints =
            remapMeshes->targetElements().size() * locationTagLength;
        if (static_cast<size_t>(forwardApproximation->npts) != expectedTargetPoints)
        {
            fmt::print(stderr,
                       "ERROR: Decoded point count {} does not match {} target elements x {} levels.\n",
                       forwardApproximation->npts, remapMeshes->targetElements().size(),
                       locationTagLength);
            std::exit(1);
        }

        if (writeLegacyRemapTags)
            output(forwardApproximation.get()).writeLegacyTargetTags(
                *forwardApproximation, locationTagLength);
        fmt::print(stderr, "ROMS3DLocations tag length: {}\n", locationTagLength);
        fmt::print(stderr, "forward approximation npts: {}\n", forwardApproximation->npts);
        fmt::print(stderr, "Expected ROMS elements: {}\n",
                   forwardApproximation->npts / locationTagLength);
        fmt::print(stderr, "Actual ROMS elements: {}\n",
                   remapMeshes->targetElements().size());

        input = remapDomain->releaseSource().release();
        remapDomain->releaseTarget();
        approx = forwardApproximation.release();
    }

  private:
    RemapOutput<T> output(mfa::PointSet<T>* forwardApproximation = nullptr)
    {
        return RemapOutput<T>(
            *remapMeshes, remapDomain->selectedSourceElements(),
            remapDomain->sourceElementLevels(), remapDomain->sourceCellIDs(),
            remapDomain->sourceCoverageFractions(), remapDomain->sourceCoverageClasses(),
            remapDomain->sourcePointEligible(), remapDomain->variableNames(), geomDim,
            remapDomain->maximumLevel(), input,
            forwardApproximation ? forwardApproximation : approx,
            remapModels.roundtripApproximation());
    }

  public:
    void writeForwardRemapH5M(const std::string& filename)
    {
        output().writeForwardRemapH5M(filename);
    }

    void writeRoundtripResults(const std::string& h5mFilename,
                               const std::string& metricsFilename)
    {
        output().writeRoundtripResults(h5mFilename, metricsFilename);
    }

    void writeVTK()
    {
        output().writeVTK();
    }

    void remap(const diy::Master::ProxyWithLink& cp, mfa::MFAInfo& info)
    {
        setParameterizations();
        this->setup_MFA(cp, info);
        computeRemap(info);
    }

    void reverseRemap(mfa::MFAInfo info)
    {
        if (!input || !approx)
        {
            fmt::print(stderr, "ERROR: Forward remap must complete before reverse remapping.\n");
            std::exit(1);
        }

        using Database = typename RemapMeshes<T>::Database;
        moab::Tag locationTag = remapMeshes->targetTag("ROMS3DLocations");
        const int locationTagLength = remapMeshes->tagLength(
            Database::Target, locationTag, "ROMS3DLocations");

        remapModels.applyReverse(
            info, *approx, *input, remapDomain->targetHeights(),
            remapMeshes->targetElements().size(), locationTagLength,
            remapDomain->minHeight(), remapDomain->maxHeight(), dom_dim, geomDim,
            remapDomain->variableNames());
    }

    // Terminal state transition: after this call the BlockBase fields describe
    // the reverse model and should only be serialized or destroyed.
    void activateReverseMFAForSerialization()
    {
        if (reverseSerializationActivated)
        {
            fmt::print(stderr, "ERROR: Reverse serialization is already active.\n");
            std::exit(1);
        }
        if (!remapModels.reverseMFA())
        {
            fmt::print(stderr, "ERROR: Reverse MFA is not available for serialization.\n");
            std::exit(1);
        }
        std::unique_ptr<mfa::MFA<T>> forwardMFA(mfa);
        mfa = remapModels.exchangeReverseMFA(std::move(forwardMFA)).release();

        std::unique_ptr<mfa::PointSet<T>> sourceInput(input);
        input = approx;
        approx = remapModels.exchangeRoundtripApproximation(std::move(sourceInput)).release();
        reverseSerializationActivated = true;
    }

    void print_model(const diy::Master::ProxyWithLink& cp)
    {
        RemapModels<T>::printModel(mfa, cp.gid());
    }
};

#endif // MFA_REMAP_CBLOCK2_HPP
