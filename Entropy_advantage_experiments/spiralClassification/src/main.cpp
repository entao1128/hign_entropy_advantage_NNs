#include <iostream>
#include <vector>
#include <random>
#include <gsl/gsl_cblas.h>
#include <cmath>
#include <sstream>
#include "Plots.h"
#include "ProgressDisplay.h"
#include "ProgressDisplay.h"
#include <omp.h>

const double pi = 3.1415926535897932384626433832795028841971693993751058209;

double parameterLb = -1.0;
double parameterUb = 1.0;
class fcNetwork
{
public:
    int inputChannel;
    int maxBatchSize;
    std::vector<int> channels;                     // number of channels for intermediate and output layers
    std::vector<double> parameters;                // weights and biases
    std::vector<double *> weights, biases;         // pointers to sections of the vector above
    std::vector<std::vector<double>> vActivations; // activations in each layer

    fcNetwork(const fcNetwork &src)
    {
        inputChannel = src.inputChannel;
        maxBatchSize = src.maxBatchSize;
        channels = src.channels;
        parameters = src.parameters;
        vActivations = src.vActivations;
        for (double *srcW : src.weights)
            weights.push_back(srcW - src.parameters.data() + parameters.data());
        for (double *srcB : src.biases)
            biases.push_back(srcB - src.parameters.data() + parameters.data());
    }
    fcNetwork(fcNetwork &&src) = default;
    fcNetwork(int inputChannel, std::vector<int> channels, int maxBatchSize) : inputChannel(inputChannel), maxBatchSize(maxBatchSize), channels(channels)
    {
        // calculate number of parameters
        int numParam = 0;
        int prevChannels = inputChannel;
        for (auto c : channels)
        {
            int numWeights = prevChannels * c;
            int numBiases = c;
            numParam += (numWeights + numBiases);
            prevChannels = c;
        }

        // allocate
        parameters.resize(numParam);
        for (auto c : channels)
        {
            int numActivations = c * maxBatchSize;
            vActivations.push_back(std::vector<double>(numActivations, 0.0));
        }
        // initialize
        std::mt19937 engine;
        std::uniform_real_distribution<double> dist(parameterLb, parameterUb);
        for (auto &p : parameters)
            p = dist(engine);

        // assign pointers
        numParam = 0;
        prevChannels = inputChannel;
        for (auto c : channels)
        {
            int numWeights = prevChannels * c;
            int numBiases = c;

            weights.push_back(parameters.data() + numParam);
            numParam += numWeights;
            biases.push_back(parameters.data() + numParam);
            numParam += numBiases;
            prevChannels = c;
        }
    }
    // evaluation results are in activations
    void evaluate(std::vector<double> &data)
    {
        int numData = data.size() / inputChannel;
        if (data.size() % inputChannel != 0 || numData > maxBatchSize)
        {
            std::cerr << "Error in fcNetwork::evaluate : incorrect input data size, data size=" << data.size() << ", exiting!\n";
            exit(1);
        }
        double *temp = data.data();
        int prevChannels = inputChannel;
        for (int i = 0; i < channels.size(); i++)
        {
            // matrix multiplication
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, channels[i], numData, prevChannels, 1.0, weights[i], prevChannels, temp, numData, 0.0, vActivations[i].data(), numData);
            // add bias
            for (int j = 0; j < channels[i]; j++)
                for (int k = 0; k < numData; k++)
                    vActivations[i][j * numData + k] += biases[i][j];
            // activation function
            if (i != channels.size() - 1)
            {
                for (auto &a : vActivations[i])
                    a = std::max(a, 0.0);
            }
            prevChannels = channels[i];
            temp = vActivations[i].data();
        }
        // last layer has softmax activation
        for (int k = 0; k < numData; k++)
        {
            double sum = 0.0;
            for (int j = 0; j < prevChannels; j++)
            {
                temp[j * numData + k] = std::exp(temp[j * numData + k]);
                sum += temp[j * numData + k];
            }
            for (int j = 0; j < prevChannels; j++)
                temp[j * numData + k] /= sum;
        }
    }
    double loss(std::vector<double> &oneHotLabels)
    {
        int numData = oneHotLabels.size() / channels.back();
        if (oneHotLabels.size() % channels.back() != 0 || numData > maxBatchSize)
        {
            std::cerr << "Error in fcNetwork::loss : incorrect input data size, data size=" << oneHotLabels.size() << ", exiting!\n";
            exit(1);
        }
        double sum = 0.0;
        for (int i = 0; i < numData * channels.back(); i++)
            sum -= oneHotLabels[i] * std::log(vActivations.back()[i]);
        return sum / numData;
    }
    double loss(std::vector<double> &data, std::vector<double> &oneHotLabels)
    {
        evaluate(data);
        return loss(oneHotLabels);
    }
    double accuracy(std::vector<double> &oneHotLabels)
    {
        int numData = oneHotLabels.size() / channels.back();
        if (oneHotLabels.size() % channels.back() != 0 || numData > maxBatchSize)
        {
            std::cerr << "Error in fcNetwork::accuracy : incorrect input data size, data size=" << oneHotLabels.size() << ", exiting!\n";
            exit(1);
        }
        double sum = 0.0;
        for (int k = 0; k < numData; k++)
        {
            double max = 0.0;
            int nClass = channels.back();
            int prediction = nClass;
            for (int j = 0; j < nClass; j++)
            {
                if (vActivations.back()[j * numData + k] > max)
                {
                    max = vActivations.back()[j * numData + k];
                    prediction = j;
                }
            }
            sum += oneHotLabels[prediction * numData + k];
        }
        return sum / numData;
    }
    double accuracy(std::vector<double> &data, std::vector<double> &oneHotLabels)
    {
        evaluate(data);
        return accuracy(oneHotLabels);
    }
};

class spiralDataSet
{
public:
    std::vector<double> x, y;
    int numClasses, numDataPerClass;
    spiralDataSet(int numClasses, int numDataPerClass, double rmin, double rmax, double dThetaDR, double noise, int seed, bool uniformR = false) : numClasses(numClasses), numDataPerClass(numDataPerClass)
    {
        int numData = numClasses * numDataPerClass;
        x = std::vector<double>(2 * numData, 0.0);
        y = std::vector<double>(numClasses * numData, 0.0);

        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> rDist(rmin, rmax);
        std::normal_distribution<double> noiseDist(0.0, noise);
        for (int i = 0; i < numClasses; i++)
        {
            for (int j = 0; j < numDataPerClass; j++)
            {
                double r;
                if (uniformR)
                    r = double(j) / numDataPerClass * (rmax - rmin) + rmin;
                else
                    r = rDist(gen);
                double theta = dThetaDR * r + (2 * pi * i) / numClasses;
                int index = i * numDataPerClass + j;

                x[index] = r * std::cos(theta) + noiseDist(gen);
                x[numData + index] = r * std::sin(theta) + noiseDist(gen);
                y[i * numData + index] = 1.0;
            }
        }
    }
    void plot(std::string outputPrefix)
    {
        std::vector<std::vector<GeometryVector>> data;
        data.resize(numClasses);
        int numData = numClasses * numDataPerClass;
        for (int i = 0; i < numData; i++)
        {
            GeometryVector xx(x[i], x[numData + i]);
            for (int j = 0; j < numClasses; j++)
                if (y[j * numData + i] == 1.0)
                    data[j].push_back(xx);
        }
        gracePlot plot;
        for (int j = 0; j < numClasses; j++)
            plot.addDataSet(data[j], "", 0, 0, -1, j + 1);

        plot.autoScaleAndTick();
        plot.outputFigure(outputPrefix);
    }
    int outputNumChannels(void)
    {
        return 2;
    }
};

const size_t AcceptanceSampleSize = 1000;
const double MaxAcceptance = 0.7;
const double MinAcceptance = 0.3;

class parameterMove
{
private:
    size_t TrialCount, AcceptCount;

public:
    size_t moved;
    double newParam;
    // the most recently calculated acceptance ratio, updated by the class
    double AcceptRatio;
    double sigma;
    double MyMinAcceptance;
    std::uniform_int_distribution<int> indexDist;
    std::uniform_real_distribution<double> moveDist;

    parameterMove(fcNetwork &net) : indexDist(0, net.parameters.size() - 1), moveDist(-1.0, 1.0)
    {
        this->sigma = 0.1;
        this->TrialCount = 0;
        this->AcceptCount = 0;
        this->MyMinAcceptance = ::MinAcceptance;
        this->AcceptRatio = 0.0;
    }

    void generateMove(fcNetwork &net, std::mt19937 &gen, bool LockStepSize) // Generate a random move, move stored in class members (moved, newcoord)
    {
        if (this->TrialCount >= ::AcceptanceSampleSize)
        {
            // TODO : THIS MAY BREAK DETAILED BALANCE. FIND A WAY TO FIX IT.
            AcceptRatio = static_cast<double>(this->AcceptCount) / this->TrialCount; // acceptance rate
            if (LockStepSize == false)
            {
                if (AcceptRatio > ::MaxAcceptance && this->sigma < 0.5)
                {
                    this->sigma *= 1.1;
                    // std::cout << "In AtomMove, step size changed to" << this->sigma << "\n";
                }
                else if (AcceptRatio < this->MyMinAcceptance)
                {
                    this->sigma *= 0.9;
                    // std::cout << "In AtomMove, step size changed to" << this->sigma << "\n";
                }
                // else
                // std::cout<<"In AtomMove, keep step size"<<this->sigma<<"\n";
            }
            // debug temp
            // std::cout<<"Sigma is:"<<this->sigma<<", Accept Count is:"<<this->AcceptCount<<'\n';
            this->TrialCount = 0;
            this->AcceptCount = 0;
        }
        this->TrialCount++;

        this->moved = indexDist(gen);
        double oldcoord = net.parameters[this->moved];
        this->newParam = oldcoord + sigma * moveDist(gen);
    }

    virtual void Accept(fcNetwork &net)
    {
        this->AcceptCount++;
        net.parameters[this->moved] = newParam;
    }
};

class binDecider
{
public:
    int numBins, numBinsPerSet;
    double min, step, max;
    spiralDataSet train, test;
    double maxTrainAccuracy;
    binDecider(double min, double step, double max, spiralDataSet train, spiralDataSet test)
        : min(min), step(step), max(max), train(train), test(test)
    {
        numBinsPerSet = std::floor((max - min) / step) + 1;
        numBins = numBinsPerSet * numBinsPerSet;
        maxTrainAccuracy = 0.0;
    }
    long getBin(fcNetwork &net)
    {
        double trainaccuracy = net.accuracy(train.x, train.y);
        if (trainaccuracy >= max || trainaccuracy < min)
            return numBins;
        int trainBin = (trainaccuracy - min) / step;

        double testaccuracy = net.accuracy(test.x, test.y);
        if (testaccuracy >= max || testaccuracy < min)
            return numBins;
        int testBin = (testaccuracy - min) / step;

        int result = testBin * numBinsPerSet + trainBin;
        return result;
    }
    std::vector<double> getSideThresholds(void)
    {
        std::vector<double> result;
        for (double x = min;; x += step)
            if (x < max)
                result.push_back(x);
            else
            {
                result.push_back(max);
                break;
            }
        return result;
    }
};

// class for Wang-Landau Monte Carlo
// class binDecider should have the following member functions:
// size_t NumBins()
// long GetBin(double E)
// double GetBinLowerBound(size_t NumBin)
// double GetBinUpperBound(size_t NumBin)
// The user is responsible for adjusting SIncrease
class WangLandauMonteCarlo
{
private:
    std::mt19937 gen;
    std::uniform_real_distribution<double> pDistribution;

public:
    binDecider bd;
    size_t NumBins;
    std::vector<double> s;
    std::vector<size_t> Histogram;
    size_t MoveCount;

    bool LockStepSize;
    int currentBin;
    fcNetwork sys;
    double SIncrease;
    std::vector<std::vector<double>> vParameters;
    WangLandauMonteCarlo(fcNetwork &sys, int RandomSeed, binDecider &d, bool RecordConfigurationsPerBin = false)
        : sys(sys), gen(RandomSeed), s(d.numBins, 0.0), Histogram(d.numBins, 0), currentBin(d.getBin(sys)), pDistribution(0.0, 1.0), bd(d)
    {
        this->MoveCount = 0;
        this->LockStepSize = false;
        this->NumBins = d.numBins;

        if (currentBin >= d.numBins || currentBin < 0)
        {
            std::cerr << "Error in WangLandauMonteCarlo : initial bin not within bounds! Bin=" << currentBin << "\n";
            assert(false);
        }
        this->SIncrease = 1.0;

        if (RecordConfigurationsPerBin)
            vParameters.resize(NumBins);
    }
    void ClearHistogram(void)
    {
        for (size_t i = 0; i < NumBins; i++)
            Histogram[i] = 0;
    }
    std::vector<GeometryVector> GetHistogram(void)
    {
        std::vector<GeometryVector> result;
        for (int i = 0; i < NumBins; i++)
        {
            GeometryVector t(2);
            t.x[0] = i;
            t.x[1] = this->Histogram[i];
            result.push_back(t);
        }
        return result;
    }
    std::vector<GeometryVector> GetEntropy(void)
    {
        std::vector<GeometryVector> result;
        for (int i = 0; i < NumBins; i++)
        {
            GeometryVector t(2);
            t.x[0] = i;
            t.x[1] = this->s[i];
            result.push_back(t);
        }
        return result;
    }
    void Move(size_t Repeat, parameterMove &move)
    {
        for (size_t i = 0; i < Repeat; i++)
        {
            MoveCount++;
            move.generateMove(sys, this->gen, LockStepSize);

            if (move.newParam > parameterLb && move.newParam < parameterUb)
            {

                double oldcoord = sys.parameters[move.moved];
                sys.parameters[move.moved] = move.newParam;
                long AfterBin = bd.getBin(sys);
                sys.parameters[move.moved] = oldcoord;

                if ((currentBin < 0) || (currentBin >= NumBins))
                {
                    std::cerr << "Error in WangLandauMonteCarlo : currentBin out of range!\n";
                    std::cerr << "currentBin=" << currentBin << ", MoveCount=" << MoveCount << '\n';
                    assert(false);
                }
                if (AfterBin >= 0 && AfterBin < NumBins && pDistribution(gen) < std::exp(s[currentBin] - s[AfterBin]))
                {
                    move.Accept(this->sys);

                    if (s[AfterBin] == 0 && vParameters.size() != 0)
                        vParameters[AfterBin] = this->sys.parameters;

                    currentBin = AfterBin;
                }
            }
            s[currentBin] += SIncrease;
            Histogram[currentBin]++;
        }
    }
    bool HistogramFlat(double tol) // return true if Histogram(E) for every possible E is not less than (1-tol)*(average)
    {
        double sum1 = 0.0, sumH = 0.0, minH = 1e300;
        for (size_t i = 0; i < NumBins; i++)
        {
            size_t temp = Histogram[i];
            if (temp > 0)
            {
                sum1 += 1.0;
                sumH += temp;
                minH = std::min(minH, (double)(temp));
            }
        }
        if (sum1 == 0.0)
            return false;
        else
            return minH >= (1.0 - tol) * sumH / sum1;
    }
};

// merge multiple WLMC objects, averaging their entropy and summing their histogram
void mergeWLMC(std::vector<WangLandauMonteCarlo> &instances)
{
    int numInstances = instances.size();
    if (numInstances == 0)
        return;
    auto nBin = instances[0].s.size();
    for (int i = 1; i < instances.size(); i++)
        if (nBin != instances[i].s.size())
        {
            std::cerr << "Error in mergeWLMC : instances do not have the same number of bins.\n";
            return;
        }
    for (int i = 0; i < nBin; i++)
    {
        double sumS = 0.0;
        for (auto &w : instances)
            sumS += w.s[i];
        for (auto &w : instances)
            w.s[i] = sumS / numInstances;

        size_t sumH = 0;
        for (auto &w : instances)
            sumH += w.Histogram[i];
        for (auto &w : instances)
            w.Histogram[i] = sumH;
    }
}

int main()
{
    int nDataPerClassTrain = 30, nClass = 2;
    int nDataPerClassTest = 30;
    double rMin = 1.0, rMax = 5.0, dThetaDR = 2.0, noise = 0.1, minTrainingAccuracyToStudy;
    int cycle = 1000, step = 100000, nThreads;
    int trainSeed, testSeed, MCSeed;
    std::vector<int> channels;
    std::cin >> nDataPerClassTrain >> nDataPerClassTest >> nClass;
    std::cin >> rMin >> rMax >> dThetaDR >> noise;
    std::cin >> nThreads >> cycle >> step >> minTrainingAccuracyToStudy;
    for (;;)
    {
        int temp = 0;
        std::cin >> temp;
        if (temp > 0)
            channels.push_back(temp);
        else
            break;
    }
    std::cin >> trainSeed >> testSeed >> MCSeed;
    std::mt19937 gen;
    gen.seed(MCSeed);

    parameterUb=2.0*std::sqrt(1.0/channels[0]);
    parameterLb=-1.0*parameterUb;
    std::cout<<"parameter LB="<<parameterLb<<", UB="<<parameterUb<<std::endl;

    // region with (training accuracy)<minTrainingAccuracyToStudy is forbidden by increasing entropy by this amount
    const double forbiddenRegionEntropyIncrease = 1e6;

    spiralDataSet train(nClass, nDataPerClassTrain, rMin, rMax, dThetaDR, noise, trainSeed, false);
    spiralDataSet test(nClass, nDataPerClassTest, rMin, rMax, dThetaDR, noise, testSeed, true);
    train.plot("train");
    test.plot("test");
    double accuracyResolution = 1.0 / std::max(nDataPerClassTrain, nDataPerClassTest) / nClass;
    progress_display pd(cycle);

    std::string OutputPrefix = "";

    std::vector<fcNetwork> networks;
    std::vector<WangLandauMonteCarlo> instances;
    std::vector<parameterMove> moves;
    for (int i = 0; i < nThreads; i++)
    {

        fcNetwork temp(train.outputNumChannels(), channels, std::max(nDataPerClassTrain, nDataPerClassTest) * nClass);
        moves.push_back(parameterMove(temp));
        binDecider temp2(-0.5 * accuracyResolution, accuracyResolution, 1.0 + 0.5 * accuracyResolution - 1e-10, train, test);
        instances.push_back(WangLandauMonteCarlo(temp, gen(), temp2, false));
    }

    int currentCycle = 0;
    std::vector<std::vector<GeometryVector>> temp1, temp2;
    for (;; currentCycle++)
    {
        std::stringstream s3;
        s3 << OutputPrefix << "Conf" << currentCycle << "_Entropy";
        ReadGraceData(temp2, s3.str());
        if (temp2.size() > 0 && temp2[0].size() > 0)
            temp1 = temp2;
        else
            break;
    }
    if (currentCycle > 0)
    {
        std::cout << "Read entropy from cycle " << currentCycle << ".\n";
        pd += currentCycle;
        for (int i = 0; i < nThreads; i++)
        {
            std::vector<double> &s = instances[i].s;
            std::vector<GeometryVector> &s2 = temp1[0];
            if (s.size() != s2.size())
            {
                std::cerr << "Error reading entropy. The length is incorrect. Exiting.\n";
                exit(1);
            }
            for (int j = 0; j < s.size(); j++)
                s[j] = s2[j].x[1];
        }
    }
    else
    {
        // increase the entropy for the forbidden region
        binDecider &tempBd = instances[0].bd;
        for (int i = 0; i < tempBd.numBinsPerSet; i++)
        {
            double trainAccuracy = tempBd.min + tempBd.step * i;
            if (trainAccuracy < minTrainingAccuracyToStudy)
                for (int j = 0; j < tempBd.numBinsPerSet; j++)
                {
                    for (int k = 0; k < nThreads; k++)
                        instances[k].s[j * tempBd.numBinsPerSet + i] += forbiddenRegionEntropyIncrease * (1 + minTrainingAccuracyToStudy - trainAccuracy);
                }
        }
    }

#pragma omp parallel num_threads(nThreads)
    {
        int threadID = omp_get_thread_num();
        parameterMove &ma = moves[threadID];
        for (size_t i = currentCycle; i < cycle; i++)
        {
            if (ma.AcceptRatio > ma.MyMinAcceptance - 0.1)
                instances[threadID].SIncrease = 5.0 / (i + 10);
            else
                instances[threadID].SIncrease = 0.001; // increase s very cautiously when the system might not be equilibrated
            // At the beginning, the move classes might not have ideal sigma, leading to sharp peaks in histograms.

            instances[threadID].ClearHistogram();
            instances[threadID].Move(step, ma);

#pragma omp barrier
#pragma omp single
            {
                mergeWLMC(instances);
                if (OutputPrefix != "No_Output")
                {
                    std::stringstream s1, s2, s3;
                    s1 << OutputPrefix << "Conf" << i;
                    s1 << "_Histogram";
                    std::vector<GeometryVector> h = instances[threadID].GetHistogram();
                    PlotFunction_Grace(h, s1.str(), "H", "p(H)", "");
                    s3 << OutputPrefix << "Conf" << i << "_Entropy";
                    h = instances[threadID].GetEntropy();
                    PlotFunction_Grace(h, s3.str(), "H", "S(H)", "");
                }
                pd++;
            }
#pragma omp barrier
        }
    }

    {
        // restore the entropy for the forbidden region by decreasing it
        binDecider &tempBd = instances[0].bd;
        for (int i = 0; i < tempBd.numBinsPerSet; i++)
        {
            double trainAccuracy = tempBd.min + tempBd.step * i;
            if (trainAccuracy < minTrainingAccuracyToStudy)
                for (int j = 0; j < tempBd.numBinsPerSet; j++)
                {
                    for (int k = 0; k < nThreads; k++)
                        instances[k].s[j * tempBd.numBinsPerSet + i] -= forbiddenRegionEntropyIncrease * (1 + minTrainingAccuracyToStudy - trainAccuracy);
                }
        }
    }

    // output result for python plotting
    auto entropy = instances[0].GetEntropy();
    std::vector<double> thresholds = instances[0].bd.getSideThresholds();
    std::fstream xfile("x.txt", std::fstream::out);
    std::fstream yfile("y.txt", std::fstream::out);
    std::fstream sfile("s.txt", std::fstream::out);
    for (auto a : thresholds)
    {
        for (auto b : thresholds)
        {
            xfile << b << ' ';
            yfile << a << ' ';
        }
        xfile << std::endl;
        yfile << std::endl;
    }
    for (int i = 0; i < instances[0].bd.numBinsPerSet; i++)
    {
        for (int j = 0; j < instances[0].bd.numBinsPerSet; j++)
            sfile << entropy[i * instances[0].bd.numBinsPerSet + j].x[1] << ' ';
        sfile << std::endl;
    }

    return 0;
}
