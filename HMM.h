#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cmath>
#include "GMM.h"
namespace Orange{
    

class HMM
{
public:
    // constructor
    HMM(int _stateNum ,int _dimNum ,int _mixNum);
    ~HMM();
    void Allocate();
    void Dispose();

    // setters
    void setStateNum(int _stateNum) { stateNum = _stateNum; }
    void setDimNum(int _dimNum) { dimNum = _dimNum; }
    void setMixNum(int _mixNum) { mixNum = _mixNum; }
    void setIterNum(int _iterNum) { iterNum = _iterNum; }

    // getters
    int getStateNum() { return stateNum; }
    int getDimNum() { return dimNum; }
    int getMixNum() { return mixNum; }
    GMM** getStateModel() { return stateModel; }
    double* getStateInit() { return stateInit; }
    double** getStateTran() { return stateTran; }
    int getIterNum() { return iterNum; }

    double getInitState(int id);
    double getFinalState(int id);
    double getTranProb(int i,int j);
    GMM* getStateMode(int id);

    // core process
    void zero();
    void norm();
    double logProb(double p);
    double calcProbability(std::vector<double*>& seq);
    double calcTransProb(int i,int j);
    double decode(std::vector<double*>& vec, std::vector<int>& states);
    void init(); 
    void train();

private:
    int stateNum;
    int dimNum;
    int mixNum;
    Orange::GMM** stateModel;
    double* stateInit;
    double** stateTran;
    int iterNum;
};

}