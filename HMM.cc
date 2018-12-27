#include "HMM.h"

Orange::HMM::HMM(int _stateNum ,int _dimNum ,int _mixNum){
    stateNum = _stateNum;
    dimNum = _dimNum;
    mixNum = _mixNum;

    Allocate();
    for(int i = 0;i < stateNum;i ++){
        stateInit[i] = 1.0 / double(stateNum);

        for(int j = 0;j <= stateNum;j ++){
            stateTran[i][j] = 1.0 / double(stateNum + 1);
        }
    }
}

Orange::HMM::~HMM(){
    Dispose();
}

void Orange::HMM::Allocate(){
    stateModel = new GMM*[stateNum];
    stateInit = new double[stateNum];
    stateTran = new double*[stateNum];

    for(int i = 0 ;i < stateNum;i ++){
        stateModel[i] = new GMM(mixNum,dimNum);
        stateTran[i] = new double[stateNum];
    }
}

void Orange::HMM::Dispose(){
    for(int i = 0 ;i < stateNum;i ++){
        delete stateModel[i];
        delete[] stateTran[i];
    }
    delete[] stateModel;
    delete[] stateInit;
    delete[] stateTran;
}

double Orange::HMM::getInitState(int id){
    return stateInit[id];
}

double Orange::HMM::getFinalState(int id){
    return stateTran[id][stateNum];
}

double Orange::HMM::getTranProb(int i,int j){
    return stateTran[i][j];
}

GMM* Orange::HMM::getStateMode(int id){
    return stateModel[id];
}

double Orange::HMM::calcProbability(std::vector<double*>& seq){
    std::vector<int> state;

}

// viterbi decode
double decode(std::vector<double*>& vec, std::vector<int>& states){

}