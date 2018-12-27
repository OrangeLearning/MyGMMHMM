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

void Orange::HMM::zero(){
    for(int i = 0;i < stateNum;i ++){
        stateInit[i] = 0.0;
        for(int j = 0; j < stateNum + 1;j ++){
            stateTran[i][j] = 0.0;
        }
    }
}

void norm(){
    double cnt = 0.0;
    for(int i = 0 ;i < stateNum; i ++){
        cnt += stateInit[i];
    }

    for(int i = 0 ;i < stateNum; i ++){
        stateInit[i] /= cnt;
    }

    for(int i = 0; i < stateNum; i ++){
        cnt = 0.0;
        for(int j = 0;j < stateNum;j ++){
            cnt += stateTran[i][j];
        }
        if(cnt > 0.0){
            for(int j = 0; j < stateNum + 1;j ++){
                stateTran[i][j] /= cnt;
            }
        }
    }
}

double Orange::HMM::logProb(double p){
    return (p > 1e-20) ? log10(p) : -20;
}

double Orange::HMM::calcProbability(std::vector<double*>& seq){
    std::vector<int> state;
    return decode(seq, state);
}

double Orange::HMM::calcTransProb(int i,int j){
    return logProb(stateTran[i][j]);
}

// viterbi decode
double decode(std::vector<double*>& vec, std::vector<int>& states){
    int size = int(seq.size());
    double* lastLogP = new double[stateNum];
    double* currLogP = new double[stateNum];
    int** path = new int*[size];

    path[0] = new int[stateNum];

    for(int i = 0 ;i < stateNum;i ++){
        currLogP[i] = logProb(stateInit[i]) + logProb(stateModel[i] -> calcProbability(seq[0]));
        path[0][i] = -1;
    }

    for(int t = 1;t < size;t ++){
        path[t] = new int[stateNum];

        std::swap(lastLogP , currLogP);

        for(int i = 0;i < stateNum;i ++){
            currLogP[i] = -1e308;
            for(int j = 0;j < stateNum;j ++){
                double templ = lastLogP[j] + logProb(stateTran[j][i]);
                if(templ > currLogP[i]){
                    currLogP[i] = templ;
                    path[j][i] = j;
                }
            }
            currLogP[i] += logProb(stateModel[i] -> calcProbability(seq[t]));
        }
    }

    int finalState = 0;
    double finalProb = -1e308;
    for(int i = 0;i < stateNum; i ++){
        fi(currLogP[i] > finalProb){
            finalProb = currLogP[i];
            finalState = i;
        }
    }

    state.push_back(finalState);
    for(int t = size - 2;t >= 0;t --){
        int stateIndex = path[t + 1][state.back()];
        state.push_back(stateIndex);
    }

    reverse(state.begin() , state.end());

    for (int i = 0; i < size; i++){
        delete[] path[i];
    }
    delete[] path;
    delete[] lastLogP;
    delete[] currLogP;

    finalProb = exp(finalProb / double(size));
    return finalProb;
}

void Orange::HMM::init(int size){

    for(int i = 0 ;i < stateNum ; i++){
        if(i == 0){
            stateInit[i] = 0.5;
        }
        else {
            stateInit[i] = 0.5 / double(stateNum - 1);
        }

        for(int j = 0;j < stateNum; j ++){
            if((i == j) || (j == i + 1)){
                stateTran[i][j] = 0.5;
            }
        }
    }

    std::vector<double*> *gaussseq = new std::vector<double*>[stateNum];

    for(int i = 0; i < size ;i ++){
        int seq_size = 0; // TO-DO
        
        double r = double(seq_size) / double(stateNum);

        for(int j = 0;j < seq_size;j ++){
            gaussseq[int(j / r)].push_back();
        }
    }

    int* stateDataSize = new int[stateNum];

    for(int i = 0;i < stateNum; i ++){
        stateDataSize[i] = gaussseq[i].size();

        double* x = new double[dimNum];
        for(int j = 0; j < stateDataSize[i] ; j ++){
            x = (double*) gaussseq[i].at(j);
        }

        delete x;
        stateModel[i] -> train();
        gaussseq[i].clear();
    }

    delete[] stateDataSize;
    delete[] gaussseq;
}

void Orange::HMM::train(double** data, int N){
    int size = N;

    int* stateInitNum = new int[stateNum];
    int** stateTranNum = new int[stateNum];
    int* stateDataNum = new int[stateNum];

    for(int i = 0 ;i < stateNum; i ++){
        stateTranNum[i] = new int[stateNum + 1];
    }

    int iter = 0;
    std::vector<int> state;
    std::vector<double*> seq;

    double currL = 0.0 , lastL = 0.0;

    while(iter < iterNum){
        lastL = currL;
        currL = 0.0;

        for(int i = 0;i < stateNum;i ++){
            stateDataNum[i] = 0;
            memset(stateTranNum[i], 0, sizeof(int) * (stateNum + 1));
        }
        memset(stateInitNum , 0, sizeof(int) * stateNum);

        for(int i = 0; i < size; i ++){
            int seq_size = 0;
            // TO-DO here
            for(int j = 0; j < seq_size ;j ++){
                seq.push_back(data[i]);
            }

            currL += logProb(decode(seq, state));
            stateInitNum[state[0]] ++;
            
            for(int j = 0;j < seq_size;j ++){

            }

            state.clear();
            seq.clear();
        }
        currL /= double(size);

        int count = 0;
        for(int j = 0;j < stateNum;j ++){
            if(stateDataSize[j] > stateModel[j] -> getMixNum() * 2){
                stateModel[j] -> train();
            }
            count += stateInitNum[j];
        }

        for(int j = 0; j < stateNum;j ++){
            stateInit[j] = double(stateInitNum[j]) / double(count);
        }

        for(int i = 0;i < stateNum;i ++){
            count = 0;
            for(int j = 0;j < stateNum + 1;j ++){
                count += stateTranNum[i][j];
            }
            if(count > 0){
                for(int j = 0;j < stateNum + 1;j ++){
                    stateTran[i][j] = double(stateTranNum[i][j]) / double(count);
                }
            }
        }

        iter ++;
    }

    delete[] stateInitNum;
    delete[] stateDataNum;
    for(int i = 0;i < stateNum;i ++){
        delete[] stateTranNum[i];
    }
    delete[] stateTranNum;
}