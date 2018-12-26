#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <KMeans.h>

using namespace Orange;

namespace Orange{

class GMM{
public:
    const double gmm_pi = acos(-1.0);
public:
    // constructor
    GMM(int,int);
    ~GMM();
    void Allocate();
    void Dispose();

    // setters
    void setIterNum(int _iterNum){ iterNum = _iterNum; }

    // getters
    int getIterNum() { return iterNum; }


    // core part
    double calcProb(double*);
    double calcSingleProb(double*,int);
    void train(double**,int);
    void init_data(double**,int);

private:
    int mixNum;
    int dimNum;
    
    double* prior;
    double** means;
    double** vars;

    double* minVars;

    int iterNum;

};

}