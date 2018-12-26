#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <ctime>
#include <cmath>
namespace Orange{

class KMeans{
public:
    KMeans(int clusterNum ,int dimNum);
    ~KMeans();
    void Allocate();
    void Dispose();

    // getters
    int getDimNum() { return dimNum; }
    int getClusterNum() { return clusterNum; }
    double** getMeans() { return means; }
    double* getMeans(int label_id) { return means[label_id]; }
    int getIterNum() { return iterNum; }

    // setters
    void setIterNum(int _iterNum) { iterNum = _iterNum; }

    // core process
    void cluster(double** data, int N, int* label);
    void init_means(double**, int N);
    void calcCurrentLabel(double* ,int& label);
    double calcDistance(double *,int id);
private:
    int dimNum;
    int clusterNum;
    double** means;

    int iterNum;
};
}