#include <GMM.h>

using namespace Orange;
Orange::GMM::GMM(int _mixNum , int _dimNum){
    mixNum = _mixNum;
    dimNum = _dimNum;

    Allocate();

    // In initial time , all of gaussian pdf is standard
    // And all the weight is the same as the rest
    for(int i = 0 ;i < mixNum ;i ++){
        prior[i] = 1.0 / double(mixNum);

        for(int j = 0 ;j < dimNum ;j ++){
            means[i][j] = 0.0;
            vars[i][j] = 1.0;
        }
    }
}

Orange::GMM::~GMM(){
    Dispose();
}

void Orange::GMM::Allocate(){
    minVars = new double[dimNum];
    prior = new double[mixNum];
    means = new double*[mixNum];
    vars = new double*[mixNum];
    
    for(int i = 0; i < mixNum; i++){
        vars[i] = new double[dimNum];
        means[i] = new double[dimNum];
    }
}

void Orange::GMM::Dispose(){
    for(int i = 0;i< mixNum ;i ++){
        delete[] vars[i];
        delete[] means[i];
    }
    delete[] prior;
    delete[] vars;
    delete[] means;
    delete[] minVars;
}

double Orange::GMM::calcSingleProb(double* x, int id){
    double p = 1;
	for (int d = 0; d < dimNum; d++){
		p *= 1 / sqrt(2 * gmm_pi * vars[id][d]);
		p *= exp(-0.5 * (x[d] - means[id][d]) * (x[d] - means[id][d]) / vars[id][d]);
	}
	return p;
}

double Orange::GMM::calcProb(double* x){
    double res = 0.0;
    for (int i = 0;i < mixNum;i ++){
        res += calcSingleProb(x , i) * prior[i];
    }
    return res;
}

// 初始化数据
// 使用kmeans 的方法得到一个基础的数据
void Orange::GMM::init_data(double** data, int N){
    KMeans* kmeans = new KMeans(mixNum , dimNum);
    // label[id] = value 的结果表示：
    //  data[id] -> label
    int* label = new int[N];
    int size = N;
    kmeans -> cluster(data , N , label);
    
    int* count = new int[mixNum];
    double* overmeans = new double[dimNum];

    
    for(int i = 0;i < mixNum; i ++){
        prior[i] = double(count[i] = 0);
        memcpy(means[i] , kmeans -> getMeans(i),sizeof(double) * dimNum);
        memset(vars[i], 0 , sizeof(double) * dimNum);
    }
    memset(overmeans , 0, sizeof(double) * dimNum);
    memset(minVars, 0 , sizeof(double) * dimNum);

    for(int i = 0 ;i < size ;i++) {
        int curLabel = label[i];

        count[curLabel] ++;
        double* curMeans = kmeans -> getMeans(curLabel);

        for(int d = 0;d < dimNum;d ++){
            vars[curLabel][d] += (data[i][d] - curMeans[curLabel][d]) * (data[i][d] - curMeans[curLabel][d]);
        }

        for(int d = 0;d < dimNum;d ++){
            overmeans[d] += data[curLabel][d];
            minVars[d] += data[curLabel][d] * data[curLabel][d];
        }
    }

    for(int d = 0;d < dimNum;d ++){
        overmeans[d] /= double(size);
        minVars[d] = max(MIN_VAR, 0.01 * (minVars[d]/ size - overmeans[d] * overmeans[d]));
    }

    for(int i = 0;i < mixNum;i ++){
        prior[i] = double(count[i]) / size;

        if(prior[i] > 0.0){
            for(int d = 0;d < dimNum;d ++){

            }
        }
        else{
            
        }
    }
    delete kmeans;
    delete[] count;
    delete[] overmeans;
    delete[] label;
}

void Orange::GMM::train(double** data ,int N){
    init_data(data, N);
    int size = N;
    // 需要训练的数据
    double* next_prior = new double[mixNum];
    double** next_means = new double*[mixNum];
    double** next_vars = new double*[mixNum];
    for(int i = 0; i < mixNum;i ++){
        next_means[i] = new double[dimNum];
        next_vars[i] = new double[dimNum];
    }

    int iter = 0;
    while(iter < iterNum){
        memset(next_prior ,0 ,sizeof(double) * mixNum);
        for(int i = 0;i < mixNum;i ++){
            memset(next_means[i] ,0 ,sizeof(double) * dimNum);
            memset(next_vars[i] ,0 ,sizeof(double) * dimNum);
        }

        /**
         * E 步骤
         * p = gmm(x | params)
         * pj = gmm(x | params , j) / p
         * np += sum{size}(sum{mixNum}(pj))
         */
        for(int i = 0; i < size;i ++){
            double p = calcProb(data[i]);

            for(int j = 0; j < mixNum;j ++){
                double pj = calcSingleProb(data[i], j) * prior[j] / p;

                next_prior[j] += pj;
                for(int k = 0; k < dimNum; k ++){
                    next_means[j][k] += pj * data[i][k];
                    next_vars[j][k]  += pj * data[i][k] * data[i][k]; 
                }
            }
        }

        /**
         * 更新数据
         */
        for(int i = 0; i < mixNum; i ++){
            prior[i] = next_prior[i] / double(size);
            if(prior[i] > 0) for(int d = 0; d < dimNum; d ++){
                means[i][d] = next_means[i][d] / next_prior[i];
                vars[i][d]  = next_vars[i][d]  / next_prior[i] - means[i][d] * means[i][d];
                // minVar 的问题
                if(vars[i][d] < minVars[d]) {
                    vars[i][d] = minVars[d];
                }
            }
        }
        iter ++;
    }

    delete[] next_prior;
    for(int i = 0;i < mixNum; i ++){
        delete[] next_means[i];
        delete[] next_vars[i];
    }
    delete[] next_means;
    delete[] next_vars;
}
