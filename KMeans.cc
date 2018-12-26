#include <KMeans.h>

Orange::KMeans::KMeans(int _clusterNum,int _dimNum){
    clusterNum = _clusterNum;
    dimNum = _dimNum;

    Allocate();
}

Orange::KMeans::~KMeans(){
    Dispose();
}

void Orange::KMeans::Allocate(){
    means = new double*[clusterNum];
    for(int i = 0;i < clusterNum;i ++){
        means[i] = new double[dimNum];
        for(int j = 0 ;j < dimNum;j ++) {
            means[i][j] = 0.0;
        }
    }
}

void Orange::KMeans::Dispose(){
    for(int i = 0 ;i < clusterNum ; i++){
        delete[] means[i];
    }
    delete[] means;
}

void Orange::KMeans::init_means(double** data,int N){
    int size = N;
    int inteval = size / clusterNum;
    double* sample = new double[dimNum];
    srand((unsigned)time(0));
    for(int i = 0; i < clusterNum; i ++){
        int select = inteval * i + (inteval - 1) * rand() / RAND_MAX;
        for(int j = 0; j < dimNum; j++){
            sample[j] = data[select*dimNum][j];
        }
        memcpy(means[i], sample, sizeof(double) * dimNum);
    }
}

double Orange::KMeans::calcDistance(double *x ,int id){
    double res = 0.0;
    for(int i = 0;i< dimNum;i ++){
        res += (x[i] - means[id][i]) * (x[i] - means[id][i]);
    }
    return sqrt(res);
}

void Orange::KMeans::calcCurrentLabel(double* x,int& label){
    label = -1;
    double minDistance = 0.0;
    for(int i = 0 ;i < clusterNum; i++){
        int dist = calcDistance(x , i);
        if(label == -1 || minDistance < dist){
            label = i;
            minDistance = dist;
        }
    }
}

void Orange::KMeans::cluster(double** data,int N,int* label){
    int size = N;
    double** next_means = new double*[clusterNum];
    for(int i = 0;i < clusterNum; i ++){
        next_means[i] = new double[dimNum];
    }

    int* label_count = new int[clusterNum];
    
    label = new int[clusterNum];

    init_means(data , N);

    int iter = 0;

    while(iter < iterNum){
        memset(label_count ,0 ,sizeof(int) * clusterNum);
        
        for(int i = 0;i < size ;i ++){
            int tmp_label = -1;
            calcCurrentLabel(data[i], tmp_label);
            label_count[tmp_label] ++;
            for(int j = 0; j< dimNum;j ++){
                next_means[tmp_label][j] += data[i][j];
            }
        }

        for(int i = 0;i < clusterNum;i ++){
            for(int j = 0;j < dimNum;j ++){
                next_means[i][j] /= double(label_count[i]);
            }
            memcpy(means[i] , next_means[i], sizeof(double) * dimNum);
        }

        iter ++ ;
    }

    for(int i = 0;i < size;i ++){
        int tmp_label = -1;
        calcCurrentLabel(data[i], tmp_label);
        label[i] = tmp_label;
    }

    delete[] label_count;
    for (int i = 0; i < clusterNum; i++){
		delete[] next_means[i];
	}
	delete[] next_means;
}
