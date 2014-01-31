#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <utility>
#include <cfloat>
#include <cmath>
#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>


#include <thrust/pair.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/fill.h>

#define ROWS 440
#define COLS 138672
#define K 5

using std::vector;
using std::string;
using std::ifstream;
using std::getline;
using std::cout;
using std::endl;


// input data to be read from file line by line needs to be split
vector<string> split(string s, char delimeter='\t') {
	vector<string> splitted;

	int num_elem = s.size();
	int i = 0;
	int j;
	while (i < num_elem) {
		j = s.find(delimeter, i);
		if (j == -1){
			splitted.push_back(s.substr(i));
			break;
		}
		splitted.push_back(s.substr(i,j-i));
		i = j+1;
	}
	return splitted;
}



// read line by line and store data in a 1d array
void readData1d(ifstream& in, thrust::host_vector<int>& data) {
	string d;
	int rows = (int)ROWS;
	for (int i = 0; i < rows; i++) {
		getline(in, d);
		data[i] = (int) atof(d.c_str());	
	}
	return ;
}


// read line by line and store data in a 2d array
void readData(ifstream& in, thrust::host_vector<float>& data, char delimeter = '\t') {
	string d;
	vector<string> dSplit;
	int rows = (int)ROWS;
	int cols = (int)COLS;

	for (int i = 0; i < rows; i++) {
		getline(in, d);
		dSplit = split(d, delimeter);
		
		for (int j = 0; j < cols; j++) {
			data[i*cols+j] = (float) atof(dSplit[j].c_str());
		}
	}
	return ;
}

// predicate for sorting vector of pair<index,distance> by distance
struct compare
{
	__host__ __device__
	bool operator()(const thrust::pair<int, float> x, const thrust::pair<int, float> y)
	{
		return x.second < y.second;
	}
};

bool compare2 (const std::pair<int, float> x, const std::pair<int, float> y)
{
	return x.second < y.second;
}

int classify(int* nn, thrust::host_vector<int>& labels) {
	int k = (int)K;
	int mid = (k - 1)/2;

	int ones = 0;
	for (int i = 0; i < k; i++)
		ones += labels[nn[i]];

	if (ones > mid) 
		return 1;
	else
		return 0;
	
}



// using cosine distance
__global__ void distances(float *d_records, float *d_distances, int* m)
{
	
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int rows = (int)ROWS;
	int cols = (int)COLS;

	if (i < rows && i != *m){
		float xDotY = 0.0;
		float normX = 0.0;
		float normY = 0.0;
		for (int k = 0; k < cols; k++){
			float x = d_records[i*cols+k];
			float y = d_records[(*m)*cols+k];

			xDotY += (x * y);
			normX += (x * x);
			normY += (y * y);
		}
		float normXY = normX * normY;
		if (normXY == 0)
			d_distances[i] = FLT_MAX;
		else
			d_distances[i] = 1-(xDotY / (normX * normY));
			
	}
	else if (i == *m)
		d_distances[i] = FLT_MAX;
}

int main()
{
	int rows = (int)ROWS;
	int cols = (int)COLS;
	int k = (int)K;

	// connect to file with records
	ifstream rec("PEMS_records01.txt");
	ifstream lab("PEMS_labels01.txt");


	// create corresponding host and device vectors
	thrust::host_vector<float> h_records(rows*cols);
	thrust::host_vector<int> h_labels(rows);
	thrust::host_vector<int> h_nn(k);
	thrust::host_vector<float> h_distances(rows);

	// keep track 
	std::vector<std::pair<int, float> > index_distance(rows);

	// keep track of accurate classification
	int accurate = 0;

	thrust::device_vector<float> d_records(rows*cols);
	thrust::device_vector<float> d_distances(rows);

	// read data from file
	readData1d(lab, h_labels);
	readData(rec, h_records);

	// copy records from host to device
	d_records = h_records;

	// pointers to pass to kernel function
	float *pd_records = thrust::raw_pointer_cast(&d_records[0]);
	float *pd_distances = thrust::raw_pointer_cast(&d_distances[0]);

	// estimate number of blocks given 512 thread per block
	int nThreads = 512;
	int nBlocks = rows/nThreads + 1;

	// variable to specify to kernel what to leave out
	int* xx;
	cudaMalloc((void**) &xx, sizeof(int));

	// variable to hold nearest neighbors
	int* nn = (int*) malloc(k*sizeof(int));

	for (int i = 0; i < rows; i++) {
		cudaMemcpy(xx, &i, sizeof(int), cudaMemcpyHostToDevice);

		distances<<<nBlocks,nThreads>>>(pd_records, pd_distances, xx);

		//synchronize
		cudaDeviceSynchronize();

		// copy distances from device to host
		h_distances = d_distances;
		
		// couple distance & index to enable sorting without loss of index info
		for (int m = 0; m < rows; m++)
			index_distance[m] = std::make_pair(m, sqrt(h_distances[m]));

		// sort
		std::sort(index_distance.begin(), index_distance.end(), compare2);

		// retrieve nearest neighbors
		for (int m = 0; m < k; m++)
			nn[m] = index_distance[m].first;

		if (classify(nn, h_labels) == h_labels[i])
			accurate++;
	}


	float accuracy = (accurate*1.0)/((float)ROWS);

	cout << "The accuracy obtained is: " << accuracy << endl;

	lab.close();
	rec.close();
	return 0;
}
