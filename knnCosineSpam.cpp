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

#define ROWS 4601
#define COLS 57
#define K 7


using namespace std;

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
ifstream& readData1d(ifstream& in, int *data) {
	string d;
	for (int i = 0; i < ROWS; i++) {
		getline(in, d);
		data[i] = (int) atof(d.c_str());	
	}
	return in;
}


// read line by line and store data in a 2d array
ifstream& readData2d(ifstream& in, float **data, char delimeter = ' ') {
	string d;
	vector<string> dSplit;
	for (int i = 0; i < ROWS; i++) {
		getline(in, d);
		dSplit = split(d, delimeter);		
		for (int j = 0; j < COLS; j++) {
			data[i][j] = (float) atof(dSplit[j].c_str());
		}
	}
	return in;
}

// predicate for sorting vector of pair<index,distance> by distance
bool compare(const pair<int, float> x, const pair<int, float> y)
{
	return x.second < y.second;
}

// distance measure
float cosine(float *v1, float *v2) {
	float distance = 0;

	float xDotY = 0.0;
	float normX = 0.0;
	float normY = 0.0;
	for (int k = 0; k < COLS; k++){
		float x = v1[k];
		float y = v2[k];
		xDotY += (x * y);
		normX += (x * x);
		normY += (y * y);
		}
	float normXY = normX * normY;
	if (normXY == 0)
		return FLT_MAX;
	else
		return 1-(xDotY / (normX * normY));
}


// k nearest neighbor
int* knn(float** records, int* labels, float* instance, int maxOut = -1)
{
	int *nn = (int*) malloc(K*sizeof(int));
	vector<pair<int,float> > distances; // keep track of the index position and the distances itself

	for (int i = 0; i < ROWS; i++){
		if (i != maxOut)
			distances.push_back(make_pair(i,cosine(records[i], instance)));
		else
			distances.push_back(make_pair(i,FLT_MAX)); // max out this record 
	}
	// sort in descending order
	//sort(distances.begin(), distances.end(), compare);
	sort(distances.begin(), distances.end(), compare);

	for (int i = 0; i < K; i++)
		nn[i] = distances[i].first;	// return index position  


	return nn;
}

// classifier based on labels of nearest neighbors
int classify(int* nn, int* labels) {
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

// using the leave-one-out cross validation
float leaveOneOut(float** records, int* labels)
{
	int accurate = 0;
	int* nn = (int*) malloc(K*sizeof(int));
	float* instance = (float*) malloc(COLS*sizeof(float));
	

	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++)
			instance[j] = records[i][j];

		nn = knn(records, labels, instance, i);

		if (classify(nn, labels) == labels[i])
			accurate += 1;
	}
	free(nn);
	free(instance);

	return (accurate*1.0)/ROWS;
}


int main(){
	// connect to file with records
	ifstream rec("spam.txt");
	ifstream lab("labels.txt");

	
	// allocate memory for records and labels
	float** records = (float**)malloc(ROWS*sizeof(float *));
	int* labels = (int*) malloc(ROWS*sizeof(int));
	for (int i = 0; i < ROWS; i++)
		records[i] = (float*) malloc(COLS*sizeof(float)); 

	// read data from file
	readData1d(lab, labels);
	readData2d(rec, records, '\t');

	float accuracy = leaveOneOut(records, labels);

	cout << accuracy << endl;

	// free memory
	for (int i = 0; i < ROWS; i++) 
		free(records[i]);
	free(labels);

	rec.close();
	lab.close();

	return 0;
}














