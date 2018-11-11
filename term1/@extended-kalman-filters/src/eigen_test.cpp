#include <iostream>
#include "Eigen/Dense"
#include <vector>

using namespace std;
using namespace Eigen;

// Script to check how Eigen works

int main(int argc, char* argv[]) {
	VectorXd vec(2);
	vec << 10, 20;
	cout << vec << endl;

	MatrixXd mat(2, 2);
	mat << 1, 2, 3, 4;
	cout << mat << endl;
	cout << mat[0] << endl;
}
