#include <string>
#include <iostream>
#include <fstream>
using namespace std;

#include "methods.h"

int main(int argc, char *argv[]) {
	// arguments
	if (argc < 3) {
		printf("incorrect args. ");
		printf("./main <data> <sample> <iter> <theta1> <theta2> ...");
		return 0;
	}
	string dataset = argv[1];
	cout << dataset << endl;

	int nsam = stoi(argv[2]);
	int niters = stoi(argv[3]);

	vector<double> thetas;
	for (int i = 4; i < argc; i++) {
		thetas.push_back(stod(argv[i]));
		cout << thetas.back() << "\t";
	}
	cout << endl;
	// arguments


	ofstream out("log/" + dataset + ".out");
	ofstream log("log/" + dataset + ".log");
	cout.rdbuf(log.rdbuf());

	random_device rd;						//non-deterministic random engine
	engine eng(rd());						//deterministic engine with a random seed

	table tbl("data/" + dataset, true);

	for (double theta : thetas) {
		cout << "theta: " << theta << endl;
		out << "theta: " << theta << endl;
		for (double epsilon : {10}) {
			for (int iter = 0; iter < niters; iter++) {
				cout << "epsilon: " << epsilon << " iter:" << iter + 1 << endl;
				bayesian bayesian(eng, tbl, epsilon, theta);
				bayesian.sampling(nsam);
				bayesian.syn.printo_file("output/syn_" + dataset + "_eps" + to_string(int(epsilon)) +
					"_theta" + to_string(int(theta)) + "_iter" + to_string(iter) + ".dat");
			}
		}
		cout << endl;
		out << endl;
	}
	out.close();
	log.close();
	return 0;
}
