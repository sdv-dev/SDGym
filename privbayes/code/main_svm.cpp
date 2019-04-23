//#include <string>
//#include <iostream>
//#include <fstream>
//using namespace std;
//
//#include "methods.h"
//
//int main(int argc, char *argv[]) {
//	// arguments
//	string dataset = argv[1];
//	string model = argv[2];
//	cout << dataset << "-" << model << endl;
//
//	int rep = atoi(argv[3]);
//
//	vector<double> thetas;
//	for (int i = 4; i < argc; i++) {
//		thetas.push_back(stod(argv[i]));
//		cout << thetas.back() << "\t";
//	}
//	cout << endl;
//	// arguments
//
//	int col;
//	set<int> positives = { 1 };
//	if (dataset == "adult") {
//		if (model == "salary") col = 14;
//		else if (model == "gender") col = 9;
//		else if (model == "education") {
//			col = 3;
//			positives = { 9, 10, 11, 12, 13, 14, 15 };
//		}
//		else if (model == "marital") {
//			col = 5;
//			positives = { 0 };
//		}
//	}
//	else if (dataset == "nltcs") {
//		if (model == "outside") col = 15;
//		else if (model == "money") col = 2;
//		else if (model == "bathing") col = 10;
//		else if (model == "traveling") col = 3;
//	}
//	else if (dataset == "br2000") {
//		if (model == "dwell") {
//			col = 0;
//			positives = { 0 };
//		}
//		else if (model == "car") {
//			col = 1;
//			positives = { 0 };
//		}
//		else if (model == "child") {
//			col = 3;
//			positives = { 0 };
//		}
//		else if (model == "age") {
//			col = 4;
//			positives = { 0, 1, 2, 3 };
//		}
//		else if (model == "religion") {
//			col = 7;
//			positives = { 0 };
//		}
//	}
//	else if (dataset == "acs") {
//		if (model == "dwell") col = 3;
//		else if (model == "mortgage") col = 4;
//		else if (model == "multigen") col = 9;
//		else if (model == "race") col = 14;
//		else if (model == "school") col = 18;
//		else if (model == "migrate") col = 20;
//	}
//
//	model = dataset + "-" + model;
//	ofstream out(model + ".out");
//	ofstream log(model + ".log");
//	cout.rdbuf(log.rdbuf());
//
//	random_device rd;						//non-deterministic random engine
//	engine eng(rd());						//deterministic engine with a random seed
//
//	table tbl("../_data/" + dataset, true);
//
//	// load testing
//	ifstream ftest("../_test/" + model + ".test");
//	string s;
//	vector<int> ytests;
//	while (getline(ftest, s)) ytests.push_back(stoi(s));
//	ftest.close();
//
//	for (double theta : thetas) {
//		cout << "theta: " << theta << endl;
//		out << "theta: " << theta << endl;
//		for (double epsilon : {0.05, 0.1, 0.2, 0.4, 0.8, 1.6}) {
//			double err = 0.0;
//			for (int i = 0; i < rep; i++) {
//				cout << "epsilon: " << epsilon << " rep: " << i << endl;
//				bayesian bayesian(eng, tbl, epsilon, theta);
//				bayesian.printo_libsvm(model, col, positives);
//				system(("svm-train -t 2 " + model).c_str());
//				system(("svm-predict ../_test/" + model + ".test " + model + ".model " + model + ".pred").c_str());
//
//				// load prediction
//				double mismatch = 0;
//				int ypred;
//				ifstream fpred(model + ".pred");
//				for (const int& ytest : ytests) {
//					fpred >> ypred;
//					if (ytest != ypred) mismatch++;
//				}
//				fpred.close();
//				err += mismatch / ytests.size();
//			}
//			out << err / rep << endl;
//		}
//		cout << endl;
//		out << endl;
//	}
//	out.close();
//	log.close();
//	return 0;
//}