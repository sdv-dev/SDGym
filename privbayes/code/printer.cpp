#include "printer.h"

printer::printer(engine& eng1, const string& dataset) : eng(eng1) {
	// read domain
	ifstream fdomain(dataset + ".domain");

	string s;
	while (getline(fdomain, s)) {
		if (s[0] == 'D')
			translators.push_back(make_shared<dtranslator>(s.substr(2)));
		else {
			size_t spos;
			double min = stod(s.substr(2), &spos);
			double max = stod(s.substr(2 + spos));
			translators.push_back(make_shared<ctranslator>(min, max, 4));		//internal parameter: cut into at most 2^4 blocks
		}
	}
	dim = translators.size();
	fdomain.close();


	// read data
	ifstream fdata(dataset + ".dat");

	string value;
	vector<string> tuple;
	while (getline(fdata, s)) {
		stringstream ss(s);
		tuple.clear();
		for (int t = 0; t < dim; t++) {
			ss >> value;
			tuple.push_back(value);
		}
		data.push_back(tuple);
	}
	fdata.close();
}

printer::~printer() {
}


void printer::printo_libsvm(const string& filename, double ratio, int col, const set<int>& positives) {
	ofstream fsvm(filename);
	for (const auto& tuple : data) {
		if (noise::nextDouble(eng) > ratio) continue;
		string output;
		int index = 1;

		for (int t = 0; t < dim; t++) {
			if (t == col) {
				if (positives.find(translators[t]->str2int(tuple[t])) != positives.end()) output = "+1" + output;
				else output = "-1" + output;
			}
			else output += " " + translators[t]->str2libsvm(tuple[t], index);
		}
		fsvm << output << endl;
	}
	fsvm.close();
}