#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <set>
#include <map>
#include <numeric>
#include <algorithm>
using namespace std;

#include "translator.h"
#include "tools.h"


struct marginal	{
	map<vector<int>, vector<double>> counts, scores, mutual;
};
// counts: lvls -> histograms		NOTE: <0, 0, ..., 0> maps to the base histogram
// scores: lvls -> score vector
// mutual: lvls -> mutual info


typedef pair<int, int> attribute;
// <col, lvl>


class dependence {
public:
	vector<attribute> p;
	attribute x;

	// marginal info
	vector<int> cols;
	vector<int> lvls;
	int ptr;

	dependence(const vector<attribute>& p1, const attribute& x1) : p(p1), x(x1) {
		ptr = tools::position(x, p);
		for (const attribute& a : p) {
			cols.push_back(a.first);
			lvls.push_back(a.second);
		}
		cols.insert(cols.begin() + ptr, x.first);
		lvls.insert(lvls.begin() + ptr, x.second);
	}
};


class table
{
public:
	table(const string&, bool);
	table();
	~table();

	int size();
	int getDepth(int);
	vector<int> getDepth(const vector<int>&);
	int getWidth(int, int = 0);
	vector<int> getWidth(const vector<int>&);
	vector<int> getWidth(const vector<int>&, const vector<int>&);
	int getWidth(const attribute&);
	double getScore(const dependence&);
	double getMutual(const dependence&);
	vector<double> getCounts(const vector<int>&);
	vector<double> getCounts(const vector<int>&, const vector<int>&);
	vector<double> getI(const vector<double>&, const vector<int>&);
	vector<double> getF(const vector<double>&, const vector<int>&);
	vector<double> getR(const vector<double>&, const vector<int>&);
	vector<double> getConditional(const dependence&, const vector<int>&);

	void initialize(const table&);
	//void materialize(const vector<int>&);
	void materialize(const vector<int>&, const vector<int>&);

	int generalize(int, int, int, int);																		// val, col, from_lvl, to_lvl;
	vector<int> generalize(const vector<int>&, const vector<int>&, const vector<int>&);						// val, col, to_lvl (from 0);
	vector<int> generalize(const vector<int>&, const vector<int>&, const vector<int>&, const vector<int>&);	// val, col, from_lvl, to_lvl;

	pair<int, int> specialize(int, int, int, int);															// val, col, from_lvl, to_lvl;
	vector<pair<int, int>> specialize(const vector<int>&, const vector<int>&, const vector<int>&);			// val, col, from_lvl (to 0);

	void printo_libsvm(const string&, int, const set<int>&);
	void printo_file(const string&);


	bool func;								// binary model
	int dim;
	vector<vector<int>> data;
	vector<shared_ptr<translator>> translators;
	map<vector<int>, marginal> margins;			// cols -> marginal
	double sens;								// sensitivity of I (general domain) and F (binary domain)
};
