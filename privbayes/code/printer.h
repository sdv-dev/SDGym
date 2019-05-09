#pragma once
using namespace std;

#include "table.h"
#include "noise.h"

class printer
{
public:
	printer(engine&, const string&);
	~printer();

	void printo_libsvm(const string&, double, int, const set<int>&);

	int dim;
	vector<vector<string>> data;
	vector<shared_ptr<translator>> translators;
	engine& eng;
};

