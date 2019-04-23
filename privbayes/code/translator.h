#pragma once
#include <string>
#include <map>
#include <vector>
#include <algorithm>
using namespace std;

#include "tools.h"

class translator
{
public:
	translator();
	~translator();

	virtual int str2int(string) = 0;
	virtual string int2str(int) = 0;
	virtual int size(int = 0) = 0;
	virtual string int2libsvm(int, int&) = 0;
	virtual string str2libsvm(string, int&) = 0;
	virtual int generalize(int, int, int) = 0;
	virtual pair<int, int> specialize(int, int, int) = 0;

	int depth = 0;
};

class dtranslator : public translator
{
public:
	dtranslator(string);
	~dtranslator();

	// lvl 0 only
	int str2int(string);
	string int2str(int);
	string int2libsvm(int, int&);
	string str2libsvm(string, int&);
	// lvl aware
	int size(int = 0);
	int generalize(int, int, int);
	pair<int, int> specialize(int, int, int);
	// tools
	void flush(string&, int);

	map<string, int> s2i;
	map<int, string> i2s;
	vector<vector<int>> segments;
};

class ctranslator : public translator
{
public:
	ctranslator(double, double, int);
	~ctranslator();

	// lvl 0 only
	int str2int(string);
	string int2str(int);
	string int2libsvm(int, int&);
	string str2libsvm(string, int&);
	// lvl aware
	int size(int = 0);
	int generalize(int, int, int);
	pair<int, int> specialize(int, int, int);

	double min, step;
	int grid;
};