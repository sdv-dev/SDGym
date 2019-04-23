#include "translator.h"

translator::translator() {
}

translator::~translator() {
}

/////////////////////////////////////////////////

dtranslator::dtranslator(string s) {
	int lvl = 1;
	for (char c : s) {
		if (c == '{') lvl++;
		else if (c == '}') lvl--;
		depth = max(depth, lvl);
	}
	segments = vector<vector<int>>(depth, vector<int>());

	lvl = depth - 1;		// at the highest abstract level
	string key = "";

	for (char c : s) {
		if (c == '{') {
			flush(key, lvl);
			lvl--;
		}
		else if (c == '}') {
			flush(key, lvl);
			lvl++;
			segments[lvl].push_back(s2i.size() - 1);
		}
		else if (c == ' ') flush(key, lvl);
		else key += c;
	}
	flush(key, lvl);
}

void dtranslator::flush(string& key, int lvl) {
	if (key.empty()) return;
	int value = s2i.size();
	s2i[key] = value;
	i2s[value] = key;
	key.clear();

	for (int t = 0; t <= lvl; t++)
		segments[t].push_back(value);
}

dtranslator::~dtranslator() {
}


int dtranslator::str2int(string key) {
	return s2i[key];
}

string dtranslator::int2str(int key) {
	return i2s[key];
}

string dtranslator::int2libsvm(int key, int& index) {
	string str = to_string(index + key) + ":1";
	index += size();
	return str;
}

string dtranslator::str2libsvm(string key, int& index) {
	string str = to_string(index + str2int(key)) + ":1";
	index += size();
	return str;
}


int dtranslator::size(int lvl) {
	return segments[lvl].size();
}

int dtranslator::generalize(int key, int from, int to) {
	return tools::position(segments[from][key], segments[to]);
}

pair<int, int> dtranslator::specialize(int key, int from, int to) {
	int lower = key ? tools::position(segments[from][key - 1], segments[to]) + 1 : 0;
	int upper = tools::position(segments[from][key], segments[to]) + 1;		// right-exclusive
	return make_pair(lower, upper);
}


/////////////////////////////////////////////////

ctranslator::ctranslator(double min1, double max1, int depth1) : min(min1) {
	depth = depth1;
	grid = size();
	step = (max1 - min1) / grid;
}

ctranslator::~ctranslator() {
}


int ctranslator::str2int(string key) {
	return int((stod(key) - min) / (step + 1e-8));		// add a small delta
}

string ctranslator::int2str(int key) {
	return to_string((key + 0.5) * step + min);
}

string ctranslator::int2libsvm(int key, int& index) {
	return to_string(index++) + ":" + to_string((key + 0.5) / grid);			//centering & [0, 1]
}

string ctranslator::str2libsvm(string key, int& index) {
	return to_string(index++) + ":" + to_string((stod(key) - min) / step / grid);
}


int ctranslator::size(int lvl) {
	return int(pow(2, depth - lvl));
}

int ctranslator::generalize(int key, int from, int to) {
	return key / pow(2, to - from);
}

pair<int, int> ctranslator::specialize(int key, int from, int to) {
	return make_pair(key * pow(2, from - to), (key + 1) * pow(2, from - to));	// right-exclusive
}
