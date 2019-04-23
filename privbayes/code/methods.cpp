#include <cmath>
#include "methods.h"

base::base(engine& eng1, table& tbl1) : eng(eng1), tbl(tbl1) {
}

base::~base() {
}



////////////////////////////// bayesian //////////////////////////////
bayesian::bayesian(engine& eng1, table& tbl1, double ep, double theta) : base(eng1, tbl1) {
	dim = tbl.dim;
	bound = ep * tbl.size() / (4.0 * dim * theta);		// bound to nr. of cells

	// for efficiency
	int sub = log2(bound);
	int count = 0;
	while (tbl.size() * tools::nCk(dim, sub) > 2e12) {		// default 2e9
		sub--;
		bound /= 2;
		count++;
	}
	if (count) cout << "Bound reduced for efficiency: " << count << "." << endl;
	// for efficiency

	model = greedy(0.5 * ep);
	addnoise(0.5 * ep);
}

bayesian::~bayesian() {
}

vector<dependence> bayesian::greedy(double ep) {
	vector<dependence> model;
	double sens = tbl.sens;

	set<int> S;
	set<int> V = tools::setize(dim);

	for (int t = 0; t < dim; t++) {
		vector<dependence> deps = S2V(S, V);
		cout << deps.size() << "\t";

		vector<double> quality;
		for (const auto& dep : deps) quality.push_back(tbl.getScore(dep));

		dependence picked = t ? deps[noise::EM(eng, quality, ep / (dim - 1), sens)] : deps[noise::EM(eng, quality, 1000.0, sens)];
		// first selection is free: all scores are zero.

		S.insert(picked.x.first);
		V.erase(picked.x.first);
		model.push_back(picked);
		cout << to_string(picked) << endl;				// debug
	}
	return model;
}

vector<dependence> bayesian::S2V(const set<int>& S, const set<int>& V) {
	vector<dependence> ans;
	for (int x : V) {
		set<vector<attribute>> exist;
		vector<vector<attribute>> parents = maximal(S, bound / tbl.getWidth(x));

		for (const vector<attribute>& p : parents)
			if (exist.find(p) == exist.end()) {
				exist.insert(p);
				ans.push_back(dependence(p, attribute(x, 0)));
			}
		if (exist.empty()) ans.push_back(dependence(vector<attribute>(), attribute(x, 0)));
	}
	return ans;
}

vector<vector<attribute>> bayesian::maximal(set<int> S, double tau) {
	vector<vector<attribute>> ans;
	if (tau < 1) return ans;
	if (S.empty()) {
		ans.push_back(vector<attribute>());
		return ans;
	}

	int last = *(--S.end());
	S.erase(--S.end());
	int depth = tbl.getDepth(last);
	set<vector<attribute>> exist;

	// with 'last' at a certain level
	for (int l = 0; l < depth; l++) {
		attribute att(last, l);
		vector<vector<attribute>> maxs = maximal(S, tau / tbl.getWidth(att));
		for (vector<attribute> z : maxs)
			if (exist.find(z) == exist.end()) {
				exist.insert(z);
				z.push_back(att);
				ans.push_back(z);
			}
	}

	// without 'last'
	vector<vector<attribute>> maxs = maximal(S, tau);
	for (vector<attribute> z : maxs)
		if (exist.find(z) == exist.end()) {
			exist.insert(z);
			ans.push_back(z);
		}

	return ans;
}

void bayesian::addnoise(double ep) {
	syn.initialize(tbl);
	for (const dependence& dep : model) {
		vector<double>& counts_syn = syn.margins[dep.cols].counts[dep.lvls];
		for (double count : tbl.getCounts(dep.cols, dep.lvls))
			counts_syn.push_back(count + noise::nextLaplace(eng, 2.0 * dim / ep));
	}
	// add consistency
}

void bayesian::sampling(int num) {
	for (int i = 0; i < num; i++) {
		vector<int> tuple(dim, 0);
		for (const dependence& dep : model) {
			vector<int> pre = tbl.generalize(
				tools::projection(tuple, dep.cols),
				dep.cols,
				dep.lvls);

			vector<double> conditional = syn.getConditional(dep, pre);
			tuple[dep.x.first] = noise::sample(eng, conditional);
		}
		syn.data.push_back(tuple);
	}
	syn.margins.clear();
}

string bayesian::to_string(const dependence& dep) {
	string ans = to_string(dep.x) + " <-";
	for (const auto& p : dep.p)
		ans += " " + to_string(p);
	return ans;
}

string bayesian::to_string(const attribute& att) {
	return std::to_string(att.first) + "(" + std::to_string(att.second) + ")";
}

void bayesian::printo_libsvm(const string& filename, const int& col, const set<int>& positives) {
	syn.printo_libsvm(filename, col, positives);
}

double bayesian::evaluate() {
	double sum = 0.0;
	for (const dependence& dep : model) sum += tbl.getMutual(dep);
	return sum;
}

// interface
vector<double> bayesian::getCounts(const vector<int>& mrg) {
	return syn.getCounts(mrg);
}



////////////////////////////	laplace //////////////////////////////
//laplace::laplace(engine& eng1, table& tbl1, double ep, const vector<vector<int>>& mrgs) : base(eng1, tbl1) {
//	double scale = 2.0 * mrgs.size() / ep;
//	for (const auto& mrg : mrgs) {
//		vector<double> counts = tbl.getCounts(mrg);
//		for (double& val : counts) val = max(0.0, val + noise::nextLaplace(eng, scale));
//		noisy[mrg] = counts;
//	}
//}
//
//laplace::~laplace() {
//}
//
//// interface
//vector<double> laplace::getCounts(const vector<int>& mrg) {
//	return noisy[mrg];
//}
//
//
//
////////////////////////////	contingency //////////////////////////////
//contingency::contingency(engine& eng1, table& tbl1, double ep) : base(eng1, tbl1) {
//	vector<int> hist = tbl.getHistogram();
//	vector<int> cells = tbl.cells(tbl.dimset());
//
//	vector<double> noisy(hist.begin(), hist.end());
//	for (double& val : noisy) val += noise::nextLaplace(eng, 2.0 / ep);
//
//	syn.copySettings(tbl);
//	vector<int> sampled = noise::sample(eng, noisy, tbl.size());
//	for (const int item : sampled)
//		syn.data().push_back(tools::decode(item, cells));
//}
//
//contingency::~contingency() {
//}
//
//// interface
//vector<double> contingency::getCounts(const vector<int>& mrg) {
//	return syn.getCounts(mrg);
//}
