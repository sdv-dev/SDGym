#pragma once
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <set>
using namespace std;

class tools
{
public:
	tools() {}
	~tools() {}

	static long long nCk(long long n, int k) {
		if (k < 0) return 0;
		if (k == 0) return 1;
		return (n * nCk(n - 1, k - 1)) / k;
	}

	static int encode_subset(const vector<int>& sub, int n) {
		int k = sub.size();
		int code = 0;
		while (k) {
			if (sub[k - 1] == n - 1) {
				code += nCk(n - 1, k);
				k--;
			}
			n--;
		}
		return code;
	}

	static vector<int> decode_subset(int code, int k, int n) {
		vector<int> sub;
		while (k) {
			if (code >= nCk(n - 1, k)) {
				code -= nCk(n - 1, k);
				sub.push_back(n - 1);
				k--;
			}
			n--;
		}
		reverse(sub.begin(), sub.end());
		return sub;
	}

	static bool inc(vector<int>& value, const vector<int>& bound) {
		int t = 0;
		while (t < value.size() && value[t] == bound[t] - 1) {
			value[t] = 0;
			t++;
		}

		if (t == value.size()) return false;
		value[t]++;
		return true;
	}

	static int encode(const vector<int>& value, const vector<int>& bound) {
		int code = 0;
		for (int t = bound.size() - 1; t > -1; t--) {
			code *= bound[t];
			code += value[t];
		}
		return code;
	}

	static vector<int> decode(int code, const vector<int>& bound) {
		vector<int> value;
		for (int t = 0; t < bound.size(); t++) {
			value.push_back(code % bound[t]);
			code /= bound[t];
		}
		return value;
	}

	static int encode_gray(const vector<int>& value) {
		int bin = encode(value, vector<int>(value.size(), 2));
		for (int mask = bin >> 1; mask != 0; mask = mask >> 1)
			bin = bin ^ mask;
		return bin;
	}

	static vector<int> decode_gray(int code, int n) {
		return decode(code ^ (code >> 1), vector<int>(n, 2));
	}

	template <typename T>
	static vector<vector<T>> kway(int k, vector<T> universe) {
		vector<vector<T>> ans;
		if (k == 0) {
			ans.push_back(vector<T>());
			return ans;
		}

		while (!universe.empty()) {
			T current = universe.back();
			universe.pop_back();
			for (vector<T> sub : kway(k - 1, universe)) {
				sub.push_back(current);
				ans.push_back(sub);
			}
		}
		return ans;
	}

	static double mutual_info(const vector<vector<double>>& joint) {
		int n = joint.size();
		int m = joint[0].size();

		double sum = 0.0;
		vector<double> nsum(n, 0.0), msum(m, 0.0);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) {
				nsum[i] += joint[i][j];
				msum[j] += joint[i][j];
				sum += joint[i][j];
			}

		double ans = 0.0;
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				if (joint[i][j]) ans += (joint[i][j] / sum) * log2(joint[i][j] * sum / (nsum[i] * msum[j]));
		return ans;
	}

	static double margin_distance(const vector<vector<double>>& joint) {
		int n = joint.size();
		int m = joint[0].size();

		double sum = 0.0;
		vector<double> nsum(n, 0.0), msum(m, 0.0);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) {
				nsum[i] += joint[i][j];
				msum[j] += joint[i][j];
				sum += joint[i][j];
			}

		double ans = 0.0;
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				ans += abs(joint[i][j] / sum - nsum[i] / sum * msum[j] / sum);
		return ans;
	}

	template <typename T>
	static vector<T> projection(const vector<T>& vec, const vector<int>& cols) {
		vector<T> projected;
		for (int col : cols) projected.push_back(vec[col]);
		return projected;
	}

	static vector<double> normalize(const vector<double>& vec) {
		double sum = accumulate(vec.begin(), vec.end(), 0.0);
		if (sum == 0.0) return vector<double>(vec.size(), 1.0 / vec.size());
		vector<double> ans;
		for (const double val : vec) ans.push_back(val / sum);
		return ans;
	}

	static double TVD(const vector<double>& a, const vector<double>& b) {
		const auto na = normalize(a);
		const auto nb = normalize(b);
		double dist = 0.0;
		for (int t = 0; t < na.size(); t++) dist += abs(na[t] - nb[t]);
		return dist / 2;
	}

	static vector<int> vectorize(int k) {
		vector<int> vec;
		for (int t = 0; t < k; t++) vec.push_back(t);
		return vec;
	}

	static set<int> setize(int k) {
		set<int> set;
		for (int t = 0; t < k; t++) set.insert(t);
		return set;
	}

	template <typename T>
	static int position(T key, const vector<T>& vec) {
		return lower_bound(vec.begin(), vec.end(), key) - vec.begin();
	}
};
