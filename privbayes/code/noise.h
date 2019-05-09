#pragma once
#include <algorithm>
#include <random>
using namespace std;

#include "tools.h"

typedef default_random_engine engine;

class noise
{
public:
	noise();
	~noise();

	// gaussian
	static double nextGaussian(engine&, double = 1.0);

	// uniform
	static double nextDouble(engine&, double = 1.0);
	static double nextDouble(engine&, double, double);
	static int nextInt(engine&, int = 2);
	static int nextInt(engine&, int, int);
	static int nextSign(engine&);

	// exponential
	static double nextExponential(engine&, double = 1.0);

	// laplace
	static double nextLaplace(engine&, double = 1.0);
	static int nextDiscreteLaplace(engine&, double = 1.0);

	// geometric
	static int nextGeometric(engine&, double = 0.5);

	// cauchy
	static double nextCauchy(engine&, double = 1.0);

	// exponential mechanism
	static int EM(engine&, const vector<double>&, double, double = 1.0);

	// random sampling
	static int sample(engine&, const vector<double>&);
	static vector<int> sample(engine&, const vector<double>&, int);
};
