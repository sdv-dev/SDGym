#include "noise.h"

noise::noise() {
}

noise::~noise() {
}


// gaussian
double noise::nextGaussian(engine& eng, double std) {
	normal_distribution<double> normal(0.0, std);
	return normal(eng);
}


// uniform
double noise::nextDouble(engine& eng, double range) {
	uniform_real_distribution<double> unif(0.0, range);
	return unif(eng);
}

double noise::nextDouble(engine& eng, double left, double right) {
	uniform_real_distribution<double> unif(left, right);
	return unif(eng);
}

int noise::nextInt(engine& eng, int range) {			// right-exclusive
	uniform_int_distribution<int> unif(0, range - 1);
	return unif(eng);
}

int noise::nextInt(engine& eng, int left, int right) {	// right-exclusive
	uniform_int_distribution<int> unif(left, right - 1);
	return unif(eng);
}

int noise::nextSign(engine& eng) {
	return 2 * nextInt(eng) - 1;
}


// exponential
double noise::nextExponential(engine& eng, double lambda) {
	exponential_distribution<double> expo(lambda);
	return expo(eng);
}


// laplace
double noise::nextLaplace(engine& eng, double scale) {
	return nextExponential(eng, 1.0 / scale) - nextExponential(eng, 1.0 / scale);
}

int noise::nextDiscreteLaplace(engine& eng, double scale) {
	double alpha = exp(-1.0 / scale);
	if (nextDouble(eng) < (1 - alpha) / (1 + alpha)) return 0;
	else return nextSign(eng) * (nextGeometric(eng, 1 - alpha) + 1);
}


// geometric
int noise::nextGeometric(engine& eng, double p) {
	geometric_distribution<int> geometric(p);
	return geometric(eng);
}


// cauchy
double noise::nextCauchy(engine& eng, double scale) {
	cauchy_distribution<double> cauchy(0.0, scale);
	return cauchy(eng);
}


// exponential mechanism
int noise::EM(engine& eng, const vector<double>& quality, double ep, double sens) {
	ep = ep / sens;
	double maxq = *max_element(quality.begin(), quality.end());		// compute maxq for the precision issue: when quality scores for ALL outputs are extremely small, exp(q) is ALWAYS zero

	vector<double> weights;
	for (const double& q : quality)
		weights.push_back(exp(ep / 2 * (q - maxq)));				// this adjustment will not affect the differences among scores & probabilities after normalization
	return sample(eng, weights);
}


// random sampling
int noise::sample(engine& eng, const vector<double>& weights) {
	double cum = 0.0;
	vector<double> cdf;
	for (const double& w : weights) {
		cum += max(w, 0.0);													// in case of noise
		cdf.push_back(cum);
	}
	if (cum == 0.0) return nextInt(eng, weights.size());					// no positive weight -> uniform sampling
	return tools::position(nextDouble(eng, cum), cdf);
}

vector<int> noise::sample(engine& eng, const vector<double>& weights, int num) {
	double cum = 0.0;
	vector<double> cdf;
	for (const double& w : weights) {
		cum += max(w, 0.0);													// in case of noise
		cdf.push_back(cum);
	}

	vector<int> sampled;
	for (int i = 0; i < num; i++) {
		if (cum == 0.0) sampled.push_back(nextInt(eng, weights.size()));	// no positive weight -> uniform sampling
		else sampled.push_back(tools::position(nextDouble(eng, cum), cdf));
	}
	return sampled;
}