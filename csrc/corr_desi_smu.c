#include <corr_desi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
Estimator whichParam_smu(const char *input) {
	if (strcmp(input,"psi3") == 0)
		return EST_PSI3;
	if (strcmp(input,"xiGG") == 0)
		return EST_XIGG;
	return EST_INVALID;
}

struct nonlin_corr calcCorr_smu(double x1, double y1, double z1, double x2, double y2, double z2,
                            double u1, double u2, double width, double mu_width){

        double delta_x, delta_y, delta_z;
        double norm, norm_a, norm_b, rahat_x, rahat_y, rahat_z, rbhat_x, rbhat_y, rbhat_z;
        double rmu_x, rmu_y, rmu_z, norm_mu, rmuhat_x, rmuhat_y, rmuhat_z;
        struct nonlin_corr result;

        delta_x = x1 - x2; delta_y = y1 - y2; delta_z = z1 - z2;
        norm   = sqrt(delta_x*delta_x + delta_y*delta_y + delta_z*delta_z);
        norm_a = sqrt(x1*x1 + y1*y1 + z1*z1);
        norm_b = sqrt(x2*x2 + y2*y2 + z2*z2);

		if(norm <= 0.0 || norm_a <= 0.0 || norm_b <= 0.0) {
			result.valid = 0;
			return result;
		}

        rahat_x = x1/norm_a; rahat_y = y1/norm_a; rahat_z = z1/norm_a;
        rbhat_x = x2/norm_b; rbhat_y = y2/norm_b; rbhat_z = z2/norm_b;

        rmu_x = (norm_a * norm_b) / (norm_a + norm_b) * (rahat_x + rbhat_x);
        rmu_y = (norm_a * norm_b) / (norm_a + norm_b) * (rahat_y + rbhat_y);
        rmu_z = (norm_a * norm_b) / (norm_a + norm_b) * (rahat_z + rbhat_z);
        norm_mu = sqrt(rmu_x*rmu_x + rmu_y*rmu_y + rmu_z*rmu_z);

		if(norm_mu <= 0.0) {
			result.valid = 0;
			return result;
		}

        rmuhat_x = rmu_x/norm_mu;
        rmuhat_y = rmu_y/norm_mu;
        rmuhat_z = rmu_z/norm_mu;
        
        result.cosA = rahat_x*(delta_x/norm)+rahat_y*(delta_y/norm)+rahat_z*(delta_z/norm);
        result.cosB = rbhat_x*(delta_x/norm)+rbhat_y*(delta_y/norm)+rbhat_z*(delta_z/norm);     
        result.cosAB = rahat_x*rbhat_x+rahat_y*rbhat_y+rahat_z*rbhat_z;
        result.cosMu = rmuhat_x*(delta_x/norm)+rmuhat_y*(delta_y/norm)+rmuhat_z*(delta_z/norm);
        result.uA = u1;
        result.uB = u2;
        result.index = (int)(norm/width);
        result.muIndex = (int)((result.cosMu+1.0) / mu_width);
		result.valid = 1;
        return result; 
}

static inline void accumulate_pairs_smu(Estimator estimator, struct nonlin_corr pair, double weight, double *num, double *den){
	switch(estimator)
	{
		case EST_PSI3:

			*num += weight * pair.uB;
			*den += weight;
			break;

		case EST_XIGG:

			*num += weight;
			break;

		case EST_INVALID:
			fprintf(stderr,
					"Invalid estimator supplied\n");
			break;

		case EST_PSI1:
		case EST_PSI2:
		case EST_GEOM:
		case EST_3D:
			break;
	}
}

struct output *pairCounter_smu(int drows, int rrows, int equiv, const double sample1[drows][4], const double sample2[rrows][4], 
                 const double weights1[drows], const double weights2[rrows], int smax, int swidth, double muwidth, 
                 const char* estimator, int nthreads, int verbose){
    long long i,j;
    struct nonlin_corr pair;
    /* allocate the memory for the output structure */

	if(swidth <= 0) {
		fprintf(stderr,
				"pairCounter_smu(): swidth must be > 0\n");
		return NULL;
	}
	if(muwidth <= 0.0) {
		fprintf(stderr,
				"pairCounter_smu(): muwidth must be > 0\n");
		return NULL;
	}
    int numBins = (int)(smax / swidth);
    int muBins = (int)(2. / muwidth);
    int vecLength = numBins * muBins;
    struct output *results = malloc(sizeof(*results));

	if (results == NULL) {
		fprintf(stderr,
				"pairCounter_smu(): failed to allocate output struct\n");
		return NULL;
	}
	results->num = calloc(vecLength, sizeof(*(results->num)));
	results->den = calloc(vecLength, sizeof(*(results->den)));

	if (results->num == NULL || results->den == NULL) {
		free(results->num);
		free(results->den);
		free(results);
		fprintf(stderr,
				"pairCounter_smu(): failed to allocate output arrays\n");
		return NULL;
	}

	double *num = results->num;
	double *den = results->den;
    
    Estimator param = whichParam_smu(estimator);
	if (param == EST_INVALID){
		free(results->num);
		free(results->den);
		free(results);
		fprintf(stderr,
				"Unknown estimator '%s'\n",estimator);
		return NULL;
	}

    if (equiv == 1){ 
        #pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) reduction(+:num[:vecLength]) reduction(+:den[:vecLength])
		for (i = 0; i < (drows-1); i++) {
			for (j = (i+1); j < drows; j++) {
				/* do the pair count */
                pair = calcCorr_smu(sample1[i][0], sample1[i][1], sample1[i][2],
                                    sample2[j][0], sample2[j][1], sample2[j][2],
                                    sample1[i][3], sample2[j][3], swidth, muwidth);
				if(!pair.valid)
					continue;
				if(pair.muIndex == muBins)
					pair.muIndex = muBins - 1;
				if (pair.index < 0 || pair.index >= numBins || pair.muIndex < 0 || pair.muIndex >= muBins)
					continue;
				int bin = pair.index * muBins + pair.muIndex;
				double weight = weights1[i]*weights2[j];
				accumulate_pairs_smu(param, pair, weight, &num[bin], &den[bin]);
			}
		}
	}
	else {
		#pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) reduction(+:num[:vecLength]) reduction(+:den[:vecLength])
		for (i = 0; i < drows; i++) {
			for (j = 0; j < rrows; j++) {
				pair = calcCorr_smu(sample1[i][0], sample1[i][1], sample1[i][2],
									sample2[j][0], sample2[j][1], sample2[j][2],
									sample1[i][3], sample2[j][3], swidth, muwidth);
				if(!pair.valid)
					continue;
				if(pair.muIndex == muBins)
					pair.muIndex = muBins - 1;
				if(pair.index < 0 || pair.index >= numBins || pair.muIndex < 0 || pair.muIndex >= muBins)
					continue;
				int bin = pair.index * muBins + pair.muIndex;
				double weight = weights1[i]*weights2[j];
				accumulate_pairs_smu(param, pair, weight, &num[bin], &den[bin]);
			}
		}
	}
	return results;
}
