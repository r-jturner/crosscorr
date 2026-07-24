#include <corr_desi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
//typedef enum {
//	EST_PSI1;
//	EST_PSI2;
//	EST_PSI3;
//	ESI_XIGG;
//	EST_GEOM;
//} Estimator;

Estimator whichParam(const char *input) {
	if (strcmp(input,"psi1") == 0)
		return EST_PSI1;
	if (strcmp(input,"psi2") == 0)
		return EST_PSI2;
	if (strcmp(input,"psi3") == 0)
		return EST_PSI3;
	if (strcmp(input,"xiGG") == 0)
		return EST_XIGG;
	if (strcmp(input,"geom") == 0)
		return EST_GEOM;
	
	return EST_INVALID;
}

void free_structmemory(struct output *ptr) {
    //free(ptr);
	if(ptr == NULL)
		return;
	
	free(ptr->num);
	free(ptr->den);
	free(ptr);
}
//void free_arraymemory(double *ptr) {
//    free(ptr);
//}

struct lin_corr calcCorr(double x1, double y1, double z1, double x2, double y2, double z2,
                     double u1, double u2, double width){

        double delta_x, delta_y, delta_z;
        double norm, norm_a, norm_b, rahat_x, rahat_y, rahat_z, rbhat_x, rbhat_y, rbhat_z;
        struct lin_corr result;

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
        
        result.cosA = rahat_x*(delta_x/norm)+rahat_y*(delta_y/norm)+rahat_z*(delta_z/norm);
        result.cosB = rbhat_x*(delta_x/norm)+rbhat_y*(delta_y/norm)+rbhat_z*(delta_z/norm);     
        result.cosAB = rahat_x*rbhat_x+rahat_y*rbhat_y+rahat_z*rbhat_z;
        result.uA = u1;
        result.uB = u2;
        result.index = (int)(norm/width);
		result.valid = 1;
        return result; 
}

static inline void accumulate_pairs(Estimator estimator, struct lin_corr pair, double weight, double *num, double *den){
	switch(estimator)
	{
		case EST_PSI1:

			*num += weight * pair.cosAB * pair.uA * pair.uB;
			*den += weight * pair.cosAB * pair.cosAB;
			break;

		case EST_PSI2:

			*num += weight * pair.cosA * pair.cosB * pair.uA * pair.uB;
			*den += weight * pair.cosA * pair.cosB * pair.cosAB;
			break;

		case EST_PSI3:
			
			*num += weight * pair.cosB * pair.uB;
			*den += weight * pair.cosB * pair.cosB;
			break;

		case EST_XIGG:

			*num += weight;
			break;

		case EST_GEOM:

			*num += weight * pair.cosA * pair.cosA * pair.cosB * pair.cosB;
			break;

		case EST_INVALID:
			fprintf(stderr,
					"Invalid estimator supplied\n");
			break;

		case EST_3D:
			break;
	}
}

struct output *pairCounter(int drows, int rrows, int equiv, const double sample1[drows][4], const double sample2[rrows][4], 
                 const double weights1[drows], const double weights2[rrows], int smax, int swidth, const char* estimator, 
                 int nthreads, int verbose){
    long long i,j;
    struct lin_corr pair;
    /* allocate the memory for the output structure */
	if(swidth <= 0) {
		fprintf(stderr,
				"pairCounter(): swidth must be > 0\n");
		return NULL;
	}

    int numBins = (int)(smax / swidth);
    struct output *results = malloc(sizeof(*results));

	if(results == NULL) {
		fprintf(stderr,
				"pairCounter(): failed to allocate output struct\n");
		return NULL;
	}
	results->num = calloc(numBins, sizeof(double));
	results->den = calloc(numBins, sizeof(double));

	if(results->num == NULL || results->den == NULL) {
		free(results->num);
		free(results->den);
		free(results);
		fprintf(stderr,
				"pairCounter(): failed to allocate output arrays\n");
		return NULL;
	}
	double *num = results->num;
	double *den = results->den;

    Estimator param = whichParam(estimator);
	if (param == EST_INVALID) {
		fprintf(stderr,
				"Unknown estimator '%s'\n",estimator);
		return NULL;
	}

    if (equiv == 1){ 
		#pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) reduction(+:num[:numBins]) reduction(+:den[:numBins])
       	for (i = 0; i < (drows-1); i++) {
			for (j = (i+1); j < drows; j++) {
            	/* do the pair count */
                pair = calcCorr(sample1[i][0], sample1[i][1], sample1[i][2],
                                sample2[j][0], sample2[j][1], sample2[j][2],
                                sample1[i][3], sample2[j][3], swidth);
				if(!pair.valid)
					continue;
				if(pair.index < 0 || pair.index >= numBins)
					continue;
				double weight = weights1[i] * weights2[j];
				accumulate_pairs(param, pair, weight, &num[pair.index], &den[pair.index]);
			}
		}
	}
	else {
		#pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) reduction(+:num[:numBins]) reduction(+:den[:numBins])
		for (i = 0; i < drows; i++) {
			for (j = 0; j < rrows; j++) {
				pair = calcCorr(sample1[i][0], sample1[i][1], sample1[i][2],
								sample2[j][0], sample2[j][1], sample2[i][2],
								sample1[i][3], sample2[j][3], swidth);
				if(!pair.valid)
					continue;
				if(pair.index < 0 || pair.index >= numBins)
					continue;
				double weight = weights1[i] * weights2[j];
				accumulate_pairs(param, pair, weight, &num[pair.index], &den[pair.index]);
			}
		}
	}
	return results;
}

