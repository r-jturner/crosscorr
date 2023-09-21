#include <corr_desi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
//#include <omp.h>
int whichParam(const char* input) {
    const char* validParams[] = {"psi1", "psi2", "psi3", "xiGG"};
    int numParams = sizeof(validParams) / sizeof(validParams[0]);

    for (int i = 0; i < numParams; ++i) {
        if (strcmp(input, validParams[i]) == 0) {
            // Return an index indicating the matching string
            // psi1 == 0, psi2 == 1, psi3 == 2, xiGG == 3
            return i;
        }
    }
    // Return -1 if no valid string is found
    return -1;
}

void free_structmemory(struct output *ptr) {
    free(ptr);
}
void free_arraymemory(double *ptr) {
    free(ptr);
}

struct lin_corr calcCorr(double x1, double y1, double z1, double x2, double y2, double z2,
                     double u1, double u2, double width){

        double delta_x, delta_y, delta_z;
        double norm, norm_a, norm_b, rahat_x, rahat_y, rahat_z, rbhat_x, rbhat_y, rbhat_z;
        struct lin_corr result;

        delta_x = x1 - x2; delta_y = y1 - y2; delta_z = z1 - z2;
        norm   = sqrt(delta_x*delta_x + delta_y*delta_y + delta_z*delta_z);
        norm_a = sqrt(x1*x1 + y1*y1 + z1*z1);
        norm_b = sqrt(x2*x2 + y2*y2 + z2*z2);
        rahat_x = x1/norm_a; rahat_y = y1/norm_a; rahat_z = z1/norm_a;
        rbhat_x = x2/norm_b; rbhat_y = y2/norm_b; rbhat_z = z1/norm_b;
        
        result.cosA = rahat_x*(delta_x/norm)+rahat_y*(delta_y/norm)+rahat_z*(delta_z/norm);
        result.cosB = rbhat_x*(delta_x/norm)+rbhat_y*(delta_y/norm)+rbhat_z*(delta_z/norm);     
        result.cosAB = rahat_x*rbhat_x+rahat_y*rbhat_y+rahat_z*rbhat_z;
        result.uA = u1;
        result.uB = u2;
        result.index = (int)(norm/width);
        return result; 
}

struct output *pairCounter(int drows, int rrows, int equiv, const double sample1[drows][4], const double sample2[rrows][4], 
                 int smax, int swidth, const char* estimator, int nthreads){
    long long i,j;
    struct lin_corr pair;
    /* allocate the memory for the output structure */
    int numBins = (int)(smax / swidth);
    struct output *results = malloc(sizeof(*results));
    results->num = malloc((numBins) * sizeof(*(results->num)));
    results->den = malloc((numBins) * sizeof(*(results->den)));

    double *num = results->num;
    double *den = results->den;
    for(int iv = 0; iv < numBins; iv++){
        num[iv] = 0.0; den[iv] = 0.0;
    }
    if(num == NULL || den == NULL){
        printf("Null pointer in output struct - exiting program.\n");
        exit(0);
    }
    int param = whichParam(estimator);
    if (equiv == 1){ 
        //if (sample1 == sample2){ <- for some reason the C code won't recognise that dat_sample == dat_sample, so pass a binary
        //                            value for the time being until it can be figured out (or just leave it as is...)

        // if samples are equivalent then we want an auto-correlation pair count (DD or RR)
        printf("Samples are equivalent, calculating auto- pair counts.\n");
        switch (param) {
            case 0:
                //printf("Calculating psi_1 estimator components.\n");
                //#pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) shared(num,den)
                #pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) reduction(+:num[:numBins]) reduction(+:den[:numBins])
                for (i = 0; i < (drows-1); i++) {
                    for (j = (i+1); j < drows; j++) {
                        /* do the pair count */
                        pair = calcCorr(sample1[i][0], sample1[i][1], sample1[i][2],
                                        sample2[j][0], sample2[j][1], sample2[j][2],
                                        sample1[i][3], sample2[j][3], swidth);
                        if (pair.index < numBins && pair.index >= 0) {
                            num[pair.index] += pair.cosAB*(pair.uA*pair.uB);
                            den[pair.index] += pair.cosAB*pair.cosAB;
                        }
                    }
                }
                break;
            case 1:
                printf("Calculating psi_2 estimator components.\n");
                #pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) reduction(+:num[:numBins]) reduction(+:den[:numBins])
                for (i = 0; i < (drows-1); i++) {
                    for (j = (i+1); j < drows; j++) {
                        /* do the pair count */
                        pair = calcCorr(sample1[i][0], sample1[i][1], sample1[i][2],
                                        sample2[j][0], sample2[j][1], sample2[j][2],
                                        sample1[i][3], sample2[j][3], swidth);
                        if (pair.index < numBins && pair.index >= 0) {
                            num[pair.index] += pair.cosA*pair.cosB*(pair.uA*pair.uB);
                            den[pair.index] += pair.cosA*pair.cosA*pair.cosAB;
                        }
                    }
                }
                break;
            case 2:
                printf("Calculating psi_3 estimator components.\n");
                #pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) reduction(+:num[:numBins]) reduction(+:den[:numBins])
                for (i = 0; i < (drows-1); i++) {
                    for (j = (i+1); j < drows; j++) {
                        /* do the pair count */
                        pair = calcCorr(sample1[i][0], sample1[i][1], sample1[i][2],
                                        sample2[j][0], sample2[j][1], sample2[j][2],
                                        sample1[i][3], sample2[j][3], swidth);
                        if (pair.index < numBins && pair.index >= 0) {
                            num[pair.index] += pair.cosB*pair.uB;
                            den[pair.index] += pair.cosB*pair.cosB;
                        }
                    }
                }
                break;
            case 3:
                printf("Calculating xi_GG estimator components.\n");
                #pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) reduction(+:num[:numBins]) reduction(+:den[:numBins])
                for (i = 0; i < (drows-1); i++) {
                    for (j = (i+1); j < drows; j++) {
                        /* do the pair count */
                        pair = calcCorr(sample1[i][0], sample1[i][1], sample1[i][2],
                                        sample2[j][0], sample2[j][1], sample2[j][2],
                                        sample1[i][3], sample2[j][3], swidth);
                        if (pair.index < numBins && pair.index >= 0) {
                            num[pair.index] += 1.0;
                            den[pair.index] += 0.0;
                        }
                    }
                }
                break;
            default:
                printf("Error occured: expected estimator = 'psi1', 'psi2', 'psi3' or 'xiGG' as an input.\n");
                break;
        }
    }
    else {
        // If samples are different then we want a cross-correlation (DR or RD)
        printf("Samples are not equivalent, calculating cross- pair counts.\n");
        switch (param) {
            case 0:
                printf("Calculating psi_1 estimator components.\n");
                #pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) reduction(+:num[:numBins]) reduction(+:den[:numBins])
                for (i = 0; i < drows; i++) {
                    for (j = 0; j < rrows; j++) {
                        /* do the pair count */
                        pair = calcCorr(sample1[i][0], sample1[i][1], sample1[i][2],
                                        sample2[j][0], sample2[j][1], sample2[j][2],
                                        sample1[i][3], sample2[j][3], swidth);
                        if (pair.index < numBins && pair.index >= 0) {
                            num[pair.index] += pair.cosAB*(pair.uA*pair.uB);
                            den[pair.index] += pair.cosAB*pair.cosAB;
                        }
                    }
                }
                break;
            case 1:
                printf("Calculating psi_2 estimator components.\n");
                #pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) reduction(+:num[:numBins]) reduction(+:den[:numBins])
                for (i = 0; i < drows; i++) {
                    for (j = 0; j < rrows; j++) {
                        /* do the pair count */
                        pair = calcCorr(sample1[i][0], sample1[i][1], sample1[i][2],
                                        sample2[j][0], sample2[j][1], sample2[j][2],
                                        sample1[i][3], sample2[j][3], swidth);
                        if (pair.index < numBins && pair.index >= 0) {
                            num[pair.index] += pair.cosA*pair.cosB*(pair.uA*pair.uB);
                            den[pair.index] += pair.cosA*pair.cosB*pair.cosAB;
                        }
                    }
                }
                break;
            case 2:
                printf("Calculating psi_3 estimator components.\n");
                #pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) reduction(+:num[:numBins]) reduction(+:den[:numBins])
                for (i = 0; i < drows; i++) {
                    for (j = 0; j < rrows; j++) {
                        /* do the pair count */
                        pair = calcCorr(sample1[i][0], sample1[i][1], sample1[i][2],
                                        sample2[j][0], sample2[j][1], sample2[j][2],
                                        sample1[i][3], sample2[j][3], swidth);
                        if (pair.index < numBins && pair.index >= 0) {
                            num[pair.index] += pair.cosB*pair.uB;
                            den[pair.index] += pair.cosB*pair.cosB;
                        }
                    }
                }
                break;
            case 3:
                printf("Calculating xi_GG estimator components.\n");
                #pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) reduction(+:num[:numBins]) reduction(+:den[:numBins])
                for (i = 0; i < drows; i++) {
                    for (j = 0; j < rrows; j++) {
                        /* do the pair count */
                        pair = calcCorr(sample1[i][0], sample1[i][1], sample1[i][2],
                                        sample2[j][0], sample2[j][1], sample2[j][2],
                                        sample1[i][3], sample2[j][3], swidth);
                        if (pair.index < numBins && pair.index >= 0) {
                            num[pair.index] += 1.0;
                            den[pair.index] += 0.0;
                        }
                    }
                }
                break;
            default:
                printf("Error occured: expected estimator = 'psi1', 'psi2', 'psi3' or 'xiGG' as an input.\n");
                break;
        }
    }
    /* 
    return the output struct, populated correctly with either auto or cross pair counts using the correct estimator
    compile call - % gcc-13 -fopenmp -shared -o desitest.so -fPIC corr_desi.c
    */
    
    return results;
}
