#include <corr_desi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

struct nonlin_corr calcCorr_xyz_smu(double x1, double y1, double z1, double x2, double y2, double z2,
                            double vx1, double vy1, double vz1, double vx2, double vy2, double vz2, double width, double mu_width){

        double delta_x, delta_y, delta_z;
        double norm, norm_a, norm_b, rahat_x, rahat_y, rahat_z, rbhat_x, rbhat_y, rbhat_z;
        double ua_x, ua_y, ua_z, ub_x, ub_y, ub_z;
        double rmu_x, rmu_y, rmu_z, norm_mu, rmuhat_x, rmuhat_y, rmuhat_z;
        struct nonlin_corr result;

        delta_x = x1 - x2; delta_y = y1 - y2; delta_z = z1 - z2;
        norm   = sqrt(delta_x*delta_x + delta_y*delta_y + delta_z*delta_z);
        norm_a = sqrt(x1*x1 + y1*y1 + z1*z1);
        norm_b = sqrt(x2*x2 + y2*y2 + z2*z2);
        rahat_x = x1/norm_a; rahat_y = y1/norm_a; rahat_z = z1/norm_a;
        rbhat_x = x2/norm_b; rbhat_y = y2/norm_b; rbhat_z = z1/norm_b;

        rmu_x = (norm_a * norm_b) / (norm_a + norm_b) * (rahat_x + rbhat_x);
        rmu_y = (norm_a * norm_b) / (norm_a + norm_b) * (rahat_y + rbhat_y);
        rmu_z = (norm_a * norm_b) / (norm_a + norm_b) * (rahat_z + rbhat_z);
        norm_mu = sqrt(rmu_x*rmu_x + rmu_y*rmu_y + rmu_z*rmu_z);
        rmuhat_x = rmu_x/norm_mu;
        rmuhat_y = rmu_y/norm_mu;
        rmuhat_z = rmu_z/norm_mu;

        ua_x = vx1*rahat_x; ua_y = vy1*rahat_y; ua_z = vz1*rahat_z;
        ub_x = vx2*rbhat_x; ub_y = vy2*rbhat_y; ub_z = vz2*rbhat_z;
        
        result.cosA = rahat_x*(delta_x/norm)+rahat_y*(delta_y/norm)+rahat_z*(delta_z/norm);
        result.cosB = rbhat_x*(delta_x/norm)+rbhat_y*(delta_y/norm)+rbhat_z*(delta_z/norm);
        result.cosAB = rahat_x*rbhat_x+rahat_y*rbhat_y+rahat_z*rbhat_z;
        result.cosMu = rmuhat_x*(delta_x/norm_mu)+rmuhat_y*(delta_y/norm_mu)+rmuhat_z*(delta_z/norm_mu);
        result.uA = ua_x + ua_y + ua_z;
        result.uB = ub_x + ub_y + ub_z;
        result.index = (int)(norm/width);
        result.muIndex = (int)(result.cosMu+1.0 / mu_width);
        return result; 
}

struct output *pairCounter_xyz_smu(int drows, int rrows, int equiv, const double sample1[drows][6], const double sample2[rrows][6], 
                 int smax, int swidth, double muwidth, const char* estimator, int nthreads){
    long long i,j;
    struct nonlin_corr pair;
    /* allocate the memory for the output structure */
    int numBins = (int)(smax / swidth);
    int muBins = (int)(2. / muwidth);
    int vecLength = numBins * muBins;
    struct output *results = malloc(sizeof(*results));
    results->num = malloc((vecLength) * sizeof(*(results->num)));
    results->den = malloc((vecLength) * sizeof(*(results->den)));

    double *num = results->num;
    double *den = results->den;
    for(int iv = 0; iv < vecLength; iv++){
        num[iv] = 0.0; den[iv] = 0.0;
    }
    if(num == NULL || den == NULL){
        printf("Null pointer in output struct - exiting program.\n");
        exit(0);
    }
    int param = whichParam(estimator);
    if (equiv == 1){ 
        printf("Samples are equivalent, calculating auto- pair counts.\n");
        switch (param) {
            case 0:
                //printf("Calculating psi_1 estimator components.\n");
                #pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) reduction(+:num[:vecLength]) reduction(+:den[:vecLength])
                for (i = 0; i < (drows-1); i++) {
                    for (j = (i+1); j < drows; j++) {
                        /* do the pair count */
                        pair = calcCorr_xyz_smu(sample1[i][0], sample1[i][1], sample1[i][2],
                                                sample2[j][0], sample2[j][1], sample2[j][2],
                                                sample1[i][3], sample1[i][4], sample1[i][5],
                                                sample2[j][3], sample2[j][4], sample2[j][5], swidth, muwidth);
                        if (pair.index < numBins && pair.muIndex < muBins && pair.index >= 0 && pair.muIndex >= 0) {
                            num[pair.index*muBins + pair.muIndex] += pair.cosAB*(pair.uA*pair.uB);
                            den[pair.index*muBins + pair.muIndex] += pair.cosAB*pair.cosAB;
                        }
                    }
                }
                break;
            case 1:
                printf("Calculating psi_2 estimator components.\n");
                #pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) reduction(+:num[:vecLength]) reduction(+:den[:vecLength])
                for (i = 0; i < (drows-1); i++) {
                    for (j = (i+1); j < drows; j++) {
                        /* do the pair count */
                        pair = calcCorr_xyz_smu(sample1[i][0], sample1[i][1], sample1[i][2],
                                                sample2[j][0], sample2[j][1], sample2[j][2],
                                                sample1[i][3], sample1[i][4], sample1[i][5],
                                                sample2[j][3], sample2[j][4], sample2[j][5], swidth, muwidth);
                        if (pair.index < numBins && pair.muIndex < muBins && pair.index >= 0 && pair.muIndex >= 0) {
                            num[pair.index*muBins + pair.muIndex] += pair.cosA*pair.cosB*(pair.uA*pair.uB);
                            den[pair.index*muBins + pair.muIndex] += pair.cosA*pair.cosA*pair.cosAB;
                        }
                    }
                }
                break;
            case 2:
                printf("Calculating psi_3 estimator components.\n");
                #pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) reduction(+:num[:vecLength]) reduction(+:den[:vecLength])
                for (i = 0; i < (drows-1); i++) {
                    for (j = (i+1); j < drows; j++) {
                        /* do the pair count */
                        pair = calcCorr_xyz_smu(sample1[i][0], sample1[i][1], sample1[i][2],
                                                sample2[j][0], sample2[j][1], sample2[j][2],
                                                sample1[i][3], sample1[i][4], sample1[i][5],
                                                sample2[j][3], sample2[j][4], sample2[j][5], swidth, muwidth);
                        if (pair.index < numBins && pair.muIndex < muBins && pair.index >= 0 && pair.muIndex >= 0) {
                            num[pair.index*muBins + pair.muIndex] += pair.cosB*pair.uB;
                            den[pair.index*muBins + pair.muIndex] += pair.cosB*pair.cosB;
                        }
                    }
                }
                break;
            case 3:
                printf("Calculating xi_GG estimator components.\n");
                #pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) reduction(+:num[:vecLength]) reduction(+:den[:vecLength])
                for (i = 0; i < (drows-1); i++) {
                    for (j = (i+1); j < drows; j++) {
                        /* do the pair count */
                        pair = calcCorr_xyz_smu(sample1[i][0], sample1[i][1], sample1[i][2],
                                                sample2[j][0], sample2[j][1], sample2[j][2],
                                                sample1[i][3], sample1[i][4], sample1[i][5],
                                                sample2[j][3], sample2[j][4], sample2[j][5], swidth, muwidth);
                        if (pair.index < numBins && pair.muIndex < muBins && pair.index >= 0 && pair.muIndex >= 0) {
                            num[pair.index*muBins + pair.muIndex] += 1.0;
                            den[pair.index*muBins + pair.muIndex] += 0.0;
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
                #pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) reduction(+:num[:vecLength]) reduction(+:den[:vecLength])
                for (i = 0; i < drows; i++) {
                    for (j = 0; j < rrows; j++) {
                        /* do the pair count */
                        pair = calcCorr_xyz_smu(sample1[i][0], sample1[i][1], sample1[i][2],
                                                sample2[j][0], sample2[j][1], sample2[j][2],
                                                sample1[i][3], sample1[i][4], sample1[i][5],
                                                sample2[j][3], sample2[j][4], sample2[j][5], swidth, muwidth);
                        if (pair.index < numBins && pair.muIndex < muBins && pair.index >= 0 && pair.muIndex >= 0) {
                            num[pair.index*muBins + pair.muIndex] += pair.cosAB*(pair.uA*pair.uB);
                            den[pair.index*muBins + pair.muIndex] += pair.cosAB*pair.cosAB;
                        }
                    }
                }
                break;
            case 1:
                printf("Calculating psi_2 estimator components.\n");
                #pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) reduction(+:num[:vecLength]) reduction(+:den[:vecLength])
                for (i = 0; i < drows; i++) {
                    for (j = 0; j < rrows; j++) {
                        /* do the pair count */
                        pair = calcCorr_xyz_smu(sample1[i][0], sample1[i][1], sample1[i][2],
                                                sample2[j][0], sample2[j][1], sample2[j][2],
                                                sample1[i][3], sample1[i][4], sample1[i][5],
                                                sample2[j][3], sample2[j][4], sample2[j][5], swidth, muwidth);
                        if (pair.index < numBins && pair.muIndex < muBins && pair.index >= 0 && pair.muIndex >= 0) {
                            num[pair.index*muBins + pair.muIndex] += pair.cosA*pair.cosB*(pair.uA*pair.uB);
                            den[pair.index*muBins + pair.muIndex] += pair.cosA*pair.cosB*pair.cosAB;
                        }
                    }
                }
                break;
            case 2:
                printf("Calculating psi_3 estimator components.\n");
                #pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) reduction(+:num[:vecLength]) reduction(+:den[:vecLength])
                for (i = 0; i < drows; i++) {
                    for (j = 0; j < rrows; j++) {
                        /* do the pair count */
                        pair = calcCorr_xyz_smu(sample1[i][0], sample1[i][1], sample1[i][2],
                                                sample2[j][0], sample2[j][1], sample2[j][2],
                                                sample1[i][3], sample1[i][4], sample1[i][5],
                                                sample2[j][3], sample2[j][4], sample2[j][5], swidth, muwidth);
                        if (pair.index < numBins && pair.muIndex < muBins && pair.index >= 0 && pair.muIndex >= 0) {
                            num[pair.index*muBins + pair.muIndex] += pair.cosB*pair.uB;
                            den[pair.index*muBins + pair.muIndex] += pair.cosB*pair.cosB;
                        }
                    }
                }
                break;
            case 3:
                printf("Calculating xi_GG estimator components.\n");
                #pragma omp parallel for num_threads(nthreads) collapse(2) private(i,j,pair) reduction(+:num[:vecLength]) reduction(+:den[:vecLength])
                for (i = 0; i < drows; i++) {
                    for (j = 0; j < rrows; j++) {
                        /* do the pair count */
                        pair = calcCorr_xyz_smu(sample1[i][0], sample1[i][1], sample1[i][2],
                                                sample2[j][0], sample2[j][1], sample2[j][2],
                                                sample1[i][3], sample1[i][4], sample1[i][5],
                                                sample2[j][3], sample2[j][4], sample2[j][5], swidth, muwidth);
                        if (pair.index < numBins && pair.muIndex < muBins && pair.index >= 0 && pair.muIndex >= 0) {
                            num[pair.index*muBins + pair.muIndex] += 1.0;
                            den[pair.index*muBins + pair.muIndex] += 0.0;
                        }
                    }
                }
                break;
            default:
                printf("Error occured: expected estimator = 'psi1', 'psi2', 'psi3' or 'xiGG' as an input.\n");
                break;
        }
    }
    return results;
    /* We can compile the individual .c files: gcc-13 -fopenmp -fPIC -c foo1.c 
    and then compile all of them into a shared object as: gcc-13 -fopenmp -shared -o object.so -fPIC foo1.o foo2.o*/
}
