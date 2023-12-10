#ifndef __CORR_H__
#define __CORR_H__

struct lin_corr {
	/* defining structure for auto-correlation function data */
	double cosA;
	double cosB;
	double cosAB;
	double vel_ij;
	double vel_ji;
	double uA;
	double uB;
	int index;
};

struct nonlin_corr {
	/* defining structure for auto-correlation function data */
	double cosA;
	double cosB;
	double cosAB;
	double cosMu;
	double uA;
	double uB;
	int index;
	int muIndex;
};

struct output {
	/* create a structure to store our results and allow us to return two vectors */
	double *num; /* pair count numerator */
	double *den; /* pair count denominator */
};

/* Functions that use datasets with radial velocity data - observational data, etc. */
struct lin_corr calcCorr(double x1, double y1, double z1, double x2, double y2, double z2,
                    double u1, double u2, double width);
struct nonlin_corr calcCorr_smu(double x1, double y1, double z1, double x2, double y2, double z2,
                        	double u1, double u2, double width, double mu_width);
struct output *pairCounter(int drows, int rrows, int equiv, const double sample1[drows][4], const double sample2[rrows][4], 
                const double weights1[drows], const double weights2[rrows], int smax, int swidth, const char* estimator, 
				int nthreads, int verbose);
struct output *pairCounter_smu(int drows, int rrows, int equiv, const double sample1[drows][4], const double sample2[rrows][4], 
                const double weights1[drows], const double weights2[rrows], int smax, int swidth, double muwidth, const char* estimator, 
				int nthreads, int verbose);

/* Functions that use datasets with 3D velocity data - simulation data, etc. */
struct lin_corr calcCorr_xyz(double x1, double y1, double z1, double x2, double y2, double z2,
                        double vx1, double vy1, double vz1, double vx2, double vy2, double vz2, double width);
struct nonlin_corr calcCorr_xyz_smu(double x1, double y1, double z1, double x2, double y2, double z2,
                            double vx1, double vy1, double vz1, double vx2, double vy2, double vz2, double width, double mu_width);
struct output *pairCounter_xyz(int drows, int rrows, int equiv, const double sample1[drows][6], const double sample2[rrows][6], 
                const double weights1[drows], const double weights2[rrows], int smax, int swidth, const char* estimator, int nthreads, int verbose);
int whichParam(const char* input);
int whichParam_smu(const char* input);
int whichParam_xyz(const char* input);
void free_structmemory(struct output *ptr);
void free_arraymemory(double *ptr);

#endif