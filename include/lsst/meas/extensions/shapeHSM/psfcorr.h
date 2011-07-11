/****************************************************************
Copyright 2003, 2004 Christopher Hirata: original code
          2007, 2009 Rachel Mandelbaum: minor modifications

For a copy of the license, see COPYING; for more information,
including contact information, see README.

This file is part of the meas_shape distribution.

Meas_shape is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

Meas_shape is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with meas_shape.  If not, see <http://www.gnu.org/licenses/>.
*******************************************************************/

/* structures and declarations for shape measurement */

/* object data type */

typedef double FTYPE;

typedef struct {
   double x0;           /* x centroid */
   double y0;           /* y centroid */
   double sigma;        /* width */
   double flux;         /* total flux */
   double e1;           /* + ellipticity */
   double e2;           /* x ellipticity */
   double responsivity; /* responsivity of ellipticity estimator */
   char meas_type;      /* type of ellipticity measurement:
                         *   'e' = Bernstein & Jarvis (2002) ellipticity
                         *   'g' = shear estimator = shear * responsivity
                         */
   double resolution;   /* resolution factor (0=unresolved, 1=resolved) */

    double mxx;
    double myy;
    double mxy;
} OBJECT_DATA;

/* rectangular image type */

typedef struct {
    long xmin; /* bounding box */
    long xmax; /* " */
    long ymin; /* " */
    long ymax; /* " */
    FTYPE **image; /* the actual map */
    int **mask; /* mask = 0 (masked) or 1 (unmasked) */
} RECT_IMAGE;

/* functions that the user will want to call from outside */

int **imatrix(long nrl, long nrh, long ncl, long nch);
void free_imatrix(int **m, long nrl, long nrh, long ncl, long nch);

void allocate_rect_image(RECT_IMAGE *A, long xmin, long xmax,
   long ymin, long ymax);

void deallocate_rect_image(RECT_IMAGE *A);

unsigned int general_shear_estimator(RECT_IMAGE *gal_image, RECT_IMAGE *PSF,
   OBJECT_DATA *gal_data, OBJECT_DATA *PSF_data, char *shear_est, unsigned long flags);

