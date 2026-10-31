#include "EXF_OPTIONS.h"
C     Reference driver: MITgcm pkg/exf EXF_BULKFORMULAE (MITgcm git 27256b2) on the points of
C     validation/bulk_points/points.py, for the LY04 comparison of N2b criterion 1 (step 7).
C     exf_bulkformulae.F is compiled unchanged with ALLOW_BULK_LARGEYEAGER04 and
C     EXF_CALC_ATMRHO (rho_a from apressure, atemp, aqh).  The state it reads lives in common
C     blocks (EXF_FIELDS.h, DYNVARS.h, EXF_PARAM.h), so this driver fills them by hand as
C     exf_wind.F and exf_readparms.F would: one point on every second row (sNx = 1,
C     sNy = 2 npts) so the C-grid current 0.5 (uVel(i)+uVel(i+1)) of each point is its own;
C     the rows between have atemp = 0 and are skipped (exf_bulkformulae.F:283, :357, :471).
C     Parameters (niter_bulk = 2 is a PARAMETER of EXF_CONSTANTS.h): the exf_readparms.F:330-363 defaults, except hu = ht = hq = 10 m (all
C     heights 10 m, as the FMS table) and rhoConstFresh = 1000 (evap in m/s with 1000 kg/m3).
C     wspeed = |u_atm - u_ocean| (exf_wind.F:129-149, useRelativeWind), sh = max(wspeed, umin)
C     (exf_wind.F:267).
C     Usage: mitgcm_exf_driver in.txt out.txt ; out: "ustress vstress hs hl evap" per point
C     (MITgcm signs: stress on the ocean, hs and hl positive into the ocean, evap > 0 = loss).
      PROGRAM MITGCM_EXF_DRIVER
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "DYNVARS.h"
#include "EXF_PARAM.h"
#include "EXF_FIELDS.h"
#include "EXF_CONSTANTS.h"
      _RL exf_Tsf(1-OLx:sNx+OLx,1-OLy:sNy+OLy,nSx,nSy)
      _RL ua, va, ta, qa, ts, pa, uo, vo
      INTEGER n, k, j
      CHARACTER*512 fin, fout
      CALL GET_COMMAND_ARGUMENT(1, fin)
      CALL GET_COMMAND_ARGUMENT(2, fout)
      myBxLo(1) = 1
      myBxHi(1) = 1
      myByLo(1) = 1
      myByHi(1) = 1
      useAtmWind = .TRUE.
      useRelativeWind = .TRUE.
      usingPCoords = .FALSE.
      wspeedfile = ' '
      rhoConstFresh = 1000. _d 0
C     exf_readparms.F:330-363 (and 357 cDalton, 603 exf_scal_BulkCdn, 1044 hq = ht)
      gravity_mks = 9.81 _d 0
      atmrho = 1.200 _d 0
      atmcp = 1005.000 _d 0
      flamb = 2500000.000 _d 0
      cvapor_fac = 640380.000 _d 0
      cvapor_exp = 5107.400 _d 0
      humid_fac = 0.606 _d 0
      gamma_blk = 0.010 _d 0
      saltsat = 0.980 _d 0
      cdrag_1 = 0.0027000 _d 0
      cdrag_2 = 0.0001420 _d 0
      cdrag_3 = 0.0000764 _d 0
      cstanton_1 = 0.0327 _d 0
      cstanton_2 = 0.0180 _d 0
      cDalton = 0.0346 _d 0
      zolmin = -100.000 _d 0
      psim_fac = 5.000 _d 0
      zref = 10.000 _d 0
      hu = 10.000 _d 0
      ht = 10.000 _d 0
      hq = 10.000 _d 0
      umin = 0.5 _d 0
      exf_scal_BulkCdn = 1.0 _d 0
      OPEN(10, FILE=fin, STATUS='OLD')
      READ(10, *) n
      IF ( 2*n .NE. sNy ) STOP 'SIZE.h sNy must be 2 npts'
      DO j = 1-OLy, sNy+OLy
        atemp(1,j,1,1) = 0. _d 0
        exf_Tsf(1,j,1,1) = 300. _d 0
        DO k = 1-OLx, sNx+OLx
          uVel(k,j,1,1,1) = 0. _d 0
          vVel(k,j,1,1,1) = 0. _d 0
        ENDDO
      ENDDO
      DO k = 1, n
        READ(10, *) ua, va, ta, qa, ts, pa, uo, vo
        j = 2*k - 1
        uwind(1,j,1,1) = ua
        vwind(1,j,1,1) = va
        atemp(1,j,1,1) = ta
        aqh(1,j,1,1) = qa
        apressure(1,j,1,1) = pa
        exf_Tsf(1,j,1,1) = ts
        uVel(1,j,1,1,1) = uo
        uVel(2,j,1,1,1) = uo
        vVel(1,j,1,1,1) = vo
        vVel(1,j+1,1,1,1) = vo
        wspeed(1,j,1,1) = SQRT((ua-uo)**2 + (va-vo)**2)
        sh(1,j,1,1) = MAX(wspeed(1,j,1,1), umin)
      ENDDO
      CLOSE(10)
      CALL EXF_BULKFORMULAE( exf_Tsf, 0. _d 0, 0, 1 )
      OPEN(11, FILE=fout, STATUS='REPLACE')
      DO k = 1, n
        j = 2*k - 1
        WRITE(11, '(5ES26.17E3)') ustress(1,j,1,1), vstress(1,j,1,1),
     &        hs(1,j,1,1), hl(1,j,1,1), evap(1,j,1,1)
      ENDDO
      CLOSE(11)
      END
