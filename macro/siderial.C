// given input time returns GMST time at UT1=0 
// Sergey Klimenko, University of Florida

double getGMST0(double gps) 
{
// Earth angular velocity (defines duration of the siderial day
// 1 siderial day = 23h 56m 04.0905s (from http://www.maa.mhn.de/Scholar/times.html)
  double gps2000 = 630720013;     // GPS time at 01/01/2000 00:00:00 UTC
  double GMST2000 = 24110.54841;  // GMST at UT1=0 on Jan 1, 2000

// UT1 = UTC + dT, where dT is less than 0.8 sec
// conversion between UT1 and GMST: http://www.cv.nrao.edu/~rfisher/Ephemerides/times.html
// GMST (in seconds at UT1=0) = 24110.54841 + 8640184.812866 * T + 0.093104 * T^2 - 0.0000062 * T^3
// where T is in Julian centuries from 2000 Jan. 1 12h  UT1 (JD2000 = 2451545.0)
// d = JD - 2451545.0, T = d / 36525

  double JD2000   = 2451545.0;    // Julian date on 12h Jan 1, 2000
  double GMST2000 = 24110.54841;  // GMST at UT1=0 on Jan 1, 2000

// estimate Julian date from gps
  double JDate = JD2000+int((gps-gps2000-12*3600)/3600./24.);
  double Jday = JDate - JD2000;      // Julian days
  double Jcen = Jday/36525.;      // Julian centuries
  double T    = Jcen;
  
  GMST = GMST2000 + 8640184.812866*T + 0.093104*T*T - 0.0000062*T*T*T; // sec
  GMST /= 3600.;
   
  return GMST-int(GMST/24)*24.; 
}


// get siderial time for given gps time

double getGMST(double gps) 
{
// Earth angular velocity (defines duration of the siderial day
// 1 siderial day = 23h 56m 04.0905s (from http://www.maa.mhn.de/Scholar/times.html)
  double omega = 7.292115090e-5;  // from http://hypertextbook.com/facts/2002/JasonAtkins.shtml
  double gps2000 = 630720013;     // GPS time at 01/01/2000 00:00:00 UTC
  double GMST2000 = 24110.54841;  // GMST at UT1=0 on Jan 1, 2000

// estimate GMST (hours) of UT1=0 from gps
  double GMST = GMST2000/3600. + (gps-gps2000)*omega*180/PI/15.; // sec
   
  return GMST-int(GMST/24)*24.; 
}

// get EFEC phi angle for given gps time and R.A

double getPHI(double gps, double RA) 
{
// Earth angular velocity (defines duration of the siderial day
// 1 siderial day = 23h 56m 04.0905s (from http://www.maa.mhn.de/Scholar/times.html)
  double omega = 7.292115090e-5;  // from http://hypertextbook.com/facts/2002/JasonAtkins.shtml
  double gps2000 = 630720013;     // GPS time at 01/01/2000 00:00:00 UTC
  double GMST2000 = 24110.54841;  // GMST at UT1=0 on Jan 1, 2000

// estimate GMST (hours) of UT1=0 from gps
  double GMST = GMST2000/3600. + (gps-gps2000)*omega*180/PI/15.; // sec
  GMST -= int(GMST/24)*24.; 
  return RA-GMST*15.; 
}

