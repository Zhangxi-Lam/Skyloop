{
double latH = 46.  + 27./60. + 18.53/3600.;
double latL = 30.  + 33./60. + 46.42/3600.;
double lonH = 119. + 24./60. + 27.57/3600.;
double lonL = 90.  + 46./60. + 27.27/3600.;

double A = (lonH - lonL)*2*PI/360.;
double c = (90-latH)*2*PI/360.;
double b = (90-latL)*2*PI/360.;

double tgBpC = cos(A/2)*cos(b/2-c/2)/sin(A/2)/cos(b/2+c/2); 
double tgBmC = cos(A/2)*sin(b/2-c/2)/sin(A/2)/sin(b/2+c/2);

double BpC = atan(tgBpC)*360./2/PI;
double BmC = atan(tgBmC)*360./2/PI;
double B = BpC+BmC;
double C = BpC-BmC;
double a = acos(cos(b)*cos(c)+sin(b)*sin(c)*cos(A));

cout<<"sin A: "<<sin(a)/sin(A)<<endl;
cout<<"sin B: "<<sin(b)/sin(B*2*PI/360.)<<endl;
cout<<"sin C: "<<sin(c)/sin(C*2*PI/360.)<<endl;
cout<<"alpha: "<<a*360./2/PI/2.<<endl;


cout<<"angle A = "<<A*360./2/PI<<"   angle b = "<<b*360./2/PI<<"   angle c = "<<c*360./2/PI<<endl;
cout<<"angle H = "<<B<<"   angle L = "<<C<<endl;
cout<<"angle H = "<<180-(B+35.9993)<<"   angle L = "<<90-C+17.7164<<endl;

}




