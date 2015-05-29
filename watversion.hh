#ifndef WATVERSION_HH
#define WATVERSION_HH

/* WAT version information */

inline char* watversion(char c='s')
{
  if(c=='s') 
    return (char*)"wat600";
  if(c=='d') 
    return (char*)"600";
  if(c=='r') 
    return (char*)"3600M";
  if(c=='x') 
    return (char*)"Last Changed Date: 2014-08-04 04:43:38 +0800 (Mon, 04 Aug 2014)";
  if(c=='k') 
    return (char*)"Linux";
  if(c=='n') 
    return (char*)"HPC";
  if(c=='q') 
    return (char*)"2.6.32-504.1.3.el6.x86_64";
  if(c=='v') 
    return (char*)"#1 SMP Tue Nov 11 14:19:04 CST 2014";
  if(c=='m') 
    return (char*)"x86_64";
  if(c=='p') 
    return (char*)"x86_64";
  if(c=='t') 
    return (char*)"Sun May 24 04:21:53 UTC 2015 ";
  else         return (char*)"wat-6.0.0";
}

#endif
