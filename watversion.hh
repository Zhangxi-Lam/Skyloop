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
    return (char*)"Last Changed Date: 2014-08-04 04:43:38 +0800 (一, 04 8月 2014)";
  if(c=='k') 
    return (char*)"Linux";
  if(c=='n') 
    return (char*)"HPC";
  if(c=='q') 
    return (char*)"2.6.32-431.23.3.el6.x86_64";
  if(c=='v') 
    return (char*)"#1 SMP Tue Jul 29 11:12:56 CDT 2014";
  if(c=='m') 
    return (char*)"x86_64";
  if(c=='p') 
    return (char*)"x86_64";
  if(c=='t') 
    return (char*)"2014年 11月 28日 星期五 13:23:52 UTC ";
  else         return (char*)"wat-6.0.0";
}

#endif
