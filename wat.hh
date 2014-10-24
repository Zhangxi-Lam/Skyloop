#ifndef WAT_HH
#define WAT_HH

#define NIFO_MAX 8

#if XIFO < 5
#define NIFO 4
#endif

#if XIFO > 4
#define NIFO 8
#endif

#if XIFO < 5
#define _NET(P1,P2) \
P1                              
#endif

#if XIFO > 4
#define _NET(P1,P2) \
P1                                      \
P2                              
#endif

#if XIFO == 1
#define NETX(P1,P2,P3,P4,P5,P6,P7,P8) \
P1                              
#endif

#if XIFO == 2
#define NETX(P1,P2,P3,P4,P5,P6,P7,P8) \
P1                              	 \
P2                              
#endif

#if XIFO == 3
#define NETX(P1,P2,P3,P4,P5,P6,P7,P8) \
P1                              	 \
P2                              	 \
P3                              
#endif

#if XIFO == 4
#define NETX(P1,P2,P3,P4,P5,P6,P7,P8) \
P1                              	 \
P2                              	 \
P3                              	 \
P4                             	  
#endif

#if XIFO == 5
#define NETX(P1,P2,P3,P4,P5,P6,P7,P8) \
P1                              	 \
P2                              	 \
P3                              	 \
P4                              	 \
P5                              
#endif

#if XIFO == 6
#define NETX(P1,P2,P3,P4,P5,P6,P7,P8) \
P1                              	 \
P2                              	 \
P3                              	 \
P4                              	 \
P5                              	 \
P6                              
#endif

#if XIFO == 7
#define NETX(P1,P2,P3,P4,P5,P6,P7,P8) \
P1                              	 \
P2                              	 \
P3                              	 \
P4                              	 \
P5                              	 \
P6                              	 \
P7                              
#endif

#if XIFO == 8
#define NETX(P1,P2,P3,P4,P5,P6,P7,P8) \
P1                              	 \
P2                              	 \
P3                              	 \
P4                              	 \
P5                              	 \
P6                              	 \
P7                              	 \
P8                              
#endif

#endif

;  // DO NOT REMOVE !!!
