/***************************************************************************
                         exception.hh  -  description
                             -------------------
    begin                : Mon Jan 30 2012
    copyright            : (C) 2012 by Gabriele Vedovato
    email                : vedovato@lnl.infn.it
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef WATEXCEPTION_H
#define WATEXCEPTION_H


/**
  *@author Gabriele Vedovato
  */

#include "TError.h"
#include "TApplication.h"
#include <stdio.h>
#include <iostream>  
#include "snprintf.h"
#include "TSystem.h"

#ifndef __CINT__
#include "Varargs.h"
#endif


using namespace std;   

#define WAT_ERR_MSG_LEN 2048


#define ExceptionDef(CLASS_NAME)							\
class CLASS_NAME##Exception : public watexception {					\
public:											\
  CLASS_NAME##Exception(int type, const char *location, const char *msgfmt, ...)	\
                       : watexception() {						\
    va_list ap;										\
    va_start(ap,va_(msgfmt));								\
    watexception::ErrorHandler(type, location, va_(msgfmt), ap);			\
    va_end(ap);										\
  }											\
  CLASS_NAME##Exception(char* msg, int type = 0) : watexception (msg, type) {}		\
  ~CLASS_NAME##Exception() {}								\
};

class watexception {
public:

  watexception(int type, const char *location, const char *msgfmt, ...);
  watexception(char* errmsg = const_cast<char*>(""), int type = 0);
  ~watexception();

  const char* msg() {return errmsg;}

  static int  error() {return errtype;}
  static void clear() {errtype=0;}

protected:

  void ErrorHandler(int level, const char *location, const char *fmt, va_list ap);

private:

  void DefaultErrorHandler(int level, Bool_t abort, const char *location, const char *errmsg);

  int  gErrorIgnoreLevel;
  int  gErrorAbortLevel;

  static int errtype;
  char errmsg[WAT_ERR_MSG_LEN];
};

#endif
