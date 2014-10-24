/***************************************************************************
                         watexception.cpp  -  description
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

#include "watexception.hh"
#include "TInterpreter.h"
#include "Getline.h"
#include "TException.h"
#include "Api.h"      
#include "TROOT.h"   

extern struct G__input_file G__ifile;

int watexception::errtype=0;

watexception::watexception(int type, const char *location, const char *msgfmt, ...) {

  errtype = type;
  gErrorIgnoreLevel = 0;
  gErrorAbortLevel  = kSysError+1;
  strcpy(this->errmsg,"");

  va_list ap;
  va_start(ap,va_(msgfmt));
  ErrorHandler(type, location, va_(msgfmt), ap);
  va_end(ap);
}

watexception::watexception(char* errmsg, int type) {

  errtype = type;
  gErrorIgnoreLevel = 0;
  gErrorAbortLevel  = kSysError+1;
  strcpy(this->errmsg,"");

  if (strlen(errmsg) < WAT_ERR_MSG_LEN-1) {
    strncpy((char*)this->errmsg,(char*)errmsg,strlen(errmsg)+1);
  } else {
    strncpy((char*)this->errmsg,(char*)errmsg,WAT_ERR_MSG_LEN-1);
    this->errmsg[0] = 0x0;
  }
}

watexception::~watexception() {
}

void
watexception::ErrorHandler(int level, const char *location, const char *fmt, va_list ap) {

  // General error handler function. It calls the user set error handler
  // unless the error is of type kFatal, in which case the
  // DefaultErrorHandler() is called which will abort the application.

  errtype = level;
  static Int_t buf_size = 2048;
  static char *buf = 0;

  char *bp;

again:
  if (!buf) buf = new char[buf_size];

  int n = vsnprintf(buf, buf_size, fmt, ap);
  // old vsnprintf's return -1 if string is truncated new ones return
  // total number of characters that would have been written
  if (n == -1 || n >= buf_size) {
     buf_size *= 2;
     delete [] buf;
     buf = 0;
     goto again;
  }
  if (level >= kSysError && level < kFatal)
     bp = Form("%s (%s)", buf, gSystem->GetError());
  else
     bp = buf;

  if (level != kFatal)
     DefaultErrorHandler(level, level >= gErrorAbortLevel, location, bp);
  else
     DefaultErrorHandler(level, kTRUE, location, bp);
}

void
watexception::DefaultErrorHandler(int level, Bool_t abort, const char *location, const char *errmsg) {

  // The default error handler function. It prints the message on errmsg and
  // if abort is set it aborts the application.

  if (level < gErrorIgnoreLevel) return;

  const char *type = 0;

  if (level >= kInfo)
     type = "Info";
  if (level >= kWarning)
     type = "Warning";
  if (level >= kError)
     type = "Error";
  if (level >= kBreak)
     type = "\n *** Break ***";
  if (level >= kSysError)
     type = "SysError";
  if (level >= kFatal)
     type = "Fatal";

  if (level >= kBreak && level < kSysError)
     sprintf(this->errmsg, "%s in <%s>: %s\n", type, location, errmsg);
//     sprintf(this->errmsg, "%s %s\n", type, errmsg);
  else if (!location || strlen(location) == 0)
     sprintf(this->errmsg, "%s: %s\n", type, errmsg);
  else
     sprintf(this->errmsg, "%s in <%s>: %s\n", type, location, errmsg);


  if (abort) {
     fprintf(stderr, "aborting\n");
     fflush(stderr);
     if (gSystem) {
        gSystem->StackTrace();
        gSystem->Abort();
     } else
        ::abort();
  }

  if (gROOT->GetApplication()!=NULL) {
    if (TString(gROOT->GetApplication()->GetName()).CompareTo("TRint")==0) {
      cerr << msg() << endl;
      if ((TROOT::Initialized()) && (level >= kBreak)) {
         printf("<watexception>: FILE:%s LINE:%d\n",G__ifile.name,G__ifile.line_number);
         Getlinem(kInit, "wat > ");
         gInterpreter->RewindDictionary();
         Throw(level);
      }
    }
  }
}
