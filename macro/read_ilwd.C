/*************************************************************/
/* Wavelet Analysis Tool                                     */
/* file read_ilwd.C                                          */
/*                                                           */
/* This macro file is for ROOT interactive environment.      */
/* Macro reads requested data object from ILWD file in       */
/* ASCII or binary format.	           		     */
/*                                                           */
/* Functions:                                                */
/* void* ReadILWD(char* f, char* n, char* t, int ins)        */
/* void* ReadILWD(int &k, char* f, char* n, char* t, int ins)*/
/* void* ReadILWD(char* n) - prints names of variables from  */
/* ILWD file
/*************************************************************/

#include <iostream.h>

void* ReadILWD( int & nd,
                char* file_name,
                char* var_name,
                char* var_type = "real_4",
                int ins = 1      // instance of variable
              )
{
  if ( file_name == "" )
  {
     cout <<" ReadILWD: file name must not be empty!"<<endl;
     return NULL;
  }

  if ( var_name == "" )
  {
     cout <<" ReadILWD: variable name must not be empty!"<<endl;
     return NULL;
  }

  if ( var_type == "" )
  {
     cout <<" ReadILWD: variable type must not be empty!"<<endl;
     return NULL;
  }

// open input file
  ifstream infile(file_name);
  if (!infile)
  {
    cout <<" Cannot open file "<<file_name<<endl;
    return NULL;
  }

// input buffer for blocks <>
  int bufsize = 4096;
  char *buf = new char[bufsize];

// pointers to area where data from file will be placed
  void *pvar = NULL;
  double *pdouble;
  float  *pfloat;
  unsigned int *pintu;

// variables to input unneeded data 
  double xdouble;
  float  xfloat;

  char c;
  char *pname;
  char *ptype;
  char *pdims;
  char *pbin;

  char *dbuf;            // pointer to data array for reading in binary format
  char *token;
  char *stoken = new char[128];
  bool new_ilwd;
  char cname[32][80];     // containers names
  char vname[1024];       // full variable name
  strcpy(cname[0],"");

  int ncon = 0;         // counter of ILWD containers
  int nblk[32];         // counts blocks for each depth
  int depth = 0;        // inclusion depth
  int iblk = 0;
  int ndim;
  int ndat;
  int nsize;
  bool fbin;            // binary format flag
  bool fbig;		// big-endian binary format flag

  while ( ! infile.eof()  )
  {
    while ( infile >> c ) { if (c == '<') break; }

    if ( infile.eof() ) return pvar;

    infile.getline(buf, bufsize, '>');
    iblk++;

    nsize = 0;

    if ( infile.fail() )
    {
       cout <<" ReadILWD failed: input block "<<iblk<<" length is over limit "
       << bufsize <<endl;
       return NULL;
    }

    if ( !strcmp(buf,"?ilwd?") ) // start of ILWD file
      continue;

    if ( !strncmp(buf,"ilwd",4) ) // start of ILWD container
    {
      depth++;
      nblk[depth] = 0;
      new_ilwd = true;
    }
    else
    {
      new_ilwd = false;

      if ( !strncmp(buf,"/",1) ) // end of block
      {
        if ( !strncmp(buf,"/ilwd",5) )  // end of ILWD container
          depth--;

        continue;
      }
      else      // not end of block
      {
        nblk[depth]++;
        ndim = 1;                       // at least one number should be
      }
    }

// break string into sequence of tokens and parse
    token=strtok(buf, " ='");

    ptype = token;
    pname = "";                      // default name
    fbin = false;

// parse block
    while( token )
    {
       if ( !strcmp(token,"name") )  // look for data names
       {
          pname = strtok(NULL, " ='");
       }

       if ( !strcmp(token,"dims") )  // look for data dimension
       {
          ndim = atoi(strtok(NULL, " ='"));
       }

       if ( !strcmp(token,"format") )  // look for data format
       {
          token = strtok(NULL, " ='");
          if ( !strcmp(token,"binary") )
             fbin = true;
       }

       if ( !strcmp(token,"size") )  // look for size token
       {
          nsize = atoi(strtok(NULL, " ='"));
       }

       if ( !strcmp(token,"byteorder") )  // look for data format
       {
          token = strtok(NULL, " ='");
          if ( !strcmp(token,"big") )
             fbig = true;
       }

       token=strtok(NULL, " ='"); // take next token

    } // end of while(token)

// new ilwd container or not?
    if ( new_ilwd )
       strcpy(cname[depth], pname ); // save container name

    else // find full variable name and it's attributes
    {
      strcpy(vname,"");            // make full variable name

      for (int j=1; j<=depth; j++)
      {
        strcat(vname, cname[j]);
        strcat(vname, ":");
      }
      strcat(vname, pname);

      cout <<" depth:"<<depth<<", blk:"<<nblk[depth]<<", name='" <<vname
           <<"'"<<", type='"<<ptype<<"'" ;

      if ( ndim > 1 ) cout <<", dims="<<ndim;

      if ( fbin ) cout <<", binary ";

      if ( !strcmp(ptype,"lstring") ) cout <<", size="<<nsize;

      cout<<endl;
    }

//cout <<" compare "<< vname <<" and "<<var_name<<endl;
//cout <<" compare "<< ptype <<" and "<<var_type<<endl;

    if ( !strcmp(vname, var_name) && !strcmp(ptype, var_type) )
// input numeric data
    {
      ndat = ndim;
      if ( ndat )
      {
        switch ( var_type ) // read in variable value according to type
        {
          case "complex_8":
            ndat = 2*ndat;

          case "real_4":
            pfloat = new float[ndat];
            dbuf = pfloat;

            if (fbin) infile.read(dbuf, ndat*4);
            else for (int i=0; i<ndat; i++) infile >> pfloat[i];

            break;

          case "complex_16":
            ndat = 2*ndat;

          case "real_8":

            pdouble = new double[ndat];
            dbuf = pdouble;

            if (fbin) infile.read(dbuf, ndat*8);
            else for (int i=0; i<ndat; i++) infile >> pdouble[i];

            break;

          case "int_4u":

            pintu = new unsigned int[ndat];
            dbuf = pintu;

            if (fbin) infile.read(dbuf, ndat*4);
            else for (int i=0; i<ndat; i++) infile >> pintu[i];

            break;

          case "lstring":

            dbuf = new char[nsize];
            infile.read(dbuf, nsize);
         }

//         if ( fbin && need_swap(fbig) ) byte_swap(dbuf,nbyte);

         if ( infile.fail() )
         {
            cout <<" ReadILWD: input for "<<vname<<" has failed! "<<endl;
            return NULL;
         }
         else 
         {
            nd = ndat;
            return dbuf;
         }

      }
    }
    else
// skipping numeric fields
    if ( ndim )
    {
      if ( fbin )
      {
        if ( !strcmp(ptype,"real_4") ) infile.ignore(ndim*4);
        if ( !strcmp(ptype,"real_8") ) infile.ignore(ndim*8);
        if ( !strcmp(ptype,"int_4u") ) infile.ignore(ndim*4);
        if ( !strcmp(ptype,"complex_8") ) infile.ignore(ndim*8);
        if ( !strcmp(ptype,"complex_16") ) infile.ignore(ndim*16);
      }
      else
      {
        if ( !strcmp(ptype,"real_4") )
          for (int m=0; m < ndim; m++) infile >> xfloat;
        if ( !strcmp(ptype,"real_8") )
          for (int m=0; m < ndim; m++) infile >> xdouble;
        if ( !strcmp(ptype,"complex_8") )
          for (int m=0; m < 2*ndim; m++) infile >> xfloat;
        if ( !strcmp(ptype,"complex_16") )
          for (int m=0; m < 2*ndim; m++) infile >> xdouble;
      }
    }
// skipping 'lstring'
    if ( !strcmp(ptype,"lstring") && (nsize > 0)) infile.ignore(nsize);

  }
  return NULL;
}

void* ReadILWD( 
                char* file_name,
                char* var_name,
                char* var_type = "real_4",
                int ins = 1
              )
{
  int n = 0;
  return ReadILWD(n, file_name, var_name, var_type, ins);
}

// function prints out ILWD variable's names

void* ReadILWD( char* file_name)
{
  if ( file_name == "" )
  {
     cout <<" ReadILWD: file name must not be empty!"<<endl;
     return NULL;
  }

  ifstream infile(file_name);
  if (!infile)
  {
    cout <<" Cannot open file "<<file_name<<endl;
    return NULL;
  }

  int bufsize = 2048;
  char *buf = new char[bufsize];

  void *pvar = NULL;
  double xdouble;
  float  xfloat;
  char c;
  char *pname;
  char *ptype;
  char *pdims;
  char *pbin;
  char *dbuf;            // pointer to data array for reading in binary format
  char *token;
  char *stoken = new char[128];
  bool new_ilwd;
  char cname[32][80];     // containers names
  char vname[1024];       // full variable name
  strcpy(cname[0],"");

  int ncon = 0;         // counter of ILWD containers
  int nblk[32];         // counts blocks for each depth
  int depth = 0;	// inclusion depth
  int iblk = 0;
  int ndim;
  int nsize;
  bool fbin;		// binary format flag

  while ( ! infile.eof()  )
  {
    while ( infile >> c ) { if (c == '<') break; }

    if ( infile.eof() ) return pvar;

    infile.getline(buf, bufsize, '>');
    iblk++;

    nsize = 0;

    if ( infile.fail() )
    {
       cout <<" ReadILWD failed: input block "<<iblk<<" length is over limit "
       << bufsize <<endl;
       return NULL;
    }

    if ( !strcmp(buf,"?ilwd?") ) // start of ILWD file
      continue;

    if ( !strncmp(buf,"ilwd",4) ) // start of ILWD container
    {
      depth++;
      nblk[depth] = 0;
      new_ilwd = true;
    }
    else
    {
      new_ilwd = false;

      if ( !strncmp(buf,"/",1) ) // end of block
      {
        if ( !strncmp(buf,"/ilwd",5) )  // end of ILWD container
          depth--;

        continue;
      }
      else	// not end of block
      {
        nblk[depth]++;
        ndim = 1;			// at least one number should be
      }
    }

// break string into sequence of tokens and parse
    token=strtok(buf, " ='");

    ptype = token;
    pname = "";			     // default name
    fbin = false;

// parse block
    while( token )
    {
       if ( !strcmp(token,"name") )  // look for data names
       {
          pname = strtok(NULL, " ='");
       }

       if ( !strcmp(token,"dims") )  // look for data dimension
       {
          ndim = atoi(strtok(NULL, " ='"));
       }

       if ( !strcmp(token,"format") )  // look for data format
       {
          token = strtok(NULL, " ='");
          if ( !strcmp(token,"binary") )
             fbin = true;
       }

       if ( !strcmp(token,"size") )  // look for size token
       {
          nsize = atoi(strtok(NULL, " ='"));
       }

       token=strtok(NULL, " ='"); // take next token
    } // end of while( token )

    if ( new_ilwd )
       strcpy(cname[depth], pname ); // save container name
    else
    {
      strcpy(vname,"");            // make full variable name

      for (int j=1; j<=depth; j++)
      {
        strcat(vname, cname[j]);
        strcat(vname, ":");
      }

      cout <<" depth:"<<depth<<", blk:"<<nblk[depth]<<", name='" <<vname
           <<pname<<"'"<<", type='"<<ptype<<"'" ;

      if ( ndim > 1 ) cout <<", dims="<<ndim;

      if ( fbin ) cout <<", binary ";

      if ( !strcmp(ptype,"lstring") ) cout <<", size="<<nsize;

      cout<<endl;
    }

// skipping numeric fields
    if ( ndim ) 
    {
      if ( fbin )
      {
        if ( !strcmp(ptype,"real_4") ) infile.ignore(ndim*4);
        if ( !strcmp(ptype,"real_8") ) infile.ignore(ndim*8);
        if ( !strcmp(ptype,"int_4u") ) infile.ignore(ndim*4);
        if ( !strcmp(ptype,"complex_8") ) infile.ignore(ndim*8);
        if ( !strcmp(ptype,"complex_16") ) infile.ignore(ndim*16);
      }
      else 
      {
        if ( !strcmp(ptype,"real_4") ) 
          for (int m=0; m < ndim; m++) infile >> xfloat;
        if ( !strcmp(ptype,"real_8") )
          for (int m=0; m < ndim; m++) infile >> xdouble;
        if ( !strcmp(ptype,"complex_8") )
          for (int m=0; m < 2*ndim; m++) infile >> xfloat;
        if ( !strcmp(ptype,"complex_16") )
          for (int m=0; m < 2*ndim; m++) infile >> xdouble;
      }
    }
// skipping 'lstring'
    if ( !strcmp(ptype,"lstring") && (nsize > 0)) infile.ignore(nsize);

  }
  return NULL;
}

