
void readframes(char *filename, char *channel, wavearray<double> &w)
{

  FrFile *ifile;
  FrVect *v;
  FrameH *frame;
  int buffersize=300;
  char *buffer=new char[buffersize];

  long start, stop, fstart,  rstart, pstart;
  int frlen, nframes, len;
  double rate;
  unsigned int wcounter=0;

  FILE *framelist=fopen(filename,"r");

  //bzero(buffer,buffersize);
  fgets(buffer,buffersize,framelist);
  nframes=(int)atoi(buffer);
  //bzero(buffer,buffersize);
  fgets(buffer,buffersize,framelist);
  frlen=(int)atoi(buffer);
  //bzero(buffer,buffersize);
  fgets(buffer,buffersize,framelist);
  start=(long)atol(buffer);
  //bzero(buffer,buffersize);
  fgets(buffer,buffersize,framelist);
  stop=(long)atol(buffer);
  //bzero(buffer,buffersize);
  fgets(buffer,buffersize,framelist);
  rate=(double)atof(buffer);

  fprintf(stdout,"-------------\n");
  fprintf(stdout," filename=%s\n channel=%s rate=%f\n frames=%d frlen=%d start=%d stop=%d\n",
	  filename, channel, rate, nframes, frlen, start, stop);
  
  w.resize(int((stop-start)*rate));
  w.start(start);
  w.rate(rate);
  
  rstart=start;
  
  for(int i=0;i<nframes;i++)
    {
      //bzero(buffer,buffersize);
      fgets(buffer,buffersize,framelist);
      //fprintf(stdout,"%s",buffer);
      ifile=FrFileINew(buffer);
      if(ifile->error)
	{
	  fprintf(stderr,"Cannot read file %s\n",buffer);
	  break;
	}
      fstart=FrFileITStart(ifile);
      if(i==0)
	{
	  pstart=fstart;
	}
      else if(fstart-pstart!=frlen && fstart!=pstart)
	{
	  fprintf(stderr,"Gap in data detected pstart=%d fstart=%d\n",pstart,fstart);
	  FrFileIEnd(ifile);
	  break;
	}
      else if(fstart==pstart && fstart<stop)
	{
	  fprintf(stderr,"Not enough frame files fstart=%d\n",fstart);
	  w.resize(int((fstart-start)*rate));
	  break;
	}
      pstart=fstart;
      if(fstart>=stop)
	{
// 	  fprintf(stdout,"fstart=%d >= stop=%d\n",fstart,stop);fflush(stdout);
	  FrFileIEnd(ifile);
	  break;
	}
      if(fstart+frlen<=start)
	{
	  fprintf(stderr,"extra frame at the beginning fstart=%d frlen=%d start=%d\n",fstart,frlen,start);
	  FrFileIEnd(ifile);
	  continue;
	}

      if(fstart<start && start-fstart<frlen)
	{
	  len=frlen-(start-fstart);
	  rstart=start;
	}
      else if(fstart+frlen>stop)
	{
	  len=stop-fstart;
	  rstart=fstart;
	}
      else
	{
	  len=frlen;
	  rstart=fstart;
	}
      //fprintf(stdout,"i=%d rstart=%d len=%d\n",i,rstart,len);
      fflush(stdout);
      v=FrFileIGetV(ifile,channel, rstart, len);
      int debuglevel=1;
//        fprintf(stdout,">>>FrVectDump,%d\n",debuglevel);
//        FrVectDump(v,stdout,debuglevel);
//        fprintf(stdout,"<<<FrVectDump,%d\n",debuglevel);
//        fprintf(stdout,"<Statistics: %s>\n",FrVectStat(v));
      if(v==NULL)
	{
	  fprintf(stderr,"Cannot read file\n");
	  exit(1);
	}
      if(v->type!=FR_VECT_4R && v->type!=FR_VECT_8R)
	{
	  fprintf(stderr,"Wrong vector type %d\n",v->type);fflush(stderr);
	  exit(1);
	}
//       fprintf(stdout,"name=%s compress=%d nData=%ld nDim=%d nx=%ld\ndx=%f startX[0]=%f unitX[0]=%s unitY=%s GTime=%f rate=%f\n",
// 	      v->name,v->compress,v->nData,v->nDim,v->nx[0],v->dx[0],v->startX[0],v->unitX[0],v->unitY,v->GTime,1./v->dx[0]);
      
      long samples=long(len*rate);
//       fprintf(stdout,"samples=%f w->size()=%ld \n",samples,w->size());fflush(stdout);

      if(v->type==FR_VECT_4R)
	{
	  for(long j=0;j<samples;j++)
	    {
	      w.data[wcounter]=double(v->dataF[j]);
	      wcounter++;
	    }
	}
      if(v->type==FR_VECT_8R)
	{
          for(long j=0;j<samples;j++)
            {
              w.data[wcounter]=v->dataD[j];
              wcounter++;
            }
	}
      
      FrVectFree(v);
      FrFileIEnd(ifile);
    }
  fclose(framelist);
  delete [] buffer;

  if(rstart+len<stop)
    {
      fprintf(stderr,"Too few frames given: rstart=%d len=%d stop=%d nframes=%d\n",rstart,len,stop,nframes);
      w.resize(int((rstart+len-start)*rate));
    }
  if(wcounter!=w.size())
    {
      fprintf(stderr,"Inconsistency: wcounter=%d w.size()=%d\n",wcounter,w.size());
    }
}


