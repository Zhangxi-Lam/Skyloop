#define XIFO 4

#pragma GCC system_header

#include "cwb.hh"
#include "cwb2G.hh"
#include "config.hh"
#include "network.hh"
#include "wavearray.hh"
#include "TString.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TRandom.h"
#include "TComplex.h"

//!SUPERCLUSTER

void 
CWB_Plugin(TFile* jfile, CWB::config* cfg, network* net, WSeries<double>* x, TString ifo, int type)  {

// This plugin implements the standard supercluster stage (only 2G)

  cout << endl;
  cout << "-----> CWB_Plugin_supercluster.C" << endl;
  cout << "ifo " << ifo.Data() << endl;
  cout << "type " << type << endl;
  cout << endl;

  if(type==CWB_PLUGIN_CONFIG) {  
    cfg->scPlugin=true;  	// disable built-in supercluster function
  }

  if(type==CWB_PLUGIN_ISUPERCLUSTER) {

    cout << "type==CWB_PLUGIN_ISUPERCLUSTER" << endl;

    // import ifile
    void* gIFILE=NULL; IMPORT(void*,gIFILE)
    cout << "-----> CWB_Plugin_wavegraph.C -> " << " gIFILE : " << gIFILE << endl;
    TFile* ifile = (TFile*)gIFILE;

    // import ifactor
    int gIFACTOR=-1; IMPORT(int,gIFACTOR)
    cout << "-----> CWB_Plugin_wavegraph.C -> " << " gIFACTOR : " << gIFACTOR << endl;
    int ifactor = gIFACTOR;

    int nIFO = net->ifoListSize();			// number of detectors
    int rateANA=cfg->inRate>>cfg->levelR;
    int nRES = net->wdmListSize();			// number of resolution levels
    int lags = net->getifo(0)->lagShift.size();

    wavearray<double>* hot[NIFO_MAX];  			// temporary time series
    for(int i=0; i<nIFO; i++) hot[i] = net->getifo(i)->getHoT();

    int nevt = 0;
    int nnn = 0;
    int mmm = 0;
    size_t count = 0;
    netcluster  wc;
    netcluster* pwc;

    for(int j=0; j<(int)lags; j++) {

      int cycle = cfg->simulation ? ifactor : Long_t(net->wc_List[j].shift);

      // read cluster metadata
      if(ifile!=NULL) wc.read(ifile,"coherence","clusters",0,cycle);
      else            wc.read(jfile,"coherence","clusters",0,cycle);
      // read clusters from temporary job file, loop over TF resolutions
      if(ifile!=NULL) {
        for(int i=nRES-1; i>=0; i--)     // reverse loop is faster loading cluster (?)
          wc.read(ifile,"coherence","clusters",-1,cycle,rateANA>>(i+cfg->l_low));
      } else {
        for(int i=nRES-1; i>=0; i--)     // reverse loop is faster loading cluster (?)
          wc.read(jfile,"coherence","clusters",-1,cycle,rateANA>>(i+cfg->l_low));
      }
      if(!cfg->simulation) cout<<"process lag "   <<cycle<<" ..."<<endl;
      cout<<"loaded clusters|pixels: "<<wc.csize()<<"|"<<wc.size()<<endl;

      // supercluster analysis
      wc.supercluster('L',net->e2or,cfg->TFgap,false);  //likehood2G
      cout<<"super  clusters|pixels: "<<wc.esize(0)<<"|"<<wc.psize(0)<<endl;

      // release all pixels
      pwc = net->getwc(j);
      pwc->cpf(wc, false);

      net->setDelayIndex(hot[0]->rate());
      pwc->setcore(false);

      // apply cuts
      int psel = 0;
      while(1) {
        count = pwc->loadTDampSSE(*net, 'a', cfg->BATCH, cfg->LOUD);
        psel += net->subNetCut((int)j,cfg->subnet,NULL);
        int ptot = pwc->psize(1)+pwc->psize(-1);
        double pfrac = ptot>0 ? double(psel)/double(ptot) : 0.;
        cout<<"selected pixels: "<<psel<<", fraction: "<<pfrac<< endl;
        if(count<10000) break;
      }

      pwc->defragment(cfg->Tgap,cfg->Fgap);    // SK added defragmentation

      nevt = net->events();
      nnn += pwc->psize(-1);
      mmm += pwc->psize(1)+pwc->psize(-1);

      if(mmm) cout<<"events in the buffer: "<<net->events()<<"|"<<nnn<<"|"<<nnn/double(mmm)<<"\n";
      else    cout<<"events in the buffer: "<<net->events()<<"\n";

      // store cluster into temporary job file [NEWSS]
      pwc->write(jfile,"supercluster","clusters",0,cycle);
      pwc->write(jfile,"supercluster","clusters",-1,cycle);
      cout<<cycle<<"|"<<pwc->csize()<<"|"<<pwc->size()<<" ";cout.flush();

      pwc->clear();
      cout<<endl;
    }
  }

  return;
}

