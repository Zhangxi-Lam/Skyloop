//---------------------------------------------------
// WAT cluster class for network analysis
// S. Klimenko, University of Florida
//---------------------------------------------------
 
#define NETCLUSTER_CC
#include <time.h>
#include <iostream>
#include <stdexcept>
#include <set>
#include "netcluster.hh"
#include "detector.hh"
#include "network.hh"
#include "constants.hh"

#include "TCanvas.h"
#include "TH2F.h"
#include "TMath.h"
#include "TRandom3.h"
#include "TTreeFormula.h"

using namespace std;

ClassImp(netcluster)

// used to check duplicated entries in vector<int>
struct dupChecker_ {
    inline dupChecker_() : tmpSet() {}
    inline bool operator()(int i) {
        return tmpSet.insert(i).second;
    }
private:
    set<int> tmpSet;
};

// sort netpixel objects on time
int compare_PIX(const void *x, const void *y){   
   netpixel* p = *((netpixel**)x);
   netpixel* q = *((netpixel**)y);
   double a = double(p->time)/p->rate/p->layers 
            - double(q->time)/q->rate/q->layers;
   if(a > 0) return 1;
   if(a < 0) return -1;
   return 0;
}

// sort netpixel objects on likelihood (decreasind)
int compareLIKE(const void *x, const void *y){   
   netpixel* p = *((netpixel**)x);
   netpixel* q = *((netpixel**)y);
   double a = double(p->likelihood - q->likelihood);
   if(a > 0) return -1;
   if(a < 0) return 1;
   return 0;
}


// structure and comparison function needed for optimizing mchirp5
typedef struct{ double value; int type;} EndPoint;

int compEndP(const void* p, const void* q)
{  EndPoint* a = (EndPoint*)p;
   EndPoint* b = (EndPoint*)q;
   if(a->value  > b->value)return 1;
   return -1;
}

// constructors

netcluster::netcluster()
{
  this->clear();
  this->rate  = 0.;
  this->start = 0.;
  this->stop  = 0.;
  this->shift = 0.;
  this->bpp   = 1.;
  this->flow  = 0.;
  this->fhigh = 1.e6;
  this->run   = 0;
  this->nPIX  = 3;
  this->pair  = true;

  SetName("netcluster");
}

netcluster::netcluster(const netcluster& value)
{
   *this = value;
}

// destructor

netcluster::~netcluster(){this->clear();}

// copyfrom function - copy selected clusters and pixels
// does not copy error regions

size_t netcluster::cpf(const netcluster& value, bool optres, int nBIG)
{
// copy function (used in operator (=) as x.cpf(y,false)
//: copy content of y into x. 
// If no clusters are reconstructed - copy all pixels.
// If clusters are reconstructed - copy selected clusters.
// value:  netcluster object y
// optres: condition to select clusters: 
//         true  - copy selected clusters with core pixels 
//         false - copy all selected clusters
// nBIG:  min size of BIG clusters
//        =0 - copy all clusters regardles of their size
//        >0 - ignore BIG clusters with size >= nBIG 
//        <0 - copy BIG clusters with size >= nBIG 
   size_t n,m,k,K,ID;
   size_t nbig = abs(nBIG);
   size_t N = 0;
   size_t cid = 0;              // new cluster ID
   size_t pid = 0;              // new pixel ID
   int R;                       // resolution

   std::vector<int> id;         // list of pixels IDs in a cluster
   std::vector<int> vr;         // rate array
   std::vector<int> refI;       // reference integer
   std::vector<float> refF;     // reference float
   std::vector<vector_int>::const_iterator it;

   netpixel pix;

   this->clear();
   this->rate   = value.rate;
   this->start  = value.start;
   this->stop   = value.stop;
   this->shift  = value.shift;
   this->bpp    = value.bpp;
   this->run    = value.run;
   this->nPIX   = value.nPIX;
   this->flow   = value.flow;
   this->fhigh  = value.fhigh;
   this->pair   = value.pair;

   if(!value.cList.size()) {           // not clustered value
      this->pList = value.pList; 
      this->cluster();
      return this->pList.size(); 
   }

   for(it=value.cList.begin(); it!=value.cList.end(); it++){

      ID = value.pList[(*it)[0]].clusterID;
      K  = it->size();                  // cluster size
      if(value.sCuts[ID-1]>0) continue; // apply selection cuts
      if(nBIG>0 && K>nbig)  continue;   // do not copy BIG clusters
      if(nBIG<0 && K<nbig)  continue;   // copy only BIG clusters

      vr = value.cRate[ID-1];           // rate array
      m  = vr.size();                   // size of rate array
      if(m && optres) {                 // count only optimal resolution pixels
         for(k=0; k<K; k++) {              
            R = int(value.pList[(*it)[k]].rate+0.1);
            if(vr[0] == R) N++;
         }
      }
      else N += K;                      // count all pixels
   }
   this->pList.reserve(N+2); 

   for(it=value.cList.begin(); it!=value.cList.end(); it++){

       K  = it->size();
      ID = value.pList[(*it)[0]].clusterID;
      if(value.sCuts[ID-1]>0) continue; // apply selection cuts

      vr = value.cRate[ID-1];           // rate array
      m  = vr.size();                   // size of rate array

      cid++;                            // increment cluster ID
      id.clear();                       // clear pixel list
      
      for(k=0; k<K; k++) {              // loop over pixels
         pix = value.pList[(*it)[k]];
         R = int(value.pList[(*it)[k]].rate+0.1);
         
         if(m && optres) {              // skip if not optimal resolution
            if(vr[0] != R) continue;
         }
         
         pix.neighbors.clear();
         std::vector<int>().swap(pix.neighbors);  // free memory
         pix.clusterID = cid;
         n = this->pList.size();
         if(id.size()) {                
            pix.append(-1);              // append a neighbor on the left
            this->pList[n-1].append(1);  // append a neighbor on the right
         }
         id.push_back(pid++);            // update pixel list
         this->append(pix);              // append pixel to pList
      }
      
      if(!id.size()) cout<<"netcluster::cpf() error: empty cluster.";
      
      cList.push_back(id);                // fill cluster list
      cData.push_back(value.cData[ID-1]); // copy cluster metadata
      sCuts.push_back(0);                 // fill selection cuts
      cRate.push_back(vr);                // fill in rate array
      p_Ind.push_back(refI);              // create other arrays
      p_Map.push_back(refF);
      sArea.push_back(refF);
      nTofF.push_back(refI);
      cTime.push_back(value.cTime[ID-1]); // save supercluster central time
      cFreq.push_back(value.cFreq[ID-1]); // save supercluster frequency
   }      
   return this->pList.size();
}

// netcluster::operator - copy only selected clusters

netcluster& netcluster::operator=(const netcluster& value)
{
  this->cpf(value,false);
  return *this;
}


size_t netcluster::setcore(bool core, int id)
{
// set pixel core field to be equal to the input parameter core
// reset cluster type to 0 except for rejecter clusters (type=1)
   const vector_int* v;
   size_t i,k,K;

   for(i=0; i<this->cList.size(); i++){
      
     v = &(this->cList[i]);
     K = v->size();

     for(k=0; k<K; k++) {              
       if(id && this->pList[(*v)[k]].clusterID != id) continue; 
       this->pList[(*v)[k]].core = core;
     }
   }      
   return this->size();
}

size_t netcluster::select(size_t Ko, double Eo)
{
// select clusters 
// Ko - minimum accepted number of pixels
// Eo - minimum accepted energy

   if(!this->pList.size()) return 0;

   vector_int* v;
   size_t i,k,K;
   size_t nn = 0;

   for(i=0; i<this->cList.size(); i++){
      
     v = &(this->cList[i]);
     K = v->size();

     double E = 0;

     for(k=0; k<K; k++)
	if(this->pList[(*v)[k]].core)
	   E += this->pList[(*v)[k]].likelihood;

     if(K<Ko && E<Eo) this->ignore(i+1);
     else nn++;
   }
   return nn;
}


// delink superclusters)
size_t netcluster::delink()
{
// delink superclusters
// - destroy supercluster neighbors links 
// - preserve links for pixels with the same wavelet resolution

   size_t K = this->pList.size();
   size_t n,k,N;

   if(!K) return 0;

   std::vector<int> v;
   std::vector<int>* u;
   netpixel* q;
   netpixel* p;

   for(k=0; k<K; k++) {
      q = &(this->pList[k]);
      u = &(this->pList[k].neighbors);
      v = *u;
      u->clear();
      N = v.size(); 
      for(n=0; n<N; n++) { 
	 p = q+v[n];          // pixel index in pList
	 if(p->clusterID != q->clusterID) {
	   cout<<"netcluster::delink(): cluster ID mismatch"<<endl;
	   continue;
	 }
         // restore neighbors for the same resolution
	 if(p->rate == q->rate) q->append(v[n]);  
      }
   }
   return this->cluster();
}


size_t netcluster::cluster(int kt, int kf)
{
// produce time-frequency clusters at a single TF resolution
// any two pixels are associated if they are closer than both kt/kf
// samples in time/ftrequency 
   size_t i,j;
   size_t n = this->pList.size();
   if(n==0)   return 0;
   size_t M = this->pList[0].layers;
   double R = this->pList[0].rate;
   
   int index;

   if(kt<=0) kt = 1;
   if(kf<=0) kf = 1;
   if(!n) return 0;
   if(n==1) return this->cluster();

   netpixel** pp = (netpixel**)malloc(n*sizeof(netpixel*));
   netpixel* p;
   netpixel* q;
  
   for(i=0; i<n; i++) { pp[i] = &(pList[i]); pp[i]->neighbors.clear();} 

   qsort(pp, n, sizeof(netpixel*), &compare_PIX);         // sorted pixels

   for(i=0; i<n; i++) {
      p = pp[i];
      bool isWavelet = (size_t(this->rate/R+0.1) == pp[i]->layers) ? true : false;	// wavelet : WDM

      if(R != p->rate || M != p->layers) 
	cout<<"netcluster::cluster(int,int) applied to mixed pixel list"<<endl;
      
      for(j=i+1; j<n; j++){
	 q = pp[j];
         if(isWavelet) {
           index = int(q->time) - int(p->time);   	// wavelet
         } else {
           index = int(q->time/M) - int(p->time/M);	// WDM
         }

	 if(index < 0) cout<<"netcluster::cluster(int,int) sort error"<<endl;
         if(isWavelet) {
           if(index/M > kt) break; 			// wavelet
         } else {
           if(index > kt) break;			// WDM
         }

	 if(abs(int(q->frequency) - int(p->frequency)) <= kf) {     // set neighbours
	    p->append(q-p);  // insert in p
	    q->append(p-q);  // insert in q
	 }
      }
   }

   free(pp);
   return this->cluster();
}


size_t netcluster::cluster()
{
// perform cluster analysis
// this function initializes the netcluster data structures and
// call recursive cluster(netpixel) function to break pixel set into
// clusters.
   size_t volume;
   size_t i,m;
   vector<int> refMask;
   vector<int> refRate;
   vector<float> refArea; 
   clusterdata refCD;
   size_t nCluster = 0;
   size_t n = pList.size();

   if(!pList.size()) return 0;

   // clear/initialize netcluster metadata

   cList.clear();
   cData.clear();
   sCuts.clear();
   cRate.clear();
   cTime.clear();
   cFreq.clear();
   sArea.clear();
   p_Ind.clear();
   p_Map.clear();
   nTofF.clear();

   for(i=0; i<n; i++) pList[i].clusterID = 0;   

   // loop over pixels

   for(i=0; i<n; i++){
      if(pList[i].clusterID) continue;   // pixel is clustered
      pList[i].clusterID = ++nCluster;   // new seed pixel - set ID
      volume = cluster(&pList[i]);       // cluster and return number of pixels
      refMask.clear();                   // to the end of the loop
      cRate.push_back(refRate);          // initialize cluster metadata
      cTime.push_back(-1.);
      cFreq.push_back(-1.);
      p_Ind.push_back(refRate);
      p_Map.push_back(refArea);
      sArea.push_back(refArea);
      nTofF.push_back(refRate);
      refMask.resize(volume);
      cList.push_back(refMask);
      cData.push_back(refCD);
      sCuts.push_back(0);
   }

   vector<vector_int>::iterator it;
   nCluster = 0;
   if(!cList.size()) return 0;

   for(it=cList.begin(); it != cList.end(); it++) {
      nCluster++;

      m = 0;
      for(i=0; i<n; i++) { 
	if(pList[i].clusterID == nCluster) (*it)[m++] = i;
      }

      if(it->size() != m) { 
	 cout<<"cluster::cluster() size mismatch error: ";
	 cout<<m<<" size="<<it->size()<<" "<<nCluster<<endl;
      }
   }
   return nCluster;
}

size_t netcluster::cluster(netpixel* p)
{
// recursive clustering function used in cluster() 
// it check neighbors and set the cluster ID field
   size_t volume = 1;
   int i = p->neighbors.size();
   netpixel* q;

   while(--i >= 0){
      q = p + p->neighbors[i];
      if(!q->clusterID){
	  q->clusterID = p->clusterID;
	  volume += cluster(q);
      }
   }
   return volume;
}



size_t netcluster::cleanhalo(bool keepid)
{
// remove halo pixels from the pixel list

   if(!pList.size() || !cList.size()) return 0;

   size_t i,n,ID;
   size_t cid = 0;              // new cluster ID
   size_t pid = 0;              // new pixel ID

   netpixel* pix = NULL;
   vector<vector_int>::iterator it;
   std::vector<int> id;         // list of pixels IDs in a cluster
   std::vector<int>* pr;        // pointer to rate array

   netcluster x(*this);

   this->clear();

   for(it=x.cList.begin(); it != x.cList.end(); it++) {  // loop over clusters

      pix = &(x.pList[((*it)[0])]);
      ID  = pix->clusterID;
      if(x.sCuts[ID-1]>0) continue;            // apply selection cuts
      n  = x.cRate.size(); 
      pr = n ? &(x.cRate[ID-1]) : NULL;        // pointer to the rate array

      cid++;
      id.clear();
      for(i=0; i<it->size(); i++) {            // loop over pixels in the cluster
	 pix = &(x.pList[((*it)[i])]);
	 if(pix->core) {
	    pix->clusterID = keepid ? cid : 0;
	    pix->neighbors.clear();
	    std::vector<int>().swap(pix->neighbors);  // free memory
	    id.push_back(pid++);
	    this->append(*pix);                // fill pixel list
	 }
      }

      i = id.size();
      if(!i) cout<<"netcluster::cleanhalo() error: empty cluster.";
      
      if(keepid) { 
	 cList.push_back(id);               // fill cluster list
	 sCuts.push_back(0);                // fill selection cuts
         cData.push_back(x.cData[ID-1]);    // fillin cluster metadata
	 if(pr) cRate.push_back(*pr);       // fill in rate array
      }

      if(i<2) continue;
      
      while(--i > 0) {                     // set neighbors
	 pList[id[i-1]].append(id[i]-id[i-1]);
	 pList[id[i]].append(id[i-1]-id[i]);
      }

      
   }
   return pList.size();
}

size_t netcluster::addhalo(bool keepid)
{
// add halo pixels to the pixel list
// may not work with WDM

   if(!pList.size() || !cList.size()) return 0;

   size_t i,j, nIFO, ID;
   int k,n,m,L,I,TIME; 
   int ML,MH,mL,mH,mF,iT;
   int N = int((stop-start)*rate+0.1);
   size_t cid = 0;              // new cluster ID
   size_t pid = 0;              // new pixel ID

   netpixel* pix = NULL;
   netpixel* PIX = NULL;
   vector<vector_int>::iterator it;
   std::vector<int> id;         // list of pixels IDs in a cluster
   std::vector<int>* pr;        // pointer to rate array
   bool skip = false;
   int index[NIFO];
   Wavelet W;

   netcluster x(*this);

   this->clear();

   for(it=x.cList.begin(); it != x.cList.end(); it++) {  // loop over clusters

      pix = &(x.pList[((*it)[0])]);
      ID  = pix->clusterID;
      if(x.sCuts[ID-1]>0) continue;                  // apply selection cuts
      n  = x.cRate.size(); 
      pr = n ? &(x.cRate[ID-1]) : NULL;              // pointer to the rate array

      cid++;
      id.clear();                                    // clear list of pixel IDs 

      for(i=0; i<it->size(); i++) {                  // loop over pixels in the cluster
	 pix = &(x.pList[((*it)[i])]);
	 I = int(pix->layers);                       // number of wavelet bands (binary wavelet)
	 L = int(log(double(I))/log(2.)+0.1);        // wavelet decomposition level (binary wavelet)
	 nIFO = pix->size();

	 ML = size_t(2*x.flow/pix->rate+0.1);        // low frequency index
         if(ML == 0) ML += 1;
	 MH = size_t(2*x.fhigh/pix->rate+0.1);       // high frequency index
         if(MH == (int)pix->layers) MH -= 1;
	 iT = pix->time;
	 mF = pix->frequency;
	 mL = mF<=ML ? ML : mF-1;                    // low boundary for index m
	 mH = mF>=MH ? MH : mF+1;                    // high boundary for index m

	 for(j=0; j<nIFO; j++) {                     // store original index
	   index[j] = pix->data[j].index;
	 }

	 for(n=-1; n<=1; n++) {
	   for(m=mL; m<=mH; m++) {
	     TIME = iT+n*I-W.convertF2O(L,mF)+W.convertF2O(L,m);    // time index
	     skip = false;
	     for(k=0; k<int(id.size()); k++) {                      // check for duplicates
	       PIX = &pList[id[k]];
	       if(pix->rate != PIX->rate) continue;
	       if(TIME<0 || TIME>=N) {
		 cout<<"netcluster error: time index is out of range\n"; 
		 exit(1); 
	       }
	       if(TIME==int(PIX->time) && m==int(PIX->frequency)) skip=true;
	     }
	     if(skip) continue;

	     pix->clusterID = keepid ? cid : 0;
	     pix->neighbors.clear();
	     std::vector<int>().swap(pix->neighbors);  // free memory
	     pix->time = TIME;
	     pix->frequency = m;
	     pix->core = true;
	     
	     TIME -= iT;
	     for(j=0; j<nIFO; j++) {
	       if(TIME+index[j]<0 || TIME+index[j]>=N) {
		 cout<<"netcluster error: time index is out of range\n"; 
		 exit(1); 
	       }
	       pix->data[j].index = TIME+index[j];
	     }

	     id.push_back(pid++);
	     this->append(*pix);           // fill pixel list
	     
	   }
	 }
      }

      i = id.size();
      if(!i) cout<<"netcluster::addhalo() error: empty cluster.";
      
      if(keepid) { 
	 cList.push_back(id);               // fill cluster list
	 sCuts.push_back(0);                // fill selection cuts
         cData.push_back(x.cData[ID-1]);    // fillin cluster metadata
	 if(pr) cRate.push_back(*pr);       // fill in rate array
      }

      if(i<2) continue;
      
      while(--i > 0) {                      // set neighbors
	 pList[id[i-1]].append(id[i]-id[i-1]);
	 pList[id[i]].append(id[i-1]-id[i]);
      }

      
   }
   return pList.size();
}

size_t netcluster::append(netcluster& w)
{
// append pixel list. 
// - to reconstruct clusters and cluster metadata run cluster() utility

   size_t i;
   size_t in = w.pList.size();
   size_t on = pList.size();

   if(!on) { 
      *this = w; 
      this->clear(); 
      return in; 
   }
   if(!in) { return on; }

   netpixel p;

   if(w.start!=start || w.shift!=shift || w.rate!=rate) {
     printf("\n netcluster::append(): cluster type mismatch");
     printf("%f / %f, %f / %f\n",w.start,start,w.shift,shift);

     return on;
   }

   this->pList.reserve(in+on+2);
   for(i=0; i<in; i++){
      p = w.pList[i];
      p.clusterID = 0;                   // update cluster ID
      this->append(p);                   // add pixel to pList
   }

   return pList.size();
}


//**************************************************************************
// construct super clusters  (used in 1G pipeline)                                               
//**************************************************************************
size_t netcluster::supercluster(char atype, double S, bool core)            
{                                                                           
   size_t i,j,k,m,M;                                                        
   size_t n = pList.size();                                                 
   int l;                                                                   
                                                                            
   if(!n) return 0;                                                         

   netpixel* p = NULL;                 // pointer to pixel structure
   netpixel* q = NULL;                 // pointer to pixel structure
   std::vector<int>* v;                                             
   float eps;                                                       
   double E;                                                        
   bool insert;                                                     
   double ptime, pfreq;                                             
   double qtime, qfreq;                                             

   netpixel** pp = (netpixel**)malloc(n*sizeof(netpixel*));
   netpixel** ppo = pp;                                    
   netpixel*  g;                                           

// sort pixels

   for(i=0; i<n; i++) { pp[i] = &(pList[i]);} 
   g = pp[0];                                 
   qsort(pp, n, sizeof(netpixel*), &compare_PIX);         // sort pixels
                                                                        
// update neighbors                                                     

   for(i=0; i<n; i++) {
      p = pp[i];       
      M = size_t(this->rate/p->rate+0.5);     // number of frequency layers
      ptime = (p->time/M+0.5)/p->rate;        // extract time              
      pfreq = (p->frequency+0.5)*p->rate;                                  
                                                                           
      for(j=i+1; j<n; j++){                                                
         q = pp[j];                                                        

         eps = 0.55/p->rate + 0.55/q->rate;
         M = size_t(this->rate/q->rate+0.5);     // number of frequency layers
         qtime = (q->time/M+0.5)/q->rate;                                     

         if(qtime<ptime-eps) cout<<"netcluster::merge() error"<<endl;

         if(qtime-ptime > 1.) break;
         if(fabs(qtime-ptime) > eps) continue;
         if(p->rate==q->rate) continue;       
         if(!(p->rate==2*q->rate || q->rate==2*p->rate)) continue;

         eps = 0.55*(p->rate+q->rate);
         qfreq = (q->frequency+0.5)*q->rate;
         if(fabs(pfreq-qfreq) > eps) continue;

// insert in p

         l = q-p;
                 
         insert = true;
         v = &(p->neighbors);
         m = v->size();      
         for(k=0; k<m; k++) {
            if((*v)[k] == l) {insert=false; break;} 
         }                                          
         if(insert) p->append(l);   // add new neighbor 

// insert in q

         l = p-q;

         insert = true;
         v = &(q->neighbors);
         m = v->size();      
         for(k=0; k<m; k++) {
            if((*v)[k] == l) {insert=false; break;} 
         }                                          
         if(insert) q->append(l);   // add new neighbor  

      }
   }   

   if(ppo==pp) free(pp);
   else {cout<<"netcluster::cluster() free()\n"; exit(1);}

//***************
   cluster();    
//***************

   std::vector<vector_int>::iterator it;
   netpixel* pix = NULL;                
   std::vector<int> rate;               
   std::vector<int> temp;               
   std::vector<int> sIZe;               
   std::vector<bool> cuts;              
   std::vector<double> ampl;            
   std::vector<double> powr;            
   std::vector<double> like;            

   double a,L,e,tt,cT,cF,nT,nF;
   size_t ID,mm;               
   size_t max = 0;             
   size_t min = 0;             
   size_t count=0;             
   bool   cut;                 
   bool   oEo = atype=='E' || atype=='P';

   for(it=cList.begin(); it != cList.end(); it++) {
      k = it->size();                              
      if(!k) cout<<"netcluster::supercluster() error: empty cluster.\n";

// fill cluster statistics

      m = 0; E = 0;
      cT=cF=nT=nF=0.;
      rate.clear();  
      ampl.clear();  
      powr.clear();  
      like.clear();  
      cuts.clear();  
      sIZe.clear();  
      temp.clear();  

      ID = pList[((*it)[0])].clusterID;

      for(i=0; i<k; i++) {            
         pix = &(pList[((*it)[i])]);  
         if(!pix->core && core) continue;
         L = pix->likelihood;            
         e = 0.;                         
         for(j=0; j<pix->size(); j++) {  
           a = pix->data[j].asnr;        
           e+= fabs(a)>1. ? a*a-1 : 0.;  
         }                               

         a   = atype=='L' ? L : e;
         tt  = 1./pix->rate;                    // wavelet time resolution
         mm  = size_t(this->rate*tt+0.1);       // number of wavelet layers
         cT += (pix->time/mm + 0.5)*a;          // pixel time sum          
         nT += a/tt;                            // use weight L/t          
         cF += (pix->frequency+0.5)*a;          // pixel frequency sum     
         nF += a*2.*tt;                         // use weight L*2t         

         insert = true;
         for(j=0; j<rate.size(); j++) {
            if(rate[j] == int(pix->rate+0.1)) {
               insert=false;                   
               ampl[j] += e;                   
               sIZe[j] += 1;                   
               like[j] += L;                   
            }                                  
         }                                     

         if(insert) {
            rate.push_back(int(pix->rate+0.1));
            ampl.push_back(e);                 
            powr.push_back(0.);                
            sIZe.push_back(1);                 
            cuts.push_back(true);              
            like.push_back(L);                 
         }                                     

         m++; E += e;

         if(ID != pix->clusterID) 
            cout<<"netcluster::merge() error: cluster ID mismatch.\n";
      }                                                               

// cut off single level clusters
// coincidence between levels   
                                
      if(rate.size()<size_t(1+this->pair) || m<nPIX){ sCuts[ID-1] = true; continue; }
                                                                                     
      cut = true;                                                                    
      for(i=0; i<rate.size(); i++) {                                                 
        if((atype=='L' && like[i]<S) || (oEo && ampl[i]<S)) continue;                
        if(!pair) { cuts[i] = cut = false; continue; }                               
        for(j=0; j<rate.size(); j++) {                                               
          if((atype=='L' && like[j]<S) || (oEo && ampl[j]<S)) continue;              
          if(rate[i]/2==rate[j] || rate[j]/2==rate[i]) {                             
            cuts[i] = cuts[j] = cut = false;                                         
          }                                                                          
        }                                                                            
      }                                                                              
      if(cut || sCuts[ID-1]) { sCuts[ID-1] = true; continue; }                       

// select optimal resolution

      a = -1.e99;
      for(j=0; j<rate.size(); j++) {  // select max excess power or likelihood
         powr[j] = ampl[j]/sIZe[j];                                           
         if(atype=='E' && ampl[j]>a && !cuts[j]) {max=j; a=ampl[j];}          
         if(atype=='L' && like[j]>a && !cuts[j]) {max=j; a=like[j];}          
         if(atype=='P' && powr[j]>a && !cuts[j]) {max=j; a=powr[j];}          
      }

      if(a<S) { sCuts[ID-1] = true; continue; }

      a = -1.e99;
      for(j=0; j<rate.size(); j++) {
         if(max==j) continue;
         if(atype=='E' && ampl[j]>a && !cuts[j]) {min=j; a=ampl[j];}
         if(atype=='L' && like[j]>a && !cuts[j]) {min=j; a=like[j];}
         if(atype=='P' && powr[j]>a && !cuts[j]) {max=j; a=powr[j];}
      }

      temp.push_back(rate[max]);
      temp.push_back(rate[min]);
      cTime[ID-1] = cT/nT;
      cFreq[ID-1] = cF/nF;
      cRate[ID-1] = temp;
      count++;

   }

   return count;
}


//**************************************************************************
// construct super clusters for 2G analysis
//**************************************************************************
size_t netcluster::supercluster(char atype, double S, double gap, bool core, TH1F* his)
{
//:reconstruct clusters at several TF resolutions (superclusters)
//!param: statistic:  'E' - excess power, 'L' - likelihood
//!param: selection threshold S defines a threshold on clusters
//!       in a superclusters.
//!param: time-frequency gap used for clustering
//!param: true - use only core pixels, false - use core & halo pixels
//!return size of pixel list of selected superclusters.
// algorithm: 
// - sort the pixel list with compare_PIX condition
// - calculate the tim-frequency gap between pixels (1&2) with the same or
//   adjacent resolutions: eps = (dT>0?dT*R:0) + (dF>0?dF*T:0), where
//   dT/dF are time/frequency gaps, R/T are average rate and time:
//   R = rate1+rate2, T = 1/rate1+1/rqte2.
// - if the clustering condition is satisfied, update the neighbor array
// - cluster
// - select superclusters with with clusters in the adjacent resolutions if 
//   this->pair is true.
// in this function the optimal resolution is defined but not used 

   size_t i,j,k,m;
   size_t n = pList.size();
   int l;
   
   if(!n) return 0;

   netpixel* p = NULL;                 // pointer to pixel structure
   netpixel* q = NULL;                 // pointer to pixel structure
   std::vector<int>* v;
   double eps,E,R,T,dT,dF,prl,qrl,aa;
   bool insert;
   double ptime, pfreq;
   double qtime, qfreq;
   double Tgap = 0.;
   size_t nIFO = pList[0].size();       // number of detectors

   netpixel** pp = (netpixel**)malloc(n*sizeof(netpixel*));
   netpixel** ppo = pp;
   netpixel*  g;

// sort pixels

   for(i=0; i<n; i++) { 
      pp[i] = &(pList[i]);
      R = pp[i]->rate;
      if(1./R > Tgap) Tgap = 1./R;       // calculate max time bin
   } 
   Tgap *= (1.+gap);

   //   printf("gap = %lf\t dF = %lf\t S = %lf\n", gap, dF, S);
   
   g = pp[0];
   qsort(pp, n, sizeof(netpixel*), &compare_PIX);         // sort pixels
   
   // update neighbors

   for(i=0; i<n; i++) {
      p = pp[i];
      prl = p->rate*p->layers;
      ptime = p->time/prl;                   // time in seconds  
      pfreq = p->frequency*p->rate;          // frequency in Hz
      
      for(j=i+1; j<n; j++){
         q = pp[j];
	 qrl = q->rate*q->layers;
         if(p->clusterID && p->clusterID==q->clusterID) continue;
         qtime = q->time/qrl;                 // time in seconds
         if(qtime-ptime > Tgap) {break;}      //printf("gap breaking!\n"); break;}
         if(p->rate/q->rate > 3) continue;                                 
         if(q->rate/p->rate > 3) continue;                                 
	 qfreq = q->frequency*q->rate;        // frequency in Hz

         if(qtime<ptime) {
            cout<<"netcluster::supercluster() error "<<qtime-ptime<<endl;
            cout<<p->rate<<" "<<q->rate<<" "<<p->time<<" "<<q->time<<endl;
         }

         R = p->rate+q->rate; 
         T = 1/p->rate+1/q->rate; 

	 dT = 0.;
	 for(k=0; k<nIFO; k++)  {               // max time gap among all detectors
	    aa = p->data[k].index/prl;          // time in seconds  
	    aa-= q->data[k].index/qrl;          // time in seconds  
	    if(fabs(aa)>dT) dT = fabs(aa);
	 }

         dT -= 0.5*T;                            // time gap between pixels          
         dF  = fabs(qfreq-pfreq) - 0.5*R;        // frequency gap between pixels     
         eps = (dT>0?dT*R:0) + (dF>0?dF*T:0);    // 2 x number of pixels
         if(his) his->Fill(eps);
         if(gap < eps) continue;         

         // insert in p

         l = q-p;
         insert = true;
         v = &(p->neighbors);
         m = v->size();

         for(k=0; k<m; k++) {
            if((*v)[k] == l) {insert=false; break;}
         } 

         if(insert) p->append(l);   // add new neighbor 
         
         // insert in q

         l = p-q;
         insert = true;
         v = &(q->neighbors);
         m = v->size();

         for(k=0; k<m; k++) {
            if((*v)[k] == l) {insert=false; break;} 
         }

         if(insert) q->append(l);   // add new neighbor  
      }
   }

   if(ppo==pp) free(pp);
   else {cout<<"netcluster::supercluster() free()\n"; exit(1);}

//***************
   cluster();
//***************

   std::vector<vector_int>::iterator it;
   netpixel* pix = NULL;
   std::vector<int> rate;
   std::vector<int> temp;
   std::vector<int> sIZe;
   std::vector<bool> cuts;
   std::vector<double> ampl;
   std::vector<double> powr;
   std::vector<double> like;

   double a,L,e,tt,cT,cF,nT,nF;
   size_t ID,mm;
   size_t max = 0;
   size_t min = 0;
   size_t count=0;
   bool   cut;
   bool   oEo = atype=='E' || atype=='P';

   for(it=cList.begin(); it != cList.end(); it++) {
      k = it->size();
      if(!k) cout<<"netcluster::supercluster() error: empty cluster.\n";

      // fill cluster statistics

      m = 0; E = 0;
      cT=cF=nT=nF=0.;
      rate.clear();
      ampl.clear();
      powr.clear();
      like.clear();
      cuts.clear();
      sIZe.clear();
      temp.clear();

      ID = pList[((*it)[0])].clusterID;

      for(i=0; i<k; i++) {            // for each pixel in the cluster
	      pix = &(pList[((*it)[i])]);
	      if(!pix->core && core) continue;
	      L = pix->likelihood;
	      e = 0.;
	      for(j=0; j<pix->size(); j++) {
	         a = pix->data[j].asnr;
	         e+= fabs(a)>1. ? a*a : 0.;
	      }

	      a   = atype=='L' ? L : e;
	      tt  = 1./pix->rate;                    // wavelet time resolution
	      mm  = pix->layers;                     // number of wavelet layers
	      cT += int(pix->time/mm)*a;             // pixel time sum
	      nT += a/tt;                            // use weight L/t
	      cF += (pix->frequency+dF)*a;           // pixel frequency sum
	      nF += a*2.*tt;                         // use weight L*2t

	      insert = true;
	      for(j=0; j<rate.size(); j++) {
	         if(rate[j] == int(pix->rate+0.1)) {
	            insert=false;
	            ampl[j] += e;
	            sIZe[j] += 1;
	            like[j] += L;
	         }	       
	      }

	      if(insert) {
	         rate.push_back(int(pix->rate+0.1));
	         ampl.push_back(e);
	         powr.push_back(0.);
	         sIZe.push_back(1);
	         cuts.push_back(true);
	         like.push_back(L);
	      }

	      m++; E += e;

	      if(ID != pix->clusterID) 
	      cout<<"netcluster::supercluster() error: cluster ID mismatch.\n";
      }

      // cut off single level clusters
      // coincidence between levels
      
      if((int)rate.size()< this->pair+1 || m<nPIX){ sCuts[ID-1] = 1; continue; }
      
      cut = true;
      for(i=0; i<rate.size(); i++) {   
	      if((atype=='L' && like[i]<S) || (oEo && ampl[i]<S)) continue;
	      if(!pair) { cuts[i] = cut = false; continue; }                               
	      for(j=0; j<rate.size(); j++) {
	         if((atype=='L' && like[j]<S) || (oEo && ampl[j]<S)) continue;
	         if(rate[i]/2==rate[j] || rate[j]/2==rate[i]) {
	            cuts[i] = cuts[j] = cut = false;	       
	         }
	      }
      }
      if(cut || sCuts[ID-1]) { sCuts[ID-1] = 1; continue; }

      // select optimal resolution

      a = -1.e99;
      for(j=0; j<rate.size(); j++) {  // select max excess power or likelihood
 	      powr[j] = ampl[j]/sIZe[j];
	      if(atype=='E' && ampl[j]>a && !cuts[j]) {max=j; a=ampl[j];}
	      if(atype=='L' && like[j]>a && !cuts[j]) {max=j; a=like[j];}
	      if(atype=='P' && powr[j]>a && !cuts[j]) {max=j; a=powr[j];}
      }

      if(a<S) { sCuts[ID-1] = 1; continue; }

      a = -1.e99;
      for(j=0; j<rate.size(); j++) {
         if(max==j) continue;
         if(atype=='E' && ampl[j]>a && !cuts[j]) {min=j; a=ampl[j];}
         if(atype=='L' && like[j]>a && !cuts[j]) {min=j; a=like[j];}
         if(atype=='P' && powr[j]>a && !cuts[j]) {min=j; a=powr[j];}
      }  

      temp.push_back(rate[max]);
      temp.push_back(rate[min]);
      temp.push_back(E);
      cTime[ID-1] = cT/nT;
      cFreq[ID-1] = cF/nF;
      cRate[ID-1] = temp;
      cData[ID-1].cTime = cT/nT;
      cData[ID-1].cFreq = cF/nF;
      cData[ID-1].energy = E;
      cData[ID-1].likenet = E;
      count++;
   
   }

   return count;
}

//**************************************************************************
// construct duperclusters for 2G analysis
//**************************************************************************
size_t netcluster::defragment(double tgap, double fgap, TH2F* his)
{
// construct duperclusters for 2G analysis
// merge clusters if they are close to each other in time and frequency
//  tgap - maximum time gap in seconds
//  fgap - maximum frequency gap in Hz

   size_t i,j,k;
   size_t I = pList.size();
   int l,n,m;
   
   if(!I) return 0;

   netpixel* p = NULL;                 // pointer to pixel structure
   netpixel* q = NULL;                 // pointer to pixel structure
   std::vector<int>* v;
   std::vector<int>* u;
   double R,T,dT,dF,x,a,E,prl,qrl;
   bool insert;
   double ptime, pfreq;
   double qtime, qfreq;
   size_t nIFO = pList[0].size();      // number of detectors

   netpixel** pp = (netpixel**)malloc(I*sizeof(netpixel*));
   netpixel** ppo = pp;

   // sort pixels

   for(i=0; i<I; i++) { 
      pp[i] = &(pList[i]);
      R = pp[i]->rate;
      if(!(pp[i]->clusterID)) {
	 cout<<"defragment: un-clustered pixel list \n"; exit(1);
      }
   }

   //printf("gap = %lf\t dF = %lf\t I = %d\n", Tgap, Fgap, I);
   
   qsort(pp, I, sizeof(netpixel*), &compare_PIX);         // sort pixels
   
   // update neighbors

   for(i=0; i<I; i++) {
      p = pp[i];
      n = p->clusterID;
      prl = p->rate*p->layers;   
      if(sCuts[n-1]==1) continue;                 // skip rejected pixels
      ptime = p->time/prl;                        // time in seconds  
      pfreq = p->frequency*p->rate/2.;            // frequency in Hz
      
      for(j=i+1; j<I; j++){
         q = pp[j];
	 m = q->clusterID;
	 qrl = q->rate*q->layers;   
	 if(sCuts[m-1]==1) continue;              // skip rejected pixels
         if(n==m) continue;                       // skip the same cluster
         qtime = q->time/qrl;                     // time in seconds
         qfreq = q->frequency*q->rate/2.;         // frequency in Hz
         T = 1/p->rate+1/q->rate; 
         R = p->rate+q->rate; 

	 dT = 0.;
	 for(k=0; k<nIFO; k++)  {                 // max time gap among all detectors
	    a  = p->data[k].index/prl;            // time in seconds  
	    a -= q->data[k].index/qrl;            // time in seconds  
	    if(fabs(a)>dT) dT = fabs(a);
	 }

         dT -= 0.5*T;                            // time gap between pixels          
         dF = fabs(qfreq-pfreq) - 0.25*R;         // frequency gap between pixels     
         if(his) his->Fill(dT,dF);
         if(dT > tgap) {break;}                   // printf("gap breaking!\n"); break;}

         if(qtime<ptime) {
            cout<<"netcluster::supercluster() error "<<qtime-ptime<<endl;
            cout<<p->rate<<" "<<q->rate<<" "<<p->time<<" "<<q->time<<endl;
         }

         if(p->rate/q->rate > 3) continue;                                 
         if(q->rate/p->rate > 3) continue;                                 
	 if(dF > fgap) continue;                   

         // insert in p

         l = q-p;
         insert = true;
         v = &(p->neighbors);

	 for(k=0; k<v->size(); k++) {
            if((*v)[k] == l) {insert=false; break;}
         } 

         if(insert) p->append(l);   // add new neighbor 
         
         // insert in q

         l = p-q;
         insert = true;
         v = &(q->neighbors);

	 for(k=0; k<v->size(); k++) {
            if((*v)[k] == l) {insert=false; break;} 
         }

         if(insert) q->append(l);   // add new neighbor  

	 // replace cluster ID
	 
	 v = &(this->cList[n-1]);                        // append to this pixel list 
	 u = &(this->cList[m-1]);                        // append from this pixel list
	 for(k=0; k<u->size(); k++) {
	    pList[(*u)[k]].clusterID = n;                // set cluster ID=n in q-pixel
	    v->push_back((*u)[k]);                       // add q-pixel to n-cluster
	 }
	 sCuts[m-1] = 1;                                 // mask m cluster

	 x = pList[(*u)[0]].ellipticity;                 // q-pixel subnet sky stat
	 E = pList[(*u)[0]].polarisation;                // q-pixel null energy
	 a = x*E/(1-x);                                  // q-pixel subnet energy
	 x = pList[(*v)[0]].ellipticity;                 // p-pixel subnet stat
	 a+= x*pList[(*v)[0]].polarisation/(1-x);        // p&q-pixel subnet energy
	 E+= pList[(*v)[0]].polarisation;                // p&q-pixel null energy
	 pList[(*v)[0]].polarisation = E;                // p&q-pixel null energy
	 pList[(*v)[0]].ellipticity = a/(a+E+0.1);       // p&q-pixel subnet sky stat x 2

	 x = pList[(*u)[1]].ellipticity;                 // q-pixel subnet MRA stat
	 E = pList[(*u)[1]].polarisation;                // q-pixel null energy
	 a = x*E/(1-x);                                  // q-pixel subnet energy
	 x = pList[(*v)[1]].ellipticity;                 // p-pixel subnet stat
	 a+= x*pList[(*v)[1]].polarisation/(1-x);        // p&q-pixel subnet energy
	 E+= pList[(*v)[1]].polarisation;                // p&q-pixel null energy
	 pList[(*v)[1]].polarisation = E;                // p&q-pixel null energy
	 pList[(*v)[1]].ellipticity = a/(a+E+0.1);       // p&q-pixel subnet MRA stat x 2

      }
   }

   if(ppo==pp) free(pp);
   else {cout<<"netcluster::supercluster() free()\n"; exit(1);}
   return esize();
}


double netcluster::mchirp(int ID)
{  
   const double G  = watconstants::GravitationalConstant();
   const double SM = watconstants::SolarMass();
   const double C  = watconstants::SpeedOfLightInVacuo();
   const double Pi = TMath::Pi();
   double kk = 256.*Pi/5*pow(G*SM*Pi/C/C/C, 5./3);
    
   std::vector<int>* vint = &this->cList[ID-1];
   int V = vint->size();
   if(!V) return -1;
   
   
   double* x = new double[V];
   double* y = new double[V];
   double* xerr = new double[V];
   double* yerr = new double[V];
   double* wgt = new double[V];
   
   double tmin=1e20;          
   double tmax=0.; 
   double T, rms;
   
   
   this->cData[ID-1].chi2chirp = 0;
   this->cData[ID-1].mchirp = 0 ;
   this->cData[ID-1].mchirperr = -1;
   this->cData[ID-1].tmrgr = 0;
   this->cData[ID-1].tmrgrerr = -1; 
   
   // scaling of frequency: in units of 128Hz
   double sF = 128; 
   kk *= pow(sF, 8./3);
   int np = 0;
   double xmin,ymin, xmax, ymax;
   xmin = ymin = 1e100;
   xmax = ymax = -1e100;
   
   for(int j=0; j<V; j++) {   
      netpixel* pix = this->getPixel(ID, j);
      if(pix->likelihood<1. || pix->frequency==0) continue;
         
      T = pix->time/pix->layers;                                      
      T = T/pix->rate;                        // time in seconds from the start
      if(T<tmin) tmin=T;                   
      if(T>tmax) tmax=T;
      
      x[np] = T ;
	   xerr[np] = 0.5/pix->rate;
	   y[np] = pix->frequency*pix->rate/2.;
	   yerr[np] = pix->rate/4;
	   y[np] /= sF;
	   yerr[np] /= sF;	    
	   wgt[np] = pix->likelihood;	    
	   yerr[np] *= 8./3/pow(y[np], 11./3);            // frequency transformation
	   //yerr[np] += 0.1;                               // mitigate high frequency pixels
	   y[np] = 1./pow(y[np], 8./3);
      
      if(x[np]>xmax)xmax = x[np];
      if(x[np]<xmin)xmin = x[np];
      if(y[np]>ymax)ymax = y[np];
      if(y[np]<ymin)ymin = y[np];
	   ++np;
   }
   if(np<5) return -1;
   
   double xcm, ycm, qxx, qyy, qxy;
   xcm = ycm  = qxx = qyy = qxy = 0;
   
   for(int i=0; i<np; ++i){
      xcm += x[i];
      ycm += y[i];
   }
   xcm /= np;
   ycm /= np;
   
   for(int i=0; i<np; ++i){
      qxx += (x[i] - xcm)*(x[i] - xcm);
      qyy += (y[i] - ycm)*(y[i] - ycm);
      qxy += (x[i] - xcm)*(y[i] - ycm);
   }
    
   //printf(" | %lf  ,  %lf |\n", qxx, qxy);
   //printf(" | %lf  ,  %lf |\n", qxy, qyy);
   
   double sq_delta = sqrt( (qxx - qyy)*(qxx - qyy) + 4*qxy*qxy );
   double lam1 = (qxx + qyy + sq_delta)/2;
   double lam2 = (qxx + qyy - sq_delta)/2;
   lam1 = sqrt(lam1);
   lam2 = sqrt(lam2);
   double ellipt = fabs((lam1 - lam2)/(lam1 + lam2));
      
      
   const double maxM = 30;
   const double stepM = 0.1;
   const int massPoints = 601;
      
   int maxMasses[massPoints];   
   int nmaxm = 0;
   
   EndPoint* bint[massPoints];      // stores b intervals for all pixels, satisfying chi2<2 
   for(int j=0; j<massPoints; ++j) bint[j] = new EndPoint[2*np];
   
   int nselmax = 0;
   int massIndex = 0;

   for(double m = -maxM; m<maxM + 0.01; m+=stepM){
      double sl = kk*pow(fabs(m), 5./3);
      if(m>0)sl = -sl; // this is real slope, proper sign
      
      int j=0;
      for(int i=0; i<np; ++i){
         double Db = sqrt( 2 * (sl*sl*xerr[i]*xerr[i] + yerr[i]*yerr[i]) );
         double bmini = y[i] - sl*x[i] - Db;
         double bmaxi = bmini + 2*Db;
         bint[massIndex][j].value = bmini;
         bint[massIndex][j].type = 1;
         bint[massIndex][++j].value = bmaxi;
         bint[massIndex][j++].type = -1;
      }
      qsort(bint[massIndex], 2*np, sizeof(EndPoint), compEndP);  
        
      int nsel = 1;
      for(j=1; j<2*np; ++j){
         bint[massIndex][j].type += bint[massIndex][j-1].type;   
         if(bint[massIndex][j].type>nsel)nsel = bint[massIndex][j].type;
      }
      if(nsel>nselmax){
         nselmax = nsel;
         maxMasses[0] = massIndex;
         nmaxm = 1;
      }
      else if(nsel == nselmax) maxMasses[nmaxm++] = massIndex;
         
      ++massIndex;
   }
      
   
   double m0, b0; 
   double chi2min = 1e100;
   
   // fine parsing in b range, optimize chi2:
   for(int j=0; j<nmaxm; ++j){
      double m =  -maxM + maxMasses[j]*stepM;
      double sl = kk*pow(fabs(m), 5./3);
      if(m>0)sl = -sl;
      
      for(int k=0; k<2*np-1; ++k)if(bint[maxMasses[j]][k].type==nselmax)
         for(double b = bint[maxMasses[j]][k].value; b<bint[maxMasses[j]][k+1].value; b+=0.0025){
            double totchi2 = 0;
            for(int i=0; i<np; ++i){
               double chi2 = y[i] - sl*x[i] -b;
               chi2 *= chi2;
               chi2 /= (sl*sl*xerr[i]*xerr[i] + yerr[i]*yerr[i]);
               if(chi2>2)continue;
               totchi2 += chi2;
            }
            if(totchi2<chi2min){
               chi2min = totchi2;
               m0 = m;
               b0 = b;
            }
         }
   }
   
   for(int j=0; j<massPoints; ++j) delete [] bint[j];
   
   
   double sl = kk*pow(fabs(m0), 5./3);
   if(m0>0)sl = -sl;
   
   
   double totEn = 0.;
   double selEn = 0.;
   double tmax2 = 0.;
   
   for(int i=0; i<np; ++i){
      totEn += wgt[i];
      double chi2 = y[i] - sl*x[i] -b0;
      chi2 *= chi2;
      chi2 /= (sl*sl*xerr[i]*xerr[i] + yerr[i]*yerr[i]);
      if(chi2>2)continue;
      selEn += wgt[i];
      if(x[i]>tmax2)tmax2 = x[i];
   }
   
   double Efrac = selEn/totEn;
   
   tmax2 += 1e-6;
   
   // shift selected pixels on first positions
   int j=0;
   double frac = 0; 
   for(int i=0; i<np; ++i){
      if(x[i]<tmax2)frac += 1.0;
      double chi2 = y[i] - sl*x[i] -b0;
      chi2 *= chi2;
      chi2 /= (sl*sl*xerr[i]*xerr[i] + yerr[i]*yerr[i]);
      
      if(chi2>2)continue;
      x[j] = x[i]; xerr[j] = xerr[i];
      y[j] = y[i]; yerr[j] = yerr[i];
      ++j;
   }
   np = j;
   frac = np/frac;   
   
   //if(Efrac > 1) printf("MCHIRP5ERR_EFRAC\n");
   //if(frac > 1)  printf("MCHIRP5ERR_FRAC\n");
   
   // recompute ellipticity
   
   xcm = ycm  = qxx = qyy = qxy = 0;
   
   for(int i=0; i<np; ++i){
      xcm += x[i];
      ycm += y[i];
   }
   xcm /= np;
   ycm /= np;
   
   for(int i=0; i<np; ++i){
      qxx += (x[i] - xcm)*(x[i] - xcm);
      qyy += (y[i] - ycm)*(y[i] - ycm);
      qxy += (x[i] - xcm)*(y[i] - ycm);
   }
    
   //printf(" | %lf  ,  %lf |\n", qxx, qxy);
   //printf(" | %lf  ,  %lf |\n", qxy, qyy);
   
   sq_delta = sqrt( (qxx - qyy)*(qxx - qyy) + 4*qxy*qxy );
   lam1 = (qxx + qyy + sq_delta)/2;
   lam2 = (qxx + qyy - sq_delta)/2;
   lam1 = sqrt(lam1);
   lam2 = sqrt(lam2);
   double ellipt2 = fabs((lam1 - lam2)/(lam1 + lam2));
   
   
   // bootstrapping section 
   
   if(np<=0)return 0;
   //printf("MCHIRP5ERR_NPZERO\n");
   
   wavearray<int> used(np);
   
   int np2 = np*0.5, Trials=500;
   if(np2<3)np2 = np-1;
   
   if(np2<8)Trials = np2;
   else if(np2<15) Trials = np2*np2/2;
   else if(np2<20) Trials = np2*np2*np2/6;
   if(Trials>500) Trials = 500;
   if(Trials<0) Trials=0; 
      
   this->cData[ID-1].mchpdf.resize(Trials);
   TRandom3 rnd(0);
   wavearray<float> slpv(Trials);
   wavearray<float> mchv(Trials);
   wavearray<float> bv(Trials);
   
   for(int ii=0; ii<Trials; ++ii){
      // generate random sequence
      used = 0;
      for(int k, i=0; i<np2; ++i){
         do k = rnd.Uniform(0. , np - 1e-10);
         while(used[k]);
         used[k]=1;
      }
   
      // linear fit, no weights for now 
      double sx=0, sy=0, sx2=0, sxy=0;
      for(int i=0; i<np; ++i)if(used[i]){
         //x[i] -= tmin;
         sx += x[i];                           
         sy += y[i];
         sx2 += x[i]*x[i];
         sxy += x[i]*y[i];
      }
      double slp = (sy/np2 - sxy/sx)/(sx2/sx - sx/np2);
      double b = (sy + slp*sx)/np2;
      
      slpv.data[ii] = slp;
      mchv.data[ii] = slp>0 ? pow(slp/kk,0.6) : -pow(-slp/kk, 0.6);
      bv.data[ii]   = b/slp;
      this->cData[ID-1].mchpdf.data[ii] = mchv.data[ii];
   }
         
   
   size_t nn = mchv.size()-1;
   size_t ns = size_t(mchv.size()*0.16);
   cData[ID-1].mchirperr = Trials>8 ? ( mchv.waveSplit(0,nn,nn-ns)-mchv.waveSplit(0,nn,ns) ) / 2 : 2*mchv.rms();
   
   // end bootstrapping section
    
   
   char name[100];
   sprintf(name, "netcluster::mchirp5:func_%d", ID);      
   TF1* f = &(this->cData[ID-1].fit);
   TGraphErrors* gr = &(this->cData[ID-1].chirp);   
   
   *gr = TGraphErrors(np, x, y, xerr, yerr);
   *f = TF1(name, "[0]*x + [1]", xmin, xmax);
   f->SetParameter(0, sl);
   f->SetParameter(1, b0);
   gr->Fit(f,"Q");
   
   
   int ndf = f->GetNDF();
   double mch = -f->GetParameter(0)/kk;
   double relerr = fabs(f->GetParError(0)/f->GetParameter(0));
   if(ndf==0) return -1;
   this->cData[ID-1].chi2chirp = Efrac; //f->GetChisquare()/ndf;
   this->cData[ID-1].mchirp = mch>0 ? pow(mch, 0.6) : 0.0001;
   //this->cData[ID-1].mchirperr = 0.6*this->cData[ID-1].mchirp*relerr; // determined by bootstrapping!
   this->cData[ID-1].tmrgr = ellipt2; //-f->GetParameter(1)/f->GetParameter(0);
   this->cData[ID-1].tmrgrerr = frac; //ellipt;

   delete [] x;
   delete [] y;
   delete [] xerr;
   delete [] yerr;
   delete [] wgt;
   
   return 0;

}




double netcluster::mchirp5(int ID)
{  
   const double G  = watconstants::GravitationalConstant();
   const double SM = watconstants::SolarMass();
   const double C  = watconstants::SpeedOfLightInVacuo();
   const double Pi = TMath::Pi();
   double kk = 256.*Pi/5*pow(G*SM*Pi/C/C/C, 5./3);
    
   std::vector<int>* vint = &this->cList[ID-1];
   int V = vint->size();
   if(!V) return -1;
   
   
   double* x = new double[V];
   double* y = new double[V];
   double* xerr = new double[V];
   double* yerr = new double[V];
   double* wgt = new double[V];
   
   double tmin=1e20;          
   double tmax=0.; 
   double T, rms;
   
   
   this->cData[ID-1].chi2chirp = 0;
   this->cData[ID-1].mchirp = 0 ;
   this->cData[ID-1].mchirperr = -1;
   this->cData[ID-1].tmrgr = 0;
   this->cData[ID-1].tmrgrerr = -1; 
   
   // scaling of frequency: in units of 128Hz
   double sF = 128; 
   kk *= pow(sF, 8./3);
   int np = 0;
   double xmin,ymin, xmax, ymax;
   xmin = ymin = 1e100;
   xmax = ymax = -1e100;
   
   for(int j=0; j<V; j++) {   
      netpixel* pix = this->getPixel(ID, j);
      if(pix->likelihood<1. || pix->frequency==0) continue;
         
      T = pix->time/pix->layers;                                      
      T = T/pix->rate;                        // time in seconds from the start
      if(T<tmin) tmin=T;                   
      if(T>tmax) tmax=T;
      
      x[np] = T ;
	   xerr[np] = 0.5/pix->rate;
	   y[np] = pix->frequency*pix->rate/2.;
	   yerr[np] = pix->rate/4;
	   y[np] /= sF;
	   yerr[np] /= sF;	    
	   wgt[np] = pix->likelihood;	    
	   yerr[np] *= 8./3/pow(y[np], 11./3);            // frequency transformation
	   //yerr[np] += 0.1;                               // mitigate high frequency pixels
	   y[np] = 1./pow(y[np], 8./3);
      
      if(x[np]>xmax)xmax = x[np];
      if(x[np]<xmin)xmin = x[np];
      if(y[np]>ymax)ymax = y[np];
      if(y[np]<ymin)ymin = y[np];
	   ++np;
   }
   if(np<5) return -1;
   
   double xcm, ycm, qxx, qyy, qxy;
   xcm = ycm  = qxx = qyy = qxy = 0;
   
   for(int i=0; i<np; ++i){
      xcm += x[i];
      ycm += y[i];
   }
   xcm /= np;
   ycm /= np;
   
   for(int i=0; i<np; ++i){
      qxx += (x[i] - xcm)*(x[i] - xcm);
      qyy += (y[i] - ycm)*(y[i] - ycm);
      qxy += (x[i] - xcm)*(y[i] - ycm);
   }
    
   //printf(" | %lf  ,  %lf |\n", qxx, qxy);
   //printf(" | %lf  ,  %lf |\n", qxy, qyy);
   
   double sq_delta = sqrt( (qxx - qyy)*(qxx - qyy) + 4*qxy*qxy );
   double lam1 = (qxx + qyy + sq_delta)/2;
   double lam2 = (qxx + qyy - sq_delta)/2;
   lam1 = sqrt(lam1);
   lam2 = sqrt(lam2);
   double ellipt = fabs((lam1 - lam2)/(lam1 + lam2));
      
      
   const double maxM = 30;
   const double stepM = 0.1;
   const int massPoints = 601;
      
   int maxMasses[massPoints];   
   int nmaxm = 0;
   
   EndPoint* bint[massPoints];      // stores b intervals for all pixels, satisfying chi2<2 
   for(int j=0; j<massPoints; ++j) bint[j] = new EndPoint[2*np];
   
   int nselmax = 0;
   int massIndex = 0;

   for(double m = -maxM; m<maxM + 0.01; m+=stepM){
      double sl = kk*pow(fabs(m), 5./3);
      if(m>0)sl = -sl; // this is real slope, proper sign
      
      int j=0;
      for(int i=0; i<np; ++i){
         double Db = sqrt( 2 * (sl*sl*xerr[i]*xerr[i] + yerr[i]*yerr[i]) );
         double bmini = y[i] - sl*x[i] - Db;
         double bmaxi = bmini + 2*Db;
         bint[massIndex][j].value = bmini;
         bint[massIndex][j].type = 1;
         bint[massIndex][++j].value = bmaxi;
         bint[massIndex][j++].type = -1;
      }
      qsort(bint[massIndex], 2*np, sizeof(EndPoint), compEndP);  
        
      int nsel = 1;
      for(j=1; j<2*np; ++j){
         bint[massIndex][j].type += bint[massIndex][j-1].type;   
         if(bint[massIndex][j].type>nsel)nsel = bint[massIndex][j].type;
      }
      if(nsel>nselmax){
         nselmax = nsel;
         maxMasses[0] = massIndex;
         nmaxm = 1;
      }
      else if(nsel == nselmax) maxMasses[nmaxm++] = massIndex;
         
      ++massIndex;
   }
      
   
   double m0, b0; 
   double chi2min = 1e100;
   
   // fine parsing in b range, optimize chi2:
   for(int j=0; j<nmaxm; ++j){
      double m =  -maxM + maxMasses[j]*stepM;
      double sl = kk*pow(fabs(m), 5./3);
      if(m>0)sl = -sl;
      
      for(int k=0; k<2*np-1; ++k)if(bint[maxMasses[j]][k].type==nselmax)
         for(double b = bint[maxMasses[j]][k].value; b<bint[maxMasses[j]][k+1].value; b+=0.0025){
            double totchi2 = 0;
            for(int i=0; i<np; ++i){
               double chi2 = y[i] - sl*x[i] -b;
               chi2 *= chi2;
               chi2 /= (sl*sl*xerr[i]*xerr[i] + yerr[i]*yerr[i]);
               if(chi2>2)continue;
               totchi2 += chi2;
            }
            if(totchi2<chi2min){
               chi2min = totchi2;
               m0 = m;
               b0 = b;
            }
         }
   }
   
   for(int j=0; j<massPoints; ++j) delete [] bint[j];
   
   
   double sl = kk*pow(fabs(m0), 5./3);
   if(m0>0)sl = -sl;
   
   
   double totEn = 0.;
   double selEn = 0.;
   double tmax2 = 0.;
   
   for(int i=0; i<np; ++i){
      totEn += wgt[i];
      double chi2 = y[i] - sl*x[i] - b0;
      chi2 *= chi2;
      chi2 /= (sl*sl*xerr[i]*xerr[i] + yerr[i]*yerr[i]);
      if(chi2>2)continue;
      selEn += wgt[i];
      if(x[i]>tmax2)tmax2 = x[i];
   }
   
   double Efrac = selEn/totEn;
   
   tmax2 += 1e-6;
   
   // shift selected pixels on first positions
   int j=0;
   double frac = 0; 
   for(int i=0; i<np; ++i){
      if(x[i]<tmax2)frac += 1.0;
      double chi2 = y[i] - sl*x[i] -b0;
      chi2 *= chi2;
      chi2 /= (sl*sl*xerr[i]*xerr[i] + yerr[i]*yerr[i]);
      
      if(chi2>2)continue;
      x[j] = x[i]; xerr[j] = xerr[i];
      y[j] = y[i]; yerr[j] = yerr[i];
      ++j;
   }
   np = j;
   frac = np/frac;   
   
   //if(Efrac > 1) printf("MCHIRP5ERR_EFRAC\n");
   //if(frac > 1)  printf("MCHIRP5ERR_FRAC\n");
   
   // recompute ellipticity
   
   xcm = ycm  = qxx = qyy = qxy = 0;
   
   for(int i=0; i<np; ++i){
      xcm += x[i];
      ycm += y[i];
   }
   xcm /= np;
   ycm /= np;
   
   for(int i=0; i<np; ++i){
      qxx += (x[i] - xcm)*(x[i] - xcm);
      qyy += (y[i] - ycm)*(y[i] - ycm);
      qxy += (x[i] - xcm)*(y[i] - ycm);
   }
    
   //printf(" | %lf  ,  %lf |\n", qxx, qxy);
   //printf(" | %lf  ,  %lf |\n", qxy, qyy);
   
   sq_delta = sqrt( (qxx - qyy)*(qxx - qyy) + 4*qxy*qxy );
   lam1 = (qxx + qyy + sq_delta)/2;
   lam2 = (qxx + qyy - sq_delta)/2;
   lam1 = sqrt(lam1);
   lam2 = sqrt(lam2);
   double ellipt2 = fabs((lam1 - lam2)/(lam1 + lam2));
   
   
   // bootstrapping section 
   
   if(np<=0)return 0;
   //printf("MCHIRP5ERR_NPZERO\n");
   
   wavearray<int> used(np);
   
   int np2 = np*0.5, Trials=500;
   if(np2<3)np2 = np-1;
   
   if(np2<8)Trials = np2;
   else if(np2<15) Trials = np2*np2/2;
   else if(np2<20) Trials = np2*np2*np2/6;
   if(Trials>500) Trials = 500;
   if(Trials<0) Trials=0; 
      
   this->cData[ID-1].mchpdf.resize(Trials);
   TRandom3 rnd(0);
   wavearray<float> slpv(Trials);
   wavearray<float> mchv(Trials);
   wavearray<float> bv(Trials);
   
   for(int ii=0; ii<Trials; ++ii){
      // generate random sequence
      used = 0;
      for(int k, i=0; i<np2; ++i){
         do k = rnd.Uniform(0. , np - 1e-10);
         while(used[k]);
         used[k]=1;
      }
   
      // linear fit, no weights for now 
      double sx=0, sy=0, sx2=0, sxy=0;
      for(int i=0; i<np; ++i)if(used[i]){
         //x[i] -= tmin;
         sx += x[i];                           
         sy += y[i];
         sx2 += x[i]*x[i];
         sxy += x[i]*y[i];
      }
      double slp = (sy/np2 - sxy/sx)/(sx2/sx - sx/np2);
      double b = (sy + slp*sx)/np2;
      
      slpv.data[ii] = slp;
      mchv.data[ii] = slp>0 ? pow(slp/kk,0.6) : -pow(-slp/kk, 0.6);
      bv.data[ii]   = b/slp;
      this->cData[ID-1].mchpdf.data[ii] = mchv.data[ii];
   }
         
   
   size_t nn = mchv.size()-1;
   size_t ns = size_t(mchv.size()*0.16);
   cData[ID-1].mchirperr = Trials>8 ? ( mchv.waveSplit(0,nn,nn-ns)-mchv.waveSplit(0,nn,ns) ) / 2 : 2*mchv.rms();
   
   // end bootstrapping section
    
   
   char name[100];
   sprintf(name, "netcluster::mchirp5:func_%d", ID);      
   TF1* f = &(this->cData[ID-1].fit);
   TGraphErrors* gr = &(this->cData[ID-1].chirp);   
   
   *gr = TGraphErrors(np, x, y, xerr, yerr);
   *f = TF1(name, "[0]*x + [1]", xmin, xmax);
   f->SetParameter(0, sl);
   f->SetParameter(1, b0);
   gr->Fit(f,"Q");
   
   
   int ndf = f->GetNDF();
   double mch = -f->GetParameter(0)/kk;
   double relerr = fabs(f->GetParError(0)/f->GetParameter(0));
   if(ndf==0) return -1;
   this->cData[ID-1].chi2chirp = Efrac; //f->GetChisquare()/ndf;
   this->cData[ID-1].mchirp = mch>0 ? pow(mch, 0.6) : 0.0001;
   //this->cData[ID-1].mchirperr = 0.6*this->cData[ID-1].mchirp*relerr; // determined by bootstrapping!
   this->cData[ID-1].tmrgr = ellipt2; //-f->GetParameter(1)/f->GetParameter(0);
   this->cData[ID-1].tmrgrerr = frac; //ellipt;

   delete [] x;
   delete [] y;
   delete [] xerr;
   delete [] yerr;
   delete [] wgt;
   
   return 0;

}

double netcluster::mchirp6(int ID)
{  
   const double G  = watconstants::GravitationalConstant();
   const double SM = watconstants::SolarMass();
   const double C  = watconstants::SpeedOfLightInVacuo();
   const double Pi = TMath::Pi();
   //double kk = 256.*Pi/5*pow(G*SM*Pi/C/C/C, 5./3); 
       
   this->cData[ID-1].chi2chirp = 0;
   this->cData[ID-1].mchirp = 0 ;
   this->cData[ID-1].mchirperr = -1;
   this->cData[ID-1].tmrgr = 0;
   this->cData[ID-1].tmrgrerr = -1; 
   
   double kf = pow(5./256, 3./8) * pow( C*C*C/G/SM, 5./8) / Pi; // f(t) = kf / Mc^(5/8) / (t0-t)^(3/8)
   double kdf = sqrt(0.6*kf*3./8);                              // df(t) = kdf / Mc^(5/16) / (t0-t)^(11/16)
    
   std::vector<int>* vint = &this->cList[ID-1];
   int V = vint->size();
   if(!V) return -1;
   
   int np=0;
   double minDT=1e10, minFreq=1e10;
	   
	for(int j=0; j<V; j++) {   
      netpixel* pix = this->getPixel(ID, j);
	   if(pix->likelihood<1. || pix->frequency==0  || !pix->core) continue;	   
      ++np;  
	   if(pix->rate/2. < minFreq) minFreq = pix->rate/2;
	   if(1./pix->rate < minDT) minDT = 1./pix->rate;
   }
   if(np<3)return -1;
   int naux = 1./(2*minDT*minFreq) + 0.5;  // division
   int Vp = naux*V;
  
	   //printf("minDT = %lf \t minFreq = %lf \t naux = %d\n", minDT, minFreq, naux);	
 
   double* x = new double[Vp];
   double* y = new double[Vp];
   double* wgt = new double[Vp];
   bool* sel = new bool[Vp];
	   
   np = 0;
   double tmin=1e20;          
   double tmax=0.; 
   double T; //, rms;
   
   double xmin,ymin, xmax, ymax;
   xmin = ymin = 1e100;
   xmax = ymax = -1e100;

   // break regular pixels
   
   for(int j=0; j<V; j++) {   
      netpixel* pix = this->getPixel(ID, j);
      if(pix->likelihood<1. || pix->frequency==0 || !pix->core) continue;
      T = pix->time/pix->layers;                                     
      T = T/pix->rate;                        // time in seconds from the start
      if(T<tmin) tmin=T;                   
      if(T>tmax) tmax=T;     
      int nx = 1./pix->rate/minDT + 0.5;
      int ny = pix->rate/2/minFreq + 0.5;
      //printf("nx = %d    ny = %d\n", nx, ny);
      for(int ix=0; ix<nx; ++ix) {
         for(int iy=0; iy<ny; ++iy){
            x[np] = T - 0.5/pix->rate + (ix+0.5)*minDT;  //xerr[np] = 0.5/pix->rate;
            y[np] = pix->frequency*pix->rate/2. - pix->rate/4 + (iy+0.5)*minFreq; //yerr[np] = pix->rate/4;         
            wgt[np] = pix->likelihood/naux;  
            
            if(x[np]>xmax)xmax = x[np];
            if(x[np]<xmin)xmin = x[np];
            if(y[np]>ymax)ymax = y[np];
            if(y[np]<ymin)ymin = y[np];       
            ++np;
   	   } 
      }   
   }
 
   const double maxM = 30;           // maximum chirp mass
   const double stepM = 0.1;         // step
   //const int massPoints = 601;
   
   int nselmax = 0;   
   double m0 = 0, t0max = 0;  
   double t0 = (tmin*2 +  tmax)/3;
   
   /*
   for(; t0<tmax+1; t0 += minDT){                   // intercept loop
      for(double m = 0.1; m<maxM; m += stepM){    // mass scan
         double cf = kf/pow(m, 5./8);
         double cdf = kdf/pow(m, 5./16);
         int nsel = 0; 
         for(int i=0; i<np; ++i){                 // scan over elementary pixels
            double a = t0 - x[i];
            if(a<=0)continue;
            double b = sqrt(sqrt(a)); 
            b *= sqrt(b); 
            double fi = cf/b; 
            double df = cdf/sqrt(b*a);            // evelope width in frequency 
            if(y[i]>fi+df) continue;
            if(y[i]<fi-df) continue;
            ++nsel; 
         }
         if(nsel>nselmax){                         // new optimum
            nselmax = nsel;
            m0 = m;
            t0max = t0;
         }
      }   
   }
   */
   
   
   int tPoints = (tmax + 1 - t0)/minDT + 1;
   double* maxTimes = new double[tPoints];  
   double* maxMasses = new double[tPoints];
   int nmaxt = 0;
   
   EndPoint** mint = new EndPoint*[tPoints];      // stores valid m intervals for all pixels
   for(int j=0; j<tPoints; ++j) mint[j] = new EndPoint[2*np];
   
   int tIndex = 0;
   for(; t0<tmax+1; t0 += minDT){                   // intercept loop
      int j=0;
      for(int i=0; i<np; ++i){
         double dt = t0 - x[i];
         if(dt<=0)continue;
         double dt38 = sqrt(sqrt(dt)); 
         dt38 *= sqrt(dt38);
         double a = kf/dt38;
         double b = kdf/sqrt(dt*dt38);   
         double Delta = sqrt(b*b + 4*a*y[i]);
         
         mint[tIndex][j].value = pow( 2*a/(Delta + b), 16./5);
         mint[tIndex][j].type = 1;
         mint[tIndex][++j].value = pow( 2*a/(Delta - b), 16./5);
         mint[tIndex][j++].type = -1;
         //printf("%lf %lf\n", mint[tIndex][j-2].value, mint[tIndex][j-1].value) ;
      }
      qsort(mint[tIndex], j, sizeof(EndPoint), compEndP);  
        
      int nsel = 1;
      double mmax;
      for(int k=1; k<j; ++k){
         mint[tIndex][k].type += mint[tIndex][k-1].type;   
         if(mint[tIndex][k].type>=nsel){ 
            mmax = (mint[tIndex][k].value + mint[tIndex][k+1].value)/2; 
            nsel = mint[tIndex][k].type;
         }
      }
      if(nsel>nselmax){
         nselmax = nsel;
         maxTimes[0] = t0;
         maxMasses[0] = mmax;
         nmaxt = 1;
      }
      else if(nsel == nselmax){ maxMasses[nmaxt] = mmax; maxTimes[nmaxt++] = t0;}
      ++tIndex;
   }
   
   t0max = maxTimes[nmaxt-1];
   m0 = maxMasses[nmaxt-1];
   
   for(int j=0; j<tPoints; ++j) delete [] mint[j];
   delete [] mint; 
   delete [] maxTimes;
   delete [] maxMasses;
   
   
   // find selected pixels, max t among them
   
   double tmax2 = 0.;
   
   double cf = kf/pow(m0, 5./8);
   double cdf = kdf/pow(m0, 5./16);
   for(int i=0; i<np; ++i){
      sel[i] = false;
      double a = t0max - x[i];
      if(a<=0) continue;
      double b = sqrt(sqrt(a)); 
      b *= sqrt(b); 
      double fi = cf/b; 
      double df = cdf/sqrt(b*a);
      if(y[i]>fi+df) continue;
      if(y[i]<fi-df) continue;      
      sel[i] = true;                                 // flag pixels
      if(x[i]>tmax2) tmax2 = x[i];
   }
   
   
   // compute energy fraction (global) , pixel fraction (among those below tmax2!)s
   // shift selected pixels on first positions, too
   
   double totEn = 0.;
   double selEn = 0.;
   int j=0;
   double frac = 0; 
   tmax2 += 1e-6;
              
   for(int i=0; i<np; ++i){
      totEn += wgt[i];
      if(x[i]<tmax2) frac += 1.0;
      if(sel[i]){
         selEn += wgt[i];
         x[j] = x[i];
         y[j] = y[i];
         ++j;
      }
   }
   
   double Efrac = selEn/totEn;
   np = j;
   frac = np/frac;   
         
   // compute ellipticity
   
   double xcm, ycm, qxx, qyy, qxy;
   xcm = ycm  = qxx = qyy = qxy = 0;
       
   for(int i=0; i<np; ++i){
      xcm += x[i];
      ycm += y[i];
   }
   xcm /= np;
   ycm /= np;
   
   for(int i=0; i<np; ++i){
      qxx += (x[i] - xcm)*(x[i] - xcm);
      qyy += (y[i] - ycm)*(y[i] - ycm);
      qxy += (x[i] - xcm)*(y[i] - ycm);
   }
    
   //printf(" | %lf  ,  %lf |\n", qxx, qxy);
   //printf(" | %lf  ,  %lf |\n", qxy, qyy);
   
   double sq_delta = sqrt( (qxx - qyy)*(qxx - qyy) + 4*qxy*qxy );
   double lam1 = (qxx + qyy + sq_delta)/2;
   double lam2 = (qxx + qyy - sq_delta)/2;
   lam1 = sqrt(lam1);
   lam2 = sqrt(lam2);
   double ellipt2 = fabs((lam1 - lam2)/(lam1 + lam2));
   
   
   /*/ bootstrapping section 
   
   if(np<=0)return 0;
   //printf("MCHIRP5ERR_NPZERO\n");
   
   wavearray<int> used(np);
   
   int np2 = np*0.5, Trials=500;
   if(np2<3)np2 = np-1;
   
   if(np2<8)Trials = np2;
   else if(np2<15) Trials = np2*np2/2;
   else if(np2<20) Trials = np2*np2*np2/6;
   if(Trials>500) Trials = 500;
   if(Trials<0) Trials=0; 
      
   this->cData[ID-1].mchpdf.resize(Trials);
   TRandom3 rnd(0);
   wavearray<float> slpv(Trials);
   wavearray<float> mchv(Trials);
   wavearray<float> bv(Trials);
   
   for(int ii=0; ii<Trials; ++ii){
      // generate random sequence
      used = 0;
      for(int k, i=0; i<np2; ++i){
         do k = rnd.Uniform(0. , np - 1e-10);
         while(used[k]);
         used[k]=1;
      }
   
      // linear fit, no weights for now 
      double sx=0, sy=0, sx2=0, sxy=0;
      for(int i=0; i<np; ++i)if(used[i]){
         //x[i] -= tmin;
         sx += x[i];                           
         sy += y[i];
         sx2 += x[i]*x[i];
         sxy += x[i]*y[i];
      }
      double slp = (sy/np2 - sxy/sx)/(sx2/sx - sx/np2);
      double b = (sy + slp*sx)/np2;
      
      slpv.data[ii] = slp;
      mchv.data[ii] = slp>0 ? pow(slp/kk,0.6) : -pow(-slp/kk, 0.6);
      bv.data[ii]   = b/slp;
      this->cData[ID-1].mchpdf.data[ii] = mchv.data[ii];
   }
         
   
   size_t nn = mchv.size()-1;
   size_t ns = size_t(mchv.size()*0.16);
   cData[ID-1].mchirperr = Trials>8 ? ( mchv.waveSplit(0,nn,nn-ns)-mchv.waveSplit(0,nn,ns) ) / 2 : 2*mchv.rms();
   
   // end bootstrapping section 
    
   
   char name[100];
   sprintf(name, "netcluster::mchirp5:func_%d", ID);      
   TF1* f = &(this->cData[ID-1].fit);
   TGraphErrors* gr = &(this->cData[ID-1].chirp);   
   
   *gr = TGraphErrors(np, x, y, xerr, yerr);
   *f = TF1(name, "[0]*x + [1]", xmin, xmax);
   f->SetParameter(0, sl);
   f->SetParameter(1, b0);
   gr->Fit(f,"Q");
   
   
   
   int ndf = f->GetNDF();
   double mch = -f->GetParameter(0)/kk;
   double relerr = fabs(f->GetParError(0)/f->GetParameter(0));
   if(ndf==0) return -1;
   */
   this->cData[ID-1].chi2chirp = Efrac; //f->GetChisquare()/ndf;
   this->cData[ID-1].mchirp = m0; // mch>0 ? pow(mch, 0.6) : 0.0001;
   //this->cData[ID-1].mchirperr = 0.6*this->cData[ID-1].mchirp*relerr; // determined by bootstrapping!
   this->cData[ID-1].tmrgr = ellipt2; //-f->GetParameter(1)/f->GetParameter(0);
   this->cData[ID-1].tmrgrerr = frac; //ellipt;

   delete [] x;
   delete [] y;
   delete [] sel;
   delete [] wgt;
   
   return 0;

}
// draw chirp cluster
void netcluster::chirpDraw(int id)
{
// Draw chirp object for cluster id
   TCanvas* c = new TCanvas("chirp", ""); c->cd(); 
   this->cData[id-1].chirp.Draw("AP"); 
   this->cData[id-1].fit.Draw("same");
   return;
}

void netcluster::PlotClusters()
{  TCanvas* c = new TCanvas("cpc", "", 1200, 800);
   TH2F* h2 = new TH2F("h2h", "", 600, 0, 600, 2048, 0, 2048);
   int cntr = 0;
   std::vector<vector_int>::iterator it;
   for(it=cList.begin(); it != cList.end(); it++){ 
      size_t ID = pList[((*it)[0])].clusterID;
      if(sCuts[ID-1]==0){
         h2->Fill(cTime[ID-1], cFreq[ID-1], cRate[ID-1][2]);
         //printf("\nID = %lu time = %lf\t freq = %lf ", ID, cTime[ID-1], cFreq[ID-1]);
         ++cntr;
      }
      else if(sCuts[ID-1]==1)printf("*");
   }
   printf("\n# of zero sCuts lusters:%d\n", cntr);
   c->cd();
   h2->Draw("colz");
}

//**************************************************************************
//**************************************************************************
wavearray<double> netcluster::get(char* name, size_t index, char atype, int type, bool core)
{
// return cluster parameters defined by name
// name="ID"       clusterID
// name="size"     core size
// name="SIZE"     core+halo size
// name="volume"   volume
// name="start"    actual start time relative to segment start
// name="stop"     actual stop time relative to segment stop
// name="low"      low frequency
// name="high"     high frequency
// name="energy"   energy;  'R' - rank, 'S' - Gauss 
// name="power"    energy/size 
// name="conf"     cluster confidence; 'R' - rank, 'S' - Gauss
// name="like"     network likelihood
// name="null"     network null statistic
// name="sign"     significance;   index<0 - rank, index>0 - Gauss
// name="corr"     xcorrelation
// name="asym"     asymmetry 
// name="grand"    grandAmplitude 
// name="rate"     cluster rate
// name="SNR"      cluster SNR: 'R' - rank, 'S' - Gaussian
// name="hrss"     log10(calibrated cluster hrss) 
// name="noise"    log10(average calibrated noise): average of sigma^2
// name="NOISE"    log10(average calibrated noise): average of 1/sigma^2
// name="elli"     ellipticity 
// name="psi"      source polarisation angle 
// name="phi"      source coordinate phi 
// name="theta"    source coordinate theta 
// name="time"     zero lag time averaged over: 'S'-gSNR, 'R'-rSNR, 'L'-likelihood
// name="TIME"     zero lag time averaged over energy and all resolutions
// name="freq"     frequency averaged over: 'S'-gSNR, 'R'-rSNR, 'L'-likelihood
// name="FREQ"     frequency averaged over energy and all resolutions 
// name="chi2"     chi2 significance: 1 - cumulative probability                      

   wavearray<double> out;
   if(!cList.size() || !pList.size()) return out;

   size_t i,k,K;
   size_t mp,mm;
   size_t it_size; 
   size_t it_core;
   size_t it_halo;
   size_t out_size = 0;
   double x,y;
   double a,b;
   double t,r;
   double sum = 0.;
   double logbpp = -log(bpp);
   
   // before Feb, 2007:  ampbpp = sqrt(2*1.07*logbpp)-1.11/2.;
   // after  Feb, 2007:  more accurate approximation
   double ampbpp = sqrt(2*logbpp-2*log(1+pow(log(1+logbpp*(1+1.5*exp(-logbpp))),2)));
   
   int ID, rate;
   int RATE = abs(type)>2 ? abs(type) : 0;

   wavearray<int> skip;
   vector<vector_int>::iterator it;
   vector_int* pv = NULL;
   size_t M = pList.size();
   size_t m = index ? index : index+1;

   out.resize(cList.size());
   out.start(start);
   out.rate(1.);
   out = 0.;

   char c = '0';         

   if(strstr(name,"ID"))     c = 'i';   // clusterID
   if(strstr(name,"size"))   c = 'k';   // core size
   if(strstr(name,"SIZE"))   c = 'K';   // core+halo size
   if(strstr(name,"volume")) c = 'v';   // volume
   if(strstr(name,"start"))  c = 's';   // actual start time relative to segment start
   if(strstr(name,"stop"))   c = 'd';   // actual stop time relative to segment stop
   if(strstr(name,"low"))    c = 'l';   // low frequency
   if(strstr(name,"high"))   c = 'h';   // high frequency
   if(strstr(name,"energy")) c = 'e';   // energy;  'R' - rank, 'S' - Gauss 
   if(strstr(name,"power"))  c = 'w';   // energy/size 
   if(strstr(name,"conf"))   c = 'Y';   // cluster confidence; 'R' - rank, 'S' - Gauss
   if(strstr(name,"like"))   c = 'L';   // network likelihood
   if(strstr(name,"null"))   c = 'N';   // network null statistic
   if(strstr(name,"sign"))   c = 'z';   // significance;   index<0 - rank, index>0 - Gauss
   if(strstr(name,"corr"))   c = 'x';   // xcorrelation
   if(strstr(name,"asym"))   c = 'a';   // asymmetry 
   if(strstr(name,"grand"))  c = 'g';   // grandAmplitude 
   if(strstr(name,"rate"))   c = 'r';   // cluster rate
   if(strstr(name,"SNR"))    c = 'S';   // cluster SNR: 'R' - rank, 'S' - Gaussian
   if(strstr(name,"hrss"))   c = 'H';   // log10(calibrated cluster hrss) 
   if(strstr(name,"noise"))  c = 'n';   // log10(average calibrated noise): average of sigma^2
   if(strstr(name,"NOISE"))  c = 'I';   // log10(average calibrated noise): average of 1/sigma^2
   if(strstr(name,"elli"))   c = 'o';   // ellipticity 
   if(strstr(name,"psi"))    c = 'O';   // source polarisation angle 
   if(strstr(name,"phi"))    c = 'P';   // source coordinate phi 
   if(strstr(name,"theta"))  c = 'p';   // source coordinate theta 
   if(strstr(name,"time"))   c = 't';   // zero lag time averaged over: 'S'-gSNR, 'R'-rSNR, 'L'-likelihood
   if(strstr(name,"TIME"))   c = 'T';   // zero lag time averaged over energy and all resolutions
   if(strstr(name,"freq"))   c = 'f';   // frequency averaged over: 'S'-gSNR, 'R'-rSNR, 'L'-likelihood
   if(strstr(name,"FREQ"))   c = 'F';   // frequency averaged over energy and all resolutions 
   if(strstr(name,"chi2"))   c = 'C';   // chi2 significance: 1 - cumulative probability                      

   if(c=='0') return out;

   k = K = 0;

   for(it=cList.begin(); it!=cList.end(); it++){

     ID = pList[((*it)[0])].clusterID;
     if(sCuts[ID-1]==1) continue;   // apply selection cuts
     it_size = it->size();
     if(!it_size) continue;

     it_core = 0;
     it_halo = 0;
     skip.resize(it_size);
     pv = cRate.size() ? &(cRate[ID-1]) : NULL;
     rate = 0;

     if(type<0) rate = -type;
     else if(!pv) rate = 0;
     else if(type==1 && pv->size()) rate = (*pv)[0];
     else if(pv->size() && RATE) rate = ((*pv)[0]==RATE) ? RATE : -1;

     for(k=0; k<it_size; k++) {          // fill skip array
	M = (*it)[k];
	skip.data[k] = 1;
	if(rate && int(pList[M].rate+0.1)!=rate) continue;
	it_halo++;
	if(!pList[M].core && core) continue;
	skip.data[k] = 0;
	it_core++;
     }

     if(!it_core) continue;     // skip cluster
     
     switch (c) {

     case 'i':          // get cluster ID 
       for(k=0; k<it_size; k++){
	 M = (*it)[k];
	 if(!skip.data[k]) { 
	   out.data[out_size++] = pList[M].clusterID; 
	   break; 
	 }
       }
       break;

     case 'k':          // get cluster core size 
     case 'K':          // get cluster core+halo size 
       out.data[out_size++] = c=='k' ? float(it_core) : float(it_halo);
       break;

     case 'P':          // get source coordinate phi 
     case 'p':          // get source coordinate theta
       for(k=0; k<it_size; k++){
	 M = (*it)[k];
	 if(!skip.data[k]) { 
	   out.data[out_size++] = (c=='p') ? pList[M].theta :  pList[M].phi;
	   break; 
	 }
       }
       break;

     case 'O':          // get source polarisation angle 
     case 'o':          // get source ellipticity
       for(k=0; k<it_size; k++){
	 M = (*it)[k];
	 if(!skip.data[k]) { 
	   out.data[out_size++] = (c=='o') ? pList[M].ellipticity :  pList[M].polarisation;
	   break; 
	 }
       }
       break;

     case 'r':          // get cluster rate 
       for(k=0; k<it_size; k++){
	 M = (*it)[k];
	 if(!skip.data[k]) { 
	   out.data[out_size++] = pList[M].rate; 
	   break; 
	 }
       }
       break;

     case 'a':          // get cluster asymmetry 
     case 'x':          // get cluster x-correlation parameter 
       mp = 0; mm = 0;
       for(k=0; k<it_size; k++){
	 M = (*it)[k];

	 if(!index) continue;
	 if(skip.data[k]) continue;
	 if(pList[M].size()<m) continue;
	 x = pList[M].getdata(atype,m-1);
	 if(x>0.) mp++;
	 else     mm++;
       }
       if(c == 'a') out.data[out_size++] = mp+mm>0 ? (float(mp)-float(mm))/(mp+mm) : 0;
       else         out.data[out_size++] = mp+mm>0 ? signPDF(mp,mm) : 0; 
       break;
       
     case 'e':          // get cluster energy
     case 'S':          // get cluster SNR
     case 'Y':          // get cluster confidence
     case 'z':          // get cluster significance
     case 'w':          // get cluster power
     case 'C':          // get cluster chi2 probability
       y = 0.;
       mm = 0;
       for(k=0; k<it_size; k++){
	 M = (*it)[k];

	 if(!index) continue;
	 if(skip.data[k]) continue;
	 if(pList[M].size()<m) continue;

	 x = pList[M].getdata(atype,m-1);
	 x/= (atype=='W' || atype=='w') ? pList[M].data[m-1].noiserms : 1.; 
	 x/= (atype=='U' || atype=='u') ? pList[M].data[m-1].noiserms : 1.; 
	 x = fabs(x);

	 if(atype=='R' || atype=='r'){                  // get rank statistics
	   a = x+logbpp;                               
	   a-= log(1+pow(log(1+a*(1+1.5*exp(-a))),2));  // new approximation for rank SNR
	   a = sqrt(2*a);                               // was a = sqrt(2*1.07*(x+logbpp))-1.11/2.;
                                                        
	   if(x==0.) a = 0.;

	   if(c=='Y' || c=='z') {
	     if(atype=='R')        { y += x; mm++; }
	     if(atype=='r' && x>0) { y += x; mm++; }
	   }
	   else if(c=='e' || c=='C') { y += a*a; mm++; }     
	   else if(a*a>1.) { y += a*a; mm++; } 
	 }

	 else {                                         // get Gaussian statistics
	   if(c=='Y' || c=='z') {
	      a = pow(x+1.11/2,2)/2./1.07;
	      y+= (a>logbpp) ? a-logbpp : 0.;
	      if(atype=='S' || atype=='W' || atype=='P' || atype=='U') mm++;  // total size
	      else if(a>logbpp) mm++;                           // reduced size
	   }
	   else if(c=='e' || c=='C' || c=='w') { 
	     if(atype=='S' || atype=='W' || atype=='P'|| atype=='U') { 
	       y += x*x; mm++;                                  // total energy
	     }
	     else if(x>ampbpp) { y += x*x; mm++; }              // reduced energy
	   } 
	   else if(x*x>1.) { y += x*x; mm++; } 
	 }

       }

       if(c=='S' && mm) y -= double(mm);
       if(c=='w' && mm) y /= double(mm);
//       if(c=='C' && mm) { y = 1-Gamma(mm/2.,y/2.);  y = (y>0.) ? -log10(y) : 20.; }
       if(c=='z' && mm) y = gammaCL(y,mm);
//       if(c=='Y' && (atype=='s' || atype=='w')) y /= double(mm);

       out.data[out_size++] = y; 
       break;
       
     case 'g':          // get cluster grand amplitude
       y = 0.;
       for(k=0; k<it_size; k++){
	 M = (*it)[k];

	 if(!index) continue;
	 if(skip.data[k]) continue;
	 if(pList[M].size()<m) continue;

	 x = fabs(pList[M].getdata(atype,m-1));
	 if(x>y) y = x;
       }
       out.data[out_size++] = y;
       break;
       
     case 's':          // get cluster start time
     case 'd':          // get cluster stop time 
       a = 1.e99;
       b = 0.;
       for(k=0; k<it_size; k++){
	 M = (*it)[k];
	 if(skip.data[k]) continue;
	 y = 1./pList[M].rate;       // time resolution
	 mm= pList[M].layers;
	 x = index ? y*int(pList[M].data[m-1].index/mm) : 
	             y*int(pList[M].time/mm);
	 if(x  <a) a = x;         // first channel may be shifted 
	 if(x+y>b) b = x+y;
       }
       out.data[out_size++] = (c=='s') ? a : b;
       break;
       
     case 'l':             // get low frequency
     case 'h':             // get high frequency
       a = 1.e99;
       b = 0.;
       for(k=0; k<it_size; k++){
	 M = (*it)[k];
	 if(skip.data[k]) continue;
         mp= size_t(this->rate/pList[M].rate+0.1);
         mm= pList[M].layers;                   // number of wavelet layers
         double dF = mm==mp ? 0. : -0.5;        // wavelet : WDM
         y = pList[M].rate/2.;                  // frequency resolution
         x = y * (pList[M].frequency+dF);       // pixel frequency
	 if(x  <a) a = x;
	 if(x+y>b) b = x+y;
       }
       out.data[out_size++] = (c=='l') ? a : b;
       break;

     case 'L':             // get network likelihood
     case 'N':             // get network null stream
       a = 0.;
       x = 0.;
       for(k=0; k<it_size; k++){
	 M = (*it)[k];
	 if(skip.data[k]) continue;

	 if(atype=='S' || atype=='s' || atype=='P' || atype=='p' || !index) {
	   for(i=0; i<pList[M].size(); i++) {
	     x += pow(pList[M].data[i].asnr,2);
	   }
	   x -= 2*pList[M].likelihood;
	 }
	 else if(c=='N'){
	     b  = pList[M].data[m-1].asnr;
	     b -= pList[M].data[m-1].wave/pList[M].data[m-1].noiserms;
	     x += b*b;
	 }

	 a += pList[M].likelihood;     // pixel network likelihood
       }
       if(c=='N') a=x;
       out.data[out_size++] = a;
       break;

     case 't':			// get central time optimal resolution
     case 'T':			// get central time for all resolutions
     case 'f':			// get central frequency for optimal resolution
     case 'F':			// get central frequency for all resolutions
       a = 0.;
       b = 0.;
       
       if(c=='F' && (int)cFreq.size()>ID-1) {   // get supercluster frequency
	 out.data[out_size++] = cFreq[ID-1];
	 break;
       }
       if(c=='T' && (int)cTime.size()>ID-1) {   // get supercluster time
	 out.data[out_size++] = cTime[ID-1];
	 break;
       }
       
       for(k=0; k<it_size; k++) {
         M = (*it)[k];

	 if(!index && atype!='L') continue;
	 if(skip.data[k]) continue;
         if(pList[M].size()<m && atype!='L') continue;

	 mp= size_t(this->rate/pList[M].rate+0.1);
	 t = 1./pList[M].rate;                // wavelet time resolution
	 mm= pList[M].layers;                 // number of wavelet layers
         if(atype=='S' || atype=='s' || atype=='P' || atype=='p') {       // get Gaussian statistics
	    x = pList[M].getdata(atype,m-1);
	    x = x*x;
	 }
	 else if (atype=='R' || atype=='r'){  // get rank statistics
	    x = pList[M].getdata(atype,m-1);
	    x = pow(sqrt(2*1.07*(fabs(x)-log(bpp)))-1.11/2.,2);   
	 }
	 else {
	    x = pList[M].likelihood;
	 }
	 if(x<0.) x = 0.;

	 if(c=='t' || c=='T') {
           double dT = mm==mp ? 0.5 : 0.;                     // wavelet : WDM
           if(index)
             a += (pList[M].data[m-1].index/mm + dT)*x;       // pixel time sum
           else
             a += (pList[M].time/mm + dT)*x;                  // pixel time sum
           b += x/t;                                          // use weight x/t
	 }
	 else {
           double dF = mm==mp ? 0.5 : 0.;                     // wavelet : WDM
	   a += (pList[M].frequency+dF)*x;                    // pixel frequency sum
	   b += x*2.*t;                                       // use weight x*2t
	 }
       }

       out.data[out_size++] = b>0. ? a/b : -1.;
       break;

     case 'H':			// get calibrated hrss
     case 'n':			// get calibrated noise sigma^2
     case 'I':			// get calibrated noise 1/sigma^2

       mp  = 0;
       sum = 0.;
       out.data[out_size] = 0.;

       for(k=0; k<it_size; k++) {
         M = (*it)[k];

	 if(!index) continue;
	 if(skip.data[k]) continue;
         if(pList[M].size()<m) continue;

	 r = pList[M].getdata('N',m-1);

	 mp++;

	 if(c == 'H'){
	    a = pow(pList[M].getdata(atype,m-1),2);
	    if(atype=='S' || atype=='s' || atype=='P' || atype=='p') { a -= 1.; a *= r*r; }
	    sum += a<0. ? 0. : a;
	 }
	 else if(r>0){
	    sum += c=='n' ? r*r : 1./r/r;
	 }

       }

       if(c != 'H' && mp) { sum = sum/double(mp); }     // noise hrss
       if(c == 'I' && mp) { sum = 1./sum; }
       out.data[out_size++] = sum>0. ? float(log(sum)/2./log(10.)) : 0.;
       break;

     case 'v':
     default:
       for(k=0; k<it_size; k++){
	 M = (*it)[k];
	 if(!skip.data[k]) { 
	   out.data[out_size++] = it_size;
	   break; 
	 }
       }
       break;
     }
   }

   out.resize(out_size);
   return out;
}


//**************************************************************************
// extract WSeries for specified cluster ID and detector index
// does not work with WDM 
//**************************************************************************
double netcluster::getwave(int cid, WSeries<double>& W, char atype, size_t n)
{   
   if(!cList.size()) return 0.;
//   if(atype != 'W' && atype != 'S') return 0.;

   int t,offset;
   int ID, R;
   int level,mm;

   size_t k,f;
   size_t mp = 0;
   size_t max_layer;
   size_t max,min;
   size_t it_size;
   size_t it_core;
   size_t pixtime;

   double a;
   double tsRate = W.rate();
   double fl = tsRate/2.;
   double fh = 0.;
   double sum = 0.;
   double temp=0.;

   slice S;

   wavearray<int> skip;
   vector<vector_int>::iterator it;
   vector_int* pv = NULL;
   netpixel* pix;
   size_t M = pList.size();

// extract cluster

   for(it=cList.begin(); it!=cList.end(); it++){

     ID = pList[((*it)[0])].clusterID;
     it_size = it->size();

     if(ID != cid) continue;   // find the cluster
     if(!it_size) continue;    // skip empty cluster

     it_core = 0;
     skip.resize(it_size);

     pv =&(cRate[ID-1]);
     R = pv->size() ? (*pv)[0] : 0;

     max = 0;           // max time index
     min = 1234567890;  // min time index

     for(k=0; k<it_size; k++) {          // fill skip array
	M = (*it)[k];
	pix = &pList[M];

	skip.data[k] = 1;
	if(!pList[M].core) continue;
	if(R && int(pix->rate+0.1)!=R) continue;
	if(!R) R = int(pix->rate+0.1);                      // case of elementary clusters
        mm = pix->layers;                                   // number of wavelet layers
	pixtime = pix->getdata('I',n)/mm;
	if(pixtime > max) max = pixtime;                    // max pixel time index
	if(pixtime < min) min = pixtime;                    // min pixel time index
	skip.data[k] = 0;
	it_core++;
     }

     k = size_t((max-min+2*W.pWavelet->m_H)/R)+1;           // W duration in seconds 

//     cout<<"duration="<<k<<" filter="<<W.pWavelet->m_L<<endl;

// setup WSeries metadata

//     cout<<"wrate="<<tsRate<<" rate="<<R<<endl;

     W.resize(size_t(k*this->rate+0.1));
     W.setLevel(0);
     level = int(log(tsRate/R)/log(2.)+0.1);
     while(W.getMaxLevel()<level) W.resize(W.size()*2);
     W.Forward(level);
     W = 0.;

     max_layer = W.maxLayer();     

//     cout<<"maxlayer="<<max_layer<<" wsize="<<W.size()<<endl;

     S = W.getSlice(0);
     offset = (int(max+min) - int(S.size()))/2;
     W.start(double(offset)/R);

//     cout<<"min="<<min<<" max="<<max<<" Ssize="<<S.size()<<endl;
//     cout<<"offset="<<offset<<" wsize="<<W.size()<<endl;

     if(!it_core) continue;     // skip cluster

     for(k=0; k<it_size; k++){
	M = (*it)[k];
	pix = &pList[M];

	if(skip.data[k]) continue;
	if(!(pix->size())) continue;
        mm = pix->layers;                                   // number of wavelet layers
	pixtime = size_t(pix->getdata('I',n)/mm);

	f = pix->frequency;           // pixel frequency
	t = int(pixtime)-offset;      // pixel time index

	if(f > max_layer) continue;
	if(f*R/2. <= fl) { fl = f*R/2.; W.setlow(fl); }
	if((f+1)*R/2. >= fh) { fh = (f+1)*R/2.; W.sethigh(fh); }

	S = W.getSlice(f);

	if(t < 0 || size_t(t) >= S.size()) continue;
	if(pix->size() <= n) continue;

	a = pix->getdata(atype,n);
      	if(atype=='W') a /= pix->getdata('N',n);
	W.data[S.start()+t*S.stride()] = a;

	a = pix->getdata('N',n);
	mp++;  sum += 1./a/a;
	temp += pList[M].time/double(R);

     }

//     cout<<"cid="<<cid<<" size="<<mp<<"  "<<pow(W.rms(),2.)*W.size()/tsRate;
//     cout<<"  time="<<temp/mp<<" rate="<<R<<"  duration="<<W.size()/tsRate<<endl;

     sum = sqrt(double(mp)/sum); // noise hrss

     return sum;
   }
   return 0.;
}


wavearray<double> netcluster::getMRAwave(network* net, int ID, size_t ifo, char atype, int mode) 
{                                                                                                       
//**************************************************************************
// !!! may violate WDM parity for non-integer seconds lags !!!!
// construct waveform from MRA pixels at different resolutions 
// extract MRA waveforms for specified cluster ID and detector index n            
// works only with WDM. Create WSeries<> objects for each resolution,                             
// find principle components, fill in waveForm and waveBand arrays                                              
// atype = 'W' - get whitened detector output (Wavelet data)
// atype = 'w' - get detector output (Wavelet data)
// atype = 'S' - get whitened reconstructed response (Signal)
// atype = 's' - get reconstructed response (Signal)
// mode: -1/0/1 - return 90/mra/0 phase
//**************************************************************************
  wavearray<double> z; 

  if(!cList.size()) return z;

  bool signal = atype=='S' || atype=='s';
  bool strain = atype=='w' || atype=='s';

  int nRES = net->wdmList.size();

  double maxfLen = 0;                                    // find max filter length
  for(int l=0;l<nRES;l++) {
    double fLen = net->wdmList[l]->m_H/this->rate;
    if(maxfLen<fLen) maxfLen=fLen;     
  }                                                                                               

  WDM<double>* wdm;
  wavearray<double> x00;
  wavearray<double> x90;
  std::vector<int>* vint;
  wavearray<double> cid;
  netpixel* pix;

  cid = this->get((char*)"ID",0,'S',0);                  // get cluster ID

  int K = cid.size();

  for(int k=0; k<K; k++) {                               // loop over clusters

    int id = size_t(cid.data[k]+0.1);
    if(id!=ID) continue;            

    vint = &(this->cList[id-1]);                         // pixel list

    int V = vint->size();
    if(!V) continue;     

// find event time interval, fill in amplitudes

    double tmin=1e20;          
    double tmax=0.; 
    double T, a00, a90, rms;
    for(int j=0; j<V; j++) {   
      pix = this->getPixel(id,j);
      T = pix->time;                                      // get time index
      T = T/pix->rate/pix->layers;                        // time in seconds from the start
      if(T<tmin) tmin=T;                   
      if(T>tmax) tmax=T;
    }                                                                               

    tmin = int(tmin-maxfLen)-1;                           // start event time in sec
    tmax = int(tmax+maxfLen)+1;                           // end event time in sec
    z.resize(size_t(this->rate*(tmax-tmin)+0.1)); 
    z.rate(this->rate); z.start(tmin); z.stop(tmax), z=0;   

    int io = int(tmin*z.rate()+0.01);                     // index offset of z-array
    int M, j00, j90;

    //cout<<"+++ "<<tmin<<" "<<tmax<<" "<<io<<" "<<z.rate()<<" "<<z.size()<<" "<<maxfLen<<endl;

    float s00=0.;
    float s90=0.;

    for(int j=0; j<V; j++) {   
      pix = this->getPixel(id,j);
      if(!pix->core) continue;             

      rms = pix->getdata('N',ifo);
      a00 = signal ? pix->getdata('s',ifo) : pix->getdata('w',ifo);
      a90 = signal ? pix->getdata('p',ifo) : pix->getdata('u',ifo);
      a00*= strain ? rms : 1.;
      a90*= strain ? rms : 1.;
      wdm = net->getwdm(pix->layers);                      // pointer to WDM transform stored in network
      j00 = wdm->getBaseWave(pix->time,x00,false)-io;
      j90 = wdm->getBaseWave(pix->time,x90,true)-io;
      if(mode < 0) a00=0.; 		                   // select phase 90
      if(mode > 0) a90=0.; 		                   // select phase 00

      s00 += a00*a00;
      s90 += a90*a90;

      for(int i=0; i<x00.size(); i++){
         z.data[j00+i] += x00[i]*a00;
         z.data[j90+i] += x90[i]*a90;
      }
      //cout<<"*** "<<pix->layers<<" "<<pix->data[ifo].index<<" "<<j00<<" "<<j90<<" "
      //    <<x00.size()<<" "<<x90.size()<<" "<<z.size()<<endl;
    }
    //cout<<mode<<" s00/s90: "<<s00<<" "<<s90<<" "<<z.rms()*z.rms()*z.size()<<endl;

    break;
  }                            
  return z;
}          

size_t netcluster::write(const char *fname, int app)
{
// write the entire pixel structure into a file
// only metadata and pixels are written, no cluster metadata is stored
// app = 0 - open new file and dump metadata and pixel structure
// app = 1 - append pixel structure to existing file

  size_t i;
  size_t I = this->pList.size();     // number of pixels;
  FILE *fp;

  if(app) fp=fopen(fname, "ab");
  else    fp=fopen(fname, "wb");

  if(!fp) {
     cout<<"netcluster::write() error : cannot open file "<<fname<<"\n";
     fclose(fp); return 0;
  }

  if(!app) {                // write metadata
     if(!write(fp,app)) { fclose(fp); return 0; };
  }

  for(i=0; i<I; i++) {      // write pixels
     if(!pList[i].write(fp)) { fclose(fp); return 0; }
  }
  fclose(fp); 
  return I; 
}

size_t netcluster::write(FILE *fp, int app)
{
// write pixel structure with TD vectors attached into a file
// only some metadata and pixels are written, no cluster metadata is stored
// app = 0 - store metadata
// app = 1 - store pixels with the TD vectors attached by setTDAmp()

   size_t i,k;
   size_t I = this->pList.size();     // number of pixels;
   size_t II= 0;
   netpixel pix;

   // write metadata

   if(!app) {
      double db[9];
      double rest = this->nPIX*10+2*this->pair;
      db[0] = this->rate;               // original Time series rate 
      db[1] = this->start;              // interval start GPS time
      db[2] = this->stop;               // interval stop GPS time 
      db[3] = this->bpp;                // black pixel probability
      db[4] = this->shift;              // time shift
      db[5] = this->flow;               // low frequency boundary
      db[6] = this->fhigh;              // high frequency boundary
      db[7] = (double)this->run;        // run ID
      db[8] = rest;                     // store the rest   
      if(fwrite(db, 9*sizeof(double), 1, fp) !=1) {
         fclose(fp); return 0;
      }
      return 1;
   }
   
   if(!I) return 0;
   
   // write pixels
   
   for(i=0; i<this->cList.size(); ++i) {
      if(sCuts[i]==1) continue;
      
      const vector_int& v = cList[i];
      size_t K = v.size();
      bool skip = true;
      if(!K) continue;
      
      for(k=0; k<K; ++k) {                           // loop over pixels           
         if(pList[v[k]].tdAmp.size()) skip=false;
      }
      if(skip) continue;
      
      netpixel** pp = (netpixel**)malloc(K*sizeof(netpixel*));
      for(k=0; k<K; k++) pp[k] = &pList[v[k]];
      qsort(pp, K, sizeof(netpixel*), &compareLIKE); // sort pixels     
      
      for(k=0; k<K; k++) {                           // loop over pixels           
         if(!pp[k]->tdAmp.size()) continue;
         pix = *pp[k];
         pp[k]->clean();                             // clean TD amplitudes
         pix.neighbors.clear();
         if(k<K-1) pix.neighbors.push_back(1);
         if(k>0) pix.neighbors.push_back(-1);   
         // set right index (set in subNetCuts) for netcc[3] after pixels sorting 
         if(k==0) pix.ellipticity = pList[v[0]].ellipticity;
         if(k==1) pix.ellipticity = pList[v[1]].ellipticity;
         if(k==0) pix.polarisation = pList[v[0]].polarisation;
         if(k==1) pix.polarisation = pList[v[1]].polarisation;
         pix.write(fp);
         II++;
         //printf("k = %lu , clID = %lu\n", k, pix->clusterID);
         //printf("pixel %d from cluster %d written\n", k, i);
      }

      free(pp); 
   } 
   return II;
}


size_t netcluster::read(const char* fname)
{
// read pixel structure from file
// the entire content is loaded into pList structure
  size_t i;
  netpixel pix;

  FILE *fp = fopen(fname,"rb");

  if (!fp) {
     cout << "netcluster::read() error : cannot open file " << fname <<". \n";
     fclose(fp); return 0;
  }

  if(!read(fp,0)) {fclose(fp); return 0;}   // read metadata

  // read pixels
  bool end = false;
  do {
    pList.push_back(pix);
    i = pList.size()-1;
    end = pList[i].read(fp);
  } while(end);
  pList.pop_back();

  size_t I = pList.size();
  fclose(fp);
  if(I) this->cluster();
  return I;
}


// read clusters from file
size_t netcluster::read(FILE* fp, int nmax)
{ 
// read metadata and pixels stored in a file on cluster by cluster basis
// clusters should be contiguous in the file (written by write(FILE*)) 
// nmax = 0 - read metadata
// nmax > 0 - read no more than maxPix pixels from a cluster

   size_t i;
   
   if(nmax==0){                         // read nextcluster metadata from file
      this->clear();
      double db[9];
      if(fread(db, 9*sizeof(double), 1, fp) !=1) return 0;
      int kk = int(db[8]+0.1);          // rest
      db[8] += db[8]>0. ? 0.1 : -0.1;   // prepare for conversion to int
      this->rate = db[0];               // original Time series rate 
      this->start = db[1];              // interval start GPS time
      this->stop = db[2];               // interval stop GPS time 
      this->bpp = db[3];                // black pixel probability
      this->shift = db[4];              // time shift
      this->flow = db[5];               // low frequency boundary
      this->fhigh = db[6];              // high frequency boundary
      this->run = int(db[7]+0.1);       // run ID
      this->nPIX = kk/10;               // recover nPIX   
      this->pair = (kk%10)/2;           // recover pair   
      return 1;
   }
  
   // clean TD amplitudes in pList structure
   size_t I = pList.size();
   for(i=0; i<I; ++i) pList[i].clean(); // clean TD amplitudes
   
   // reads first pixel
   netpixel pix;
   bool stop = false;
   size_t II = 0;
   int ID = cList.size()+1;
   
   while(pix.read(fp)){
      II++;                                          // update counter
      if(II<2 && pix.neighbors.size()<1) stop=true;  // just one pixel
      if(II>1 && pix.neighbors.size()<2) stop=true;  // last pixel
      if((int)II>nmax) pix.clean();                  // clean cluster tail
      pix.clusterID = ID;                            // setup new ID
      pList.push_back(pix);                          // update pList and counter
      if(stop) break;
   }
   
   if(!II) return II;
   sCuts.push_back(0);                               // to be processed by likelihood
   
   std::vector<int> list(II);                        // update cluster list
   std::vector<int> vtof(NIFO);                      // recreate time configuration array
   std::vector<int> vtmp;                            // recreate sky index array
   for(i=0; i<II; ++i) list[i] = I++;
   cList.push_back(list);
   nTofF.push_back(vtof);
   p_Ind.push_back(vtmp);
   return II;
}


//**************************************************************************
// set arrays for time-delayed amplitudes in collected coherent pixels 
// !!! works only with WDM transformation
//**************************************************************************
size_t netcluster::loadTDamp(network &net, char c, size_t BATCH, size_t LOUD)
{
// set time-delayed amplitude vectors in collected coherent pixels 
// returns number of pixels to process, if zero - nothing to process
// net: network
// c: 'a','A' - delayed amplitudes, 'p','P' - delayed power
// BATCH - max number of pixels to process in one batch
// LOUD - max number of loudest pixels to process in a cluster
// state of pixel core field (true/false) indicates if the pixel was 
// visited/not_visited by loadTDamp. Time-delay amplitudes are attached only
// if core=false. Therefore, before the first loadTDamp call the core status 
// of pixels should be set to false.

   if(!net.ifoListSize()) { 
      cout<<"netcluster::setTDvec() error: empty network."; 
      exit(1); 
   }
   if(net.rTDF<=0.) { 
      cout<<"netcluster::setTDvec() error: run network::setDelayIndex() first."; 
      exit(1); 
   }

   WSeries<double>* pTF = net.getifo(0)->getTFmap();

   size_t i,j,k,K,KK;
   size_t batch = BATCH ? BATCH : this->size()+1;
   size_t L     = size_t(net.getDelay((char*)"MAX")*net.rTDF)+1;
   size_t nIFO  = net.ifoListSize();
   size_t loud  = LOUD ? LOUD : batch;
   size_t count = 0;
   size_t npix  = 0;
   int    waveR = int(pTF->wrate()+0.1);  // wavelet rate
   int ind;
   
   netpixel* pix;                         // pointer to pixel
   const vector_int* v;                   // pointer to cluster array of pixels
   wavearray<float> vec;                  // storage for delayed amplitudes

// set time delayed amplitudes

   for(i=0; i<this->cList.size(); i++){
      
      if(this->sCuts[i]==1) continue;                // skip rejected clusters
      v  = &(this->cList[i]);
      K  = v->size();
      if(!K) continue;
      KK = LOUD && K>loud ? loud : K;
      if(this->sCuts[i]==-2) {count+=KK; continue;}  // skip loaded clusters

      bool skip = true;
      npix = 0;
      for(k=0; k<K; k++) {                           // loop over pixels           
         pix = &(this->pList[(*v)[k]]);
         if(!pix->core) skip=false;     
         if(pix->tdAmp.size()) {skip=false; npix++;}     
      }
      if(npix==K) {count+=K; continue;}              // skip loaded clusters
      if(skip) continue;                             // skip processed clusters

// sort pixels

      netpixel** pp = (netpixel**)malloc(K*sizeof(netpixel*));
      for(k=0; k<K; k++) pp[k] = &(pList[(*v)[k]]);
      qsort(pp, K, sizeof(netpixel*), &compareLIKE);

      skip = false;
      npix = 0;
      for(k=0; k<KK; k++) {                          // loop over loud pixels           
         pix = pp[k];

         if(count>=batch) {skip = true; break;} 
         if(pix->tdAmp.size() && pix->core) {                     
            npix++;                                  // count loaded pixels in the cluster
            count++;                                 // count loaded pixels in the batch
            continue;                                // skip loaded pixels
         }
         if(!pix->core) count++;                     //
         else continue;                              // skip processed pixels
         if(int(pix->rate+0.1)!= waveR) continue;    // skip wrong resolutions

         for(j=0; j<nIFO; j++) {                     // loop over detectors           
            pTF = net.getifo(j)->getTFmap();         // pointer to TF array
            ind = int(pix->data[j].index);           // index in TF array
            vec = pTF->pWavelet->getTDvec(ind,L,c);  // obtain TD vector
            pix->tdAmp.push_back(vec);               // store TD vector
         }

         pix->core = true;                           // mark pixel as processed 
         npix++;                                     
      }
      //cout<<i<<" "<<npix<<" "<<KK<<" "<<K<<" "<<LOUD<<endl;
      free(pp);
      if(LOUD && npix==KK) this->sCuts[i] = -2;      // mark fully loaded cluster 
      if(skip) break;
   }      
   return count;
}


//**************************************************************************
// set arrays for time-delayed amplitudes in collected coherent pixels 
// !!! works only with WDM transformation
//**************************************************************************
size_t netcluster::loadTDampSSE(network &net, char c, size_t BATCH, size_t LOUD)
{
// set time-delayed amplitude vectors in collected coherent pixels 
// fast version by using sparse TF arrays and SSE instructions
// returns number of pixels to process, if zero - nothing to process
// net: network
//   c: 'a','A' - delayed amplitudes, 'p','P' - delayed power
// BATCH - max number of pixels to process in one batch
// LOUD - max number of loudest pixels to process in a cluster
// state of pixel core field (true/false) indicates if the pixel was 
// visited/not_visited by loadTDamp. Time-delay amplitudes are attached only
// if core=false. Therefore, before the first loadTDamp call the core status 
// of pixels should be set to false.

   if(!net.ifoListSize()) { 
      cout<<"netcluster::setTDvec() error: empty network."; 
      exit(1); 
   }
   if(net.rTDF<=0.) { 
      cout<<"netcluster::setTDvec() error: run network::setDelayIndex() first."; 
      exit(1); 
   }

   size_t i,j,k,K,KK,M;
   size_t batch = BATCH ? BATCH : this->size()+1;
   size_t L     = size_t(net.getDelay((char*)"MAX")*net.rTDF)+1;
   size_t nIFO  = net.ifoListSize();
   size_t loud  = LOUD ? LOUD : batch;
   size_t count = 0;
   size_t npix  = 0;
   size_t nres  = net.getifo(0)->ssize();  // number of resolutions
   int ind;
   
   netpixel* pix;                          // pointer to pixel
   const vector_int* v;                    // pointer to cluster array of pixels
   wavearray<float> vec;                   // storage for delayed amplitudes
   WDM<double>* wdm;                       // pointer to wdm transform
   SSeries<double>* pSS;                   // pointer to sparse TF map

// set time delayed amplitudes

   for(i=0; i<this->cList.size(); i++){
      
      if(this->sCuts[i]==-1) continue;               // skip analized clusters
      if(this->sCuts[i]==1) continue;                // skip rejected clusters
      v  = &(this->cList[i]);
      K  = v->size();
      if(!K) continue;
      KK = LOUD && K>loud ? loud : K;
      if(this->sCuts[i]==-2) {count+=KK; continue;}  // skip loaded clusters

      bool skip = true;
      npix = 0;
      for(k=0; k<K; k++) {                           // loop over pixels           
         pix = &(this->pList[(*v)[k]]);
         if(!pix->core) skip=false;     
         if(pix->tdAmp.size()) {skip=false; npix++;}     
      }
      if(npix==K) {count+=K; continue;}              // skip loaded clusters
      if(skip) continue;                             // skip processed clusters

// sort pixels

      netpixel** pp = (netpixel**)malloc(K*sizeof(netpixel*));
      for(k=0; k<K; k++) pp[k] = &(pList[(*v)[k]]);
      qsort(pp, K, sizeof(netpixel*), &compareLIKE);

      skip = false;
      npix = 0;
      for(k=0; k<KK; k++) {                             // loop over loud pixels           
         pix = pp[k];

         if(count>=batch) {skip = true; break;} 
         if(pix->tdAmp.size() && pix->core) {                     
            npix++;                                     // count loaded pixels in the cluster
            count++;                                    // count loaded pixels in the batch
            continue;                                   // skip loaded pixels
         }
         if(!pix->core) count++;                     
         else continue;                                 // skip processed pixels

         for(j=0; j<nIFO; j++) {                        // loop over detectors           
            ind = net.getifo(j)->getSTFind(pix->rate);  // pointer to sparse TF array
            pSS = net.getifo(j)->getSTFmap(ind);        // pointer to sparse TF array
              M = pSS->maxLayer()+1;                    // number of layers
            wdm = net.getwdm(M);                        // pointer to WDM transform stored in network
            ind = int(pix->data[j].index);              // index in TF array
            vec = wdm->getTDvecSSE(ind,L,c,pSS);        // obtain TD vector

            for(int qqq=0; qqq<vec.size(); qqq++){
               if(fabs(vec.data[qqq])>1.e6) cout<<vec.data[qqq]<<" nun in loadTDampSSE\n";
            }

            pix->tdAmp.push_back(vec);                  // store TD vector
         }

         pix->core = true;                              // mark pixel as processed 
         npix++;                                     
      }
      //cout<<i<<" "<<npix<<" "<<KK<<" "<<K<<endl;
      free(pp);
      if(LOUD && npix==KK) this->sCuts[i] = -2;         // mark fully loaded cluster 
      if(skip) break;
   }      
   return count;
}


size_t netcluster::write(TFile *froot, TString tdir, TString tname, int app, int cycle, int cID)
{                                                                                                 
// write pixel structure with TD vectors attached into a file                                     
// froot    - root file pointer                                                                   
// tdir    - internal root directory where the tree is located                                    
// tname   - name of tree containing the cluster                                                  
// app = 0 - store light netcluster                                                               
// app = 1 - store pixels with the TD vectors attached by setTDAmp()                              
// app < 0 - store all pixels                                                                     
// cycle   - sim -> it is factor id : prod -> it is the lag number                                
// cID     - cluster id (cID=0 -> write all clusters)

   size_t i,k;
   size_t I = this->pList.size();     // number of pixels;
   size_t II= 0;                                          

   // check root file mode
   if(TString(froot->GetOption())!="CREATE" && TString(froot->GetOption())!="UPDATE") {
      cout<<"netcluster::write error: input root file wrong mode (must be CREATE or UPDATE) "
          <<froot->GetPath()<<endl;                                                          
      exit(1);                                                                               
   } else froot->cd();                                                                       

   // check if tdir exists otherwise it is created
   TDirectory* cdtree = tdir!="" ? (TDirectory*)froot->Get(tdir) : (TDirectory*)froot;
   if(cdtree==NULL) cdtree = froot->mkdir(tdir);      
   cdtree->cd();                                      

   int cid;
   int rate;					  // rate = pix->rate 
   int orate;					  // optimal rate
   float ctime;        				  // supercluster central time
   float cfreq;        				  // supercluster central frequency
   netpixel* pix = new netpixel;

   // check if tree tname exists otherwise it is created
   char trName[64];sprintf(trName,"%s-cycle:%d",tname.Data(),cycle);
   TTree* tree = (TTree*)cdtree->Get(trName);            
   if(tree==NULL) {                                     
     tree = new TTree(trName,trName);                     
     tree->Branch("cid",&cid,"cid/I");                  
     tree->Branch("rate",&rate,"rate/I");            
     tree->Branch("orate",&orate,"orate/I");            
     tree->Branch("ctime",&ctime,"ctime/F");            
     tree->Branch("cfreq",&cfreq,"cfreq/F");            
     tree->Branch("pix","netpixel",&pix,32000,0);       
   } else {                                             
     tree->SetBranchAddress("cid",&cid);                
     tree->SetBranchAddress("rate",&rate);            
     tree->SetBranchAddress("orate",&orate);            
     tree->SetBranchAddress("ctime",&ctime);            
     tree->SetBranchAddress("cfreq",&cfreq);            
     tree->SetBranchAddress("pix",&pix);                
   }                                                    

   // disable the AutoFlush mechanism : corrupts the tree (probably a ROOT bug)
   tree->SetAutoFlush(0);

   // write metadata netcluster object to the tree user info 
                                                             
   if(!app) {                                                
      TList* ulist = tree->GetUserInfo();
      if(ulist->GetSize()>0) return 1;              // cluster header already presents
      netcluster* nc = new netcluster;                                                
      nc->cpf(*this,false,0);                                                          
      nc->clear();                                                                    
      char title[256];sprintf(title,"cycle:%d",cycle);
      nc->SetTitle(title);
      ulist->Add(nc);                                                   
      return 0;                                                                       
   }                                                                                  

   if(!I) return 0;                                                                   
                                                                                      
   // write pixels to the cluster tree                                                
                                                                                      
   for(i=0; i<this->cList.size(); ++i) {                                              
      if(sCuts[i]==1) continue;                                                       
                                                                                      
      const vector_int& r = cRate[i];                                                 
      orate = r.size()>0 ? r[0] : 0;	    	    // optimal rate

      ctime = cTime[i];				    // supercluster central time
      cfreq = cFreq[i];				    // supercluster central frequency

      const vector_int& v = cList[i];                                                 
      size_t K = v.size();                                                            
      if(!K) continue;                              // skip empty clusters

      if((cID!=0)&&(pList[v[0]].clusterID!=cID)) continue;

      bool skip = true;                                                               
      for(k=0; k<K; ++k) {                           // loop over pixels              
         if(pList[v[k]].tdAmp.size()) skip=false;                                     
      }                                                                               
      if(app>0 && skip) continue;                                                     
                                                                                      
      netpixel** pp = (netpixel**)malloc(K*sizeof(netpixel*));                        
      for(k=0; k<K; k++) pp[k] = &pList[v[k]];                                        
      qsort(pp, K, sizeof(netpixel*), &compareLIKE); // sort pixels                   
                                                                                      
      for(k=0; k<K; k++) {                           // loop over pixels              
         if(app>0 && !pp[k]->tdAmp.size()) continue;                                  
         *pix = *pp[k];                                                               
         pp[k]->clean();                             // clean TD amplitudes           
         pix->neighbors.clear();                                                      
         if(k<K-1) pix->neighbors.push_back(1);                                       
         if(k>0) pix->neighbors.push_back(-1);                                        

         // set right index (set in subNetCuts) for netcc[3] after pixels sorting 
         if(k==0) pix->ellipticity = pList[v[0]].ellipticity;
         if(k==1) pix->ellipticity = pList[v[1]].ellipticity;
         if(k==0) pix->polarisation = pList[v[0]].polarisation;
         if(k==1) pix->polarisation = pList[v[1]].polarisation;
         cid = pix->clusterID;                       // setup new ID                 
         rate = pix->rate;
         tree->Fill();                                                                
         II++;                                                                        

         //printf("k = %lu , clID = %lu\n", k, pix->clusterID);                       
         //printf("pixel %d from cluster %d written\n", k, i);                        
      }                                                                               

      free(pp);
   }                                                                                  

   delete pix;

   return II;
}            

// read clusters from root file
std::vector<int>               
netcluster::read(TFile* froot, TString tdir, TString tname, int nmax, int cycle, int rate, int cID)
{                                                                                                   
// read metadata and pixels stored in a file on cluster by cluster basis                            
// clusters should be contiguous in the file (written by write(FILE*))                              
// froot    - root file pointer                                                                     
// tdir     - internal root directory where the tree is located                                     
// tname    - name of tree containing the cluster                                                   
// nmax = 0 - read metadata                                                                         
// nmax > 0 - read no more than maxPix heavy pixels from a cluster                                  
// nmax < 0 - read all heavy pixels from a cluster                                                  
// cycle    - sim -> it is factor id : prod -> it is the lag number                                 
// rate     - wavelet layer rate 
// cID      - cluster ID                                                                            

   size_t i;
   int cid; 
   int orate; 
   float ctime; 
   float cfreq; 
   netpixel* pix = new netpixel;

   std::vector<int> list;                            // cluster list
   std::vector<int> vtof(NIFO);                      // time configuration array
   std::vector<int> vtmp;                            // sky index array
   std::vector<float> varea;                         // sky error regions array
   std::vector<float> vpmap;                         // sky pixel map array
   clusterdata cd;                                   // dummy cluster data

   if(nmax<0) nmax=std::numeric_limits<int>::max();

   // check if dir tdir exist 
   TObject* obj = tdir!="" ? froot->Get(tdir) : (TObject*)froot;
   if(obj==NULL) {
      cout<<"netcluster::read error: input dir " << tdir << " not exist" << endl;
      exit(1);
   }
   if(tdir!="" && !TString(obj->ClassName()).Contains("TDirectory")) {
      cout<<"netcluster::read error: input dir " << tdir << " is not a directory" << endl;
      exit(1);                                                                            
   }                                                                                      
   TDirectory* cdtree = tdir!="" ? (TDirectory*)froot->Get(tdir) : (TDirectory*)froot;
   if(cdtree==NULL) {                                                                     
      cout<<"netcluster::read error: tree dir " << tdir                                   
          << " not present in input root file " << froot->GetPath() << endl;              
      exit(1);                                                                            
   } else cdtree->cd();                                                                   

   // check if tree tname exist 
   char trName[64];sprintf(trName,"%s-cycle:%d",tname.Data(),cycle);
   TTree* tree = (TTree*)cdtree->Get(trName);
   if(tree==NULL) {                         
     cout<<"netcluster::read error: tree " << trName 
         << " not present in input root file " << froot->GetPath() << endl;
     exit(1);                                                              
   } else {                                                                
     tree->SetBranchAddress("cid",&cid);                                   
     tree->SetBranchAddress("orate",&orate);                                   
     tree->SetBranchAddress("ctime",&ctime);                                   
     tree->SetBranchAddress("cfreq",&cfreq);                                   
     tree->SetBranchAddress("pix",&pix);                                   
   }                                                                       
                                                                           
   // Extract requested infos from tree                                    
   char sel[128]="";
   if(rate>0) sprintf(sel,"rate==%d ",rate);        
   if(cID) {if(TString(sel)=="") sprintf(sel,"cid==%d",cID); 
            else                 sprintf(sel,"%s && cid==%d",sel,cID);}
   if(TString(sel)=="") sprintf(sel,"1==1");	// TTreeFormula needs a non empty string
   TTreeFormula cut("cuts", sel, tree);	        // Draw substituted with TTreeFormula because of bug 
                                                // sstrace -> -1 ENOENT (No such file or directory)
   // this get rid of the unwanted htemp;1 in root file created by Draw
   if(cdtree->Get("htemp")!=NULL) cdtree->Delete("htemp");

   // fill entry,icint arrays with the selected entries 
   size_t size = tree->GetEntries();
   wavearray<double> entry(size); 
   wavearray<double> icid(size); 
   int cnt=0;
   tree->SetBranchStatus("orate",false);
   tree->SetBranchStatus("ctime",false);
   tree->SetBranchStatus("cfreq",false);
   tree->SetBranchStatus("pix",false);
   for(i=0;i<size;i++) {                                                 
     tree->GetEntry(i);                                           
     if(cut.EvalInstance()==0) continue;
     entry[cnt]=i;
     icid[cnt]=cid;
     cnt++;
   }
   tree->SetBranchStatus("orate",true);
   tree->SetBranchStatus("ctime",true);
   tree->SetBranchStatus("cfreq",true);
   tree->SetBranchStatus("pix",true);
   size=cnt;

   // fill vector with cluster ids
   std::vector<int> clist;        
   for(i=0;i<size;i++) clist.push_back(int(icid[i]+0.5));
   // erase duplicate entries
   clist.erase(stable_partition(clist.begin(), clist.end(), dupChecker_()), clist.end());                
                                                                  
   if(nmax==0) {                                                  
      if(rate<=0 && cID==0){                   // read nextcluster metadata from file
        TList* ulist = tree->GetUserInfo();
        if(ulist->GetSize()==0) {  
          cout<<"netcluster::read error: header is null" << endl; exit(1);       
        }
        this->clear();                                                               
        *this = *(netcluster*)ulist->At(0);
        delete tree;
        return clist;  		       	       // return the full list of clusters in the tree
      } else {                                                                                
        delete tree;
        return clist;                          // return the selected list of clusters in the tree
      }                                                                                           
   }                                                                                              

   if(rate<0) {cout<<"netcluster::read error: input rate par must be >= 0" << endl; exit(1);}

   int os=0;
   for(size_t k=0;k<clist.size();k++) {        // read cluster id in the list

     int id = clist[k];
                                                                           
     // clean TD amplitudes in pList structure                             
     size_t I = pList.size();                                              
     for(i=0; i<I; ++i) pList[i].clean();      // clean TD amplitudes           
                                                                           
     // reads first pixel                                                  
     bool stop = false;                                                    
     size_t II = 0;                                                           
     int ID = cList.size()+1;                                              
                                                                           
     std::vector<int> crate;        
     for(i=os;i<size;i++) {
       if(int(icid[i]+0.5)!=id) continue;
       tree->GetEntry(int(entry[i]+0.5));                                           
       II++;                                           // update counter   
       if(II<2 && pix->neighbors.size()<1) stop=true;  // just one pixel   
       if(II>1 && pix->neighbors.size()<2) stop=true;  // last pixel       
       if(II>nmax) pix->clean();                       // clean cluster tail
       pix->clusterID = ID;                            // setup new ID      
       pList.push_back(*pix);                          // update pList and counter
       if(orate&&!crate.size()) crate.push_back(orate);// update cluster rate
       if(os==i) os++;				       // updated the already processed entry index 
       if(stop) break;                                                            
     }                                                                            
                                                                                  
     if(!II) {delete tree;return clist;}                                                        
     sCuts.push_back(0);                               // to be processed by likelihood

     list.clear(); 
     for(i=0; i<II; ++i) list.push_back(I++); 	// update cluster list
     cList.push_back(list);				
     cRate.push_back(crate);				
     cTime.push_back(ctime);
     cFreq.push_back(cfreq);
     sArea.push_back(varea);			// recreate sky error regions array
     p_Map.push_back(vpmap);			// recreate sky pixel map array
     nTofF.push_back(vtof);			// recreate time configuration array
     p_Ind.push_back(vtmp);			// recreate sky index array
     cData.push_back(cd);			// recreate dummy cluster data
   }                                                                                   
   delete pix;                                                                         
   delete tree;
   return clist;                                     // return the selected list of clusters in the tree
}                                                                                                       

void netcluster::print()
{                       
  int nhpix = 0;        // total number of heavy pixels in the netcluster object 
  for(size_t i=0; i<this->cList.size(); ++i) {                                      
    if(sCuts[i]==1) continue;                                                    
                                                                                 
    const vector_int& v = cList[i];                                              
    size_t K = v.size();   
    if(!K) continue;
                                                                                 
    for(size_t k=0; k<K; ++k) {                      // loop over pixels           
      if(pList[v[k]].tdAmp.size()) nhpix++;
    }
  }
      ;
  cout << endl;
  cout.precision(14);
  cout << "rate\t= "  << this->rate  << endl;
  cout << "start\t= " << this->start << endl;
  cout << "stop\t= "  << this->stop  << endl;
  cout << "shift\t= " << this->shift << endl;
  cout << "bpp\t= "   << this->bpp   << endl;
  cout << "flow\t= "  << this->flow  << endl;
  cout << "ghigh\t= " << this->fhigh << endl;
  cout << "run\t= "   << this->run   << endl;
  cout << "nPIX\t= "  << this->nPIX  << endl;
  cout << "pair\t= "  << this->pair  << endl;
  cout << endl;
  cout << "cList\t= "  << this->cList.size() << endl;
  cout << "pList\t= "  << this->pList.size() << endl;
  cout << "hpList\t= " << nhpix << endl;
  cout << endl;

  return;
}



