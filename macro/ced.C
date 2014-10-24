/** 

Addendum on embedded Coherent Event Display. 

Coherent Event Display is now able to run in a standalone mode via a macro called from CED.C. It has the function prototype:

int CED(network* NET, char* base_web_dir, double rho_t, int lag=-1)

The parameters:
network* NET : pointer to the current state of the network. The way the function is writen, the network must be in the same state (delay filters, search type, resolution) as the most recently produced events (and hence those you wish to potentially output). The most convenient place for this call is here:

		// remove weak glitches
		cout<<"rejected weak pixels: "<<NET.netcut(netRHO,'r',0,1)<<"\n";
		// remove loud glitches
		cout<<"rejected loud pixels: "<<NET.netcut(netCC,'c',0,1)<<"\n"; 
		cout<<"reconstructed events: "<<NET.events()<<"\n";
- 
-				if(CED) CED(&NET, base_web_dir, ced_rho);
- 
		if(i<l_high) NET.Forward(1);
	}

	system("date");

char* base_web_dir : This is the base web directory for the output. Each event which is selected for output will create a new directory here in the following format:

	base_web_dir/#{ifo0}#{ifo1}#{..}_#{ifo0_gps}_#{ifo1_gps}_#{..}

double rho_t : Threshold on rho[1]. This controls the output rate. Setting this low will increase (exponentially) the number of events output. Placing this around 3.0 or higher is safest.

int lag : If only a specific lag is desired, set this. -1 corresponds to all lags.

There is a second function:

int CED(network* NET, char* base_web_dir, bool (*cut)(netevent *n),  int lag=-1)

Instead of a rho threshold, it takes a pointer to a function with one parameter, the current netevent, and returns a boolean. This can be used for more precise cuts.

OTHER NOTES: 
- There is an additional global variable CED_VERBOSE which is set inside the script, at the top. Activating this will activate verbose output. It is false by default and should run silently.

- A simulation version of the script is possible with the current script, but will output at all factors (which is probably undesirable). That will be fixed soon.

**/

/*** Compiled mode only ****
#include "../wavearray.hh"
#include "../wseries.hh"
#include "../network.hh"
#include "../netevent.hh"
#include "../skymap.hh"

#include "WTSpectrum.C"
#include "Plot.C"

#include "TCanvas.h"
***************************/

#include <vector>

/*** set to true for verbose output *****/
bool CED_VERBOSE = false;
/****************************************/

int CED(network* NET, char* base_web_dir, double rho_t, double factor, int lag=-1){

  TFplot pF; null(pF);
  Tplot  pT;  null(pT);
  Tplot  tP;  null(tP);

  int outdir = 0;
  if ( CED_VERBOSE )
    cout<<"Starting Coherent Event Display (CED)." << endl;

  int nIFO = NET->ifoListSize(),
    rate = int(2*NET->getifo(0)->TFmap.resolution(0)+0.5);
  
  //Fill in all skymaps
  double old_cc = NET->netCC, old_rho = NET->netRHO;
  NET->netCC = -1; 
  NET->netRHO = 0;
  
  if ( CED_VERBOSE )
    cout << "Search rate: " << rate << endl;
  
  std::vector<char*> ifonames = NET->ifoName;
  // Compiled mode only
  //std::vector<detector*> pD;
  detector **pD = new detector*[nIFO];

  for(int i=0; i<nIFO; i++)
    pD[i] = NET->getifo(i);

  //Compiled mode only
  //for(int i=0; i<nIFO; i++)
  //pD.push_back(NET->getifo(i));
  
  // netevent object for holding event parameters
  netevent event(nIFO);
  // plot production object
  TCanvas *tcanvas_ptr;
  
  if( CED_VERBOSE )
    cout << "Base web directory: " << base_web_dir << endl;
  
  int lags = NET->nLag;
  for(int lag_index=0; lag_index<lags; lag_index++){
    // If a specific lag is desired:
    if ( lag >= 0 )
      if ( lag != lag_index ) continue;
    
    if( CED_VERBOSE )
      cout << "Lag: " << lag_index << endl;
    // Get event IDs
    wavearray<double> wa_id = NET->getwc(lag_index)->get("ID", 0, 'L', rate);
    
    // Selection of event ids
    wavearray<double> *ifo_start = new wavearray<double>[nIFO], 
      *ifo_stop = new wavearray<double>[nIFO], 
      *ifo_lag = new wavearray<double>[nIFO], 
      cluster_ind;
    
    if( CED_VERBOSE ){
      cout << "Lag: " << lag_index << " with " << wa_id.size() << " events." 
	   << endl;
    }
    
    for(int cluster_index=0; cluster_index < (int)wa_id.size(); cluster_index++){
      
      int event_cluster_id = int(wa_id[cluster_index]+0.5);
      if( CED_VERBOSE )
	cout << "Clust ID: " << cluster_index << "/" << event_cluster_id << endl;
      
      event.output(NULL,NET, factor, event_cluster_id, lag_index);
      
      /**** Find event IDs to run on ***/
      // rho cut
      if( event.netcc[0] < 0 | event.ECOR < 0 | event.rho[1] < rho_t ) continue;
      
      if( CED_VERBOSE ){
	cout << "Found: " << cluster_index << endl;
	printf("time: %f\n", event.time[0]);
	cout << "rho: " << event.rho[1] << endl;
	cout << "netcc: " << event.netcc[0] << endl;
	cout << "neted0: " << event.neted[0] << endl;
	cout << "neted1: " << event.neted[1] << endl;
	cout << "penalty: " << event.penalty << endl;
	cout << "---" << endl;
      }
      
      // store selecteed event parameters
      for(int j=0; j<nIFO; j++){
	ifo_start[j].append(event.start[j]);
	ifo_stop[j].append(event.stop[j]);
	ifo_lag[j].append(event.lag[j]);
      }
      cluster_ind.append(cluster_index);
    }
    
    // Loop over found events
    if( CED_VERBOSE )
      cout << cluster_ind.size() << " events to display this lag." << endl;
    
    for(int ijk=0; ijk<int(cluster_ind.size()); ijk++){
      
      // Retrieve the index in the cluster array
      int cluster_index = int(cluster_ind[ijk]+0.5);
      // Retrieve its ID
      int event_cluster_id = int(wa_id[cluster_index]+0.5);
      
      char web_dir[500] = "", command[500] = "", ifostr[20] = "";
      sprintf(ifostr, "");
      for(int d=0; d<(int)ifonames.size(); d++){
	sprintf(ifostr,"%s%s", ifostr, ifonames[d]);
      }
      sprintf(web_dir, "%s/%s", base_web_dir, ifostr);
      for(int d=0; d<nIFO; d++){
	sprintf(web_dir, "%s_%.3f", web_dir, ifo_start[d].data[ijk]);
      }
      
      if( CED_VERBOSE )
	cout << "Creating event output directory " << web_dir << endl;
      
      sprintf(command,"mkdir -p %s", web_dir);
      system(command);
      outdir = 1;                    // created output directory
      
      // filename for plots
      char plots_filename[FILENAME_MAX], dump_filename[FILENAME_MAX],
	file[FILENAME_MAX];
      
      //wavearray<double> event_start, event_stop; 
      
      if( CED_VERBOSE )
	cout << "Doing " << cluster_index << " in lag " << lag_index << endl;
      
      double event_start = 1e50, event_stop = 0;
      int d = 0;
      do{
	event_start = ifo_start[d].data[ijk] - ifo_lag[d].data[ijk] 
	  < event_start ?  ifo_start[d].data[ijk] - ifo_lag[d].data[ijk] 
	  : event_start ;
	event_stop = ifo_stop[d].data[ijk] - ifo_lag[d].data[ijk]
	  > event_stop ?  ifo_stop[d].data[ijk] - ifo_lag[d].data[ijk] 
	  : event_stop ;
	
	if( CED_VERBOSE ){
	  cout << "Event " << cluster_index << " start" << endl;
	  cout << "Detector: " << ifonames[d] << endl;
	  printf("gps start time = %f\n", ifo_start[d].data[ijk]);
	  printf("gps stop time = %f\n", ifo_stop[d].data[ijk]);
	  printf("lag = %f\n", ifo_lag[d].data[ijk]);
	}
	
      } while( ++d < nIFO );
      
      //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      // start of website plot production
      //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      
      if( CED_VERBOSE )
	cout << "Website plot production..." << endl;
      
      //get segment_start_time
      double segment_start_time = NET->getifo(0)->TFmap.start();
      
      if( CED_VERBOSE )
	printf("Segment start time %f\n", segment_start_time);
      
      // event has been found
      if( CED_VERBOSE ){
	cout << "lag_index " << lag_index << " cluster index " << cluster_index 
	     << " event ID " << event_cluster_id << endl;
	cout << " Resolution " << NET->getifo(0)->TFmap.resolution(0) << endl;  
	cout << "Likelihood, index " << cluster_index << " id " 
	     << event_cluster_id << " with type " << NET->tYPe << endl;
      }
      
      NET->likelihood(NET->tYPe, NET->acor, event_cluster_id, lag_index);
      
      // get event parameters
      event.output(NULL,NET, factor, event_cluster_id, lag_index);
      
      /****** calculate extra parameters ******/
      
      // geometric significance, set up for all detector combinations
      if( CED_VERBOSE )
	cout << "Significance." << endl;
      
      double log_gs = 0.0;
      for(int d_indx=0; d_indx < nIFO; d_indx++)
	log_gs *= event.rSF[d_indx];
      if(log_gs != 0.0)
	log_gs = log(log_gs)/nIFO;
      
      // event parameters file
      if( CED_VERBOSE )
	cout << "Parameters." << endl;
      FILE *event_params;
      
      sprintf(file, "%s/event.dat", web_dir);
      if ((event_params = fopen(file, "w")) == NULL) {
	cerr << "ERROR: Can't open output file: " << web_dir << "/event.dat" 
	     << endl;
      }
      
      for(int d=0; d<nIFO; d++) {
	fprintf(event_params, "EVENT\n");
	fprintf(event_params, "ifo %s\n", ifonames[d]);
	fprintf(event_params, "search waveburst\n");
	if( NET->mdcList.size() > 0 )
	  fprintf(event_params, "type simulation\n");
	else fprintf(event_params, "type production\n");
	fprintf(event_params, "start_time %d\n", int(event.start[d]));
	fprintf(event_params, "start_time_ns %d\n", int((event.start[d] - int(event.start[d])) * 1000000000));
	fprintf(event_params, "stop_time %d\n", int(event.stop[d]));
	fprintf(event_params, "stop_time_ns %d\n", int((event.stop[d] - int(event.stop[d])) * 1000000000));
	fprintf(event_params, "central_time %d\n", int(event.time[d]));
	fprintf(event_params, "central_time_ns %d\n", int((event.time[d] - int(event.time[d])) * 1000000000));
	fprintf(event_params, "hrss %g\n", event.hrss[d]);
	fprintf(event_params, "time_lag %f\n", event.lag[d]);
	fprintf(event_params, "snr %f\n", event.snr[d]);
	fprintf(event_params, "rank_snr %f\n", event.rSNR[d]);
	fprintf(event_params, "rank_significance %f\n", event.rSF[d]);
	fprintf(event_params, "gaussian_significance %f\n", event.gSF[d]);
	fprintf(event_params, "noise %g\n", event.noise[d]);
	fprintf(event_params, "likelihood %f\n", event.likelihood);
	fprintf(event_params, "phi %f\n", event.phi[0]);
	fprintf(event_params, "theta %f\n", event.theta[0]);
	fprintf(event_params, "psi %f\n", event.psi[0]);
	fprintf(event_params, "size %d\n", event.size[0]);
	fprintf(event_params, "volume %d\n", event.volume[0]);
	fprintf(event_params, "flow %f\n", event.low[0]);
	fprintf(event_params, "fhigh %f\n", event.high[0]);
	fprintf(event_params, "segment_time %d\n", (int)segment_start_time);
	fprintf(event_params, "segment_time_ns 0\n");
	fprintf(event_params, "geometric_significance %f\n", log_gs);
	fprintf(event_params, "duration %f\n", event.duration[0]);
	fprintf(event_params, "central_freq %f\n", event.frequency[0]);
	fprintf(event_params, "bandwidth %f\n", event.bandwidth[0]);
	fprintf(event_params, "wavelet_rate %d\n", event.rate[0]);
	fprintf(event_params, "ndm_0 %f\n", event.ndm[0]);
	fprintf(event_params, "ndm_1 %f\n", event.ndm[1]);
	fprintf(event_params, "ndm_2 %f\n", event.ndm[2]);
	fprintf(event_params, "ndm_3 %f\n", event.ndm[3]);
	fprintf(event_params, "ndm_4 %f\n", event.ndm[4]);
	fprintf(event_params, "ndm_5 %f\n", event.ndm[5]);
	fprintf(event_params, "ndm_6 %f\n", event.ndm[6]);
	fprintf(event_params, "ndm_7 %f\n", event.ndm[7]);
	fprintf(event_params, "ndm_8 %f\n", event.ndm[8]);
	fprintf(event_params, "ndm_9 %f\n", event.ndm[9]);
	fprintf(event_params, "bp %f\n", event.bp[d]);
	fprintf(event_params, "bx %f\n", event.bx[d]);
	fprintf(event_params, "null %f\n", event.null[d]);
	fprintf(event_params, "central %f\n", event.time[d] - segment_start_time);
	fprintf(event_params, "norm %f\n", event.norm);
	fprintf(event_params, "pearson_cc %f\n", event.netcc[1]);
	fprintf(event_params, "net_cc %f\n", event.netcc[0]);
	fprintf(event_params, "ecor %f\n", event.ecor);
	fprintf(event_params, "effective_ecor %f\n", event.ECOR);
	fprintf(event_params, "nill %f\n", event.nill[d]);
	fprintf(event_params, "rho_0 %f\n", event.rho[0]);
	fprintf(event_params, "rho_1 %f\n", event.rho[1]);
	fprintf(event_params, "neted_0 %f\n", event.neted[0]);
	fprintf(event_params, "neted_1 %f\n", event.neted[1]);
	fprintf(event_params, "neted_2 %f\n", event.neted[2]);
	fprintf(event_params, "neted_3 %f\n", event.neted[3]);
	fprintf(event_params, "neted_4 %f\n", event.neted[4]);
	fprintf(event_params, "inj_central_time %d\n", 
		int(event.time[d+nIFO]));
	fprintf(event_params, "inj_central_time_ns %d\n", 
		int((event.time[d+nIFO] - int(event.time[d+nIFO])) * 1000000000));
	fprintf(event_params, "inj_hrss %g\n", event.hrss[d+nIFO]);
	fprintf(event_params, "inj_psi %f\n", event.psi[1]);
	fprintf(event_params, "inj_phi %f\n", event.phi[1]);
	fprintf(event_params, "inj_theta %f\n", event.theta[1]);
	fprintf(event_params, "inj_bp %f\n", event.bp[d+nIFO]);
	fprintf(event_params, "inj_bx %f\n", event.bx[d+nIFO]);
	fprintf(event_params, "inj_central %f\n", event.time[d+nIFO] - segment_start_time);
	
	fprintf(event_params, "distance %f\n", event.distance);
	fprintf(event_params, "mchirp %f\n", event.mchirp);
	fprintf(event_params, "mass_0 %f\n", event.mass[0]);
	fprintf(event_params, "mass_1 %f\n", event.mass[1]);
	//fprintf(event_params, "spin_0 %f\n", event.spin[0]);
	fprintf(event_params, "spin_0 %f\n", 0.0);
	//fprintf(event_params, "spin_1 %f\n", event.spin[1]);
	fprintf(event_params, "spin_1 %f\n", 0.0);
	//fprintf(event_params, "spin_2 %f\n", event.spin[2]);
	fprintf(event_params, "spin_2 %f\n", 0.0);
      }
      fclose(event_params);
      
      if( CED_VERBOSE )
	cout<< " Parameters file written " << endl;
      
      //pixel maps
      for(int d=0;d<nIFO;d++) {
	if( CED_VERBOSE )
	  cout << "Producing pixel TF maps..." << endl;
	sprintf(plots_filename, "%s/%s_pixel_1.png", web_dir, ifonames[d]);
	pF = WTSpectrum(*(pD[d]->getTFmap()), 2, 2, event.time[d]-1, event.time[d]+1,"colz");
	pF.canvas->SaveAs(plots_filename);
	clear(pF);
      }
      
      //shaded maps
      for(int d=0;d<nIFO;d++) {
	if( CED_VERBOSE )
	  cout << "Producing shaded TF maps..." << endl;
	sprintf(plots_filename, "%s/%s_shaded_1.png", web_dir, ifonames[d]);
	pF = WTSpectrum(*(pD[d]->getTFmap()), 2, 2, event.time[d]-1, event.time[d]+1,"cont4z");
	pF.canvas->SaveAs(plots_filename);
	clear(pF);
      }
      
      //pixel maps
      for(int d=0;d<nIFO;d++) {
	sprintf(plots_filename, "%s/%s_pixel_05.png", web_dir, ifonames[d]);
	pF = WTSpectrum(*(pD[d]->getTFmap()), 2, 2,event.time[d]-0.5, 
		   event.time[d]+0.5, "colz");
	pF.canvas->SaveAs(plots_filename);
	clear(pF);
      }
      
      //shaded maps
      for(int d=0;d<nIFO;d++) {
	if( CED_VERBOSE )
	  cout << "Producing shaded TF maps..." << endl;
	sprintf(plots_filename, "%s/%s_shaded_05.png", web_dir, ifonames[d]);
	pF = WTSpectrum(*(pD[d]->getTFmap()), 2, 2,event.time[d]-0.5, 
		   event.time[d]+0.5, "cont4z");
	pF.canvas->SaveAs(plots_filename);
	clear(pF);
      }
      
      // get reconstructed wave forms, signal
      if( CED_VERBOSE )
	cout << "Producing reconstructed waveforms..." << endl;
      NET->getwave(event_cluster_id, lag_index, 'W');
      
      // produce plots of reconstructed wave forms
      for(int d=0;d<nIFO;d++) {
	if( CED_VERBOSE ){
	  cout << "Producing plots of reconstructed wave forms, signal..." 
	       << endl;
	}
	sprintf(plots_filename, "%s/%s_wf_signal.png", web_dir, ifonames[d]);
	pT = Plot(pD[d]->waveBand, 0, 1);
	tP = Plot(pD[d]->waveForm, 1, 2);
	pT.canvas->SaveAs(plots_filename);
        clear(pT); clear(tP);	
      }
      
      // get reconstructed waveforms, signal + noise
      for(int d=0;d<nIFO;d++) {
	pD[d]->waveBand.sethigh(0);
      }
      NET->getwave(event_cluster_id, lag_index, 'S');
      
      // produce plots of reconstructed wave forms, signal + noise
      for(int d=0;d<nIFO;d++) {
	if( CED_VERBOSE ){
	  cout << "Producing plots of reconstructed wave forms, signal+noise..."
	       << endl;
	}
	sprintf(plots_filename, "%s/%s_wf_noise.png", web_dir, ifonames[d]);
	pT = Plot(pD[d]->waveBand, 0, 1);
	tP = Plot(pD[d]->waveForm, 1, 2);
	pT.canvas->SaveAs(plots_filename);
        clear(pT); clear(tP);	
      }
      
      // get reconstructed waveforms, strain
      for(int d=0;d<nIFO;d++) {
	pD[d]->waveBand.sethigh(0);
      }
      NET->getwave(event_cluster_id, lag_index, 'w');
      
      // produce plots of reconstructed wave forms, strain
      for(int d=0;d<nIFO;d++) {
	if( CED_VERBOSE ){
	  cout << "Producing plots of reconstructed waveforms, strain..." 
	       << endl;
	}
	sprintf(plots_filename, "%s/%s_wf_strain.png", web_dir, ifonames[d]);
	pT = Plot(pD[d]->waveForm, 0, 2);
	pT.canvas->SaveAs(plots_filename);
	clear(pT);
      }
      
      //save waveforms
      for(int d=0;d<nIFO;d++) {
	sprintf(dump_filename, "%s/%s_wf_strain.dat", web_dir, ifonames[d]);
	pD[d]->waveForm.Dump(dump_filename);
      }
      
      //likelihood skymaps
      if( CED_VERBOSE )
	cout << "Producing Likelihood skymaps..." << endl;
      sprintf(plots_filename, "%s/sensitivity_plus.png", web_dir);
      pF = SMSpectrum(NET->nSensitivity, 0, 2);
      pF.canvas->SaveAs(plots_filename); clear(pF);
      sprintf(plots_filename, "%s/sensitivity_cross.png", web_dir);
      pF = SMSpectrum(NET->nAlignment, 0, 2);
      pF.canvas->SaveAs(plots_filename); clear(pF);
      sprintf(plots_filename, "%s/skystat.png", web_dir);
      pF = SMSpectrum(NET->nSkyStat, 0, 2);
      pF.canvas->SaveAs(plots_filename); clear(pF);
      sprintf(plots_filename, "%s/likelihood.png", web_dir);
      pF = SMSpectrum(NET->nLikelihood, 0, 2);
      pF.canvas->SaveAs(plots_filename); clear(pF);
      sprintf(plots_filename, "%s/null_energy.png", web_dir);
      pF = SMSpectrum(NET->nNullEnergy, 0, 2);
      pF.canvas->SaveAs(plots_filename); clear(pF);
      sprintf(plots_filename, "%s/corr_energy.png", web_dir);
      pF = SMSpectrum(NET->nCorrEnergy, 0, 2);
      pF.canvas->SaveAs(plots_filename); clear(pF);
      sprintf(plots_filename, "%s/penalty.png", web_dir);
      pF = SMSpectrum(NET->nPenalty, 0, 2);
      pF.canvas->SaveAs(plots_filename); clear(pF);
      sprintf(plots_filename, "%s/disbalance.png", web_dir);
      pF = SMSpectrum(NET->nDisbalance, 0, 2);
      pF.canvas->SaveAs(plots_filename); clear(pF);
      sprintf(plots_filename, "%s/correlation.png", web_dir);
      pF = SMSpectrum(NET->nCorrelation, 0, 2);
      pF.canvas->SaveAs(plots_filename); clear(pF);
      sprintf(plots_filename, "%s/probability.png", web_dir);
      pF = SMSpectrum(NET->nProbability, 0, 2);
      pF.canvas->SaveAs(plots_filename); clear(pF);
      sprintf(plots_filename, "%s/netindex.png", web_dir);
      pF = SMSpectrum(NET->nNetIndex, 0, 2);
      pF.canvas->SaveAs(plots_filename); clear(pF);
      
      // get network time
      double gps_start =  event_start,
	gps_stop  =  event_stop;
      
      if( CED_VERBOSE ){
	cout << "Producing Likelihood TF maps..." << endl;
	printf("gps start time %f\n", gps_start);
	printf("gps stop time %f\n", gps_stop);
      }
      
      //likelihood tf maps
      sprintf(plots_filename, "%s/l_tfmap_cluster_1.png", web_dir);
      pF = WTSpectrum(NET->pixeLHood, 0, 2, gps_start, gps_stop,"colz");
      pF.canvas->SaveAs(plots_filename); clear(pF);
      sprintf(plots_filename, "%s/l_tfmap_cluster_05.png", web_dir);
      pF = WTSpectrum(NET->pixeLHood, 0, 2, gps_start-0.5, gps_stop+0.5,"colz");
      pF.canvas->SaveAs(plots_filename); clear(pF);
      sprintf(plots_filename, "%s/l_tfmap_pixel_1.png", web_dir);
      pF = WTSpectrum(NET->pixeLHood, 0, 2, gps_start, gps_stop,"colz");
      pF.canvas->SaveAs(plots_filename); clear(pF);
      sprintf(plots_filename, "%s/l_tfmap_shaded_1.png", web_dir);
      pF = WTSpectrum(NET->pixeLHood, 0, 2, gps_start, gps_stop,"cont4z");
      pF.canvas->SaveAs(plots_filename); clear(pF);
      sprintf(plots_filename, "%s/l_tfmap_pixel_05.png", web_dir);
      pF = WTSpectrum(NET->pixeLHood, 0, 2, gps_start-0.5, gps_stop+0.5,"colz");
      pF.canvas->SaveAs(plots_filename); clear(pF);
      sprintf(plots_filename, "%s/l_tfmap_shaded_05.png", web_dir);
      pF = WTSpectrum(NET->pixeLHood, 0, 2, gps_start-0.5, gps_stop+0.5,"cont4z");
      pF.canvas->SaveAs(plots_filename); clear(pF);
      
    } // End loop on found events
    
    delete[] ifo_start; 
    delete[] ifo_stop;
    delete[] ifo_lag;
    
    if( CED_VERBOSE )
      cout << "End lag " << lag_index << endl;
  } // End loop on lags
  
  delete[] pD;
  
  if( CED_VERBOSE ){
    cout << " Stopping Coherent Event Display " << endl;  
  }
  
  NET->netCC = old_cc;
  NET->netRHO = old_rho;
  
  return outdir;
} 

/*
int CED(network* NET, char* base_web_dir, bool (*cut)(netevent *n),  int lag=-1){
  
  if ( CED_VERBOSE )
    cout<<"Starting Coherent Event Display (CED)." << endl;
  
  int nIFO = NET->ifoListSize(),
    rate = int(2*NET->getifo(0)->TFmap.resolution(0)+0.5);
	
	//Fill in all skymaps
	double old_cc = NET->netCC, old_rho = NET->netRHO;
	NET->netCC = -1; 
	NET->netRHO = 0;

	if ( CED_VERBOSE )
		cout << "Search rate: " << rate << endl;

	std::vector<char*> ifonames = NET->ifoName;
	// Compiled mode only
	//std::vector<detector*> pD;
	detector **pD = new detector*[nIFO];

	for(int i=0; i<nIFO; i++)
		pD[i] = NET->getifo(i);
	//Compiled mode only
	//for(int i=0; i<nIFO; i++)
		//pD.push_back(NET->getifo(i));
	
	// netevent object for holding event parameters
	netevent event(nIFO);
	// plot production object
	TCanvas *tcanvas_ptr;

	if( CED_VERBOSE )
		cout << "Base web directory: " << base_web_dir << endl;

	int lags = NET->nLag;
	for(int lag_index=0; lag_index<lags; lag_index++){
		// If a specific lag is desired:
		if ( lag >= 0 )
			if ( lag != lag_index ) continue;

		if( CED_VERBOSE )
			cout << "Lag: " << lag_index << endl;
		// Get event IDs
		wavearray<double> wa_id = NET->getwc(lag_index)->get("ID", 0, 'L', rate);

		// Selection of event ids
		wavearray<double> *ifo_start = new wavearray<double>[nIFO], 
		                  *ifo_stop = new wavearray<double>[nIFO], 
											*ifo_lag = new wavearray<double>[nIFO], 
		                  cluster_ind;

		if( CED_VERBOSE ){
			cout << "Lag: " << lag_index << " with " << wa_id.size() << " events." 
				<< endl;
		}

		for(int cluster_index=0; cluster_index < (int)wa_id.size(); cluster_index++){

			int event_cluster_id = int(wa_id[cluster_index]+0.5);
			if( CED_VERBOSE )
				cout << "Clust ID: " << cluster_index << "/" << event_cluster_id << endl;

			event.fill(NET, event_cluster_id, lag_index, 1);

			// cut
			bool select = (*cut)(&event);
			if( !select ) continue;

			if( CED_VERBOSE ){
				cout << "Found: " << cluster_index << endl;
				printf("time: %f\n", event.time[0]);
				cout << "rho: " << event.rho[1] << endl;
				cout << "netcc: " << event.netcc[0] << endl;
				cout << "neted0: " << event.neted[0] << endl;
				cout << "neted1: " << event.neted[1] << endl;
				cout << "penalty: " << event.penalty << endl;
				cout << "---" << endl;
			}

			// store selecteed event parameters
			for(int j=0; j<nIFO; j++){
				ifo_start[j].append(event.start[j]);
				ifo_stop[j].append(event.stop[j]);
				ifo_lag[j].append(event.lag[j]);
			}
			cluster_ind.append(cluster_index);
		}

		// Loop over found events
		if( CED_VERBOSE )
			cout << cluster_ind.size() << " events to display this lag." << endl;

		for(int ijk=0; ijk<int(cluster_ind.size()); ijk++){

			// Retrieve the index in the cluster array
			int cluster_index = int(cluster_ind[ijk]+0.5);
			// Retrieve its ID
			int event_cluster_id = int(wa_id[cluster_index]+0.5);

			char web_dir[500] = "", command[500] = "", ifostr[20] = "";
			sprintf(ifostr, "");
			for(int d=0; d<(int)ifonames.size(); d++){
				sprintf(ifostr,"%s%s", ifostr, ifonames[d]);
			}
			sprintf(web_dir, "%s/%s", base_web_dir, ifostr);
			for(int d=0; d<nIFO; d++){
			  sprintf(web_dir, "%s_%.3f", web_dir, ifo_start[d].data[ijk]);
			}

			if( CED_VERBOSE )
				cout << "Creating event output directory " << web_dir << endl;

			sprintf(command,"mkdir -p %s", web_dir);
			system(command);

			// filename for plots
			char plots_filename[FILENAME_MAX], dump_filename[FILENAME_MAX],
					 file[FILENAME_MAX];

			//wavearray<double> event_start, event_stop; 

			if( CED_VERBOSE )
				cout << "Doing " << cluster_index << " in lag " << lag_index << endl;

			double event_start = 1e50, event_stop = 0;
			int d = 0;
			do{
				event_start = ifo_start[d].data[ijk] - ifo_lag[d].data[ijk] 
					< event_start ?  ifo_start[d].data[ijk] - ifo_lag[d].data[ijk] 
					: event_start ;
				event_stop = ifo_stop[d].data[ijk] - ifo_lag[d].data[ijk]
					> event_stop ?  ifo_stop[d].data[ijk] - ifo_lag[d].data[ijk] 
					: event_stop ;

				if( CED_VERBOSE ){
					cout << "Event " << cluster_index << " start" << endl;
					cout << "Detector: " << ifonames[d] << endl;
					printf("gps start time = %f\n", ifo_start[d].data[ijk]);
					printf("gps stop time = %f\n", ifo_stop[d].data[ijk]);
					printf("lag = %f\n", ifo_lag[d].data[ijk]);
				}

			} while( ++d < nIFO );

			//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
			// start of website plot production
			//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

			if( CED_VERBOSE )
				cout << "Website plot production..." << endl;

			//get segment_start_time
			double segment_start_time = NET->getifo(0)->TFmap.start();

			if( CED_VERBOSE )
				printf("Segment start time %f\n", segment_start_time);

			// event has been found
			if( CED_VERBOSE ){
				cout << "lag_index " << lag_index << " cluster index " << cluster_index 
					   << " event ID " << event_cluster_id << endl;
				cout << " Resolution " << NET->getifo(0)->TFmap.resolution(0) << endl;  
				cout << "Likelihood, index " << cluster_index << " id " 
					   << event_cluster_id << " with type " << NET->tYPe << endl;
			}

			NET->likelihood(NET->tYPe, false, NET->acor, event_cluster_id, lag_index);

			// get event parameters
			event.fill(NET, event_cluster_id, lag_index, 1);


			// geometric significance, set up for all detector combinations
			if( CED_VERBOSE )
				cout << "Significance." << endl;

			double log_gs = 0.0;
			for(int d_indx=0; d_indx < nIFO; d_indx++)
				log_gs *= event.rSF[d_indx];
			if(log_gs != 0.0)
				log_gs = log(log_gs)/nIFO;

			// event parameters file
			if( CED_VERBOSE )
				cout << "Parameters." << endl;
			FILE *event_params;

			sprintf(file, "%s/event.dat", web_dir);
			if ((event_params = fopen(file, "w")) == NULL) {
				cerr << "ERROR: Can't open output file: " << web_dir << "/event.dat" 
				     << endl;
			}

			for(int d=0; d<nIFO; d++) {
				fprintf(event_params, "EVENT\n");
				fprintf(event_params, "ifo %s\n", ifonames[d]);
				fprintf(event_params, "search waveburst\n");
				if( NET->mdcList.size() > 0 )
					fprintf(event_params, "type simulation\n");
				else fprintf(event_params, "type production\n");
				fprintf(event_params, "start_time %d\n", int(event.start[d]));
				fprintf(event_params, "start_time_ns %d\n", int((event.start[d] - int(event.start[d])) * 1000000000));
				fprintf(event_params, "stop_time %d\n", int(event.stop[d]));
				fprintf(event_params, "stop_time_ns %d\n", int((event.stop[d] - int(event.stop[d])) * 1000000000));
				fprintf(event_params, "central_time %d\n", int(event.time[d]));
				fprintf(event_params, "central_time_ns %d\n", int((event.time[d] - int(event.time[d])) * 1000000000));
				fprintf(event_params, "hrss %g\n", event.hrss[d]);
				fprintf(event_params, "time_lag %f\n", event.lag[d]);
				fprintf(event_params, "snr %f\n", event.snr[d]);
				fprintf(event_params, "rank_snr %f\n", event.rSNR[d]);
				fprintf(event_params, "rank_significance %f\n", event.rSF[d]);
				fprintf(event_params, "gaussian_significance %f\n", event.gSF[d]);
				fprintf(event_params, "noise %g\n", event.noise[d]);
				fprintf(event_params, "likelihood %f\n", event.likelihood);
				fprintf(event_params, "phi %f\n", event.phi[0]);
				fprintf(event_params, "theta %f\n", event.theta[0]);
				fprintf(event_params, "psi %f\n", event.psi[0]);
				fprintf(event_params, "size %d\n", event.size[0]);
				fprintf(event_params, "volume %d\n", event.volume[0]);
				fprintf(event_params, "flow %f\n", event.low[0]);
				//fprintf(event_params, "flow %f\n", fLow);
				fprintf(event_params, "fhigh %f\n", event.high[0]);
				fprintf(event_params, "segment_time %d\n", (int)segment_start_time);
				fprintf(event_params, "segment_time_ns 0\n");
				fprintf(event_params, "geometric_significance %f\n", log_gs);
				fprintf(event_params, "duration %f\n", event.duration[0]);
				fprintf(event_params, "central_freq %f\n", event.frequency[0]);
				fprintf(event_params, "bandwidth %f\n", event.bandwidth[0]);
				fprintf(event_params, "wavelet_rate %d\n", event.rate[0]);
				fprintf(event_params, "ndm_0 %f\n", event.ndm[0]);
				fprintf(event_params, "ndm_1 %f\n", event.ndm[1]);
				fprintf(event_params, "ndm_2 %f\n", event.ndm[2]);
				fprintf(event_params, "ndm_3 %f\n", event.ndm[3]);
				fprintf(event_params, "ndm_4 %f\n", event.ndm[4]);
				fprintf(event_params, "ndm_5 %f\n", event.ndm[5]);
				fprintf(event_params, "ndm_6 %f\n", event.ndm[6]);
				fprintf(event_params, "ndm_7 %f\n", event.ndm[7]);
				fprintf(event_params, "ndm_8 %f\n", event.ndm[8]);
				fprintf(event_params, "ndm_9 %f\n", event.ndm[9]);
				fprintf(event_params, "bp %f\n", event.bp[d]);
				fprintf(event_params, "bx %f\n", event.bx[d]);
				fprintf(event_params, "null %f\n", event.null[d]);
				fprintf(event_params, "central %f\n", event.time[d] - segment_start_time);
				fprintf(event_params, "norm %f\n", event.norm);
				fprintf(event_params, "pearson_cc %f\n", event.netcc[1]);
				fprintf(event_params, "net_cc %f\n", event.netcc[0]);
				fprintf(event_params, "ecor %f\n", event.ecor);
				fprintf(event_params, "effective_ecor %f\n", event.ECOR);
				fprintf(event_params, "nill %f\n", event.nill[d]);
				fprintf(event_params, "rho_0 %f\n", event.rho[0]);
				fprintf(event_params, "rho_1 %f\n", event.rho[1]);
				fprintf(event_params, "neted_0 %f\n", event.neted[0]);
				fprintf(event_params, "neted_1 %f\n", event.neted[1]);
				fprintf(event_params, "neted_2 %f\n", event.neted[2]);
				fprintf(event_params, "neted_3 %f\n", event.neted[3]);
				fprintf(event_params, "neted_4 %f\n", event.neted[4]);
				fprintf(event_params, "inj_central_time %d\n", 
						int(event.time[d+nIFO]));
				fprintf(event_params, "inj_central_time_ns %d\n", 
						int((event.time[d+nIFO] - int(event.time[d+nIFO])) * 1000000000));
				fprintf(event_params, "inj_hrss %g\n", event.hrss[d+nIFO]);
				fprintf(event_params, "inj_psi %f\n", event.psi[1]);
				fprintf(event_params, "inj_phi %f\n", event.phi[1]);
				fprintf(event_params, "inj_theta %f\n", event.theta[1]);
				fprintf(event_params, "inj_bp %f\n", event.bp[d+nIFO]);
				fprintf(event_params, "inj_bx %f\n", event.bx[d+nIFO]);
				fprintf(event_params, "inj_central %f\n", event.time[d+nIFO] - segment_start_time);

				fprintf(event_params, "distance %f\n", event.distance);
				fprintf(event_params, "mchirp %f\n", event.mchirp);
				fprintf(event_params, "mass_0 %f\n", event.mass[0]);
				fprintf(event_params, "mass_1 %f\n", event.mass[1]);
				//fprintf(event_params, "spin_0 %f\n", event.spin[0]);
				fprintf(event_params, "spin_0 %f\n", 0.0);
				//fprintf(event_params, "spin_1 %f\n", event.spin[1]);
				fprintf(event_params, "spin_1 %f\n", 0.0);
				//fprintf(event_params, "spin_2 %f\n", event.spin[2]);
				fprintf(event_params, "spin_2 %f\n", 0.0);
			}
			fclose(event_params);

			if( CED_VERBOSE )
				cout<< " Parameters file written " << endl;

			//pixel maps
			for(int d=0;d<nIFO;d++) {
				if( CED_VERBOSE )
					cout << "Producing pixel TF maps..." << endl;
				sprintf(plots_filename, "%s/%s_pixel_1.png", web_dir, ifonames[d]);
				WTSpectrum(*(pD[d]->getTFmap()), 2, 2, event.time[d]-1, event.time[d]+1,
						"colz")->SaveAs(plots_filename);
			}

			//shaded maps
			for(int d=0;d<nIFO;d++) {
				if( CED_VERBOSE )
					cout << "Producing shaded TF maps..." << endl;
				sprintf(plots_filename, "%s/%s_shaded_1.png", web_dir, ifonames[d]);
				WTSpectrum(*(pD[d]->getTFmap()), 2, 2, event.time[d]-1, event.time[d]+1,
						"cont4z")->SaveAs(plots_filename);
			}

			//pixel maps
			for(int d=0;d<nIFO;d++) {
				sprintf(plots_filename, "%s/%s_pixel_05.png", web_dir, ifonames[d]);
				WTSpectrum(*(pD[d]->getTFmap()), 2, 2,event.time[d]-0.5, 
						event.time[d]+0.5, "colz")->SaveAs(plots_filename);
			}

			//shaded maps
			for(int d=0;d<nIFO;d++) {
				if( CED_VERBOSE )
					cout << "Producing shaded TF maps..." << endl;
				sprintf(plots_filename, "%s/%s_shaded_05.png", web_dir, ifonames[d]);
				WTSpectrum(*(pD[d]->getTFmap()), 2, 2,event.time[d]-0.5, 
						event.time[d]+0.5, "cont4z")->SaveAs(plots_filename);
			}

			// get reconstructed wave forms, signal
			if( CED_VERBOSE )
				cout << "Producing reconstructed waveforms..." << endl;
			NET->getwave(event_cluster_id, lag_index, 'W');

			// produce plots of reconstructed wave forms
			for(int d=0;d<nIFO;d++) {
				if( CED_VERBOSE ){
					cout << "Producing plots of reconstructed wave forms, signal..." 
						<< endl;
				}
				sprintf(plots_filename, "%s/%s_wf_signal.png", web_dir, ifonames[d]);
				tcanvas_ptr = Plot(pD[d]->waveBand, 0, 1);
				Plot(pD[d]->waveForm, 1, 2);
				tcanvas_ptr->SaveAs(plots_filename);
			}

			// get reconstructed waveforms, signal + noise
			for(int d=0;d<nIFO;d++) {
				pD[d]->waveBand.sethigh(0);
			}
			NET->getwave(event_cluster_id, lag_index, 'S');

			// produce plots of reconstructed wave forms, signal + noise
			for(int d=0;d<nIFO;d++) {
				if( CED_VERBOSE ){
					cout << "Producing plots of reconstructed wave forms, signal+noise..."
						<< endl;
				}
				sprintf(plots_filename, "%s/%s_wf_noise.png", web_dir, ifonames[d]);
				tcanvas_ptr = Plot(pD[d]->waveBand, 0, 1);
				Plot(pD[d]->waveForm, 1, 2);
				tcanvas_ptr->SaveAs(plots_filename);
			}

			// get reconstructed waveforms, strain
			for(int d=0;d<nIFO;d++) {
				pD[d]->waveBand.sethigh(0);
			}
			NET->getwave(event_cluster_id, lag_index, 'w');

			// produce plots of reconstructed wave forms, strain
			for(int d=0;d<nIFO;d++) {
				if( CED_VERBOSE ){
					cout << "Producing plots of reconstructed waveforms, strain..." 
						<< endl;
				}
				sprintf(plots_filename, "%s/%s_wf_strain.png", web_dir, ifonames[d]);
				Plot(pD[d]->waveForm, 0, 2)->SaveAs(plots_filename);
			}

			//save waveforms
			for(int d=0;d<nIFO;d++) {
				sprintf(dump_filename, "%s/%s_wf_strain.dat", web_dir, ifonames[d]);
				pD[d]->waveForm.Dump(dump_filename);
			}

			//likelihood skymaps
			if( CED_VERBOSE )
				cout << "Producing Likelihood skymaps..." << endl;
			sprintf(plots_filename, "%s/sensitivity_plus.png", web_dir);
			SMSpectrum(NET->nSensitivity, 0, 2)->SaveAs(plots_filename);
			sprintf(plots_filename, "%s/sensitivity_cross.png", web_dir);
			SMSpectrum(NET->nAlignment, 0, 2)->SaveAs(plots_filename);
			sprintf(plots_filename, "%s/skystat.png", web_dir);
			SMSpectrum(NET->nSkyStat, 0, 2)->SaveAs(plots_filename);
			sprintf(plots_filename, "%s/likelihood.png", web_dir);
			SMSpectrum(NET->nLikelihood, 0, 2)->SaveAs(plots_filename);
			sprintf(plots_filename, "%s/null_energy.png", web_dir);
			SMSpectrum(NET->nNullEnergy, 0, 2)->SaveAs(plots_filename);
			sprintf(plots_filename, "%s/corr_energy.png", web_dir);
			SMSpectrum(NET->nCorrEnergy, 0, 2)->SaveAs(plots_filename);
			sprintf(plots_filename, "%s/penalty.png", web_dir);
			SMSpectrum(NET->nPenalty, 0, 2)->SaveAs(plots_filename);
			sprintf(plots_filename, "%s/disbalance.png", web_dir);
			SMSpectrum(NET->nDisbalance, 0, 2)->SaveAs(plots_filename);
			sprintf(plots_filename, "%s/correlation.png", web_dir);
			SMSpectrum(NET->nCorrelation, 0, 2)->SaveAs(plots_filename);

			// get network time
			double gps_start = event_start,
			       gps_stop  = event_stop;

			if( CED_VERBOSE ){
				cout << "Producing Likelihood TF maps..." << endl;
				cout << "gps start time " << gps_start << endl;
				cout << "gps stop time " << gps_stop << endl;
			}

			//likelihood tf maps
			// These are identical to the ones below --- ???
			sprintf(plots_filename, "%s/l_tfmap_cluster_1.png", web_dir);
			WTSpectrum(NET->pixeLHood, 0, 2, gps_start, gps_stop,
					"colz")->SaveAs(plots_filename);
			sprintf(plots_filename, "%s/l_tfmap_cluster_05.png", web_dir);
			WTSpectrum(NET->pixeLHood, 0, 2, gps_start - 0.5, gps_stop + 0.5,
					"colz")->SaveAs(plots_filename);

			sprintf(plots_filename, "%s/l_tfmap_pixel_1.png", web_dir);
			WTSpectrum(NET->pixeLHood, 0, 2, gps_start, gps_stop,
					"colz")->SaveAs(plots_filename);
			sprintf(plots_filename, "%s/l_tfmap_shaded_1.png", web_dir);
			WTSpectrum(NET->pixeLHood, 0, 2, gps_start, gps_stop,
					"cont4z")->SaveAs(plots_filename);
			sprintf(plots_filename, "%s/l_tfmap_pixel_05.png", web_dir);
			WTSpectrum(NET->pixeLHood, 0, 2, gps_start - 0.5, gps_stop + 0.5,
					"colz")->SaveAs(plots_filename);
			sprintf(plots_filename, "%s/l_tfmap_shaded_05.png", web_dir);
			WTSpectrum(NET->pixeLHood, 0, 2, gps_start - 0.5, gps_stop + 0.5,
					"cont4z")->SaveAs(plots_filename);

		} // End loop on found events

		delete[] ifo_start; 
		delete[] ifo_stop;
		delete[] ifo_lag;

		if( CED_VERBOSE )
			cout << "End lag " << lag_index << endl;
	} // End loop on lags
	
	delete[] pD;

	if( CED_VERBOSE ){
		cout << " Stopping Coherent Event Display " << endl;  
	}

	NET->netCC = old_cc;
	NET->netRHO = old_rho;

	return 0;
} 
*/
