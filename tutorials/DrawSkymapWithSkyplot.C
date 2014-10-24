{ // This example show how to plot skymap with watplot

  skymap sm(int(4)); 		// create skymap object
  sm=0;				// set skymap to 0

  // fill sm 
  for(int l=0;l<sm.size();l++) { 
    if(l%2==0) sm.set(l,0.75); else sm.set(l,1);
  }

  watplot pl;			// create skyplot object
  pl.plot(sm);			// plot skymap

  return pl.canvas;		// used by THtml doc
}
