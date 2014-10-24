{
gSystem->Load("libPhysics.so");
gSystem->Load("lib/wavelet.so");

THtml html;

html.SetOutputDir("html");
html.SetSourceDir(".");
html.MakeClass("Wavelet");
html.MakeClass("Biorthogonal<double>");
html.MakeClass("Biorthogonal<float>");
html.MakeClass("Daubechies<double>");
html.MakeClass("Daubechies<float>");
html.MakeClass("Haar<double>");
html.MakeClass("Haar<float>");
html.MakeClass("Meyer<double>");
html.MakeClass("Meyer<float>");
html.MakeClass("Symlet<double>");
html.MakeClass("Symlet<float>");
html.MakeClass("wavearray<double>");
html.MakeClass("wavearray<float>");
html.MakeClass("wavearray<int>");
html.MakeClass("wavearray<short>");
html.MakeClass("wavecomplex");
html.MakeClass("WSeries<double>");
html.MakeClass("WSeries<float>");
html.MakeClass("WaveDWT<double>");
html.MakeClass("WaveDWT<float>");
html.MakeClass("WaveRDC");
html.MakeClass("linefilter");
html.MakeClass("wavepixel");
html.MakeClass("wavecluster");
html.MakeClass("netpixel");
html.MakeClass("netcluster");
html.MakeClass("skymap");
html.MakeClass("detector");
html.MakeClass("network");
// html.MakeClass("LineFilter");
html.MakeIndex();

return;
}
