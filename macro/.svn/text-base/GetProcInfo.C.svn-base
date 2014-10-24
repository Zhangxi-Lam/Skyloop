void GetProcInfo() {
   TString s;
   FILE *f = fopen(Form("/proc/%d/statm", gSystem->GetPid()), "r");
   s.Gets(f);
   Long_t total, rss;
   sscanf(s.Data(), "%ld %ld", &total, &rss);
   cout << " virtual : " <<  total * 4 / 1024 << " (mb)  rss  : " <<  rss * 4 /
1024 << " (mb)" << endl;
   fclose(f);
   return;
}
