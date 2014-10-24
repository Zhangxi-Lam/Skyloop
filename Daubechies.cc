// Wavelet Analysis Tool
//--------------------------------------------------------------------
// Implementation of 
// Daubeches wavelets using Fast Wavelet Transform 
// References:
//   I.Daubechies, Ten lectures on wavelets
//   ISBN 0-89871-274-2, 1992
//--------------------------------------------------------------------
//$Id: Daubechies.cc,v 1.6 2004/07/12 07:05:20 jzweizig Exp $

#define DAUBECHIES_CC

#include "Daubechies.hh"
#include <time.h>
#include <iostream>

//namespace datacondAPI {
//namespace wat {

ClassImp(Daubechies<double>)

extern const double dbc1[2]={0.70710678118655,0.70710678118655};

extern const double dbc2[4]=
{0.48296291314469,0.83651630373747, 0.22414386804186,-0.12940952255092};

extern const double dbc3[6]=
{0.33267055295096,0.80689150931334,0.45987750211933,-0.13501102001039,
-0.08544127388224,0.03522629188210};

extern const double dbc4[8]=
{0.23037781330886,0.71484657055254,0.63088076792959,-0.02798376941698,
-0.18703481171888,0.03084138183599,0.03288301166698,-0.01059740178500};

extern const double dbc5[10]=
{0.16010239797413,0.60382926979747,0.72430852843857,0.13842814590110,
-0.24229488706619,-0.03224486958503,0.07757149384007,-0.00624149021301,
-0.01258075199902,0.00333572528500};

extern const double dbc6[12]=
{0.11154074335008,0.49462389039839,0.75113390802158,0.31525035170924,
-0.22626469396517,-0.12976686756710,0.09750160558708,0.02752286553002,
-0.03158203931803,0.00055384220099,0.00477725751101,-0.00107730108500};

extern const double dbc7[14]=
{0.07785205408506,0.39653931948231,0.72913209084656,0.46978228740536,
-0.14390600392911,-0.22403618499417,0.07130921926705,0.08061260915107,
-0.03802993693503,-0.01657454163102,0.01255099855601,0.00042957797300,
-0.00180164070400,0.00035371380000};

extern const double dbc8[16]=
{0.05441584224308,0.31287159091447,0.67563073629801,0.58535468365487,
-0.01582910525602,-0.28401554296243,0.00047248457400,0.12874742662019,
-0.01736930100202,-0.04408825393106,0.01398102791702,0.00874609404702,
-0.00487035299301,-0.00039174037300,0.00067544940600,-0.00011747678400};

extern const double dbc9[18]=
{0.03807794736317,0.24383467463767,0.60482312367678,0.65728807803664,
0.13319738582209,-0.29327378327259,-0.09684078322088,0.14854074933476,
0.03072568147832,-0.06763282905952,0.00025094711499,0.02236166212352,
-0.00472320475789,-0.00428150368190,0.00184764688296,0.00023038576400,
-0.00025196318900,0.00003934732000};

extern const double dbc10[20]=
{0.02667005790095,0.18817680007762,0.52720118893092,0.68845903945259,
0.28117234366043,-0.24984642432649,-0.19594627437660,0.12736934033574,
0.09305736460381,-0.07139414716586,-0.02945753682195,0.03321267405893,
0.00360655356699,-0.01073317548298,0.00139535174699,0.00199240529499,
-0.00068585669500,-0.00011646685499,0.00009358867000,-0.00001326420300};

extern const double dbc11[22]=
{0.01869429776147,0.14406702115062,0.44989976435603,0.68568677491617,
0.41196436894790,-0.16227524502747,-0.27423084681793,0.06604358819669,
0.14981201246638,-0.04647995511667,-0.06643878569502,0.03133509021905,
0.02084090436018,-0.01536482090620,-0.00334085887301,0.00492841765606,
-0.00030859285882,-0.00089302325067,0.00024915252355,0.00005443907470,
-0.00003463498419,0.00000449427428};

extern const double dbc12[24]=
{0.01311225795723,0.10956627282118,0.37735513521419,0.65719872257928,
0.51588647842779,-0.04476388565377,-0.31617845375276,-0.02377925725606,
0.18247860592758,0.00535956967437,-0.09643212009649,0.01084913025583,
0.04154627749509,-0.01221864906975,-0.01284082519830,0.00671149900880,
0.00224860724100,-0.00217950361863,0.00000654512821,0.00038865306282,
-0.00008850410921,-0.00002424154576,0.00001277695222,-0.00000152907176};

extern const double dbc13[26]=
{0.00920213353896,0.08286124387290,0.31199632216043,0.61105585115878,
0.58888957043121,0.08698572617964,-0.31497290771138,-0.12457673075080,
0.17947607942935,0.07294893365679,-0.10580761818792,-0.02648840647534,
0.05613947710028,0.00237997225405,-0.02383142071033,0.00392394144879,
0.00725558940162,-0.00276191123466,-0.00131567391189,0.00093232613087,
0.00004925152513,-0.00016512898856,0.00003067853758,0.00001044193057,
-0.00000470041648,0.00000052200351};

extern const double dbc14[28]=
{0.00646115346009,0.06236475884939,0.25485026779257,0.55430561794077,
0.63118784910472,0.21867068775886,-0.27168855227868,-0.21803352999321,
0.13839521386480,0.13998901658447,-0.08674841156811,-0.07154895550399,
0.05523712625925,0.02698140830794,-0.03018535154036,-0.00561504953034,
0.01278949326634,-0.00074621898927,-0.00384963886802,0.00106169108561,
0.00070802115424,-0.00038683194731,-0.00004177724577,0.00006875504253,
-0.00001033720918,-0.00000438970490,0.00000172499468,-0.00000017871400};

extern const double dbc15[30]=
{0.00453853736158,0.04674339489277,0.20602386398700,0.49263177170816,
0.64581314035744,0.33900253545474,-0.19320413960915,-0.28888259656697,
0.06528295284880,0.19014671400716,-0.03966617655576,-0.11112093603722,
0.03387714392350,0.05478055058448,-0.02576700732848,-0.02081005016973,
0.01508391802781,0.00510100036039,-0.00648773456032,-0.00024175649076,
0.00194332398038,-0.00037348235414,-0.00035956524436,0.00015589648992,
0.00002579269916,-0.00002813329627,0.00000336298718,0.00000181127041,
-0.00000063168823,0.00000006133360};

extern const double dbc16[32]=
{0.00318922092535,0.03490771432367,0.16506428348885,0.43031272284599,
0.63735633208377,0.44029025688634,-0.08975108940249,-0.32706331052792,
-0.02791820813305,0.21119069394708,0.02734026375270,-0.13238830556381,
-0.00623972275247,0.07592423604429,-0.00758897436884,-0.03688839769171,
0.01029765964098,0.01399376885985,-0.00699001456340,-0.00364427962149,
0.00312802338121,0.00040789698085,-0.00094102174936,0.00011424152004,
0.00017478724523,-0.00006103596621,-0.00001394566899,0.00001133660866,
-0.00000104357134,-0.00000073636568,0.00000023087841,-0.00000002109340};

extern const double dbc17[34]=
{0.00224180700104,0.02598539370361,0.13121490330784,0.37035072415269,
0.61099661568471,0.51831576405701,0.02731497040330,-0.32832074836400,
-0.12659975221589,0.19731058956506,0.10113548917751,-0.12681569177829,
-0.05709141963170,0.08110598665412,0.02231233617804,-0.04692243838934,
-0.00327095553587,0.02273367658392,-0.00304298998137,-0.00860292152033,
0.00296799669153,0.00230120524216,-0.00143684530480,-0.00032813251941,
0.00043946542777,-0.00002561010957,-0.00008204803202,0.00002318681380,
0.00000699060099,-0.00000450594248,0.00000030165496,0.00000029577009,
-0.00000008423948,0.00000000726749};

extern const double dbc18[36]=
{0.00157631021844,0.01928853172418,0.10358846582258,0.31467894133751,
0.57182680776747,0.57180165488952,0.14722311197014,-0.29365404073704,
-0.21648093400555,0.14953397556546,0.16708131276330,-0.09233188415126,
-0.10675224666029,0.06488721621171,0.05705124773838,-0.04452614190323,
-0.02373321039602,0.02667070592642,0.00626216795425,-0.01305148094667,
0.00011863003383,0.00494334360546,-0.00111873266700,-0.00134059629834,
0.00062846568297,0.00021358156191,-0.00019864855231,-0.00000015359171,
0.00003741237881,-0.00000852060254,-0.00000333263448,0.00000176871298,
-0.00000007691633,-0.00000011760988,0.00000003068836,-0.00000000250793};

extern const double dbc19[38]=
{0.00110866976318,0.01428109845070,0.08127811326510,0.26438843173972,
0.52443637746232,0.60170454912487,0.26089495264991,-0.22809139421438,
-0.28583863175437,0.07465226970810,0.21234974330584,-0.03351854190149,
-0.14278569503735,0.02758435062625,0.08690675555608,-0.02650123624947,
-0.04567422627659,0.02162376740985,0.01937554988940,-0.01398838867821,
-0.00586692228080,0.00704074736719,0.00076895435932,-0.00268755180066,
0.00034180865347,0.00073580252051,-0.00026067613568,-0.00012460079173,
0.00008711270467,0.00000510595049,-0.00001664017630,0.00000301096432,
0.00000153193148,-0.00000068627557,0.00000001447088,0.00000004636938,
-0.00000001116402,0.00000000086668};

extern const double dbc20[40]=
{0.00077995361366,0.01054939462487,0.06342378045858,0.21994211354965,
0.47269618530714,0.61049323893374,0.36150229873647,-0.13921208801035,
-0.32678680043141,-0.01672708830897,0.22829105081798,0.03985024645728,
-0.15545875070610,-0.02471682733811,0.10229171917456,0.00563224685888,
-0.06172289962210,0.00587468181393,0.03229429953237,-0.00878932492251,
-0.01381052613626,0.00672162730258,0.00442054238715,-0.00358149425955,
-0.00083156217282,0.00139255961930,-0.00005349759845,-0.00038510474870,
0.00010153288973,0.00006774280828,-0.00003710586183,-0.00000437614386,
0.00000724124829,-0.00000101199401,-0.00000068470796,0.00000026339242,
0.00000000020143,-0.00000001814843,0.00000000405613,-0.00000000029988};

extern const double dbc21[42]=
{0.00054882250986,0.00777663905246,0.04924777153883,0.18135962544278,
0.41968794494492,0.60150609494296,0.44459045193348,-0.03572291961772,
-0.33566408953492,-0.11239707156980,0.21156452768396,0.11523329844160,
-0.13994042493413,-0.08177594298219,0.09660039032382,0.04572340574753,
-0.06497750489781,-0.01865385920593,0.03972683542509,0.00335775638782,
-0.02089205367997,0.00240347091984,0.00898882438159,-0.00289133434884,
-0.00295837403905,0.00171660704064,0.00063941850053,-0.00069067111708,
-0.00003196406277,0.00019366465042,-0.00003635520250,-0.00003499665985,
0.00001535482509,0.00000279033054,-0.00000309001716,0.00000031660954,
0.00000029921366,-0.00000010004009,-0.00000000225401,0.00000000705803,
-0.00000000147195,0.00000000010388};

extern const double dbc22[44]=
{0.00038626323150,0.00572185463145,0.03806993723721,0.14836754089323,
0.36772868345377,0.57843273102169,0.50790109063283,0.07372450118513,
-0.31272658043501,-0.20056840610944,0.16409318810947,0.17997318799515,
-0.09711079841387,-0.13176813769363,0.06807631438861,0.08455737636227,
-0.05136425430463,-0.04653081183330,0.03697084661864,0.02058670762720,
-0.02348000134426,-0.00621378284801,0.01256472522016,0.00030013739972,
-0.00545569198552,0.00104426073960,0.00182701049587,-0.00077069098807,
-0.00042378739982,0.00032860941423,0.00004345899905,-0.00009405223635,
0.00001137434966,0.00001737375696,-0.00000616672932,-0.00000156517913,
0.00000129518206,-0.00000008779880,-0.00000012833362,0.00000003761229,
0.00000000168017,-0.00000000272962,0.00000000053359,-0.00000000003602};

extern const double dbc23[46]=
{0.00027190419416,0.00420274889370,0.02931000366146,0.12051553179868,
0.31845081389173,0.54493114794003,0.55101851730917,0.18139262538597,
-0.26139214806257,-0.27140209864102,0.09212540709353,0.22357365826899,
-0.03303744709916,-0.16401132155390,0.02028307457378,0.11229704362359,
-0.02112621237227,-0.07020739160254,0.02176585681370,0.03849533250212,
-0.01852351367601,-0.01753710102460,0.01275194391902,0.00603184064178,
-0.00707531927963,-0.00113486547603,0.00312287644907,-0.00024650140100,
-0.00106123122919,0.00031942049269,0.00025676245202,-0.00015002185037,
-0.00003378894835,0.00004426071204,-0.00000263520789,-0.00000834787557,
0.00000239756955,0.00000081475748,-0.00000053390054,0.00000001853092,
0.00000005417549,-0.00000001399935,-0.00000000094729,0.00000000105045,
-0.00000000019324,0.00000000001250};

extern const double dbc24[48]=
{0.00019143580095,0.00308208171500,0.02248233995038,0.09726223583651,
0.27290891607582,0.50437104085489,0.57493922111260,0.28098555324203,
-0.18727140689076,-0.31794307900890,0.00477661368430,0.23923738878710,
0.04252872964227,-0.17117535137613,-0.03877717358004,0.12101630347157,
0.02098011370824,-0.08216165421234,-0.00457843624443,0.05130162003823,
-0.00494470943241,-0.02821310710056,0.00766172187681,0.01304997086673,
-0.00629143537412,-0.00474656878930,0.00373604617663,0.00115376493595,
-0.00169645681942,-0.00004416184869,0.00058612705931,-0.00011812332380,
-0.00014600798177,0.00006559388640,0.00002183241461,-0.00002022888293,
0.00000001341158,0.00000390110034,-0.00000089802531,-0.00000040325078,
0.00000021663397,-0.00000000050576,-0.00000002255740,0.00000000515778,
0.00000000047484,-0.00000000040247,0.00000000006992,-0.00000000000434};

extern const double dbc25[50]=
{0.00013480297936,0.00225695959203,0.01718674125537,0.07803586287819,
0.23169350790401,0.45968341518179,0.58163689679121,0.36788507483148,
-0.09717464097229,-0.33647307966812,-0.08758761459496,0.22453781976164,
0.11815528672764,-0.15056021376482,-0.09850861530222,0.10663380501832,
0.06675216448493,-0.07708411108361,-0.03717396289023,0.05361790937603,
0.01554260590837,-0.03404232047610,-0.00307983679799,0.01892280448261,
-0.00198942577434,-0.00886070261061,0.00272693626530,0.00332270777835,
-0.00184248428806,-0.00089997742272,0.00087725819420,0.00011532124422,
-0.00030988009907,0.00003543714525,0.00007904640005,-0.00002733048120,
-0.00001277195293,0.00000899066139,0.00000052328277,-0.00000177920133,
0.00000032120375,0.00000019228068,-0.00000008656942,-0.00000000261160,
0.00000000927922,-0.00000000188042,-0.00000000022285,0.00000000015359,
-0.00000000002528,0.00000000000151};

extern const double dbc26[52]=
{0.00009493795747,0.00165052023287,0.01309755428729,0.06227474400009,
0.19503943863827,0.41329296211202,0.57366904280335,0.43915831161244,
0.00177407678034,-0.32638459356024,-0.17483996121861,0.18129183223897,
0.18275540951738,-0.10432390024244,-0.14797719321441,0.06982318608594,
0.10648240520785,-0.05344856165621,-0.06865475956471,0.04223218580538,
0.03853571600107,-0.03137811028460,-0.01776090348004,0.02073492025806,
0.00582958063302,-0.01178549783685,-0.00052873835369,0.00560194726473,
-0.00093905823572,-0.00214553027435,0.00083834880766,0.00061613822092,
-0.00043195570707,-0.00010605747474,0.00015747952381,-0.00000527779549,
-0.00004109673995,0.00001074221541,0.00000700007868,-0.00000388740016,
-0.00000046504632,0.00000079392106,-0.00000010790042,-0.00000008904466,
0.00000003407796,0.00000000216933,-0.00000000377601,0.00000000067800,
0.00000000010023,-0.00000000005840,0.00000000000913,-0.00000000000053};

extern const double dbc27[54]=
{0.00006687131379,0.00120553123053,0.00995258877145,0.04945259993605,
0.16292202734804,0.36711021377760,0.55384986046579,0.49340612221061,
0.10284085496457,-0.28971680303965,-0.24826458166690,0.11482301941169,
0.22727328820471,-0.03878641858319,-0.17803174076834,0.01579939748221,
0.13119797164867,-0.01406275146647,-0.09102290634917,0.01731101835678,
0.05796940579574,-0.01851249342470,-0.03273906647991,0.01614696702403,
0.01566559574354,-0.01157718635089,-0.00586209625930,0.00685663566511,
0.00134262691970,-0.00333285443828,0.00014575297882,0.00130117745760,
-0.00034183511836,-0.00038790185541,0.00020197198834,0.00007660058395,
-0.00007711145508,-0.00000351748362,0.00002063442645,-0.00000390116407,
-0.00000365750091,0.00000163436962,0.00000030508807,-0.00000034724681,
0.00000003286559,0.00000004026255,-0.00000001321332,-0.00000000130947,
0.00000000152161,-0.00000000024155,-0.00000000004375,0.00000000002214,
-0.00000000000330,0.00000000000018};

extern const double dbc28[56]=
{0.00004710807766,0.00087949851431,0.00754265036328,0.03909260804095,
0.13513791399624,0.32256336067113,0.52499823063037,0.53051629243100,
0.20017614366470,-0.23049895360849,-0.30132780895814,0.03285787910353,
0.24580815091531,0.03690688527328,-0.18287733031840,-0.04683823352168,
0.13462756788564,0.03447863155804,-0.09768535516323,-0.01734192227806,
0.06774789588669,0.00344801938223,-0.04333336823364,0.00443173305187,
0.02468805998294,-0.00681554982076,-0.01206359205683,0.00583881650115,
0.00478486300787,-0.00372546130632,-0.00136037388200,0.00187599864580,
0.00014156723219,-0.00074867495568,0.00011546560683,0.00022957909867,
-0.00008903901415,-0.00004907713377,0.00003641401216,0.00000463866501,
-0.00001004326038,0.00000124790032,0.00000184036373,-0.00000066702155,
-0.00000017574612,0.00000014906600,-0.00000000826239,-0.00000001784139,
0.00000000504405,0.00000000069445,-0.00000000060770,0.00000000008492,
0.00000000001867,-0.00000000000837,0.00000000000119,-0.00000000000006};

extern const double dbc29[58]=
{0.00003318966297,0.00064095168358,0.00570212654694,0.03077358037883,
0.11137011752144,0.28065345740663,0.48975880726750,0.55137443557886,
0.28910523981468,-0.15402873524802,-0.33004095060681,-0.05570680036129,
0.23610523735052,0.11241917542149,-0.16087798947795,-0.10784595061226,
0.11447229626215,0.08322074724110,-0.08512549355142,-0.05502749046173,
0.06347916412676,0.03053154257401,-0.04518798236422,-0.01291714340149,
0.02947043137455,0.00264832683582,-0.01704122498871,0.00173788013922,
0.00846972541737,-0.00255080721472,-0.00347379905891,0.00187712089643,
0.00108705391903,-0.00100077835149,-0.00020007114870,0.00041112834174,
-0.00002292018316,-0.00012930448572,0.00003645026049,0.00002913344752,
-0.00001657328408,-0.00000359364483,0.00000475060927,-0.00000030290546,
-0.00000089757018,0.00000026338984,0.00000009387197,-0.00000006286157,
0.00000000107659,0.00000000776898,-0.00000000189400,-0.00000000034268,
0.00000000024071,-0.00000000002941,-0.00000000000783,0.00000000000315,
-0.00000000000043,0.00000000000002};

extern const double dbc30[60]=
{0.00002338616182,0.00046663795235,0.00430079718272,0.02413083277074,
0.09123830444189,0.24202067193463,0.45048782370427,0.55757223520372,
0.36624268487617,-0.06618367104886,-0.33296697639194,-0.14196851392136,
0.19946212238116,0.17782987393397,-0.11455821998655,-0.15723681875947,
0.07277865900869,0.12274774614278,-0.05380646625820,-0.08765869114446,
0.04380166398048,0.05671236507154,-0.03567339847522,-0.03226375971045,
0.02707861927863,0.01528796063552,-0.01839974396002,-0.00529685959906,
0.01091563182436,0.00061967186322,-0.00553073010417,0.00084338462127,
0.00232452011131,-0.00086092770350,-0.00076787825930,0.00050509482141,
0.00017248258226,-0.00021617183228,-0.00000854830600,0.00006982008384,
-0.00001339716873,-0.00001636152487,0.00000725214556,0.00000232754911,
-0.00000218726769,0.00000001099474,0.00000042616623,-0.00000010004147,
-0.00000004764380,0.00000002605443,0.00000000055534,-0.00000000333111,
0.00000000069849,0.00000000016136,-0.00000000009461,0.00000000001000,
0.00000000000324,-0.00000000000119,0.00000000000015,-0.00000000000001};

// constructors

template<class DataType_t> Daubechies<DataType_t>::
Daubechies(const Wavelet &w) : 
WaveDWT<DataType_t>(w) 
{ 
   setFilter();
}

template<class DataType_t> Daubechies<DataType_t>::
Daubechies(const Daubechies<DataType_t> &w) : 
WaveDWT<DataType_t>(w) 
{ 
   setFilter();
}

template<class DataType_t> Daubechies<DataType_t>::
Daubechies(int m, int tree, enum BORDER border) :
WaveDWT<DataType_t>(m,m,tree,border) 
{
   setFilter();
}

// destructor
template<class DataType_t>
Daubechies<DataType_t>::~Daubechies()
{ 
   if(pLForward) delete [] pLForward;
   if(pLInverse) delete [] pLInverse;
   if(pHForward) delete [] pHForward;
   if(pHInverse) delete [] pHInverse;
}

// clone
template<class DataType_t>
Daubechies<DataType_t>* Daubechies<DataType_t>::Clone() const
{
  return new Daubechies<DataType_t>(*this);
}

template<class DataType_t>
void Daubechies<DataType_t>::setFilter()
{
   const double* pF;
   this->m_H = (this->m_H>>1)<<1;
   int n = this->m_H/2;
   switch(n)
   {
      case  1: pF =  dbc1; this->m_H =  2; break;
      case  2: pF =  dbc2; this->m_H =  4; break;
      case  3: pF =  dbc3; this->m_H =  6; break;
      case  4: pF =  dbc4; this->m_H =  8; break;
      case  5: pF =  dbc5; this->m_H = 10; break;
      case  6: pF =  dbc6; this->m_H = 12; break;
      case  7: pF =  dbc7; this->m_H = 14; break;
      case  8: pF =  dbc8; this->m_H = 16; break;
      case  9: pF =  dbc9; this->m_H = 18; break;
      case 10: pF = dbc10; this->m_H = 20; break;
      case 11: pF = dbc11; this->m_H = 22; break;
      case 12: pF = dbc12; this->m_H = 24; break;
      case 13: pF = dbc13; this->m_H = 26; break;
      case 14: pF = dbc14; this->m_H = 28; break;
      case 15: pF = dbc15; this->m_H = 30; break;
      case 16: pF = dbc16; this->m_H = 32; break;
      case 17: pF = dbc17; this->m_H = 34; break;
      case 18: pF = dbc18; this->m_H = 36; break;
      case 19: pF = dbc19; this->m_H = 38; break;
      case 20: pF = dbc20; this->m_H = 40; break;
      case 21: pF = dbc21; this->m_H = 42; break;
      case 22: pF = dbc22; this->m_H = 44; break;
      case 23: pF = dbc23; this->m_H = 46; break;
      case 24: pF = dbc24; this->m_H = 48; break;
      case 25: pF = dbc25; this->m_H = 50; break;
      case 26: pF = dbc26; this->m_H = 52; break;
      case 27: pF = dbc27; this->m_H = 54; break;
      case 28: pF = dbc28; this->m_H = 56; break;
      case 29: pF = dbc29; this->m_H = 58; break;
      case 30: pF = dbc30; this->m_H = 60; break;
      default: pF =  dbc4; this->m_H =  8; break;
   }
   
   pLInverse = new double[this->m_H];
   pLForward = new double[this->m_H];
   pHInverse = new double[this->m_H];
   pHForward = new double[this->m_H];

//  LP filter for db3:  h0  h1  h2  h3  h4  h5
//  HP filter for db3:  h5 -h4  h3 -h2  h1 -h0
// iLP filter for db3:  h4  h1  h2  h3  h0  h5
// iHP filter for db3:  h5 -h0  h3 -h2  h1 -h4

   for(int i=0; i<this->m_H; i+=2){

      pLForward[i]   = pF[i];
      pLForward[i+1] = pF[i+1];
      pHForward[i]   = pF[this->m_H-1-i];
      pHForward[i+1] = -pF[this->m_H-2-i];

      pLInverse[i]   = pF[this->m_H-2-i];
      pLInverse[i+1] = pF[i+1];
      pHInverse[i]   = pF[this->m_H-1-i];
      pHInverse[i+1] = -pF[i];	 

   }

   this->m_WaveType = DAUBECHIES;
}

// forward function does one step of forward transformation.
// <level> input parameter is the level to be reconstructed
// <layer> input parameter is the layer to be reconstructed.
template<class DataType_t>
void Daubechies<DataType_t>::forward(int level,int layer)
{ 
   this->forwardFWT(level, layer, pLForward, pHForward); 
}

// inverse function does one step of inverse transformation.
// <level> input parameter is the level to be reconstructed
// <layer> input parameter is the layer to be reconstructed.
template<class DataType_t>
void Daubechies<DataType_t>::inverse(int level,int layer)
{
   this->inverseFWT(level, layer, pLInverse, pHInverse); 
}

// instantiations

#define CLASS_INSTANTIATION(class_) template class Daubechies< class_ >;

CLASS_INSTANTIATION(float)
CLASS_INSTANTIATION(double)
//CLASS_INSTANTIATION(std::complex<float>)
//CLASS_INSTANTIATION(std::complex<double>)

#undef CLASS_INSTANTIATION

//template Daubechies<float>::
//Daubechies(const Daubechies<float> &);
//template Daubechies<double>::
//Daubechies(const Daubechies<double> &);

//}  // end namespace wat
//}  // end namespace datacondAPI






