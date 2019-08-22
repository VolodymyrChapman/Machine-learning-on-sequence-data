import math
import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
from sklearn.pipeline import Pipeline
import pickle
os.chdir(r'C:\Users\chapmanvl\Documents\VC 2019 projects')
print(os.getcwd())

# Extract sequence name and sequence 
def getseq(file):
    pure_chunk = '' 
    with open (file) as seq:
        path, name = os.path.split(file)
        name_stripped = os.path.splitext(name)[0]
        for line in seq:
                upperLine = line.upper()
                pure_upper_line = ''.join([ s for s in upperLine if s in 'ATGC'])
                pure_chunk += pure_upper_line
    return pure_chunk, name_stripped

#get CGR Z co-ordinates only
def getz(oligomer):
    ruleX = {'A': 0, 'C' : 0, 'G': 1, 'T' : 1}
    ruleY = {'A': 0, 'C' : 1, 'G': 1, 'T' : 0}
    X = [ruleX[i] for i in oligomer ] 
    X.insert(0,0.5)
    Y = [ruleY[i] for i in oligomer ] 
    Y.insert(0,0.5)
    
    for i in range(len(X[1:])):
        h = i + 1
        X[h] = (X[i] + X[h])/2
    for i in range(len(Y[1:])):
        h = i + 1
        Y[h] = (Y[i] + Y[h])/2
    Z = []
    for x,y in zip(X[1:],Y[1:]):
         Z.append(complex(x,y))
    return Z


# Read all files in a folder and find fourier, save to pickle file in a new folder

def multiple_seq_to_fourier(folder):
    files = [file for file in listdir(folder) if isfile(join(folder, file))] #only use files
    # initiate details list
    names = []
    ffts = []
    for file in files:
        sequence, name = getseq(join(folder,file))
        names.append(name)
        encoded = getz(sequence)
        ffts.append(np.fft.fft(encoded))
    return names, ffts

# test case: details = multiple_seq_to_ps_file(r'C:\Users\chapmanvl\Documents\VC 2019 projects\testing_cut_random', 'power_signals')

names, ffts = multiple_seq_to_fourier(r'C:\Users\chapmanvl\Documents\VC 2019 projects\Influenza A segment 6 NA\extended_list_57_tagged')
df_ffts = pd.DataFrame(ffts, index = names).T

# relabelling rows (index) to be shorter
black = ['blackCY0147881InfluenzaAvirusAturkeyMinnesota11988H7N9segment6completesequence',
       'blackCY1860041InfluenzaAvirusAmallardMinnesotaAI0937702009H7N9neuraminidaseNAgenecompletecds',
       'blackCY2353641InfluenzaAvirusAreassortantIDCDCRG56BHongKong1252017XPuertoRico81934H7N9neuraminidaseNAgenecompletecds',
       'blackGU0604841InfluenzaAVirusAgooseCzechRepublic1848K92009H7N9segment6neuraminidaseNAgenecompletecds',
       'blackKC6098011InfluenzaAvirusAwildduckKoreaSH19472010H7N9segment6neuraminidaseNAgenecompletecds',
       'blackKC8532311InfluenzaAvirusAShanghai4664T2013H7N9segment6neuraminidaseNAgenecompletecds',
       'blackKF2596881InfluenzaAvirusAduckJiangxi30962009H7N9segment6neuraminidaseNAgenecompletecds',
       'blackKF2597341InfluenzaAvirusAchickenRizhao7132013H7N9segment6neuraminidaseNAgenecompletecds',
       'blackKF6095291InfluenzaAvirusAtreesparrowShanghai012013H7N9segment6neuraminidaseNAgenecompletecds',
       'blackKF9389451InfluenzaAvirusAchickenJiangsu10212013H7N9segment6neuraminidaseNAgenecompletecds',
       'blackKY7511241InfluenzaAvirusAchickenGuangdongGD152016H7N9segment6neuraminidaseNAgenecompletecds']
red = ['redAB6841611InfluenzaAvirusAchickenMiyazaki102011H5N1NAgeneforneuraminidasecompletecds',
       'redAF5091022InfluenzaAvirusAChickenHongKong822101H5N1neuraminidaseNAgenecompletecds',
       'redAM9140171InfluenzaAvirusAdomesticduckGermanyR177207H5N1N1geneforneuraminidasegenomicRNA',
       'redEF5414641InfluenzaAvirusAchickenKoreaes2003H5N1segment6neuraminidaseNAgenecompletecds',
       'redEU6358751InfluenzaAvirusAchickenYunnanchuxiong012005H5N1neuraminidaseNAgenecompletecds',
       'redFM1771211InfluenzaAvirusAchickenGermanyR32342007H5N1NAgeneforneuraminidase',
       'redGU1865111InfluenzaAvirusAturkeyVA505477182007H5N1segment6neuraminidaseNAgenecompletecds',
       'redHQ1853811InfluenzaAvirusAchickenEasternChinaXH2222008H5N1neuraminidaseNAgenecompletecds',
       'redHQ1853831InfluenzaAvirusAduckEasternChinaJS0172009H5N1segment6neuraminidaseNAgenecompletecds',
       'redJF6996771InfluenzaAvirusAmandarinduckKoreaK104832010H5N1segment6neuraminidaseNAgenecompletecds',
       'redKF5724351InfluenzaAvirusAwildbirdHongKong0703512011H5N1segment6neuraminidaseNAgenecompletecds']
blue = ['blueAB4706631InfluenzaAvirusAduckHokkaidow732007H1N1NAgeneforneuraminidasecompletecds',
       'blueAB5461591InfluenzaAvirusApintailMiyagi14722008H1N1NAgeneforneuraminidasecompletecds',
       'blueAM1573581InfluenzaAvirusAmallardFrance6912002H1N1NAgeneforneuraminidasegenomicRNA',
       'blueCY1385621InfluenzaAvirusAmallardNovaScotia000882010H1N1neuraminidaseNAgenecompletecds',
       'blueCY1400471InfluenzaAvirusAmallardMinnesotaSg006202008H1N1neuraminidaseNAgenecompletecds',
       'blueCY1496301InfluenzaAvirusAthickbilledmurreCanada18712011H1N1neuraminidaseNAgenecompletecds',
       'blueEU0260462InfluenzaAvirusAmallardMaryland3522002H1N1segment6neuraminidaseNAgenecompletecds',
       'blueFJ3571141InfluenzaAvirusAmallardMD262003H1N1segment6neuraminidaseNAgenecompletecds',
       'blueGQ4118941InfluenzaAvirusAdunlinAlaska444216602008H1N1segment6neuraminidaseNAgenecompletecds',
       'blueHM3709691InfluenzaAvirusAturkeyOntarioFAV11042009H1N1segment6neuraminidaseNAgenecompletecds',
       'blueHQ8979661InfluenzaAvirusAmallardKoreaKNUYP092009H1N1segment6neuraminidaseNAgenecompletecds',
       'blueKC6081601InfluenzaAvirusAduckGuangxi030D2009H1N1segment6neuraminidaseNAgenecompletecds',
       'blueKM2440781InfluenzaAvirusAturkeyVirginia41352014H1N1segment6neuraminidaseNAgenecompletecds']
green = ['greenAY6460801InfluenzaAvirusAchickenBritishColumbiaGSChumanB04H7N3neuraminidaseNAgenecompletecds',
       'greenCY0393211InfluenzaAvirusAavianDelawareBay2262006H7N3segment6completesequence',
       'greenCY0762311InfluenzaAvirusAAmericangreenwingedtealCalifornia442429062007H7N3segment6completesequence',
       'greenCY1293361InfluenzaAvirusAAmericanblackduckNewBrunswick024902007H7N3neuraminidaseNAgenecompletecds',
       'greenEU0309701InfluenzaAvirusAshorebirdDelaware2206H7N3segment6neuraminidaseNANAgenecompletecds',
       'greenEU0309861InfluenzaAvirusAlaughinggullDelaware4206H7N3segment6neuraminidaseNANAgenecompletecds',
       'greenEU5008541InfluenzaAvirusAAmericanblackduckNB25382007H7N3segment6completesequence',
       'greenFJ6383281InfluenzaAvirusAchickenKarachiSPVC52004H7N3segment6neuraminidaseNAgenecompletecds',
       'greenHM1507381InfluenzaAvirusAchickenMurreeNARC11995H7N3segment6neuraminidaseNAgenecompletecds',
       'greenKF0420671InfluenzaAvirusAduckZhejiangDK102013H7N3segment6neuraminidaseNAgenecompletecds',
       'greenKX2896531InfluenzaAvirusAchickenPueblaCPA0330916CENASA950762016H7N3segment6neuraminidaseNAgenecompletecds',
       'greenMN2080521InfluenzaAvirusAnorthernshovelerEgyptMBD695C2016H7N3segment6neuraminidaseNAgenecompletecds']
purple = ['purpleAY2099251InfluenzaAvirusATaiwan167H2N2neuraminidaseNAgenecompletecds',
       'purpleAY2099281InfluenzaAvirusAGeorgia167H2N2neuraminidaseNAgenecompletecds',
       'purpleAY2099321InfluenzaAvirusAKorea42668H2N2neuraminidaseNAgenecompletecds',
       'purpleAY2099331InfluenzaAvirusABerkeley168H2N2neuraminidaseNAgenecompletecds',
       'purpleCY0055401InfluenzaAvirusAduckHongKong3191978H2N2segment6completesequence',
       'purpleCY1219611InfluenzaAvirusAwhitefrontedgooseNetherlands221999H2N2neuraminidaseNAgenecompletecds',
       'purpleCY1258881InfluenzaAvirusANed651963H2N2neuraminidaseNAgenecompletecds',
       'purpleDQ0174871InfluenzaAvirusAmallardPostdam178483H2N2fromGermanysegment6completesequence',
       'purpleJX0811421InfluenzaAvirusAemperorgooseAlaska442972602007H2N2segment6neuraminidaseNAgenecompletecds',
       'purpleL373292InfluenzaAvirusALeningrad13457H2N2neuraminidaseNAgenecompletecds']       


labelled = df_ffts[:]
labelled = labelled.T
labelled.columns = range(1,1468)
labelled['colour'] = 0
labelled.loc[black, 'colour'] = 1
labelled.loc[blue, 'colour'] = 2
labelled.loc[green, 'colour'] = 3
labelled.loc[purple, 'colour'] = 4
labelled.loc[red, 'colour'] = 5
labelled.tail()

# change index (row) names to be shorter
blacknum = []
bluenum = []
greennum = []
purplenum = []
rednum = []
for i in range(14):
    blacknum.append(join(f'black{i}'))
    bluenum.append(join(f'blue{i}')) 
    greennum.append(join(f'green{i}')) 
    purplenum.append(join(f'purple{i}')) 
    rednum.append(join(f'red{i}')) 

colnames = []
colnames.extend(blacknum[1:12])
colnames.extend(bluenum[1:14]) 
colnames.extend(greennum[1:13]) 
colnames.extend(purplenum[1:11]) 
colnames.extend(rednum[1:12])
labelled.index = colnames

# convert frequencies to periods and reverse and relabel the dataframe -duh, need to calculate
#period for EACH sequence as dependent on length (1/f for each value)
## will need: fig,ax = subplots();  ax.scatter(a.real,a.imag) to plot real and imaginary separately 

frequencies_string = labelled.columns[:1467]
frequencies_float = [float(i) for i in frequencies_string]
periods = [1 / i for i in frequencies_float]

periods_df = labelled[:]
periods_df = periods_df.drop('colour', axis = 1)
periods_df = periods_df.reindex(columns=periods_df.columns[::-1])
periods_df.columns = periods