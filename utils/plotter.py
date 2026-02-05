import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json, re, glob
from tqdm import tqdm
import seaborn as sns





def datacollector_realtest(json_files_path):
    #datafile = pd.read_csv(csv_file_path)
    
    DRW_filenames = glob.glob(json_files_path + '/*[0-9]_DRW.*'); DRW_filenames.sort()
    DRWsine_filenames = glob.glob(json_files_path + '/*[0-9]_DRWsine.*'); DRWsine_filenames.sort()
    CARMA21_filenames = glob.glob(json_files_path+'/*[0-9]_CARMA21.*'); CARMA21_filenames.sort()
    CARMA21sine_filenames = glob.glob(json_files_path+'/*[0-9]_CARMA21sine.*'); CARMA21sine_filenames.sort()
    OBPL10_filenames = glob.glob(json_files_path + '/*[0-9]_OBPL10.*'); OBPL10_filenames.sort()
    OBPLsine10_filenames = glob.glob(json_files_path+'/*[0-9]_OBPL10sine.*'); OBPLsine10_filenames.sort()
    
    
    
    DRW_IDs, DRWsine_IDs, CARMA21_IDs, CARMA21sine_IDs, OBPL10_IDs, OBPLsine10_IDs = [],[],[],[],[],[]
    
    for val in zip(DRW_filenames, DRWsine_filenames, CARMA21_filenames, CARMA21sine_filenames, OBPL10_filenames, OBPLsine10_filenames):
        DRW_IDs.append(int(re.search('/[0-9]*[0-9]*[0-9]*[0-9]*[0-9]_',val[0]).group()[1:-1]))
        DRWsine_IDs.append(int(re.search('/[0-9]*[0-9]*[0-9]*[0-9]*[0-9]_',val[1]).group()[1:-1]))
        CARMA21_IDs.append(int(re.search('/[0-9]*[0-9]*[0-9]*[0-9]*[0-9]_',val[2]).group()[1:-1]))
        CARMA21sine_IDs.append(int(re.search('/[0-9]*[0-9]*[0-9]*[0-9]*[0-9]_',val[3]).group()[1:-1]))
        OBPL10_IDs.append(int(re.search('/[0-9]*[0-9]*[0-9]*[0-9]*[0-9]_',val[4]).group()[1:-1]))
        OBPLsine10_IDs.append(int(re.search('/[0-9]*[0-9]*[0-9]*[0-9]*[0-9]_',val[5]).group()[1:-1]))

    print('Collected all json files...')


    
    # if len(datafile) != len(DRW_IDs):
    #     common_IDs = datafile.ID.to_list()
         
    #     print('Not all json IDs are present in the datafile... Continuing with crossmatched IDs\n')      
    #     DRW_filenames = []
    #     DRWsine_filenames = []
    #     CARMA21_filenames = []
    #     CARMA21sine_filenames = []
    #     OBPL10_filenames = []
    #     OBPLsine10_filenames = []
    #     for id in common_IDs:
    #         DRW_filenames.append(json_files_path+str(int(id))+'_DRW.json')
    #         DRWsine_filenames.append(json_files_path+str(int(id))+'_DRWsine.json')
    #         CARMA21_filenames.append(json_files_path+str(int(id))+'_CARMA21.json')
    #         CARMA21sine_filenames.append(json_files_path+str(int(id))+'_CARMA21sine.json')
    #         OBPL10_filenames.append(json_files_path+str(int(id))+'_OBPL10.json')
    #         OBPLsine10_filenames.append(json_files_path+str(int(id))+'_OBPLsine10.json')

    
    def load_json(file_path):
        with open(file_path) as json_file:
            return json.load(json_file)
        
    datafile = pd.DataFrame({'ID':DRW_IDs})
    
    columns_to_drop = ['DRW_log_Z_mean', 'DRW_log_Z_uncert', 'DRWsine_log_Z_mean', 'DRWsine_log_Z_uncert', 'CARMA21_log_Z_mean', 'CARMA21_log_Z_uncert',
                 'CARMA21sine_log_Z_mean','CARMA21sine_log_Z_uncert', 'OBPL10_log_Z_mean', 'OBPL10_log_Z_uncert', 'OBPLsine10_log_Z_mean','OBPLsine10_log_Z_uncert' ,
                 'DRW_bf', 'DRWsine_bf', 'OBPL10_bf', 'OBPLsine10_bf' ]
    
    datafile.drop(columns=columns_to_drop, errors='ignore', inplace=True)
    
    update_data = {
    'ID': [],
    'DRW_log_Z_mean': [], 'DRW_log_Z_uncert': [],
    'DRWsine_log_Z_mean': [], 'DRWsine_log_Z_uncert': [],
    'CARMA21_log_Z_mean': [], 'CARMA21_log_Z_uncert': [], 
    'CARMA21sine_log_Z_mean': [], 'CARMA21sine_log_Z_uncert': [],
    'OBPL10_log_Z_mean': [], 'OBPL10_log_Z_uncert': [], 'OBPL10_alpha_h':[],'OBPL10_alpha_l':[],'OBPL10_alpha_l_84':[], 'OBPL10_alpha_l_16':[], 'OBPL10_alpha_h_84':[], 'OBPL10_alpha_h_16':[],
    'OBPLsine10_log_Z_mean': [], 'OBPLsine10_log_Z_uncert': [], 'OBPLsine10_alpha_h':[], 'OBPLsine10_alpha_l':[], 'OBPLsine10_alpha_l_84':[], 'OBPLsine10_alpha_l_16':[], 'OBPLsine10_alpha_h_84':[], 'OBPLsine10_alpha_h_16':[] }
    
    for i in tqdm(range(len(DRW_filenames)), desc = 'Gathering data from json files to dataframe'):
        
        val = DRW_filenames[i], DRWsine_filenames[i], CARMA21_filenames[i], CARMA21sine_filenames[i], OBPL10_filenames[i], OBPLsine10_filenames[i]
        ID = int(re.search('[0-9]*[0-9]*[0-9]*[0-9]*[0-9]_',val[0]).group()[:-1])
        
        DRW_dict = load_json(val[0])
        DRWsine_dict = load_json(val[1])
        CARMA21_dict = load_json(val[2])
        CARMA21sine_dict = load_json(val[3])
        OBPL10_dict = load_json(val[4])
        OBPLsine10_dict = load_json(val[5])
        
        update_data['ID'].append(ID)
        update_data['DRW_log_Z_mean'].append(DRW_dict['log_Z_mean'])
        update_data['DRW_log_Z_uncert'].append(DRW_dict['log_Z_uncert'])
        #update_data['DRW_bf'].append(np.power(10, np.median(DRW_dict['log_bend_freq'])))
    
        update_data['DRWsine_log_Z_mean'].append(DRWsine_dict['log_Z_mean'])
        update_data['DRWsine_log_Z_uncert'].append(DRWsine_dict['log_Z_uncert'])
        #update_data['DRWsine_bf'].append(np.power(10, np.median(DRWsine_dict['log_bend_freq'])))
        
        update_data['CARMA21_log_Z_mean'].append(CARMA21_dict['log_Z_mean'])
        update_data['CARMA21_log_Z_uncert'].append(CARMA21_dict['log_Z_uncert'])
        
        update_data['CARMA21sine_log_Z_mean'].append(CARMA21sine_dict['log_Z_mean'])
        update_data['CARMA21sine_log_Z_uncert'].append(CARMA21sine_dict['log_Z_uncert'])
        
        update_data['OBPL10_log_Z_mean'].append(OBPL10_dict['log_Z_mean'])
        update_data['OBPL10_log_Z_uncert'].append(OBPL10_dict['log_Z_uncert'])
        update_data['OBPL10_alpha_h'].append(np.percentile(OBPL10_dict['alpha_h'], q=[50])[0])
        update_data['OBPL10_alpha_h_16'].append(np.percentile(OBPL10_dict['alpha_h'], q=[16])[0])
        update_data['OBPL10_alpha_h_84'].append(np.percentile(OBPL10_dict['alpha_h'], q=[84])[0])
        update_data['OBPL10_alpha_l'].append(np.percentile(OBPL10_dict['alpha_l'], q=[50])[0])
        update_data['OBPL10_alpha_l_16'].append(np.percentile(OBPL10_dict['alpha_l'], q=[16])[0])
        update_data['OBPL10_alpha_l_84'].append(np.percentile(OBPL10_dict['alpha_l'], q=[84])[0])
        #update_data['OBPL10_bf'].append(np.power(10, np.median(OBPL10_dict['log_bend_freq'])))
        
        update_data['OBPLsine10_log_Z_mean'].append(OBPLsine10_dict['log_Z_mean'])
        update_data['OBPLsine10_log_Z_uncert'].append(OBPLsine10_dict['log_Z_uncert'])
        update_data['OBPLsine10_alpha_h'].append(np.percentile(OBPLsine10_dict['alpha_h'], q=[50])[0])
        update_data['OBPLsine10_alpha_h_16'].append(np.percentile(OBPLsine10_dict['alpha_h'], q=[16])[0])
        update_data['OBPLsine10_alpha_h_84'].append(np.percentile(OBPLsine10_dict['alpha_h'], q=[84])[0])
        update_data['OBPLsine10_alpha_l'].append(np.percentile(OBPLsine10_dict['alpha_l'], q=[50])[0])
        update_data['OBPLsine10_alpha_l_16'].append(np.percentile(OBPLsine10_dict['alpha_l'], q=[16])[0])
        update_data['OBPLsine10_alpha_l_84'].append(np.percentile(OBPLsine10_dict['alpha_l'], q=[84])[0])
        #update_data['OBPLsine10_bf'].append(np.power(10, np.median(OBPLsine10_dict['log_bend_freq'])))
        
    update_df = pd.DataFrame(update_data)

    datafile = datafile.merge(update_df, on='ID', how='right', suffixes=(None, '_new'))
    
    datafile['DRW_DRWsine_bayes'] = np.exp(datafile.DRW_log_Z_mean - datafile.DRWsine_log_Z_mean)
    datafile['CARMA21_CARMA21sine_bayes'] = np.exp(datafile.CARMA21_log_Z_mean - datafile.CARMA21sine_log_Z_mean)
    datafile['OBPL10_OBPLsine10_bayes'] = np.exp(datafile.OBPL10_log_Z_mean - datafile.OBPLsine10_log_Z_mean)
    
    datafile[['DRWsine_bool', 'CARMA21sine_bool','OBPLsine_bool',]] = 0, 0, 0
    bayes_threshold = 2
    for i, row in datafile.iterrows():
        if np.log10(row.DRW_DRWsine_bayes)<(-1 * bayes_threshold):
            datafile.loc[datafile.ID == row.ID,'DRWsine_bool'] = 1
        if np.log10(row.DRW_DRWsine_bayes)>bayes_threshold:
            datafile.loc[datafile.ID == row.ID,'DRWsine_bool'] = -1
        if np.log10(row.CARMA21_CARMA21sine_bayes)<(-1 * bayes_threshold):
            datafile.loc[datafile.ID == row.ID,'CARMA21sine_bool'] = 1
        if np.log10(row.CARMA21_CARMA21sine_bayes)>bayes_threshold:
            datafile.loc[datafile.ID == row.ID,'CARMA21sine_bool'] = -1
        if np.log10(row.OBPL10_OBPLsine10_bayes)<(-1 * bayes_threshold):
            datafile.loc[datafile.ID == row.ID,'OBPLsine_bool'] = 1
        if np.log10(row.OBPL10_OBPLsine10_bayes)>bayes_threshold:
            datafile.loc[datafile.ID == row.ID,'OBPLsine_bool'] = -1
    
    datafile.sort_values(by='ID', inplace=True)
    datafile.reset_index(inplace=True, drop=True)
    
    
    return datafile.copy()





def datacollector(csv_file_path, json_files_path):
    datafile = pd.read_csv(csv_file_path)
    
    DRW_filenames = glob.glob(json_files_path + '/*DRW.*'); DRW_filenames.sort()
    DRWsine_filenames = glob.glob(json_files_path + '/*DRWsine.*'); DRWsine_filenames.sort()
    CARMA21_filenames = glob.glob(json_files_path+'/*CARMA21.*'); CARMA21_filenames.sort()
    CARMA21sine_filenames = glob.glob(json_files_path+'/*CARMA21sine.*'); CARMA21sine_filenames.sort()
    OBPL10_filenames = glob.glob(json_files_path + '/*OBPL10.*'); OBPL10_filenames.sort()
    OBPLsine10_filenames = glob.glob(json_files_path+'/*OBPLsine10.*'); OBPLsine10_filenames.sort()
    
    DRW_IDs, DRWsine_IDs, CARMA21_IDs, CARMA21sine_IDs, OBPL10_IDs, OBPLsine10_IDs = [],[],[],[],[],[]
    
    for val in zip(DRW_filenames, DRWsine_filenames, CARMA21_filenames, CARMA21sine_filenames, OBPL10_filenames, OBPLsine10_filenames):
        DRW_IDs.append(int(re.search('[0-9][0-9][0-9][0-9]*[0-9]',val[0]).group()))
        DRWsine_IDs.append(int(re.search('[0-9][0-9][0-9][0-9]*[0-9]',val[1]).group()))
        CARMA21_IDs.append(int(re.search('[0-9][0-9][0-9][0-9]*[0-9]',val[2]).group()))
        CARMA21sine_IDs.append(int(re.search('[0-9][0-9][0-9][0-9]*[0-9]',val[3]).group()))
        OBPL10_IDs.append(int(re.search('[0-9][0-9][0-9][0-9]*[0-9]',val[4]).group()))
        OBPLsine10_IDs.append(int(re.search('[0-9][0-9][0-9][0-9]*[0-9]',val[5]).group()))

    print('Collected all json files...')


    if len(datafile) != len(DRW_IDs):
        common_IDs = datafile.ID.to_list()
         
        print('Not all json IDs are present in the datafile... Continuing with crossmatched IDs\n')      
        DRW_filenames = []
        DRWsine_filenames = []
        CARMA21_filenames = []
        CARMA21sine_filenames = []
        OBPL10_filenames = []
        OBPLsine10_filenames = []
        for id in common_IDs:
            DRW_filenames.append(json_files_path+str(int(id))+'_DRW.json')
            DRWsine_filenames.append(json_files_path+str(int(id))+'_DRWsine.json')
            CARMA21_filenames.append(json_files_path+str(int(id))+'_CARMA21.json')
            CARMA21sine_filenames.append(json_files_path+str(int(id))+'_CARMA21sine.json')
            OBPL10_filenames.append(json_files_path+str(int(id))+'_OBPL10.json')
            OBPLsine10_filenames.append(json_files_path+str(int(id))+'_OBPLsine10.json')

    
    def load_json(file_path):
        with open(file_path) as json_file:
            return json.load(json_file)
    
    columns_to_drop = ['DRW_log_Z_mean', 'DRW_log_Z_uncert', 'DRWsine_log_Z_mean', 'DRWsine_log_Z_uncert', 'CARMA21_log_Z_mean', 'CARMA21_log_Z_uncert',
                 'CARMA21sine_log_Z_mean','CARMA21sine_log_Z_uncert', 'OBPL10_log_Z_mean', 'OBPL10_log_Z_uncert', 'OBPLsine10_log_Z_mean','OBPLsine10_log_Z_uncert' ,
                 'DRW_bf', 'DRWsine_bf', 'OBPL10_bf', 'OBPLsine10_bf' ]
    
    datafile.drop(columns=columns_to_drop, errors='ignore', inplace=True)
    
    update_data = {
    'ID': [],
    'DRW_log_Z_mean': [], 'DRW_log_Z_uncert': [], 'DRW_bf': [],
    'DRWsine_log_Z_mean': [], 'DRWsine_log_Z_uncert': [], 'DRWsine_bf': [], 'DRWsine_t0' : [],
    'CARMA21_log_Z_mean': [], 'CARMA21_log_Z_uncert': [], 
    'CARMA21sine_log_Z_mean': [], 'CARMA21sine_log_Z_uncert': [], 'CARMA21sine_t0' : [], 
    'OBPL10_log_Z_mean': [], 'OBPL10_log_Z_uncert': [], 'OBPL10_bf': [],'OBPL10_alpha_h': [],
    'OBPLsine10_log_Z_mean': [], 'OBPLsine10_log_Z_uncert': [], 'OBPLsine10_bf': [], 'OBPLsine10_alpha_h': [], 'OBPLsine10_t0' : []}
    
    for i in tqdm(range(len(DRW_filenames)), desc = 'Gathering data from json files to dataframe'):
        val = DRW_filenames[i], DRWsine_filenames[i], CARMA21_filenames[i], CARMA21sine_filenames[i], OBPL10_filenames[i], OBPLsine10_filenames[i]
        ID = int(re.search('[0-9][0-9][0-9][0-9]*[0-9]',val[0]).group())
        
        DRW_dict = load_json(val[0])
        DRWsine_dict = load_json(val[1])
        CARMA21_dict = load_json(val[2])
        CARMA21sine_dict = load_json(val[3])
        OBPL10_dict = load_json(val[4])
        OBPLsine10_dict = load_json(val[5])
        
        update_data['ID'].append(ID)
        update_data['DRW_log_Z_mean'].append(DRW_dict['log_Z_mean'])
        update_data['DRW_log_Z_uncert'].append(DRW_dict['log_Z_uncert'])
        update_data['DRW_bf'].append(np.power(10, np.median(DRW_dict['log_bend_freq'])))
    
        update_data['DRWsine_log_Z_mean'].append(DRWsine_dict['log_Z_mean'])
        update_data['DRWsine_log_Z_uncert'].append(DRWsine_dict['log_Z_uncert'])
        update_data['DRWsine_bf'].append(np.power(10, np.median(DRWsine_dict['log_bend_freq'])))
        update_data['DRWsine_t0'].append( np.median(DRWsine_dict['t0']))
        
        update_data['CARMA21_log_Z_mean'].append(CARMA21_dict['log_Z_mean'])
        update_data['CARMA21_log_Z_uncert'].append(CARMA21_dict['log_Z_uncert'])
        
        update_data['CARMA21sine_log_Z_mean'].append(CARMA21sine_dict['log_Z_mean'])
        update_data['CARMA21sine_log_Z_uncert'].append(CARMA21sine_dict['log_Z_uncert'])
        update_data['CARMA21sine_t0'].append( np.median(CARMA21sine_dict['t0']))
        
        update_data['OBPL10_log_Z_mean'].append(OBPL10_dict['log_Z_mean'])
        update_data['OBPL10_log_Z_uncert'].append(OBPL10_dict['log_Z_uncert'])
        update_data['OBPL10_bf'].append(np.power(10, np.median(OBPL10_dict['log_bend_freq'])))
        update_data['OBPL10_alpha_h'].append(np.median(OBPL10_dict['alpha_h']))
        
        update_data['OBPLsine10_log_Z_mean'].append(OBPLsine10_dict['log_Z_mean'])
        update_data['OBPLsine10_log_Z_uncert'].append(OBPLsine10_dict['log_Z_uncert'])
        update_data['OBPLsine10_bf'].append(np.power(10, np.median(OBPLsine10_dict['log_bend_freq'])))
        update_data['OBPLsine10_alpha_h'].append(np.median(OBPLsine10_dict['alpha_h']))
        update_data['OBPLsine10_t0'].append( np.median(OBPLsine10_dict['t0']))
        
    update_df = pd.DataFrame(update_data)

    datafile = datafile.merge(update_df, on='ID', how='right', suffixes=(None, '_new'))
    
    datafile['DRW_DRWsine_bayes'] = np.exp(datafile.DRW_log_Z_mean - datafile.DRWsine_log_Z_mean)
    datafile['CARMA21_CARMA21sine_bayes'] = np.exp(datafile.CARMA21_log_Z_mean - datafile.CARMA21sine_log_Z_mean)
    datafile['OBPL10_OBPLsine10_bayes'] = np.exp(datafile.OBPL10_log_Z_mean - datafile.OBPLsine10_log_Z_mean)
    
    datafile[['DRWsine_bool', 'CARMA21sine_bool','OBPLsine_bool',]] = 0, 0, 0
    bayes_threshold = 2
    for i, row in datafile.iterrows():
        if np.log10(row.DRW_DRWsine_bayes)<(-1 * bayes_threshold):
            datafile.loc[datafile.ID == row.ID,'DRWsine_bool'] = 1
        if np.log10(row.DRW_DRWsine_bayes)>bayes_threshold:
            datafile.loc[datafile.ID == row.ID,'DRWsine_bool'] = -1
        if np.log10(row.CARMA21_CARMA21sine_bayes)<(-1 * bayes_threshold):
            datafile.loc[datafile.ID == row.ID,'CARMA21sine_bool'] = 1
        if np.log10(row.CARMA21_CARMA21sine_bayes)>bayes_threshold:
            datafile.loc[datafile.ID == row.ID,'CARMA21sine_bool'] = -1
        if np.log10(row.OBPL10_OBPLsine10_bayes)<(-1 * bayes_threshold):
            datafile.loc[datafile.ID == row.ID,'OBPLsine_bool'] = 1
        if np.log10(row.OBPL10_OBPLsine10_bayes)>bayes_threshold:
            datafile.loc[datafile.ID == row.ID,'OBPLsine_bool'] = -1
    
    return datafile.copy()



class statPLOTS:
    def __init__(self, data_file) -> None:
        if isinstance(data_file, str):
            self.datafile = pd.read_csv(data_file)
            
        if isinstance(data_file, pd.DataFrame):
            self.datafile = data_file
            
    
    def get_dataframe(self):
        return self.datafile
    
    def fillna_model_counts(self, dictionary):
    
        BMCdict = dictionary.copy()
        for key in dictionary:

            if (key[0], 0) not in dictionary:
                BMCdict[(key[0], 0)] = 0
            if (key[0], 1) not in dictionary:
                BMCdict[(key[0], 1)] = 0
            if (key[0], -1) not in dictionary:
                BMCdict[(key[0], -1)] = 0

        return BMCdict
    
    
    
    
    def plot_barchart(self, param, model = 'DRW', xarray = [-99]):
        if model =='DRW':
            boolvar = 'DRWsine_bool'
        if model =='CARMA21':
            boolvar = 'CARMA21sine_bool'
        if model =='OBPL':
            boolvar = 'OBPLsine_bool'
        
        BMCdict =  self.fillna_model_counts(self.datafile.groupby(param)[[boolvar]].value_counts().to_dict())
        BMCdict = dict(sorted(BMCdict.items(), key=lambda x: x[0]))
        x_array = []
        noise_value_counts = []
        noisesine_value_counts = []
        inc_value_counts = []
        for key in BMCdict:
            if key[1]==0:
                x_array.append(key[0])
                inc_value_counts.append(BMCdict[key])
            elif key[1]==1:
                noisesine_value_counts.append(BMCdict[key])
            elif key[1]==-1:
                noise_value_counts.append(BMCdict[key])

        if xarray[0] == -99:
            pass
        else:
            x_array = xarray
        noise_value_counts = np.array(noise_value_counts)
        noisesine_value_counts = np.array(noisesine_value_counts)
        inc_value_counts = np.array(inc_value_counts)  

        sumofall = noise_value_counts + noisesine_value_counts + inc_value_counts
        plt.figure(figsize=[12, 6])
        plt.barh(np.round(x_array,decimals=5).astype('str'), noise_value_counts*100/sumofall, height=0.5, color='mediumblue' )
        plt.barh(np.round(x_array,decimals=5).astype('str'), noisesine_value_counts*100/sumofall, left=noise_value_counts*100/sumofall, height=0.5,  color='firebrick')
        plt.barh(np.round(x_array,decimals=5).astype('str'), inc_value_counts*100/sumofall, left= np.array(noise_value_counts*100/sumofall) + np.array(noisesine_value_counts*100/sumofall), height=0.5,  color='gray')
        plt.tight_layout()
        plt.legend(['$'+ model+'$', '$'+ model+' + sine$', '$inconclusive$'], fontsize=13, loc = 'upper right')
        plt.ylabel('$'+ param+'$', fontsize=15)
        plt.xlabel('$Model\;preference\;[\%]$', fontsize=15)
        plt.gca().invert_yaxis()
        plt.tick_params(labelsize=14)
        plt.tight_layout()
        #plt.savefig('plots/DRWsine_highalpha_2_4.png', )
        plt.show()
        return pd.DataFrame(np.array([noise_value_counts,noisesine_value_counts, inc_value_counts]).T,
                   index=np.round(x_array, decimals=2),
                   columns=[r"$red\;noise\;only$", r"$red\;noise + period$", r"$inconclusive$"])
        
    def plot_bayesDistri(self, param , model ='DRW', plottype = 'violin', 
                        xlabel = r"$Length\;of\;Observation\;[years]$",
                        xlim = (), x_tick_fac = 1):
        if model =='DRW':
            bayesvar_arr = ['DRW_DRWsine_bayes']
        if model =='CARMA21':
            bayesvar_arr = ['CARMA21_CARMA21sine_bayes']
        if model =='OBPL':
            bayesvar_arr =['OBPL10_OBPLsine10_bayes' ] 
        if model == 'all':
            bayesvar_arr =['DRW_DRWsine_bayes', 'CARMA21_CARMA21sine_bayes', 'OBPL10_OBPLsine10_bayes' ] 
            
        if model!='all':
            plt.figure(figsize=[10, 6])
            
            colors = ['purple', 'red', 'green']
            for j, bayesvar in enumerate(bayesvar_arr):
                box_plot_stack = []
                # x_array = list(set(self.datafile[param].to_list())) ; x_array.sort()
                # x_array = np.round(x_array, decimals=5)
                x_array = list(set(self.datafile[param].to_numpy()))
                x_array.sort()
                x_array.reverse()

                x_array = x_array[0::1]
#                x_array.append(-4)
                for i in x_array:
                    box_plot_stack.append(self.datafile.loc[np.round(self.datafile[param], decimals=5)== i][bayesvar].to_numpy())
                    
            
                # for i in x_array:
                #     y_array = self.datafile.loc[np.round(self.datafile[param], decimals=5) == i][bayesvar]
                #     plt.plot(np.ones(len(y_array))*i, y_array, 'ko', alpha = 0.1  )
                
                diff = np.diff(x_array,n =1)[2]
                if plottype =='violin':
                    plt.violinplot(box_plot_stack, positions=x_array, widths=diff/3)
                elif plottype =='box':
                    bplot = plt.boxplot(box_plot_stack, positions=x_array, widths=diff/3, patch_artist=True)

                    for patch in bplot['boxes']:
                        patch.set_color(colors[int(j)])
                        patch.set_alpha(0.5)
                        
            for i in [0.,  2, -2]:
                plt.axhline(10**i, label = 'log K = '+str(i), color = 'k', ls = '--')
                    
                    
            plt.xticks(x_array, labels=x_array)
            plt.yscale('log')
            #plt.gca().set_xticklabels()
            plt.legend(loc = 'lower right')
            plt.xlim(min(x_array) - diff/3, max(x_array) + diff/3)
            plt.xlabel(''+ param +'', fontsize = 13)
            plt.ylabel(r'$B_{10}\left(\frac{}{}\right)$'.format("{"+model+"}", "{"+model+" + sine }" ), fontsize =13)
            plt.tight_layout()
            plt.show()
            
            
        else:
            fig, ax = plt.subplots(3, 1, figsize=[16, 9], sharex=True, sharey=True)
            colors = ['purple', 'red', 'green']
            models = ['DRW', 'CARMA', 'OBPL']
            for j, bayesvar in enumerate(bayesvar_arr):
                box_plot_stack = []
                # x_array = list(set(self.datafile[param].to_list())) ; x_array.sort()
                # x_array = np.round(x_array, decimals=5)
                x_array = list(set(self.datafile[param].to_numpy()))
                x_array.sort()
                x_array.reverse()

                # x_array = x_array[0::2]
                # x_array.append(-4)
                x_array = np.round(x_array, decimals=5)
                for i in x_array:
                    box_plot_stack.append(self.datafile.loc[np.round(self.datafile[param], decimals=5)== i][bayesvar].to_numpy())
                    
                for i in x_array:
                    y_array = self.datafile.loc[np.round(self.datafile[param], decimals=5) == i][bayesvar]
                    ax[j].plot(np.ones(len(y_array))*i, y_array, 'ko', alpha = 0.1  )
                
                diff = np.diff(x_array,n =1)[2]
                if plottype =='violin':
                    ax[j].violinplot(box_plot_stack, positions=x_array, widths=diff/3)
                elif plottype =='box':
                    bplot = ax[j].boxplot(box_plot_stack, positions=x_array, widths=diff/3, patch_artist=True)

                    for patch in bplot['boxes']:
                        patch.set_color(colors[int(j)])
                        patch.set_alpha(0.5)
                        
                for i in [ 2, -2]:
                    ax[j].axhline(10**i, label = '$log_{10}\;B_{10} = $'+str(i), color = 'k', ls = '--',alpha=0.3)
                    
                ax[j].set_xticks(x_array, labels=np.round(x_array* x_tick_fac, decimals=2))
                ax[j].set_yscale('log')
                ax[j].tick_params(labelsize = 14)
                ax[j].legend(loc = 'lower left', fontsize = 13)
                if len(xlim)!=0:
                    ax[j].set_xlim(xlim[0], xlim[1])
                if j ==2:
                    ax[j].set_xlabel(xlabel, fontsize = 19)
                ax[j].grid(axis = 'x')
                ax[j].set_ylabel(r'$B_{}\left(\frac{}{}\right)$'.format("{"+str(10)+"}","{"+models[j]+"}", "{"+models[j]+" + sine }" ), fontsize =15)
                ax[j].invert_xaxis()
            plt.tight_layout()
            plt.show()
                
                    
                    
                    
                    
                    
                    
    def confusion_matrix(self,  no_period_dataset, period_dataset, model='DRW'):
        if model =='DRW':
            boolvar = 'DRWsine_bool'
        if model =='CARMA21':
            boolvar = 'CARMA21sine_bool'
        if model =='OBPL':
            boolvar = 'OBPLsine_bool'
            
        no_period_prediction = no_period_dataset[boolvar].value_counts().to_dict()
        period_prediction = period_dataset[boolvar].value_counts().to_dict()
        
        for i in [-1,0,1]:
            if i not in no_period_prediction.keys():
                no_period_prediction[i] = 0
            if i not in period_prediction.keys():
                period_prediction[i] = 0
        
        confusion_matrix = np.zeros((2, 3), dtype=int)

        confusion_matrix[0, 0] = no_period_prediction[0]
        confusion_matrix[0, 1] = no_period_prediction[-1]
        confusion_matrix[0, 2] = no_period_prediction[1]
        confusion_matrix[1, 0] = period_prediction[0]
        confusion_matrix[1, 1] = period_prediction[-1]
        confusion_matrix[1, 2] = period_prediction[1]
        
        plt.figure(figsize=(12, 7))
        sns.set(font_scale=1.1)
        sns.heatmap(confusion_matrix, annot=True, cmap='Blues',
                    xticklabels=['$inconclusive$', '$Noise\;only\;prefered$', '$periodic\;model\;prefered$'],
                    yticklabels=['$No\;periodicity\;simulated$', '$Periodicity\;simulated$'], square=True, annot_kws={'size': 12})

        plt.xlabel('Predicted', fontsize = 16, weight = 'bold')
        plt.ylabel('True', fontsize = 16, weight = 'bold')
        plt.tight_layout()
        plt.show()