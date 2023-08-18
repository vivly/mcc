import utils
import torch
import pandas as pd
import numpy as np
import geopy.distance
import os
from pandarallel import pandarallel

jhu_csse_confirm_path = './data/time_series_covid19_confirmed_US.csv'
jhu_csse_death_path = './data/time_series_covid19_deaths_US.csv'
cagdp1_path = './data/CAGDP1_gdp_us.csv'
oxcgrt_path = './data/OxCGRT_USA_latest.csv'
location_csv_path = './data/locations.csv'
geo_info_idx = ['Admin2', 'Province_State', 'Lat', 'Long_', 'Combined_Key']
jhu_csse_key = ['blank_idx', 'geo_info', 'confirm', 'death', 'population_info']
node_rep_key = ['geo', 'confirm', 'death', 'population', 'gdp', 'policy']
state_idx_str_dict = {'Alabama': '01', 'Alaska': '02', 'Arizona': '04', 'Arkansas': '05', 'California': '06',
                      'Colorado': '08', 'Connecticut': '09', 'Delaware': '10', 'District of Columbia': '11',
                      'Florida': '12', 'Georgia': '13', 'Hawaii': '15', 'Idaho': '16', 'Illinois': '17',
                      'Indiana': '18', 'Iowa': '19', 'Kansas': '20', 'Kentucky': '21', 'Louisiana': '22',
                      'Maine': '23', 'Maryland': '24', 'Massachusetts': '25', 'Michigan': '26', 'Minnesota': '27',
                      'Mississippi': '28', 'Missouri': '29', 'Montana': '30', 'Nebraska': '31', 'Nevada': '32',
                      'New Hampshire': '33', 'New Jersey': '34', 'New Mexico': '35', 'New York': '36',
                      'North Carolina': '37', 'North Dakota': '38', 'Ohio': '39', 'Oklahoma': '40',
                      'Oregon': '41', 'Pennsylvania': '42', 'Rhode Island': '44', 'South Carolina': '45',
                      'Tennessee': '47', 'Texas': '48', 'Utah': '49', 'Vermont': '50', 'Virginia': '51',
                      'Washington DC': '53', 'West Virginia': '54', 'Wisconsin': '55', 'Wyoming': '56'}
policy_indicator_list = ['C4M_Restrictions on gatherings']


def load_JHU_CSSE_dataset():
    jhu_csse_confirm_df = pd.read_csv(jhu_csse_confirm_path, low_memory=False)
    jhu_csse_death_df = pd.read_csv(jhu_csse_death_path, low_memory=False)
    jhu_csse_death_df.set_index('FIPS', inplace=True)
    jhu_csse_confirm_df.set_index('FIPS', inplace=True)

    # Split the JHU dataset into four parts: confirm sequence, population, death sequence and location information.
    geo_info_df = jhu_csse_confirm_df[geo_info_idx]
    confirm_df = jhu_csse_confirm_df.iloc[:, 10:]
    death_df = jhu_csse_death_df.iloc[:, 11:]
    population_info_series = jhu_csse_death_df['Population']

    # Filter blank locations
    blank_idx = population_info_series[population_info_series.values == 0].index.dropna().values
    geo_info_df = geo_info_df.drop(index=blank_idx)
    confirm_df = confirm_df.drop(index=blank_idx)
    death_df = death_df.drop(index=blank_idx)
    population_info_series = population_info_series.drop(index=blank_idx)

    return_dict = {jhu_csse_key[0]: blank_idx, jhu_csse_key[1]: geo_info_df, jhu_csse_key[2]: confirm_df,
                   jhu_csse_key[3]: death_df, jhu_csse_key[4]: population_info_series}
    print('JHU dataset load complete!')
    return return_dict


def load_GDP_dataset():
    gdp_df = pd.read_csv(cagdp1_path, low_memory=False)
    gdp_df['GeoFips'] = pd.to_numeric(gdp_df['GeoFips'], downcast='float')
    gdp_df = gdp_df[gdp_df['LineCode'] == 1].set_index('GeoFips')
    print('GDP dataset load complete!')
    return gdp_df


def load_OxCGRT_policy_data(jhu_csse_data):
    print('loading OxCGRT policy dataset...')
    OxCGRT_df = pd.read_csv(oxcgrt_path, low_memory=False)
    OxCGRT_df = OxCGRT_df[OxCGRT_df['Jurisdiction'] == 'STATE_TOTAL']
    location_df = pd.read_csv(location_csv_path, low_memory=False)
    policy_df = jhu_csse_data['confirm'].copy(deep=True)
    max_date = policy_df.columns[-1]
    min_date = policy_df.columns[0]
    min_date2 = utils.reform_date_str(min_date)
    max_date2 = utils.reform_date_str(max_date)
    # Generate a list of date that included in the max and min date
    date_list2 = pd.date_range(start=min_date2, end=max_date2).strftime('%Y%m%d').tolist()
    date_list = pd.date_range(start=min_date, end=max_date).strftime('%-m/%-d/%y').tolist()

    # Query the policy ratings according to state index
    tmp_list1 = []
    for idx_str in state_idx_str_dict.values():
        state_name = location_df[location_df['location'] == idx_str]['location_name'].tolist()[0]
        state_index = state_idx_str_dict[state_name]
        state_index_int = int(state_index)
        df1 = policy_df[(policy_df.index >= state_index_int * 1000) & (policy_df.index < (state_index_int + 1) * 1000)]
        tmp_list2 = []
        for idx, date in enumerate(date_list2):
            series = OxCGRT_df[(OxCGRT_df['RegionName'] == state_name) & (OxCGRT_df['Date'] == int(date))]
            series = series.fillna(0.0)
            try:
                index = series['C4M_Restrictions on gatherings'].tolist()[0]
            except IndexError:
                index = 0.0
            tmp_series = df1[date_list[idx]].astype('object')
            for i in df1.index.tolist():
                tmp_series.at[i] = index
            tmp_list2.append(tmp_series)
        tmp_df1 = pd.concat(tmp_list2, axis=1, ignore_index=False)
        tmp_list1.append(tmp_df1)
    tmp_df2 = pd.concat(tmp_list1, axis=0, ignore_index=False)
    print('OxCGRT policy dataset load complete!')
    # Two layers of recurrences: first iterate the location, i.e. the states; second recurrence iterate the date
    return tmp_df2


def clean(jhu_dataset, gdp_dataset, policy_dataset):
    geo = jhu_dataset[jhu_csse_key[1]]
    confirm = jhu_dataset[jhu_csse_key[2]]
    death = jhu_dataset[jhu_csse_key[3]]
    population = jhu_dataset[jhu_csse_key[4]]

    # align all the dataframes to make sure the number of counties is equal.
    gdp, policy = gdp_dataset.align(policy_dataset, join='inner', axis=0)
    gdp, geo = gdp.align(geo, join='inner', axis=0)
    _, population = gdp.align(population, join='inner', axis=0)
    _, confirm = gdp.align(confirm, join='inner', axis=0)
    _, death = gdp.align(death, join='inner', axis=0)
    assert len(geo) == len(gdp) == len(confirm) == len(death) == len(population) == len(policy)
    # assertion: the length of dataframes ought to be the same

    return dict(zip(node_rep_key, [geo, confirm, death, population, gdp, policy]))


def calculate_geo_topology(node_rep, max_state_neighbour=5):
    print('Calculating geographical topology...')
    # Since it is computationally prohibitive to compute pairwise distances at county level
    # We only compute county-level distances on filtered state pairs

    # 1. Calculate and rank pairwise distances at state level
    geo_df = node_rep['geo']
    state_str_list = state_idx_str_dict.keys()
    state_location_df = pd.DataFrame(columns=['state_str', 'state_id', 'Lat', 'Long_'])
    for state_str in state_str_list:
        county_df = geo_df[geo_df['Province_State'] == state_str]
        state_lat = county_df['Lat'].mean()
        state_long = county_df['Long_'].mean()
        state_id = state_idx_str_dict[state_str]
        tmp_dict = {
            'state_str': state_str,
            'state_id': state_id,
            'Lat': state_lat,
            'Long_': state_long
        }
        state_location_df.loc[len(state_location_df)] = tmp_dict
    state_pair = pd.MultiIndex.from_product([state_location_df['state_str'].values,
                                             state_location_df['state_str'].values],
                                             names=['State_1', 'State_2']).to_frame(index=False)
    state_pair = state_pair[state_pair['State_1'] != state_pair['State_2']].reset_index(drop=True)
    state_pair = pd.merge(state_pair,
                          state_location_df.rename(columns={'state_str': 'State_1', 'state_id': 'State_1_id',
                                                            'Lat': 'Lat_1', 'Long_': 'Long_1'}), how='left',
                                                            on='State_1')
    state_pair = pd.merge(state_pair,
                          state_location_df.rename(columns={'state_str': 'State_2', 'state_id': 'State_2_id',
                                                            'Lat': 'Lat_2', 'Long_': 'Long_2'}), how='left',
                                                            on='State_2')
    state_pair['distance'] = state_pair.apply(lambda x: geopy.distance.distance((x['Lat_1'], x['Long_1']),
                                                                      (x['Lat_2'], x['Long_2'])).km, axis=1)
    state_pair['rank'] = state_pair.groupby('State_1')['distance'].rank()
    print('State topology calculation complete!')

    # 2. Calculate approximate pairwise distances at county level
    filtered_state_pair = state_pair[state_pair['rank'] <= max_state_neighbour][['State_1', 'State_2', 'State_1_id',
                                                                                 'State_2_id']]
    self_state_pair = pd.concat([state_location_df[['state_str']].rename(columns={'state_str': 'State_1'}),
                                 state_location_df[['state_str']].rename(columns={'state_str': 'State_2'})], axis=1)
    filtered_state_pair = pd.concat([filtered_state_pair, self_state_pair], axis=0, ignore_index=True)
    # Calculate pairwise distances among counties within filtered state pairs
    geo_df['County'] = geo_df.apply(lambda x: '{} ~ {}'.format(x['Province_State'], x['Admin2']), axis=1)
    county_geo = geo_df[['County', 'Lat', 'Long_']].groupby('County')[['Lat', 'Long_']].median().reset_index(drop=False)
    county_geo['State'] = county_geo['County'].map(lambda x: x.split(' ~ ')[0])
    county_pair = pd.merge(filtered_state_pair, county_geo.rename(
        columns={'County': 'County_1', 'State': 'State_1', 'Lat': 'Lat_1', 'Long_': 'Long_1'}),
                           how='left', on='State_1')
    county_pair = pd.merge(county_pair, county_geo.rename(
        columns={'County': 'County_2', 'State': 'State_2', 'Lat': 'Lat_2', 'Long_': 'Long_2'}),
                           how='left', on='State_2')
    county_pair = county_pair[county_pair['County_1'] != county_pair['County_2']].sort_values(
        ['County_1', 'County_2']).reset_index(drop=True)
    county_pair['distance'] = county_pair.parallel_apply(
        lambda x: geopy.distance.distance((x['Lat_1'], x['Long_1']), (x['Lat_2'], x['Long_2'])).km, axis=1)
    county_pair['rank'] = county_pair.groupby('County_1')['distance'].rank()

    return county_pair[['County_1', 'County_2', 'distance', 'rank']].\
           rename(columns={'County_1': 'Node', 'County_2': 'Node_1'})


def generate_us_graph(topo, node_rep, dump_fp, max_neighbor_num=50, dump_flag=True):
    geo_df = node_rep['geo']
    geo_df['County'] = geo_df.apply(lambda x: '{} ~ {}'.format(x['Province_State'], x['Admin2']), axis=1)
    county_name_list = list(geo_df['County'])
    county_idx_list = list(geo_df.index)
    node2idx = dict(zip(county_name_list, county_idx_list))
    node_levels = [2] * len(county_name_list)

    topo = topo[topo['Node'].isin(node2idx) & topo['Node_1'].isin(node2idx) & (topo['rank'] <= max_neighbor_num)]

    # graph defined by geographical information
    geo_index_src = list(topo['Node_1'].map(node2idx).values)
    geo_index_tgt = list(topo['Node'].map(node2idx).values)
    geo_weight = list(topo['distance'].map(lambda x: 1.0/np.sqrt(x)).values)

    # dump the graph into cpt file
    graph_info = {
        'edge_index': torch.LongTensor([geo_index_src, geo_index_tgt]),
        'edge_weight': torch.FloatTensor(geo_weight),
        'edge_type': torch.LongTensor(np.array([0]*len(geo_weight))),
        'node_name': county_name_list,
        'node_type': torch.LongTensor(node_levels)
    }
    if dump_flag:
        print('\nGenerate us graph to {}'.format(dump_fp))
        torch.save(graph_info, dump_fp)

    return graph_info


def load_data():
    fp_graph = './data/us_graph.cpt'
    pandarallel.initialize(progress_bar=False)
    # load data from JHU CSSE dataset
    JHU_CSSE_data = load_JHU_CSSE_dataset()
    # load data from CAGDP1 GDP dataset
    gdp_data = load_GDP_dataset()
    # load data from policy dataset
    OxCGRT_data = load_OxCGRT_policy_data(JHU_CSSE_data)
    # clean and align different datasets
    node_rep_dict = clean(JHU_CSSE_data, gdp_data, OxCGRT_data)

    # calculate the geology topo.
    geo_county = calculate_geo_topology(node_rep=node_rep_dict, max_state_neighbour=5)
    # generate the graph for all counties
    graph_info = generate_us_graph(topo=geo_county, node_rep=node_rep_dict, dump_fp=fp_graph)

    return node_rep_dict, graph_info

