# análiseF1
Análise de Dados da Fórmula 1

# Bibliotecas e Dados

!pip install kagglehub

#Bibliotecas

import pandas as pd
import os
from pandas import json_normalize
import numpy as np
import matplotlib.pyplot as plt
import requests
import kagglehub
import seaborn as sns

#Importando o DataSet via Kaggle
path = kagglehub.dataset_download("rohanrao/formula-1-world-championship-1950-2020")
print("Path to dataset files:", path)

directory_path = '/root/.cache/kagglehub/datasets/rohanrao/formula-1-world-championship-1950-2020/versions/23'

# Listando os arquivos no diretório
files = os.listdir(directory_path)
print("Arquivos disponíveis:", files)


# Conhecendo os Dados

#Identificando Pilotos Brasileiros
drivers_df = pd.read_csv(directory_path + '/drivers.csv')
brazilian_df = drivers_df[drivers_df['nationality'] == 'Brazilian']
brazilian_df

# Entendendo a disposição da base de dados de voltas

laps_df =pd.read_csv(directory_path + '/lap_times.csv')
laps_df

# Entendendo a disposição da base de dados de circuitos

circuits_df = pd.read_csv(directory_path + '/circuits.csv')
circuits_df

# Entendendo a disposição da base de dados de corridas

race_df = pd.read_csv(directory_path + '/races.csv')
race_df

# Identificando onde ocorreu cada corrida

# Mesclando os DataFrames
races_and_circuits_df = pd.merge(
    race_df,
    circuits_df[['circuitId', 'name', 'location', 'country']],
    on='circuitId',
    how='left'
)

# Removendo colunas indesejadas
races_and_circuits_df = races_and_circuits_df.drop(columns=[
    'date', 'time', 'url',
    'fp1_date', 'fp1_time',
    'fp2_date', 'fp2_time',
    'fp3_date', 'fp3_time',
    'quali_date', 'sprint_date',
    'sprint_time', 'quali_time'
])

races_and_circuits_df


# Identificando voltas dos pilotos


# Mesclando os DataFrames

laps_and_drivers_df = pd.merge(laps_df, drivers_df[['driverId', 'driverRef', 'surname', 'nationality']], on='driverId', how='left')
laps_and_drivers_df

# Data Frame sobre Voltas e Circuitos

full_races_df = pd.merge(races_and_circuits_df, laps_and_drivers_df, on='raceId', how='left')
full_races_df

# Inferências

num_brazilian_racers = len(brazilian_df)
print(f"{num_brazilian_racers} Pilotos Brasileiros já pilotaram na fórmula 1.")

# Ranqueando as nacionalidades que mais deram voltas

laps_by_nationality = full_races_df.groupby('nationality')['lap'].count().reset_index()
laps_by_nationality.rename(columns={'lap': 'total_laps'}, inplace=True)
print(laps_by_nationality.sort_values(by=['total_laps'], ascending=False))

laps_by_nationality = laps_by_nationality.sort_values(by=['total_laps'], ascending=False)

# Gráfico das Nacionalidades
plt.figure(figsize=(12, 6))
plt.bar(laps_by_nationality['nationality'], laps_by_nationality['total_laps'])
plt.xlabel('Nationality')
plt.ylabel('Total Laps')
plt.title('Total Laps by Nationality')
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.show()

# Comportamento do Brasil

brazilian_laps_df = full_races_df.loc[full_races_df['nationality'] == 'Brazilian']
min_lap_time = brazilian_laps_df['time'].min()

# Identificando a menor volta (Volta mais rápida)
min_lap_row = brazilian_laps_df.loc[brazilian_laps_df['time'] == min_lap_time]

race_id_min_lap = min_lap_row['raceId'].values[0]

# Nome do GP
grand_prix_name = races_and_circuits_df.loc[races_and_circuits_df['raceId'] == race_id_min_lap, 'name_y'].values[0]

# Nome do Piloto

drivers_name = min_lap_row['driverRef'].values[0]

print(f"A volta mais rápida foi feita em: {min_lap_time} segundos. No Grande Prêmio do {grand_prix_name}, pelo {drivers_name} ")

# Identificando o maior número de qualificações executadas por brasileiros

qualy_df = pd.read_csv(directory_path + '/qualifying.csv')
qualy_drivers_df = pd.merge(qualy_df, drivers_df[['driverId', 'driverRef', 'surname', 'nationality']], on='driverId', how='left')
br_qualy_drivers_df = qualy_drivers_df.loc[qualy_drivers_df['nationality'] == 'Brazilian']

qualy_counts = br_qualy_drivers_df.groupby('driverRef')['qualifyId'].count().reset_index()
qualy_counts.rename(columns={'qualifyId': 'num_qualifying_sessions'}, inplace=True)


qualy_counts = qualy_counts.sort_values(by='num_qualifying_sessions', ascending=False)
print(qualy_counts)
# Obs: Acredito que a base não esteja completa com esse dado, visto que o Ayrton Senna executou várias qualificações

br_qualy_drivers_df

# Identificando os brasileiros com mais "Pole Position"

pole_positions_df = br_qualy_drivers_df[br_qualy_drivers_df['position'] == 1]

pole_counts = pole_positions_df.groupby('driverRef')['qualifyId'].count().reset_index()
pole_counts.rename(columns={'qualifyId': 'num_pole_positions'}, inplace=True)

pole_counts = pole_counts.sort_values(by='num_pole_positions', ascending=False)
print(pole_counts)

# Identificando pontuação por equipe

race_constructor_standings_df = pd.read_csv(directory_path + '/constructor_standings.csv')
race_constructor_standings_df

# Identificando resultados
results_df = pd.read_csv(directory_path + '/results.csv')
results_df


# Mesclando com os pilotos para identificação de pontuação individual
results_with_nationality = pd.merge(results_df, drivers_df[['driverId', 'nationality']], on='driverId', how='left')

# Filtrando brasileiros
brazilian_results = results_with_nationality[results_with_nationality['nationality'] == 'Brazilian']

# Identificando pontuação dos pilotos brasileiros por equipe (e a soma dos pontos)
constructor_points = brazilian_results.groupby(['constructorId', 'driverId'])['points'].sum().reset_index()

# Achando o máximo de pontos por construtor
max_points_by_constructor = constructor_points.loc[constructor_points.groupby('constructorId')['points'].idxmax()]

print(max_points_by_constructor)

# Traduzindo os ID's para nomes de pilotos e equipes

constructor_df = pd.read_csv(directory_path + '/constructors.csv')
constructors_df = pd.merge(max_points_by_constructor, constructor_df[['constructorId', 'name']], on='constructorId', how='left')
max_points_with_names = pd.merge(max_points_by_constructor, drivers_df[['driverId', 'driverRef']], on='driverId', how='left')
max_points_with_names = pd.merge(max_points_with_names, constructors_df[['constructorId', 'name']], on='constructorId', how='left')

print(max_points_with_names[['name','driverRef', 'points']])

# Total de Pontos por Piloto Brasileiro

brazilian_results = results_with_nationality[results_with_nationality['nationality'] == 'Brazilian']

brazilian_driver_points = brazilian_results.groupby('driverId')['points'].sum().reset_index()

brazilian_driver_points = pd.merge(brazilian_driver_points, drivers_df[['driverId', 'driverRef']], on='driverId', how='left').sort_values(by='points', ascending=False)

print(brazilian_driver_points[['driverRef', 'points']])

# Dúvidas sobre a pontuação

races_per_driver = brazilian_results.groupby('driverId')['raceId'].nunique().reset_index()
races_per_driver.rename(columns={'raceId': 'num_races'}, inplace=True)

driver_points_races = pd.merge(brazilian_driver_points, races_per_driver, on='driverId', how='left')

driver_points_races['points_per_race'] = driver_points_races['points'] / driver_points_races['num_races']

print(driver_points_races[['driverRef', 'points_per_race']].sort_values(by=['points_per_race'], ascending=False))

# Massa tem realmente mais pontos por corrida do que o Senna?

plt.figure(figsize=(10, 6))
sns.barplot(x='driverRef', y='points_per_race', data=driver_points_races.sort_values(by=['points_per_race'], ascending=False))
plt.title('Points per Race for Brazilian F1 Drivers')
plt.xlabel('Driver')
plt.ylabel('Points per Race')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Após uma análise dos dados, identifiquei que a pontuação era diferente,
# então igualei tudo para a base das pontuações atuais

points_per_position = {
    'position': ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'),
    'points': (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)
}

points_df = pd.DataFrame(points_per_position)
points_df

position_points_map = dict(zip(points_df['position'], points_df['points']))
print(position_points_map)

# Pontos Atuais para todas as corridas

results_df['points'] = results_df['position'].map(position_points_map).fillna(results_df['points'])
results_df

# Mesclando com os pilotos para identificação de pontuação individual
results_with_nationality = pd.merge(results_df, drivers_df[['driverId', 'nationality']], on='driverId', how='left')

# Filtrando brasileiros
brazilian_results = results_with_nationality[results_with_nationality['nationality'] == 'Brazilian']

# Identificando pontuação dos pilotos brasileiros por equipe (e a soma dos pontos)
constructor_points = brazilian_results.groupby(['constructorId', 'driverId'])['points'].sum().reset_index()

# Achando o máximo de pontos por construtor
max_points_by_constructor = constructor_points.loc[constructor_points.groupby('constructorId')['points'].idxmax()]

print(max_points_by_constructor)

# Traduzindo os ID's para nomes de pilotos e equipes

constructor_df = pd.read_csv(directory_path + '/constructors.csv')
constructors_df = pd.merge(max_points_by_constructor, constructor_df[['constructorId', 'name']], on='constructorId', how='left')
max_points_with_names = pd.merge(max_points_by_constructor, drivers_df[['driverId', 'driverRef']], on='driverId', how='left')
max_points_with_names = pd.merge(max_points_with_names, constructors_df[['constructorId', 'name']], on='constructorId', how='left')

print(max_points_with_names[['name','driverRef', 'points']])

# Total de Pontos por Piloto Brasileiro

brazilian_results = results_with_nationality[results_with_nationality['nationality'] == 'Brazilian']

brazilian_driver_points = brazilian_results.groupby('driverId')['points'].sum().reset_index()

brazilian_driver_points = pd.merge(brazilian_driver_points, drivers_df[['driverId', 'driverRef']], on='driverId', how='left').sort_values(by='points', ascending=False)

print(brazilian_driver_points[['driverRef', 'points']])

races_per_driver = brazilian_results.groupby('driverId')['raceId'].nunique().reset_index()
races_per_driver.rename(columns={'raceId': 'num_races'}, inplace=True)

driver_points_races = pd.merge(brazilian_driver_points, races_per_driver, on='driverId', how='left')

driver_points_races['points_per_race'] = driver_points_races['points'] / driver_points_races['num_races']

print(driver_points_races[['driverRef', 'points_per_race']].sort_values(by=['points_per_race'], ascending=False))

plt.figure(figsize=(10, 6))
sns.barplot(x='driverRef', y='points_per_race', data=driver_points_races.sort_values(by=['points_per_race'], ascending=False))
plt.title('Points per Race for Brazilian F1 Drivers')
plt.xlabel('Driver')
plt.ylabel('Points per Race')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

driver_points_races = pd.merge(brazilian_driver_points, races_per_driver, on='driverId', how='left')
print(driver_points_races[['driverRef', 'num_races']].sort_values(by=['num_races'], ascending=False))

Xplt.figure(figsize=(10, 6))
sns.barplot(x='driverRef', y='num_races', data=driver_points_races.sort_values(by=['num_races'], ascending=False))
plt.title('Race per Drivers for Brazilian F1 Drivers')
plt.xlabel('Driver')
plt.ylabel('Races')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
