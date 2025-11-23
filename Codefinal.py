import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.markers as mk
import matplotlib.ticker as mtick
from sympy import symbols, diff
from numpy.random import normal
import sklearn
import csv
import random
import os
from concurrent.futures import ProcessPoolExecutor 
from scipy.optimize import fmin
import seaborn as sns




##################################################################
# Question 1: Estimation de la courbe de rendement NS et Svenson #
##################################################################
                                



# Lecture des données à partir de fichiers CSV

fichier1 = pd.read_csv('data/data.csv')
fichier1['Time'].head()
fichier1.shape
# Définition de la plage de dates
date_debut = '2019-03-07'
date_fin = '2022-03-08'

# Convertion des dates de début et de fin en datetime
date_debut = pd.to_datetime(date_debut)
date_fin = pd.to_datetime(date_fin)


# Filtration les lignes de fichier qui correspondent à la plage de dates spécifiée
filtre1 = fichier1[(pd.to_datetime(fichier1['Time']) >= date_debut) & (pd.to_datetime(fichier1['Time']) <= date_fin)]

filtre1.shape

donnees = filtre1.iloc[:,1:]

n = len(donnees)
donnees.shape
donnees.head()

#donnees = donnees.dropna()

time = [1,2,3,4,5,6,7,8,9,10,20,30]

last = donnees.iloc[-1,:]/100
last
dd = {'Maturity' : time, 'Yield' : last}
dd = pd.DataFrame(dd)
df = dd.copy()
df.style.format({'Maturity': '{:,.0f}'.format,'Yield': '{:,.2%}'})
df.head()
dd.head()




# Tracer la courbe des taux du marché

#Image 1
sf = df.copy()
sf = df.dropna()
sf1 = sf.copy()
sf1['Y'] = round(sf['Yield']*100,4)
sf = sf.style.format({'Maturity': '{:,.2f}'.format,'Yield': '{:,.4%}'})

fontsize=15
fig = plt.figure(figsize=(13,7))
plt.title("Courbe des taux estimée de notre échantillon",fontsize=fontsize)
ax = plt.axes()
ax.set_facecolor("black")
fig.patch.set_facecolor('white')
X = sf1["Maturity"]
Y = sf1["Y"]
plt.scatter(X, Y, marker="o", c="blue")
plt.xlabel('Maturité',fontsize=fontsize)
plt.ylabel('Interest',fontsize=fontsize)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.xaxis.set_ticks(np.arange(0, 30, 5))
ax.yaxis.set_ticks(np.arange(0, 4, 0.5))
ax.legend(loc="lower right", title="Yield")
plt.grid()
plt.show()


#Image 2
plt.figure(figsize=(10, 6))
plt.plot(df['Maturity'], df['Yield'],  label='Données observées')
plt.plot(extended_maturities, predicted_yields, 'Yield', label='Courbe des taux estimée')
plt.xlabel('Maturité (en années)')
plt.ylabel('Rendement')
plt.title('Courbe des taux estimée avec le modèle de Nelson-Siegel Augmenté')
plt.legend()
plt.grid(True)
plt.show()



#Estimation des paramètres du modèle de Nelson Siegel

### Exemple illustratif sur une journée 


#Initialisation

β0 = 0.01
β1 = 0.01
β2 = 0.01
λ = 1.00



df['NS'] =(β0)+(β1*((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))+(β2*((((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))-(np.exp(-df['Maturity']/λ))))
df.style.format({'Maturity': '{:,.0f}'.format,'Yield': '{:,.2%}','NS': '{:,.2%}'})
df.head()


#Visualisation des taux calculé à partir de la fonction de Nelson Siegel sans ajustement

df1 = df.copy()
df['Y'] = round(df['Yield']*100,4)
df['NS'] =(β0)+(β1*((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))+(β2*((((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))-(np.exp(-df['Maturity']/λ))))
df['N'] = round(df['NS']*100,4)
df2 = df.copy()
df2 = df2.style.format({'Maturity': '{:,.2f}'.format,'Y': '{:,.2%}', 'N': '{:,.2%}'})
fontsize=15
fig = plt.figure(figsize=(13,7))
plt.title("Modèle Nelson-Siegel non ajusté Vs Donnée du marché ",fontsize=fontsize)
ax = plt.axes()
ax.set_facecolor("black")
fig.patch.set_facecolor('white')
X = df["Maturity"]
Y = df["Y"]
x = df["Maturity"]
y = df["N"]
ax.plot(x, y, color="orange", label="NS")
plt.scatter(x, y, marker="o", c="orange")
plt.scatter(X, Y, marker="o", c="blue")
plt.xlabel('Maturité',fontsize=fontsize)
plt.ylabel('Interest',fontsize=fontsize)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.xaxis.set_ticks(np.arange(0, 30, 5))
ax.yaxis.set_ticks(np.arange(0, 4, 0.5))
ax.legend(loc="lower right", title="Yield")
plt.grid()
plt.show()



#Ajout de la colonne des erreurs (MSE)

df['Residual'] =  (df['Yield'] - df['NS'])**2
df22 = df[['Maturity','Yield','NS','Residual']]  
df22.style.format({'Maturity': '{:,.0f}'.format,'Yield': '{:,.2%}','NS': '{:,.2%}','Residual': '{:,.9f}'})
df22.head()
df.head()

np.sum(df['Residual'])

#Lancement de l'estimation
def Error(params):
    df = dd.copy()
    df['NS'] =(params[0])+(params[1]*((1-np.exp(-df['Maturity']/params[3]))/(df['Maturity']/params[3])))+(params[2]*((((1-np.exp(-df['Maturity']/params[3]))/(df['Maturity']/params[3])))-(np.exp(-df['Maturity']/params[3]))))
    df['Residual'] =  (df['Yield'] - df['NS'])**2
    error = np.sum(df['Residual'])
    print("[β0, β1, β2, λ]=",params,", SUM:", error)
    return(error)
    
params = fmin(Error, [0.01, 0.00, -0.01, 1.0])

#Les paramètres estimés sont:

β0 = params[0]
β1 = params[1]
β2 = params[2]
λ = params[3]
print("[β0, β1, β2, λ]=", [params[0].round(2), params[1].round(2), params[2].round(2), params[3].round(2)])



#Visualisation de la courbe des taux calculés à partir des paramètres estimés

df = df1.copy()
df['NS'] =(β0)+(β1*((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))+(β2*((((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))-(np.exp(-df['Maturity']/λ))))
sf4 = df.copy()
sf5 = sf4.copy()
sf5['Y'] = round(sf4['Yield']*100,4)
sf5['N'] = round(sf4['NS']*100,4)
sf4 = sf4.style.format({'Maturity': '{:,.2f}'.format,'Yield': '{:,.2%}', 'NS': '{:,.2%}'})
M0 = 0.00
M1 = 3.50

fontsize=15
fig = plt.figure(figsize=(13,7))
plt.title("Modèle de Nelson-Siegel estimé",fontsize=fontsize)
ax = plt.axes()
ax.set_facecolor("black")
fig.patch.set_facecolor('white')
X = sf5["Maturity"]
Y = sf5["Y"]
x = sf5["Maturity"]
y = sf5["N"]
ax.plot(x, y, color="orange", label="NS")
plt.scatter(x, y, marker="o", c="orange")
plt.scatter(X, Y, marker="o", c="blue")
plt.xlabel('Maturité',fontsize=fontsize)
plt.ylabel('Interest',fontsize=fontsize)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.xaxis.set_ticks(np.arange(0, 30, 5))
ax.yaxis.set_ticks(np.arange(0, 4, 0.4))
ax.legend(loc="lower right", title="Yield")
plt.grid()
plt.show()









###Roule sur plage de date de 3 ans


resultat1 = pd.DataFrame()
resultat1.shape

Params1 = pd.DataFrame()
Params1.shape

n

for i in range(n):
    first = donnees.iloc[i-1,:]/100

    dd = {'Maturity' : time, 'Yield' : first}
    dd = pd.DataFrame(dd)
    df = dd.copy()
    df.style.format({'Maturity': '{:,.0f}'.format,'Yield': '{:,.2%}'})
    
    #Initialisation

    β0 = 0.01
    β1 = 0.01
    β2 = 0.01
    λ = 1.00


    df['NS'] =(β0)+(β1*((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))+(β2*((((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))-(np.exp(-df['Maturity']/λ))))
    df.style.format({'Maturity': '{:,.0f}'.format,'Yield': '{:,.2%}','NS': '{:,.2%}'})
    df.head()
    df['Y'] = round(df['Yield']*100,4)
    df['N'] = round(df['NS']*100,4)
    df['Residual'] =  (df['Yield'] - df['NS'])**2


    df22 = df[['Maturity','Yield','NS','Residual']]  
    df22.style.format({'Maturity': '{:,.0f}'.format,'Yield': '{:,.2%}','NS': '{:,.2%}','Residual': '{:,.9f}'})
    df22.head()
    df.head()

    np.sum(df['Residual'])


    #Lancement de l'estimation
    def Error(params):
        df = dd.copy()
        df['NS'] =(params[0])+(params[1]*((1-np.exp(-df['Maturity']/params[3]))/(df['Maturity']/params[3])))+(params[2]*((((1-np.exp(-df['Maturity']/params[3]))/(df['Maturity']/params[3])))-(np.exp(-df['Maturity']/params[3]))))
        df['Residual'] =  (df['Yield'] - df['NS'])**2
        error = np.sum(df['Residual'])
        print("[β0, β1, β2, λ]=",params,", SUM:", error)
        return(error)
        
    params = fmin(Error, [0.01, 0.00, -0.01, 1.0])

    #Les paramètres estimés sont:

    β0 = params[0]
    β1 = params[1]
    β2 = params[2]
    λ = params[3]

    #Calcul des taux avec les paramètres estimés
    df['NS'] =(β0)+(β1*((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))+(β2*((((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))-(np.exp(-df['Maturity']/λ))))

    # Créer un dictionnaire contenant les valeurs de la nouvelle ligne des paramètres
    nouvelle_ligne_param = {'β0': β0, 'β1': β1, 'β2': β2, 'λ': λ}
    
    # Ajout de la nouvelle ligne au DataFrame Params en utilisant la méthode append()
    Params1 = Params1.append(nouvelle_ligne_param, ignore_index=True)
    
    # Ajout de la nouvelle ligne au DataFrame Params en utilisant la méthode append()
    resultat1 = resultat1.append(df['NS'], ignore_index=True)

Params1.shape
resultat1.shape




#Sauvegarde du fichier des paramètres et des résultats de Nelson Siegel estimé

Resultat1 = pd.concat([fichier1['Time'],resultat1, Params1], axis = 1)
Resultat1.head()
Resultat1.to_excel('output/data_treated_ns.xlsx', index=False)







#Estimation des paramètres du modèle de Nvenson

### Exemple illustratif sur une journée 


#Initialisation

β0 = 0.01
β1 = 0.01
β2 = 0.01
β3 = 0.01

λ = 1.00
k = 1.00


df['SV'] = β0 + (β1 + β2) * ((1 - np.exp(-df['Maturity'] / λ)) / (df['Maturity'] / λ)) + β2 * np.exp(-df['Maturity'] / λ) + β3 * ((1 - np.exp(-df['Maturity'] / k)) / (df['Maturity'] / k)) * np.exp(-df['Maturity'] / k)

df.style.format({'Maturity': '{:,.0f}'.format,'Yield': '{:,.2%}','SV': '{:,.2%}'})
df.head()


#Visualisation des taux calculé à partir de la fonction de Nelson Siegel sans ajustement

df1 = df.copy()
df['Y'] = round(df['Yield']*100,4)
df['SV'] =β0 + (β1 + β2) * ((1 - np.exp(-df['Maturity'] / λ)) / (df['Maturity'] / λ)) + β2 * np.exp(-df['Maturity'] / λ) + β3 * ((1 - np.exp(-df['Maturity'] / k)) / (df['Maturity'] / k)) * np.exp(-df['Maturity'] / k)
df['S'] = round(df['SV']*100,4)
df2 = df.copy()
df2 = df2.style.format({'Maturity': '{:,.2f}'.format,'Y': '{:,.2%}', 'S': '{:,.2%}'})
fontsize=15
fig = plt.figure(figsize=(13,7))
plt.title("Modèle Nvenson non ajusté Vs Donnée du marché ",fontsize=fontsize)
ax = plt.axes()
ax.set_facecolor("black")
fig.patch.set_facecolor('white')
X = df["Maturity"]
Y = df["Y"]
x = df["Maturity"]
y = df["S"]
ax.plot(x, y, color="red", label="SV")
plt.scatter(x, y, marker="o", c="red")
plt.scatter(X, Y, marker="o", c="blue")
plt.xlabel('Maturité',fontsize=fontsize)
plt.ylabel('Interest',fontsize=fontsize)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.xaxis.set_ticks(np.arange(0, 30, 5))
ax.yaxis.set_ticks(np.arange(0, 4, 0.5))
ax.legend(loc="lower right", title="Yield")
plt.grid()
plt.show()



#Ajout de la colonne des erreurs (MSE)

df['Residual'] =  (df['Yield'] - df['SV'])**2
df22 = df[['Maturity','Yield','SV','Residual']]  
df22.style.format({'Maturity': '{:,.0f}'.format,'Yield': '{:,.2%}','SV': '{:,.2%}','Residual': '{:,.9f}'})
df22.head()
df.head()

np.sum(df['Residual'])

#Lancement de l'estimation
def Error(params):
    df = dd.copy()
    df['SV'] =params[0] + (params[1] + params[2]) * ((1 - np.exp(-df['Maturity'] / params[4])) / (df['Maturity'] / params[4])) + params[2] * np.exp(-df['Maturity'] / params[4]) + params[3] * ((1 - np.exp(-df['Maturity'] / params[5])) / (df['Maturity'] / params[5])) * np.exp(-df['Maturity'] / params[5])
    df['Residual'] =  (df['Yield'] - df['SV'])**2
    error = np.sum(df['Residual'])
    print("[β0, β1, β2, β3, λ, k]=",params,", SUM:", error)
    return(error)
    
params = fmin(Error, [0.01, 0.01, -0.01, 0.01, 1.0, 1.0])

#Les paramètres estimés sont:

β0 = params[0]
β1 = params[1]
β2 = params[2]
β3 = params[3]
λ = params[4]
k = params[5]

print("[β0, β1, β2, β3, λ, k]=", [params[0].round(2), params[1].round(2), params[2].round(2), params[3].round(2), params[4].round(2), params[5].round(2)])



#Visualisation de la courbe des taux calculés à partir des paramètres estimés

df = df1.copy()
df['SV'] = β0 + (β1 + β2) * ((1 - np.exp(-df['Maturity'] / λ)) / (df['Maturity'] / λ)) + β2 * np.exp(-df['Maturity'] / λ) + β3 * ((1 - np.exp(-df['Maturity'] / k)) / (df['Maturity'] / k)) * np.exp(-df['Maturity'] / k)
sf4 = df.copy()
sf5 = sf4.copy()
sf5['Y'] = round(sf4['Yield']*100,4)
sf5['S'] = round(sf4['SV']*100,4)
sf4 = sf4.style.format({'Maturity': '{:,.2f}'.format,'Yield': '{:,.2%}', 'SV': '{:,.2%}'})
M0 = 0.00
M1 = 3.50

fontsize=15
fig = plt.figure(figsize=(13,7))
plt.title("Svenson Model - Fitted Yield Curve",fontsize=fontsize)
ax = plt.axes()
ax.set_facecolor("black")
fig.patch.set_facecolor('white')
X = sf5["Maturity"]
Y = sf5["Y"]
x = sf5["Maturity"]
y = sf5["S"]
ax.plot(x, y, color="red", label="SV")
plt.scatter(x, y, marker="o", c="red")
plt.scatter(X, Y, marker="o", c="blue")
plt.xlabel('Maturité',fontsize=fontsize)
plt.ylabel('Interest',fontsize=fontsize)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.xaxis.set_ticks(np.arange(0, 30, 5))
ax.yaxis.set_ticks(np.arange(0, 4, 0.5))
ax.legend(loc="lower right", title="Yield")
plt.grid()
plt.show()









###Roule sur plage de date


resultat1 = pd.DataFrame()
resultat1.shape

Params1 = pd.DataFrame()
Params1.shape
Resultat1 = pd.DataFrame()
n

for i in range(n):
    first = donnees.iloc[i-1,:]/100

    dd = {'Maturity' : time, 'Yield' : first}
    dd = pd.DataFrame(dd)
    df = dd.copy()
    df.style.format({'Maturity': '{:,.0f}'.format,'Yield': '{:,.2%}'})
    
    #Initialisation

    β0 = 0.01
    β1 = 0.01
    β2 = 0.01
    β3 = 0.01

    λ = 1.00
    k = 1.00


    df['SV'] = β0 + (β1 + β2) * ((1 - np.exp(-df['Maturity'] / λ)) / (df['Maturity'] / λ)) - β2 * np.exp(-df['Maturity'] / λ) + β3 * ((1 - np.exp(-df['Maturity'] / k)) / (df['Maturity'] / k)) * np.exp(-df['Maturity'] / k)
    df.style.format({'Maturity': '{:,.0f}'.format,'Yield': '{:,.2%}','SV': '{:,.2%}'})
    df.head()
    df['Y'] = round(df['Yield']*100,4)
    df['S'] = round(df['SV']*100,4)
    df['Residual'] =  (df['Yield'] - df['SV'])**2


    df22 = df[['Maturity','Yield','SV','Residual']]  
    df22.style.format({'Maturity': '{:,.0f}'.format,'Yield': '{:,.2%}','SV': '{:,.2%}','Residual': '{:,.9f}'})
    df22.head()
    df.head()

    np.sum(df['Residual'])


    #Lancement de l'estimation
    def Error(params):
        df = dd.copy()
        df['SV'] =params[0] + (params[1] + params[2]) * ((1 - np.exp(-df['Maturity'] / params[4])) / (df['Maturity'] / params[4])) - params[2] * np.exp(-df['Maturity'] / params[4]) + params[3] * ((1 - np.exp(-df['Maturity'] / params[5])) / (df['Maturity'] / params[5])) * np.exp(-df['Maturity'] / params[5])
        df['Residual'] =  (df['Yield'] - df['SV'])**2
        error = np.sum(df['Residual'])
        print("[β0, β1, β2, β3, λ, k]=",params,", SUM:", error)
        return(error)
        
    params = fmin(Error, [0.01, 0.01, -0.01, 0.01, 1.0, 1.0])

    #Les paramètres estimés sont:

    β0 = params[0]
    β1 = params[1]
    β2 = params[2]
    β3 = params[3]
    λ = params[4]
    k = params[5]



    #Calcul des taux avec les paramètres estimés
    df['SV'] = β0 + (β1 + β2) * ((1 - np.exp(-df['Maturity'] / λ)) / (df['Maturity'] / λ)) - β2 * np.exp(-df['Maturity'] / λ) + β3 * ((1 - np.exp(-df['Maturity'] / k)) / (df['Maturity'] / k)) * np.exp(-df['Maturity'] / k)

    # Créer un dictionnaire contenant les valeurs de la nouvelle ligne des paramètres
    nouvelle_ligne_param = {'β0': β0, 'β1': β1, 'β2': β2, 'β3': β3, 'λ': λ, 'k': k}
    
    # Ajout de la nouvelle ligne au DataFrame Params en utilisant la méthode append()
    Params1 = Params1.append(nouvelle_ligne_param, ignore_index=True)
    
    # Ajout de la nouvelle ligne au DataFrame Params en utilisant la méthode append()
    resultat1 = resultat1.append(df['SV'], ignore_index=True)

Params1.shape
resultat1.shape




#Sauvegarde du fichier des paramètres et des résultats de Svenson estimé

Resultat1 = pd.concat([fichier1['Time'],resultat1, Params1], axis = 1)
Resultat1.head()
Resultat1.to_excel('output/data_treated_svenson.xlsx', index=False)







#############################################################
# Question 2: Comparaison de nos résultats à ceux de la FED #
#############################################################


#Calculons d'abord la moyenne par colonne des taux estimés de notre échantillon

moyenne_echantillon = resultat1.mean()



# Lecture des données à partir de fichiers CSV de la FED

fichier2 = pd.read_csv('data/feds200628.csv')

fichier2.shape

# Définition de la plage de dates
date_debut = '2019-03-07'
date_fin = '2022-03-08'

# Convertion des dates de début et de fin en datetime
date_debut = pd.to_datetime(date_debut)
date_fin = pd.to_datetime(date_fin)


# Filtration les lignes de fichier qui correspondent à la plage de dates spécifiée
filtre2 = fichier2[(pd.to_datetime(fichier2['Time']) >= date_debut) & (pd.to_datetime(fichier2['Time']) <= date_fin)]

filtre2.shape

donnees = filtre2.iloc[:,68:98]



n = len(donnees)
donnees.shape
donnees.head()

#donnees = donnees.dropna()

time = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

last = donnees.iloc[-1,:]/100
last
dd = {'Maturity' : time, 'Yield' : last}
dd = pd.DataFrame(dd)
df = dd.copy()
df.style.format({'Maturity': '{:,.0f}'.format,'Yield': '{:,.2%}'})
df.head()
dd.head()




# Tracer la courbe des taux du marché

#Image 1
sf = df.copy()
sf = df.dropna()
sf1 = sf.copy()
sf1['Y'] = round(sf['Yield']*100,4)
sf = sf.style.format({'Maturity': '{:,.2f}'.format,'Yield': '{:,.4%}'})

fontsize=15
fig = plt.figure(figsize=(13,7))
plt.title("Courbe des taux estimée avec le modèle de Nelson-Siegel sur les données de la FED",fontsize=fontsize)
ax = plt.axes()
ax.set_facecolor("black")
fig.patch.set_facecolor('white')
X = sf1["Maturity"]
Y = sf1["Y"]
plt.scatter(X, Y, marker="o", c="blue")
plt.xlabel('Maturité',fontsize=fontsize)
plt.ylabel('Interest',fontsize=fontsize)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.xaxis.set_ticks(np.arange(0, 30, 5))
ax.yaxis.set_ticks(np.arange(0, 4, 0.5))
ax.legend(loc="lower right", title="Yield")
plt.grid()
plt.show()


#Image 2
plt.figure(figsize=(10, 6))
plt.plot(df['Maturity'], df['Yield'],  label='Données observées')
plt.plot(extended_maturities, predicted_yields, 'Yield', label='Courbe des taux estimée')
plt.xlabel('Maturité (en années)')
plt.ylabel('Rendement')
plt.title('Courbe des taux estimée avec le modèle de Nelson-Siegel Augmenté')
plt.legend()
plt.grid(True)
plt.show()



#Estimation des paramètres

#initial_params = [0.01, 0.01, 0.01, 0.01, 1.0, 1.0]

#Initialisation

β0 = 0.01
β1 = 0.01
β2 = 0.01
λ = 1.00

df['NS'] =(β0)+(β1*((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))+(β2*((((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))-(np.exp(-df['Maturity']/λ))))
df.style.format({'Maturity': '{:,.0f}'.format,'Yield': '{:,.2%}','NS': '{:,.2%}'})
df.head()


#Visualisation des taux calculé à partir de la fonction de Nelson Siegel sans ajustement

df1 = df.copy()
df['Y'] = round(df['Yield']*100,4)
df['NS'] =(β0)+(β1*((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))+(β2*((((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))-(np.exp(-df['Maturity']/λ))))
df['N'] = round(df['NS']*100,4)
df2 = df.copy()
df2 = df2.style.format({'Maturity': '{:,.2f}'.format,'Y': '{:,.2%}', 'N': '{:,.2%}'})
import matplotlib.pyplot as plt
import matplotlib.markers as mk
import matplotlib.ticker as mtick
fontsize=15
fig = plt.figure(figsize=(13,7))
plt.title("Modèle Nelson-Siegel non ajusté Vs Donnée de la FED ",fontsize=fontsize)
ax = plt.axes()
ax.set_facecolor("black")
fig.patch.set_facecolor('white')
X = df["Maturity"]
Y = df["Y"]
x = df["Maturity"]
y = df["N"]
ax.plot(x, y, color="orange", label="NS")
plt.scatter(x, y, marker="o", c="orange")
plt.scatter(X, Y, marker="o", c="blue")
plt.xlabel('Maturité',fontsize=fontsize)
plt.ylabel('Interest',fontsize=fontsize)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.xaxis.set_ticks(np.arange(0, 30, 5))
ax.yaxis.set_ticks(np.arange(0, 4, 0.5))
ax.legend(loc="lower right", title="Yield")
plt.grid()
plt.show()



#Ajout de la colonne des erreurs (MSE)

df['Residual'] =  (df['Yield'] - df['NS'])**2
df22 = df[['Maturity','Yield','NS','Residual']]  
df22.style.format({'Maturity': '{:,.0f}'.format,'Yield': '{:,.2%}','NS': '{:,.2%}','Residual': '{:,.9f}'})
df22.head()
df.head()

np.sum(df['Residual'])

#Lancement de l'estimation
def Error(params):
    df = dd.copy()
    df['NS'] =(params[0])+(params[1]*((1-np.exp(-df['Maturity']/params[3]))/(df['Maturity']/params[3])))+(params[2]*((((1-np.exp(-df['Maturity']/params[3]))/(df['Maturity']/params[3])))-(np.exp(-df['Maturity']/params[3]))))
    df['Residual'] =  (df['Yield'] - df['NS'])**2
    error = np.sum(df['Residual'])
    print("[β0, β1, β2, λ]=",params,", SUM:", error)
    return(error)
    
params = fmin(Error, [0.01, 0.00, -0.01, 1.0])

#Les paramètres estimés sont:

β0 = params[0]
β1 = params[1]
β2 = params[2]
λ = params[3]
print("[β0, β1, β2, λ]=", [params[0].round(2), params[1].round(2), params[2].round(2), params[3].round(2)])



#Visualisation de la courbe des taux calculés à partir des paramètres estimés

df = df1.copy()
df['NS'] =(β0)+(β1*((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))+(β2*((((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))-(np.exp(-df['Maturity']/λ))))
sf4 = df.copy()
sf5 = sf4.copy()
sf5['Y'] = round(sf4['Yield']*100,4)
sf5['N'] = round(sf4['NS']*100,4)
sf4 = sf4.style.format({'Maturity': '{:,.2f}'.format,'Yield': '{:,.2%}', 'NS': '{:,.2%}'})
M0 = 0.00
M1 = 3.50

fontsize=15
fig = plt.figure(figsize=(13,7))
plt.title("Modèle de Nelson-Siegel estimé sur les données de la FED ",fontsize=fontsize)
ax = plt.axes()
ax.set_facecolor("black")
fig.patch.set_facecolor('white')
X = sf5["Maturity"]
Y = sf5["Y"]
x = sf5["Maturity"]
y = sf5["N"]
ax.plot(x, y, color="orange", label="NS")
plt.scatter(x, y, marker="o", c="orange")
plt.scatter(X, Y, marker="o", c="blue")
plt.xlabel('Maturité',fontsize=fontsize)
plt.ylabel('Interest',fontsize=fontsize)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.xaxis.set_ticks(np.arange(0, 30, 5))
ax.yaxis.set_ticks(np.arange(0, 4, 0.5))
ax.legend(loc="lower right", title="Yield")
plt.grid()
plt.show()



#Visualisation de la courbe du spread des  taux calculés et des taux de la FED

df['spread']= df['Yield'] - df['NS']

#Afficher les statistiques du spread de taux entre la FED et le NS

print(df['spread'].describe())


#Distribution de la série des spread de taux entre la FED et le NS
plt.figure(figsize=(10, 6))
sns.histplot(df['spread'], kde=True, color='skyblue')
plt.title('Distribution de la série des spread de taux entre la FED et le NS')
plt.xlabel('spread')
plt.ylabel('Densité')
plt.show()





###Roule sur plage de date de 3 ans


resultat2 = pd.DataFrame()
resultat2.shape

Params2 = pd.DataFrame()
Params2.shape


Spread = pd.DataFrame()
Spread.shape

n

for i in range(n):
    first = donnees.iloc[i-1,:]/100

    dd = {'Maturity' : time, 'Yield' : first}
    dd = pd.DataFrame(dd)
    df = dd.copy()
    df.style.format({'Maturity': '{:,.0f}'.format,'Yield': '{:,.2%}'})
    
    #Initialisation

    β0 = 0.01
    β1 = 0.01
    β2 = 0.01
    λ = 1.00


    df['NS'] =(β0)+(β1*((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))+(β2*((((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))-(np.exp(-df['Maturity']/λ))))
    df.style.format({'Maturity': '{:,.0f}'.format,'Yield': '{:,.2%}','NS': '{:,.2%}'})
    df.head()
    df['Y'] = round(df['Yield']*100,4)
    df['N'] = round(df['NS']*100,4)
    df['Residual'] =  (df['Yield'] - df['NS'])**2


    df22 = df[['Maturity','Yield','NS','Residual']]  
    df22.style.format({'Maturity': '{:,.0f}'.format,'Yield': '{:,.2%}','NS': '{:,.2%}','Residual': '{:,.9f}'})
    df22.head()
    df.head()

    np.sum(df['Residual'])


    #Lancement de l'estimation
    def Error(params):
        df = dd.copy()
        df['NS'] =(params[0])+(params[1]*((1-np.exp(-df['Maturity']/params[3]))/(df['Maturity']/params[3])))+(params[2]*((((1-np.exp(-df['Maturity']/params[3]))/(df['Maturity']/params[3])))-(np.exp(-df['Maturity']/params[3]))))
        df['Residual'] =  (df['Yield'] - df['NS'])**2
        error = np.sum(df['Residual'])
        print("[β0, β1, β2, λ]=",params,", SUM:", error)
        return(error)
        
    params = fmin(Error, [0.01, 0.00, -0.01, 1.0])

    #Les paramètres estimés sont:

    β0 = params[0]
    β1 = params[1]
    β2 = params[2]
    λ = params[3]

    #Calcul des taux avec les paramètres estimés
    df['NS'] =(β0)+(β1*((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))+(β2*((((1-np.exp(-df['Maturity']/λ))/(df['Maturity']/λ)))-(np.exp(-df['Maturity']/λ))))

    #Calcul du spread des taux FED et NS
    df['spread']= df['Yield'] - df['NS']

    # Créer un dictionnaire contenant les valeurs de la nouvelle ligne des paramètres
    nouvelle_ligne_param = {'β0': β0, 'β1': β1, 'β2': β2, 'λ': λ}
    
    # Ajout de la nouvelle ligne au DataFrame Params en utilisant la méthode append()
    Params2 = Params2.append(nouvelle_ligne_param, ignore_index=True)
    
    # Ajout de la nouvelle ligne au DataFrame Params en utilisant la méthode append()
    resultat2 = resultat2.append(df['NS'], ignore_index=True)

    # Ajout de la nouvelle ligne au DataFrame Spread en utilisant la méthode append()
    Spread = Spread.append(df['spread'], ignore_index=True)

Params2.shape
resultat2.shape
Spread.shape



#Afficher les statistiques du spread de taux entre la FED et le NS

#Par échéance
print(Spread.describe())

#Par date
print(Spread.T.describe())



#Calculons enfin la moyenne par colonne des taux estimés de la FED

moyenne_fed = resultat2.mean()

#Nous allons enfin comparer les deux moyennes, en explorant leurs statistiques et leur distribution respective

#Pour l'échantillon
moyenne_echantillon
print(moyenne_echantillon.describe())

#Distribution de la série des moyennes de taux de l'échantillon
plt.figure(figsize=(10, 6))
sns.histplot(moyenne_echantillon, kde=True, color='skyblue')
plt.title('Distribution de la série des moyennes de taux de l échantillon estimé avec Svenson')
plt.xlabel('Time')
plt.ylabel('Moyenne')
plt.show()


#Pour la fed
moyenne_fed
print(moyenne_fed.describe())

#Distribution de la série des moyennes de taux de la FED 
plt.figure(figsize=(10, 6))
sns.histplot(moyenne_fed, kde=True, color='skyblue')
plt.title('Distribution de la série des moyennes de taux de la FED estimé avec Nelson Siegel')
plt.xlabel('Time')
plt.ylabel('Moyenne')
plt.show()




# Tracer les séries temporelles
plt.plot( moyenne_fed.values, label='moyenne_fed')
plt.plot( moyenne_echantillon.values, label='moyenne_echantillon')

# Ajouter une légende
plt.legend()

# Ajouter un titre et des étiquettes d'axe
plt.title('Graphique des moyennes des taux de la FED et de notre échantillon WRDS')
plt.xlabel('Date')
plt.ylabel('Moyennes')

# Afficher le graphique
plt.show()



#Sauvegarde du fichier des paramètres et des spread de taux

Resultat2 = pd.concat([fichier1['Time'],resultat2, Params2, Spread], axis = 1)
Resultat2.head()
Resultat2.to_excel('output/data_spread.xlsx', index=False)






######################################################################################
# Question 3: Discussion de deux changement majeur dans la structure à terme des taux#
######################################################################################



resultat2.shape

# Diviser le DataFrame en trois parties en fonction des échéances
court_terme = resultat2.iloc[:, :9]
intermediaire = resultat2.iloc[:, 10:19]
long_terme = resultat2.iloc[:, 20:29]

                    

# Tracer l'évolution de chacune des parties dans le temps
plt.figure(figsize=(10, 6))

# Courbe pour les obligations à court terme
plt.plot(court_terme.index, court_terme.mean(axis=1), label='Court Terme')

# Courbe pour les obligations intermédiaires
plt.plot(intermediaire.index, intermediaire.mean(axis=1), label='Intermédiaire')

# Courbe pour les obligations à long terme
plt.plot(long_terme.index, long_terme.mean(axis=1), label='Long Terme')

# Ajouter une légende, un titre et des étiquettes d'axe
plt.legend()
plt.title('Évolution des taux d\'obligations par échéance')
plt.xlabel('Date')
plt.ylabel('Taux d\'obligation')
plt.grid(True)
plt.show()




######################################################################################
# Question 4: Analyse de l'évolution de la structure à terme des spreads de crédit   #
# pour les obligations à haut rendement par rapport aux obligations                  #
# de première qualité de 2007 à 2010                                                 #
######################################################################################



# Chargement des données du fichier CSV dans un DataFrame

# Lecture des données à partir de fichiers CSV de la FED

fichier3 = pd.read_csv('data/trace.csv')

fichier3.shape


# Sélection des colonnes pertinentes
filtre3 = fichier3[['TRD_EXCTN_DT', 'YLD_SPREAD', 'RATING_1','YLD_PT']]
filtre3.shape

# Convertions la colonne 'TRD_EXCTN_DT' en datetime
filtre3['TRD_EXCTN_DT'] = pd.to_datetime(filtre3['TRD_EXCTN_DT'], format='%Y%m%d')
filtre3.head()
filtre3['TRD_EXCTN_DT'].head()

donnees = filtre3[(filtre3['TRD_EXCTN_DT'].dt.year >= 2007) & (filtre3['TRD_EXCTN_DT'].dt.year <= 2010)]

donnees.shape
donnees.head()

df = donnees.tail(1500000)


df.shape
df.head()



# Divisions des données en deux groupes : obligations à haut rendement et obligations de première qualité


# Tri les données dans l'ordre décroissant
df_sorted_by_yield = df.sort_values(by='YLD_PT', ascending=False)
df_sorted_by_rate = df.sort_values(by='RATING_1', ascending=False)

# Calcul de l'index correspondant au 20e percentile
index_20_percentile_yield = int(0.2 * len(df_sorted_by_yield))
index_20_percentile_rate = int(0.2 * len(df_sorted_by_rate))


# Sélectionner les données jusqu'à cet index
yield_20_percent = df_sorted_by_yield.head(index_20_percentile_yield) #obligations_haut_rendement
rate_20_percent = df_sorted_by_rate.head(index_20_percentile_rate) #obligations_premiere_qualite



# Calcul des moyennes des spreads de crédit pour chaque groupe pour chaque année
moyennes_haut_rendement = yield_20_percent.groupby(yield_20_percent['TRD_EXCTN_DT'].dt.year)['YLD_SPREAD'].mean()
moyennes_premiere_qualite = rate_20_percent.groupby(rate_20_percent['TRD_EXCTN_DT'].dt.year)['YLD_SPREAD'].mean()


#Statistique descriptive de chaque groupe

moyennes_haut_rendement.describe()

moyennes_premiere_qualite.describe()



# Visualiser l'évolution des moyennes des spreads de crédit au fil du temps pour chaque groupe
plt.figure(figsize=(10, 6))
plt.plot(moyennes_haut_rendement, label='Obligations à haut rendement', linestyle='-')
plt.plot(moyennes_premiere_qualite, label='Obligations de première qualité', linestyle='-')
plt.xlabel('Année')
plt.ylabel('Moyenne des spreads de crédit')
plt.title("Évolution de la structure à terme des spreads de crédit (2007-2010)")
plt.legend()
plt.grid(True)
plt.show()

